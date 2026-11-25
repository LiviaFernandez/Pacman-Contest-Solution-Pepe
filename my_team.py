# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import os
import random

import contest.distance_calculator as distance_calculator
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions, Configuration
from contest.util import nearest_point

SIGHT_RANGE = 5 #maximum visibility range according to rules
MIN_FOOD = 2 # according to rules it is over when all food but 2 is eaten
UNSEEN_SLACK = 3      # margin of error for unseen ghots  
_DEBUG = os.environ.get('PACMAN_DEBUG') == '1'   # solo para pruebas locales


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##################################
# info compartida por los dos agentes
##################################

class TeamInfo:
    def __init__(self):
        self.registered = set()
        self.ready = False
        self.beliefs = {}
        self.last_defended_food = None
        self.last_defended_caps = None
        self.last_eaten = None          # (pos, turno) de la ultima comida nuestra que se comio el rival
        self.turn = 0
        self.targets = {}               # indice agente -> casilla objetivo actual
        self.invader_seen_turn = -float('inf')   # ultimo turno con algun invasor
        self.helping = {}               # indice agente -> ultimo turno en que ayudo a defender
        self.last_pacman_turn = {}      # rival -> ultimo turno en que estuvo en nuestro campo


_TEAMS = {}


##########
# Agents #
##########

class BaseAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None


    def register_initial_state(self, game_state):

        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)

        team = _TEAMS.get(self.red) #shared memory

        if team is None or self.index in team.registered:
            team = TeamInfo()            #crea una partida nueva
            _TEAMS[self.red] = team

        team.registered.add(self.index) #luego simplemente se añade
        self.team = team

        if not team.ready: #analiza el mapa, una vez por equipo (con team.ready)
            self._precompute(game_state)
            team.ready = True
        self.walls, self.cells, self.mid = team.walls, team.cells, team.mid #get info
        self.home_border, self.home_distancer = team.home_border, team.home_distancer
        self.depth, self.exit = team.depth, team.exit

        self.opponents = self.get_opponents(game_state)
        self.teammate = [i for i in self.get_team(game_state) if i != self.index][0]
        self.opp_start = {o: game_state.get_initial_agent_position(o) for o in self.opponents}
        for opp in self.opponents:
            # al empezar, cada rival esta seguro en su casilla de salida
            team.beliefs.setdefault(opp, util.Counter({self.opp_start[opp]: 1.0}))
        self.initial_enemy_food = len(self.get_food(game_state).as_list())

        self.last_positions = []
        self.stuck = 0
        self.last_carrying = 0
        self.help_streak = 0
        self.help_cooldown = 0

    #helper function: studies the map at the start of the game and saves the results in the team's shared memory
    def _precompute(self, gs):
        t = self.team
        walls = gs.get_walls()
        t.walls = walls
        t.mid = walls.width / 2.0        # middle line of the grid 
        t.cells = walls.as_list(False)  #open cells

        # Distance inside our camp (defender diesn't leave)
        home_layout = gs.data.layout.deep_copy()
        for x in range(walls.width):
            if (x < t.mid) != self.red: #next cellis  from rival
                for y in range(walls.height):
                    home_layout.walls[x][y] = True
        t.home_distancer = distance_calculator.Distancer(home_layout)
        t.home_distancer.get_maze_distances()

        red_x = (walls.width - 1) // 2      # last red colu
        blue_x = (walls.width + 1) // 2     # first blue col
        bx, ox = (red_x, blue_x) if self.red else (blue_x, red_x)
        t.home_border = [(bx, y) for y in range(walls.height)
                         if not walls[bx][y] and not walls[ox][y]]

        # no way out
        # get_legal_neighbors has its own cell
        nbrs = {c: [n for n in Actions.get_legal_neighbors(c, walls) if n != c] for c in t.cells}
        deg = {c: len(nbrs[c]) for c in t.cells}
        removed = set()
        q = util.Queue()
        for c in t.cells:
            if deg[c] <= 1:
                q.push(c)
        while not q.is_empty():
            c = q.pop()
            if c in removed:
                continue
            removed.add(c)
            for n in nbrs[c]:
                if n not in removed:
                    deg[n] -= 1
                    if deg[n] <= 1:
                        q.push(n)
        depth = {c: 0 for c in t.cells}
        exit_ = {c: c for c in t.cells}
        if len(removed) < len(t.cells):
            q = util.Queue()
            for c in t.cells:
                if c not in removed:
                    for n in nbrs[c]:
                        if n in removed and exit_[n] == n:
                            depth[n] = 1
                            exit_[n] = c
                            q.push(n)
            while not q.is_empty():
                c = q.pop()
                for n in nbrs[c]:
                    if n in removed and exit_[n] == n:
                        depth[n] = depth[c] + 1
                        exit_[n] = exit_[c]
                        q.push(n)
        t.depth, t.exit = depth, exit_

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------
    def hdist(self, a, b):
        #Distance without leaving our camp, (si no hay camino, la normal + penalizacion)
        if self.on_home_side(a) and self.on_home_side(b):
            d = self.home_distancer.get_distance(a, b)
            if d < 10 ** 6: 
                return d
        return self.get_maze_distance(a, b) + 50

    def on_home_side(self, pos):
        return (pos[0] < self.mid) == self.red

    def home_dist(self, pos):
        return min(self.get_maze_distance(pos, b) for b in self.home_border)

    # ------------------------------------------------------------------
    # Seguimiento de rivales
    # ------------------------------------------------------------------
    def update_beliefs(self, gs):
        team = self.team
        team.turn += 1
        my_pos = gs.get_agent_position(self.index)
        mate_pos = gs.get_agent_position(self.teammate)
        noisy = gs.get_agent_distances()
        mover = (self.index - 1) % 4   # el rival que acaba de mover

        # 1. Prediccion: el rival que movio se difunde a sus vecinos
        if mover in team.beliefs:
            new = util.Counter()
            for p, pr in team.beliefs[mover].items():
                nb = [n for n in Actions.get_legal_neighbors(p, self.walls) if n != p]
                share = pr / (len(nb) + 1)
                new[p] += share
                for n in nb:
                    new[n] += share
            team.beliefs[mover] = new

        # 2. Comida nuestra que ha desaparecido -> el rival que movio esta ahi
        defended = set(self.get_food_you_are_defending(gs).as_list())
        caps = set(self.get_capsules_you_are_defending(gs))
        eaten_spot = None
        if team.last_defended_food is not None:
            eaten = (team.last_defended_food - defended) | (team.last_defended_caps - caps)
            if eaten:
                likely = team.beliefs[mover].arg_max() if mover in team.beliefs else None
                eaten_spot = min(eaten, key=lambda p: self.get_maze_distance(p, likely) if likely else 0)
                team.last_eaten = (eaten_spot, team.turn)
        team.last_defended_food = defended
        team.last_defended_caps = caps

        # 3. Observacion
        for opp in self.opponents:
            st = gs.get_agent_state(opp)
            if st.is_pacman:
                team.last_pacman_turn[opp] = team.turn
            pos = gs.get_agent_position(opp)
            if pos is not None:
                b = util.Counter()
                b[pos] = 1.0
                team.beliefs[opp] = b
                continue
            if opp == mover and eaten_spot is not None and st.is_pacman:
                b = util.Counter()
                b[eaten_spot] = 1.0
                team.beliefs[opp] = b
                continue
            reading = noisy[opp] if noisy else None
            b = self._observe(gs, team.beliefs[opp], st.is_pacman, my_pos, mate_pos, reading)
            if not b:
                # Creencia vacia: casi siempre porque el rival murio y reaparecio
                start = self.opp_start[opp]
                seed = util.Counter()
                for c in self.cells:
                    if self.get_maze_distance(c, start) <= 4:
                        seed[c] = 1.0
                b = self._observe(gs, seed, st.is_pacman, my_pos, mate_pos, reading)
                if not b:
                    seed = util.Counter()
                    seed.increment_all(self.cells, 1.0)
                    b = self._observe(gs, seed, st.is_pacman, my_pos, mate_pos, reading)
                if not b:
                    b = util.Counter()
                    b.increment_all([c for c in self.cells if self.on_home_side(c) == st.is_pacman] or [start], 1.0)
                    b.normalize()
            team.beliefs[opp] = b

    def _observe(self, gs, belief, is_pacman, my_pos, mate_pos, reading):
        out = util.Counter()
        for p, pr in belief.items():
            if pr <= 0:
                continue
            if self.on_home_side(p) != is_pacman:
                continue
            d_me = util.manhattan_distance(p, my_pos)
            if d_me <= SIGHT_RANGE or (mate_pos and util.manhattan_distance(p, mate_pos) <= SIGHT_RANGE):
                continue
            if reading is not None and gs.get_distance_prob(d_me, reading) == 0:
                continue
            out[p] = pr
        if out.total_count() <= 0:
            return util.Counter()
        out.normalize()
        # se poda lo muy improbable para que la creencia siga siendo pequena
        pruned = util.Counter()
        for p, v in out.items():
            if v > 1e-4:
                pruned[p] = v
        pruned.normalize()
        return pruned

    def estimate(self, gs, opp):
        """Posicion exacta si se ve; si no, la mas probable segun la creencia."""
        pos = gs.get_agent_position(opp)
        if pos is not None:
            return pos, True
        b = self.team.beliefs.get(opp)
        if not b:
            return self.opp_start[opp], False
        return b.arg_max(), False

    # ------------------------------------------------------------------
    # Accion
    # ------------------------------------------------------------------
    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        try:
            self.update_beliefs(game_state)
            action = self.decide(game_state)
            if action not in legal:
                action = None
        except Exception:
            if _DEBUG:
                raise
            action = None
        if action is None:
            moving = [a for a in legal if a != Directions.STOP]
            action = random.choice(moving or legal)
        self.last_positions = (self.last_positions + [game_state.get_agent_position(self.index)])[-12:]
        return action


    # ------------------------------------------------------------------
    # Piezas comunes
    # ------------------------------------------------------------------
    def threats(self, gs):
        """Fantasmas rivales (en su campo) como lista de (pos, timer_susto, visible)."""
        res = []
        for opp in self.opponents:
            st = gs.get_agent_state(opp)
            if st.is_pacman:
                continue
            pos, seen = self.estimate(gs, opp)
            res.append((pos, st.scared_timer, seen))
        return res

    def danger_map(self, threats):
        """danger[c] = primer instante en que un fantasma no asustado puede estar en la casilla c."""
        danger = {c: float('inf') for c in self.cells}
        for pos, scared, seen in threats:
            slack = 0 if seen else UNSEEN_SLACK   # un fantasma no visto esta mal localizado
            for c in self.cells:
                v = max(self.get_maze_distance(pos, c), scared) + slack
                if v < danger[c]:
                    danger[c] = v
        return danger

    def safe_bfs(self, start, danger, slack=0):
        """BFS desde start; devuelve {casilla: (t, primera_accion)} alcanzables con seguridad.
        slack > 0 acepta casillas a las que el fantasma podria llegar un poco antes (riesgo)."""
        visited = {start: (0, None)}
        q = util.Queue()
        q.push(start)
        while not q.is_empty():
            c = q.pop()
            t, a0 = visited[c]
            nt = t + 1
            for a in Actions.get_possible_actions(Configuration(c, Directions.STOP), self.walls):
                if a == Directions.STOP:
                    continue
                n = nearest_point(Actions.get_successor(c, a))
                if n in visited:
                    continue
                if not self.on_home_side(n) and danger[n] <= nt - slack:
                    continue
                visited[n] = (nt, a if a0 is None else a0)
                q.push(n)
        return visited

    def escape_margin(self, pos, t, danger, capsules):
        """Margen (en pasos) con el que, llegando a pos en t, seguimos por delante
        de cualquier fantasma. Si pos esta en un callejon, lo que cuenta es poder
        salir del callejon antes de que un fantasma llegue a su entrada."""
        if self.on_home_side(pos):
            return float('inf')
        if self.depth[pos] > 0:
            m = danger[self.exit[pos]] - (t + self.depth[pos])
        else:
            m = danger[pos] - t
        # una capsula cercana tambien sirve de salida
        if m < 3:
            for c in capsules:
                m = max(m, danger[c] - (t + self.get_maze_distance(pos, c)))
        return m

    def evade(self, gs, threats):
        """Ultimo recurso: alejarse de los fantasmas, evitando callejones."""
        my_pos = gs.get_agent_position(self.index)
        best, best_score = None, -float('inf')
        active = [p for p, s, _ in threats if s <= 1]
        for a in gs.get_legal_actions(self.index):
            nxt = nearest_point(Actions.get_successor(my_pos, a))
            d = min([self.get_maze_distance(nxt, p) for p in active] or [20])
            score = d * 10 - self.depth[nxt] * 3
            if d <= 1:
                score -= 1000
            if self.on_home_side(nxt):
                score += 500
            if a == Directions.STOP:
                score -= 5
            score -= self.home_dist(nxt) * 0.5
            if score > best_score:
                best, best_score = a, score
        return best

    def invaders(self, gs):
        res = []
        for opp in self.opponents:
            if gs.get_agent_state(opp).is_pacman:
                pos, seen = self.estimate(gs, opp)
                res.append((opp, pos, seen))
        return res

    def step_towards(self, gs, target, avoid_enemy_side=False, keep_dist=None):
        """Accion que minimiza la distancia de laberinto a target."""
        my_pos = gs.get_agent_position(self.index)
        best, best_score = [], float('inf')
        for a in gs.get_legal_actions(self.index):
            nxt = nearest_point(Actions.get_successor(my_pos, a))
            if avoid_enemy_side and not self.on_home_side(nxt):
                continue
            d = self.hdist(nxt, target) if avoid_enemy_side else self.get_maze_distance(nxt, target)
            if keep_dist is not None:
                score = abs(d - keep_dist) * 10 + (1000 if d < keep_dist else 0)
            else:
                score = d
            if a == Directions.STOP:
                score += 0.5
            if score < best_score:
                best, best_score = [a], score
            elif score == best_score:
                best.append(a)
        return random.choice(best) if best else None

    def chase_invader(self, gs, inv_pos, role='auto', inv_carry=0):
        """Perseguir/interceptar a un invasor quedandose en casa.
        role: 'auto' (perseguir / control de zona) o 'cut' (cerrar la pinza)."""
        my_pos = gs.get_agent_position(self.index)
        my_state = gs.get_agent_state(self.index)
        if my_state.scared_timer > 0:
            return self.step_towards(gs, inv_pos, avoid_enemy_side=True, keep_dist=2)
        d_me = self.hdist(my_pos, inv_pos)
        if d_me <= 2 and role != 'cut':
            return self.step_towards(gs, inv_pos, avoid_enemy_side=True)
        if role == 'cut':
            # Pinza: bloquear la via de escape del invasor por el lado opuesto al
            # companero que lo persigue (casilla a la que llego antes que el invasor).
            chaser = gs.get_agent_position(self.teammate)
            best, best_d = None, float('inf')
            for c in self.cells:
                if not self.on_home_side(c):
                    continue
                di = self.get_maze_distance(inv_pos, c)
                if di < 1 or di > 10:
                    continue
                if self.hdist(chaser, c) <= di:
                    continue          # ese lado ya lo cubre el companero
                if self.hdist(my_pos, c) > di:
                    continue          # no llego a tiempo
                if di < best_d:
                    best, best_d = c, di
            if best is not None:
                return self.step_towards(gs, best, avoid_enemy_side=True)
            exit_b = min(self.home_border, key=lambda b: self.hdist(inv_pos, b))
            return self.step_towards(gs, exit_b, avoid_enemy_side=True)
        # Control de zona: ir a donde menos comida quede "segura" para el invasor
        # (comida a la que el llega con >= 2 pasos de ventaja sobre nosotros) y,
        # si lleva comida, donde menos salidas a su casa le queden libres.
        food_d = [(f, self.get_maze_distance(inv_pos, f)) for f in self.get_food_you_are_defending(gs).as_list()]
        cap_d = [(c, self.get_maze_distance(inv_pos, c)) for c in self.get_capsules_you_are_defending(gs)]
        border_d = [(b, self.get_maze_distance(inv_pos, b)) for b in self.home_border]
        best, best_score = [], float('inf')
        for a in gs.get_legal_actions(self.index):
            n = nearest_point(Actions.get_successor(my_pos, a))
            if not self.on_home_side(n):
                continue
            free_food = sum(1 for f, di in food_d if self.hdist(n, f) - di >= 2)
            free_caps = sum(1 for c, di in cap_d if self.hdist(n, c) - di >= 1)
            free_exits = sum(1 for b, di in border_d if self.hdist(n, b) - di >= 1)
            score = free_food * 3 + free_caps * 8 + self.hdist(n, inv_pos) * 0.6
            if inv_carry > 0:
                score += free_exits * (0.5 + 0.3 * min(inv_carry, 10))
            if a == Directions.STOP:
                score += 0.3
            if score < best_score - 1e-9:
                best, best_score = [a], score
            elif abs(score - best_score) <= 1e-9:
                best.append(a)
        if best:
            return random.choice(best)
        return self.step_towards(gs, inv_pos, avoid_enemy_side=True)

    def defenders_plan(self, gs, helpers):
        """Reparto de papeles cuando hay invasores. helpers: indices de nuestros
        agentes que estan defendiendo. Devuelve accion para self o None."""
        invs = self.invaders(gs)
        if not invs:
            return None
        my_pos = gs.get_agent_position(self.index)
        # invasor principal: visto y con mas comida, o el mas cercano
        invs.sort(key=lambda x: (0 if x[2] else 1, -gs.get_agent_state(x[0]).num_carrying,
                                 self.hdist(my_pos, x[1])))
        if len(helpers) >= 2 and len(invs) >= 2:
            # cada uno al suyo: el reparto con menor distancia total
            other = [h for h in helpers if h != self.index][0]
            opos = gs.get_agent_position(other)
            a, b = invs[0], invs[1]
            cost_ab = self.hdist(my_pos, a[1]) + self.hdist(opos, b[1])
            cost_ba = self.hdist(my_pos, b[1]) + self.hdist(opos, a[1])
            target = a if cost_ab <= cost_ba else b
            return self.chase_invader(gs, target[1], 'auto', gs.get_agent_state(target[0]).num_carrying)
        opp, ipos, seen = invs[0]
        carry = gs.get_agent_state(opp).num_carrying
        if len(helpers) >= 2:
            other = [h for h in helpers if h != self.index][0]
            opos = gs.get_agent_position(other)
            # el mas cercano persigue, el otro corta la salida
            if self.hdist(my_pos, ipos) <= self.hdist(opos, ipos):
                return self.chase_invader(gs, ipos, 'auto', carry)
            return self.chase_invader(gs, ipos, 'cut', carry)
        return self.chase_invader(gs, ipos, 'auto', carry)


class OffensiveAgent(BaseAgent):
    """Atacante con planificacion segura en el tiempo."""

    def decide(self, gs):
        my_state = gs.get_agent_state(self.index)
        my_pos = gs.get_agent_position(self.index)
        team = self.team

        # ¿Ayudar a defender? (pinza con el defensor), con duracion limitada
        if self.help_streak > 40:
            self.help_cooldown = 60
            self.help_streak = 0
        if self.help_cooldown > 0:
            self.help_cooldown -= 1
        helping_now = False
        if my_state.scared_timer == 0 and my_state.num_carrying <= 2 and self.help_cooldown == 0:
            mate_pos = gs.get_agent_position(self.teammate)
            for opp, ipos, seen in self.invaders(gs):
                carry = gs.get_agent_state(opp).num_carrying
                d = self.get_maze_distance(my_pos, ipos)
                d_mate = self.get_maze_distance(mate_pos, ipos)
                close_help = (not my_state.is_pacman) and d <= 6 and (carry >= 1 or d_mate > 5)
                big_threat = carry >= 4 and d <= 14 and d_mate > 2
                if close_help or big_threat:
                    helping_now = True
                    self.help_streak += 1
                    team.helping[self.index] = team.turn
                    if my_state.is_pacman:
                        return self.attack(gs, force_home=True)
                    helpers = [self.index]
                    if not gs.get_agent_state(self.teammate).is_pacman:
                        helpers.append(self.teammate)
                    return self.defenders_plan(gs, helpers)
        if not helping_now:
            self.help_streak = 0

        return self.attack(gs)

    def attack(self, gs, force_home=False):
        my_state = gs.get_agent_state(self.index)
        my_pos = gs.get_agent_position(self.index)
        team = self.team
        threats = self.threats(gs)
        danger = self.danger_map(threats)
        capsules = self.get_capsules(gs)
        food = self.get_food(gs).as_list()
        carrying = my_state.num_carrying
        moves_left = gs.data.timeleft // 4

        reach = self.safe_bfs(my_pos, danger)
        if self.stuck >= 20 and carrying == 0:
            # bloqueado mucho tiempo sin nada encima: aceptar pasar cerca de fantasmas
            reach = self.safe_bfs(my_pos, danger, slack=min(3, 1 + (self.stuck - 20) // 10))

        # Proximidad del peligro (fantasmas activos)
        active = [(p, s, seen) for p, s, seen in threats if s <= 3]
        near_threat = min([self.get_maze_distance(my_pos, p) for p, s, seen in active if seen] or [float('inf')])

        # Tiempo seguro para volver a casa
        home_opts = [(reach[b][0], b) for b in self.home_border if b in reach]
        home_t, home_b = min(home_opts) if home_opts else (float('inf'), None)

        # Cuantas tenemos ya en casa + encima
        returned = sum(gs.get_agent_state(i).num_returned for i in self.get_team(gs))
        enough_to_win = returned + carrying >= self.initial_enemy_food - MIN_FOOD

        # --- Decidir si volver a casa ---
        go_home = force_home
        if carrying > 0:
            if len(food) <= MIN_FOOD or enough_to_win:
                go_home = True
            elif moves_left <= self.home_dist(my_pos) + 6:
                go_home = True
            elif near_threat <= 4 and carrying >= 2:
                go_home = True
            elif carrying >= self.carry_limit(gs, near_threat, len(active)):
                go_home = True

        # --- Elegir comida ---
        best_food, best_score, best_margin = None, float('inf'), -float('inf')
        # Si llevamos varios turnos sin objetivo, aceptamos algo de riesgo
        need = 2 + min(carrying, 6) // 3
        if self.stuck >= 4 and carrying <= 2:
            need = 1 if self.stuck < 12 else 0
            if carrying == 0 and self.stuck >= 20:
                need = -float('inf')     # sin nada que perder: arriesgar
        if not go_home:
            mate_target = team.targets.get(self.teammate)
            for f in food:
                if f not in reach:
                    continue
                t, _ = reach[f]
                margin = self.escape_margin(f, t, danger, capsules)
                if margin < need:
                    continue
                # no podemos volver a tiempo antes del final
                if t + self.home_dist(f) >= moves_left - 2:
                    continue
                score = t - 1.5 * min(margin, 6)
                if need == -float('inf'):
                    score = t - 4 * min(margin, 6)
                if self.depth[f] > 0 and active:
                    score += self.depth[f]
                if mate_target is not None and self.get_maze_distance(f, mate_target) <= 3:
                    score += 6
                if carrying > 0:
                    score += 0.15 * carrying * self.home_dist(f) / 3.0
                if score < best_score:
                    best_food, best_score, best_margin = f, score, margin

            # Capsula cuando hay amenaza cerca
            if active and near_threat <= 6:
                for c in capsules:
                    if c in reach and reach[c][0] - 4 < best_score:
                        best_food, best_score, best_margin = c, reach[c][0] - 4, float('inf')

            # Comer un fantasma asustado si esta muy a mano
            for p, s, seen in threats:
                if seen and s > 2 and p in reach and reach[p][0] <= 2 and reach[p][0] < s:
                    best_food, best_score, best_margin = p, -10, float('inf')

            # Si llevamos mucho y la comida esta mas lejos que casa, asegurar puntos
            if best_food is not None and carrying >= 3 and home_t < float('inf') and best_score > home_t + 4 and active:
                go_home = True

            if best_food is None and (carrying > 0 or my_state.is_pacman):
                go_home = True

        normal_need = 2 + min(carrying, 6) // 3
        if best_food is None and not go_home:
            self.stuck += 1
        elif go_home or carrying > self.last_carrying:
            self.stuck = 0
        elif best_food is not None and best_margin >= normal_need:
            self.stuck = 0
        self.last_carrying = carrying
        if not go_home and best_food is not None:
            team.targets[self.index] = best_food
            return reach[best_food][1] or self.step_towards(gs, best_food)

        team.targets.pop(self.index, None)

        if go_home:
            caps_r = [(reach[c][0], c) for c in capsules if c in reach]
            if home_b is not None:
                # ¿es mejor una capsula?
                if active and near_threat <= 3 and caps_r and min(caps_r)[0] < home_t:
                    return reach[min(caps_r)[1]][1]
                if reach[home_b][1] is not None:
                    return reach[home_b][1]
            if caps_r:
                return reach[min(caps_r)[1]][1]
            if my_state.is_pacman:
                # Sin ruta segura: probar ignorando fantasmas no vistos y luego con menos margen
                dvis = self.danger_map([th for th in threats if th[2]])
                for slack in (0, 1):
                    r2 = self.safe_bfs(my_pos, dvis, slack=slack)
                    goals = [(r2[b][0], b) for b in self.home_border + list(capsules) if b in r2]
                    if goals and r2[min(goals)[1]][1] is not None:
                        return r2[min(goals)[1]][1]
                return self.evade(gs, threats)

        # En casa sin comida segura: buscar un hueco en la frontera
        return self.probe_border(gs, danger)

    def carry_limit(self, gs, near_threat, n_ghosts=1):
        if near_threat <= 6:
            return 3
        if n_ghosts == 0:
            return 12          # campo rival sin defensores
        base = 4 if len(self.get_food(gs).as_list()) < 8 else 7
        if self.get_score(gs) < 0:
            base += 2
        return base

    def probe_border(self, gs, danger):
        """Elegir punto de cruce: lejos de los fantasmas y cerca de mi."""
        my_pos = gs.get_agent_position(self.index)
        best, best_score = None, -float('inf')
        for b in self.home_border:
            nb = [n for n in Actions.get_legal_neighbors(b, self.walls) if n != b and not self.on_home_side(n)]
            dg = min([danger[n] for n in nb] + [15])
            score = dg * 2 - self.get_maze_distance(my_pos, b)
            if b in self.last_positions and my_pos != b:
                score -= 1
            if score > best_score:
                best, best_score = b, score
        if best is None:
            return None
        return self.step_towards(gs, best, avoid_enemy_side=True)


class DefensiveAgent(OffensiveAgent):
    """Defensor: protege la comida propia. Solo sale a atacar si el rival lleva
    mucho tiempo sin atacar y no vamos ganando (para romper bloqueos)."""

    def decide(self, gs):
        my_pos = gs.get_agent_position(self.index)
        my_state = gs.get_agent_state(self.index)
        team = self.team
        if self.invaders(gs):
            team.invader_seen_turn = team.turn
            if my_state.is_pacman:
                # estabamos atacando: volver a casa a defender
                return self.attack(gs, force_home=True)
            helpers = [self.index]
            mate_state = gs.get_agent_state(self.teammate)
            if team.helping.get(self.teammate, -float('inf')) >= team.turn - 2 and not mate_state.is_pacman:
                helpers.append(self.teammate)
            return self.defenders_plan(gs, helpers)

        if self.should_raid(gs):
            return self.attack(gs)
        if my_state.is_pacman:
            return self.attack(gs, force_home=True)
        team.targets.pop(self.index, None)

        # Comida recien comida: ir alli
        if team.last_eaten is not None and team.turn - team.last_eaten[1] <= 10:
            spot = team.last_eaten[0]
            if self.get_maze_distance(my_pos, spot) > 1:
                return self.step_towards(gs, spot, avoid_enemy_side=True)

        return self.patrol(gs)

    def should_raid(self, gs):
        """¿Salir a atacar con los dos? Solo si el rival no ataca y no vamos ganando."""
        team = self.team
        if team.turn < 80 or team.turn - team.invader_seen_turn < 60:
            return False
        if self.get_score(gs) > 0 or gs.data.timeleft // 4 < 30:
            return False
        # solo contra equipos que no atacan: ningun rival ha cruzado en mucho tiempo
        return all(team.turn - team.last_pacman_turn.get(opp, 0) >= 100 for opp in self.opponents)

    def patrol(self, gs):
        my_pos = gs.get_agent_position(self.index)
        defended = self.get_food_you_are_defending(gs).as_list()
        # rival que esta mas cerca de entrar
        positions = [self.estimate(gs, opp)[0] for opp in self.opponents]
        entering = min(positions, key=self.home_dist) if positions else None

        def border_value(b):
            v = self.get_maze_distance(b, entering) if entering is not None else 0.0
            if defended:
                # cerca de la comida que defendemos
                v += 0.3 * min(self.get_maze_distance(b, f) for f in defended)
            return v

        target = min(self.home_border, key=border_value)
        if my_pos == target:
            return Directions.STOP
        return self.step_towards(gs, target, avoid_enemy_side=True)
