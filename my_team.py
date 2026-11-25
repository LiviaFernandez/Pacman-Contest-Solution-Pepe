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
from contest.game import Directions

SIGHT = 5            # rules: we only see rivals at <= 5 (manhattan)
MIN_FOOD = 2         # game is over when all food but 2 is eaten
UNSEEN_SLACK = 3     #extra margin for ghosts we dont see (our guess can be wrong)
BIG = 10 ** 6        # infinity
_DEBUG = os.environ.get('PACMAN_DEBUG') == '1'   # only for local tests

# Actions are just N S E W
DIRS = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


# shared memory of our 2 agents
class TeamInfo:
    def __init__(self):
        self.registered = set()       
        self.beliefs = {}                
        self.last_food = None          
        self.last_eaten = None           # (cell, turn) where they ate our food
        self.targets = {}                # agent -> food it goes for (so they dont go to the same)
        self.turn = 0
        self.last_invader_turn = 0       # last turn a rival was on our side


_TEAMS = {} 


##########
# Agents #
##########

class BaseAgent(CaptureAgent):

    # runs once at the start 
    def register_initial_state(self, gs):
        CaptureAgent.register_initial_state(self, gs)   # gives self.red and get_maze_distance

        team = _TEAMS.get(self.red)
        if team is None or self.index in team.registered:
            team = _TEAMS[self.red] = TeamInfo()
        team.registered.add(self.index)
        self.team = team

        self.start = gs.get_agent_position(self.index)
        self.walls = gs.get_walls()
        self.cells = self.walls.as_list(False)   #open cells
        self.mid = self.walls.width / 2.0        # middle line, red is x < mid
        self.opponents = self.get_opponents(gs)
        self.mate = [i for i in self.get_team(gs) if i != self.index][0]
        for o in self.opponents:
            team.beliefs.setdefault(o, util.Counter({gs.get_initial_agent_position(o): 1.0}))
        self.total_food = len(self.get_food(gs).as_list())
        self.stuck = 0     # turns the attacker had nothing safe to eat

        self.study_map(gs)

    # helper function: studies the map at the start and saves what we need
    def study_map(self, gs):
        w = self.walls

        # border = our last column, only where we can cross to the other side
        bx = (w.width - 1) // 2 if self.red else (w.width + 1) // 2
        ox = bx + 1 if self.red else bx - 1
        self.border = [(bx, y) for y in range(w.height) if not w[bx][y] and not w[ox][y]]

        # distances inside our camp (defender doesn't leave): copy of the map with the rival side as wall
        home_layout = gs.data.layout.deep_copy()
        for x in range(w.width):
            if (x < self.mid) != self.red:
                for y in range(w.height):
                    home_layout.walls[x][y] = True
        self.home_distancer = distance_calculator.Distancer(home_layout)
        self.home_distancer.get_maze_distances()

        # dead ends: keep removing cells with 0 or 1 free neighbour until nothing changes
        dead = set()
        changed = True
        while changed:
            changed = False
            for c in self.cells:
                if c not in dead and len([n for n in self.neighbors(c) if n not in dead]) <= 1:
                    dead.add(c)
                    changed = True

        #for each dead end cell: exit = closest normal cell, depth = steps to it
        alive = [c for c in self.cells if c not in dead]
        self.exit, self.depth = {}, {}
        for c in self.cells:
            if c in dead and alive:
                self.exit[c] = min(alive, key=lambda a: self.get_maze_distance(c, a))
                self.depth[c] = self.get_maze_distance(c, self.exit[c])
            else:
                self.exit[c], self.depth[c] = c, 0


    def actions(self, c):
        return [a for a, (dx, dy) in DIRS.items() if not self.walls[c[0] + dx][c[1] + dy]]

    def result(self, c, a):
        dx, dy = DIRS.get(a, (0, 0))    # STOP -> same cell
        return (c[0] + dx, c[1] + dy)

    def neighbors(self, c):
        return [self.result(c, a) for a in self.actions(c)]

    def on_my_side(self, c):
        return (c[0] < self.mid) == self.red

    def home_distance(self, c):
        return min(self.get_maze_distance(c, b) for b in self.border)

    def hdist(self, a, b):
        # distance without leaving our side (if there is no path, normal one + 50)
        if self.on_my_side(a) and self.on_my_side(b):
            d = self.home_distancer.get_distance(a, b)
            if d < BIG:
                return d
        return self.get_maze_distance(a, b) + 50

    def choose_action(self, gs):
        #rational agent: look at the world, update beliefs, then decide
        legal = gs.get_legal_actions(self.index)
        try:
            self.update_beliefs(gs)
            action = self.decide(gs)
        except Exception:
            if _DEBUG:
                raise
            action = None
        if action not in legal:   # something went wrong -> random move so we dont lose the game
            action = random.choice([a for a in legal if a != Directions.STOP] or legal)
        return action

    def decide(self, gs):
        raise NotImplementedError

    # where can the rivals we dont see be?
    def update_beliefs(self, gs):
        t = self.team
        t.turn += 1
        me = gs.get_agent_position(self.index)
        mate = gs.get_agent_position(self.mate)
        noisy = gs.get_agent_distances()
        mover = (self.index - 1) % 4    # the rival that moved just before us

        # 1. he moved: stays or goes to a neighbour, all the same probability 
        new = util.Counter()
        for c, p in t.beliefs[mover].items():
            options = [c] + self.neighbors(c)
            for n in options:
                new[n] += p / len(options)
        t.beliefs[mover] = new

        # 2. if some of our food is gone, he is there
        food = set(self.get_food_you_are_defending(gs).as_list())
        if t.last_food is not None and t.last_food - food:
            spot = (t.last_food - food).pop()
            t.last_eaten = (spot, t.turn)
            t.beliefs[mover] = util.Counter({spot: 1.0})
        t.last_food = food

        # 3. keep only cells that agree with what we see
        for o in self.opponents:
            pos = gs.get_agent_position(o)
            pacman = gs.get_agent_state(o).is_pacman
            if pacman:
                t.last_invader_turn = t.turn
            if pos is not None:      # we see him, easy
                t.beliefs[o] = util.Counter({pos: 1.0})
                continue

            def fits(c):
                d = util.manhattan_distance(c, me)
                return (self.on_my_side(c) == pacman            # pacman = he is on our side
                        and d > SIGHT and util.manhattan_distance(c, mate) > SIGHT   # else we would see him
                        and gs.get_distance_prob(d, noisy[o]) > 0)          
            
            b = util.Counter({c: p for c, p in t.beliefs[o].items() if fits(c)})
            if b.total_count() == 0:
                # lost him, most times he died -> try near his start, then anywhere
                b = util.Counter({c: 1.0 for c in self.cells
                                  if fits(c) and self.get_maze_distance(c, gs.get_initial_agent_position(o)) <= 4})
            if b.total_count() == 0:
                b = util.Counter({c: 1.0 for c in self.cells if fits(c)})
            if b.total_count() == 0:
                b = util.Counter({gs.get_initial_agent_position(o): 1.0})
            b.normalize()
            t.beliefs[o] = b

    def estimate(self, gs, o):
        pos = gs.get_agent_position(o)
        if pos is not None:
            return pos, True
        return self.team.beliefs[o].arg_max(), False

    def ghosts(self, gs):
        # rivals on their side: (pos, scared timer, seen)
        res = []
        for o in self.opponents:
            st = gs.get_agent_state(o)
            if not st.is_pacman:
                pos, seen = self.estimate(gs, o)
                res.append((pos, st.scared_timer, seen))
        return res

    def invaders(self, gs):
        # rivals on our side: (index, pos, seen)
        return [(o,) + self.estimate(gs, o) for o in self.opponents if gs.get_agent_state(o).is_pacman]


    def step_towards(self, gs, target, stay_home=False, keep=None):
        """greedy (lecture 4): one step to the neighbour with the smallest h = maze distance
        keep=k -> try to stay at distance k (when we are scared)"""
        me = gs.get_agent_position(self.index)
        best, best_h = [], BIG
        for a in gs.get_legal_actions(self.index):
            n = self.result(me, a)
            if stay_home and not self.on_my_side(n):
                continue
            h = self.hdist(n, target) if stay_home else self.get_maze_distance(n, target)
            if keep is not None:
                h = abs(h - keep) * 10 + (1000 if h < keep else 0)
            if a == Directions.STOP:
                h += 0.5      #moving is better than waiting
            if h < best_h:
                best, best_h = [a], h
            elif h == best_h:
                best.append(a)
        return random.choice(best) if best else None

    def chase(self, gs, inv_pos, inv_carry):
        """defend vs an invader without crossing"""
        me = gs.get_agent_position(self.index)
        if gs.get_agent_state(self.index).scared_timer > 0:
            return self.step_towards(gs, inv_pos, stay_home=True, keep=2)   # he can eat us, follow from 2
        if self.hdist(me, inv_pos) <= 2:
            return self.step_towards(gs, inv_pos, stay_home=True)            # close -> go for him

        # far: we cant catch him running (same speed) so we stand where he has
        # less "safe" food (food he reaches 2+ steps before us), capsules and exits
        food = [(f, self.get_maze_distance(inv_pos, f)) for f in self.get_food_you_are_defending(gs).as_list()]
        caps = [(c, self.get_maze_distance(inv_pos, c)) for c in self.get_capsules_you_are_defending(gs)]
        exits = [(b, self.get_maze_distance(inv_pos, b)) for b in self.border]
        best, best_score = [], BIG
        for a in gs.get_legal_actions(self.index):
            n = self.result(me, a)
            if not self.on_my_side(n):
                continue
            score = (3 * sum(1 for f, d in food if self.hdist(n, f) - d >= 2)
                     + 8 * sum(1 for c, d in caps if self.hdist(n, c) - d >= 1)
                     + 0.6 * self.hdist(n, inv_pos))
            if inv_carry > 0:   # with food he wants to go home -> block exits
                score += (0.5 + 0.3 * min(inv_carry, 10)) * sum(1 for b, d in exits if self.hdist(n, b) - d >= 1)
            if a == Directions.STOP:
                score += 0.3
            if score < best_score - 1e-9:
                best, best_score = [a], score
            elif abs(score - best_score) <= 1e-9:
                best.append(a)
        return random.choice(best) if best else self.step_towards(gs, inv_pos, stay_home=True)

    def defend(self, gs, invaders):
        # first the ones we see, then the one with more food, then the closest
        me = gs.get_agent_position(self.index)
        o, pos, seen = min(invaders, key=lambda x: (not x[2], -gs.get_agent_state(x[0]).num_carrying,
                                                    self.hdist(me, x[1])))
        return self.chase(gs, pos, gs.get_agent_state(o).num_carrying)


class OffensiveAgent(BaseAgent):
    #goes to eat their food, planning safe paths

    def decide(self, gs):
        me = gs.get_agent_position(self.index)
        st = gs.get_agent_state(self.index)
        # help the defender: if an invader is close to us, or he is taking a lot of our food
        if st.scared_timer == 0 and st.num_carrying <= 2:
            mate = gs.get_agent_position(self.mate)
            for o, pos, seen in self.invaders(gs):
                carry = gs.get_agent_state(o).num_carrying
                d, d_mate = self.get_maze_distance(me, pos), self.get_maze_distance(mate, pos)
                close = not st.is_pacman and d <= 6 and (carry >= 1 or d_mate > 5)
                big_threat = carry >= 4 and d <= 14 and d_mate > 2
                if close or big_threat:
                    if st.is_pacman:
                        return self.attack(gs, go_home=True)   # first come back
                    return self.chase(gs, pos, carry)
        return self.attack(gs)

    # danger[c] = first move a ghost (not scared) could be in cell c
    def danger_map(self, gs):
        danger = {c: BIG for c in self.cells}
        for pos, scared, seen in self.ghosts(gs):
            extra = 0 if seen else UNSEEN_SLACK
            for c in self.cells:
                danger[c] = min(danger[c], max(self.get_maze_distance(pos, c), scared) + extra)
        return danger

    def safe_search(self, start, danger, risk=0):
        found = {start: (0, None)}     # expanded + frontier, so no cell twice
        frontier = util.Queue()
        frontier.push(start)
        while not frontier.is_empty():
            c = frontier.pop()
            moves, first = found[c]
            for a in self.actions(c):
                n = self.result(c, a)
                if n in found:
                    continue
                if not self.on_my_side(n) and danger[n] <= moves + 1 - risk:
                    continue    # ghost could be there first
                found[n] = (moves + 1, first or a)
                frontier.push(n)
        return found

    def margin(self, f, moves, danger, caps):
        if self.depth[f] > 0:
            m = danger[self.exit[f]] - (moves + self.depth[f])
        else:
            m = danger[f] - moves
        if m < 3:   # a capsule close by is also a way out
            for c in caps:
                m = max(m, danger[c] - (moves + self.get_maze_distance(f, c)))
        return m

    def attack(self, gs, go_home=False):
        me = gs.get_agent_position(self.index)
        st = gs.get_agent_state(self.index)
        carrying = st.num_carrying
        food = self.get_food(gs).as_list()
        caps = self.get_capsules(gs)
        moves_left = gs.data.timeleft // 4
        ghosts = self.ghosts(gs)
        active = [g for g in ghosts if g[1] <= 3]                     # not scared (or almost)
        near = min([self.get_maze_distance(me, p) for p, s, seen in active if seen] or [BIG])

        danger = self.danger_map(gs)
        risk = min(3, 1 + (self.stuck - 20) // 10) if self.stuck >= 20 and carrying == 0 else 0
        plan = self.safe_search(me, danger, risk)

        # time to go home?
        returned = sum(gs.get_agent_state(i).num_returned for i in self.get_team(gs))
        if carrying > 0 and (len(food) <= MIN_FOOD
                             or returned + carrying >= self.total_food - MIN_FOOD   # enough to win
                             or moves_left <= self.home_distance(me) + 6            # game ending
                             or (near <= 4 and carrying >= 2)
                             or carrying >= self.carry_limit(gs, near, len(active))):
            go_home = True

        target, m = (None, 0) if go_home else self.pick_food(gs, food, plan, danger, caps, carrying, active)
        if not go_home:
            if active and near <= 6:     # ghost close -> capsule is great (if not much further than the food)
                for c in caps:
                    if c in plan and (target is None or plan[c][0] < plan[target][0] + 4):
                        target, m = c, BIG
            for p, s, seen in ghosts:    # scared ghost right next to us -> eat it
                if seen and s > 2 and p in plan and plan[p][0] <= 2 and plan[p][0] < s:
                    target, m = p, BIG

        # stuck counter (used to take more risk)
        if target is None and not go_home and not st.is_pacman:
            self.stuck += 1
        elif target is None or go_home or m >= 2 + min(carrying, 6) // 3:
            self.stuck = 0

        if target is not None:
            self.team.targets[self.index] = target
            return plan[target][1]
        self.team.targets.pop(self.index, None)
        if go_home or st.is_pacman:
            return self.run_home(gs, plan, caps)
        return self.find_gap(gs, danger)

    def carry_limit(self, gs, near, n_ghosts):
        if near <= 6:
            return 3
        if n_ghosts == 0:
            return 12            # nobody defending
        limit = 4 if len(self.get_food(gs).as_list()) < 8 else 7
        if self.get_score(gs) < 0:
            limit += 2           # losing -> take more
        return limit

    def pick_food(self, gs, food, plan, danger, caps, carrying, active):
        need = 2 + min(carrying, 6) // 3      # margin we want, more food on us = more careful
        if self.stuck >= 4 and carrying <= 2:
            need = 1 if self.stuck < 12 else 0
        if self.stuck >= 20 and carrying == 0:
            need = -BIG                        # nothing to lose
        moves_left = gs.data.timeleft // 4
        mate_target = self.team.targets.get(self.mate)

        best, best_score, best_m = None, BIG, 0
        for f in food:
            if f not in plan:
                continue
            moves = plan[f][0]
            m = self.margin(f, moves, danger, caps)
            if m < need or moves + self.home_distance(f) >= moves_left - 2:
                continue
            # close + safe is good
            score = moves - (4 if need == -BIG else 1.5) * min(m, 6)
            if self.depth[f] > 0 and active:
                score += self.depth[f]
            if mate_target is not None and self.get_maze_distance(f, mate_target) <= 3:
                score += 6                      # dont go to the same food as the mate
            score += 0.05 * carrying * self.home_distance(f)
            if score < best_score:
                best, best_score, best_m = f, score, m
        return best, best_m

    def run_home(self, gs, plan, caps):
        # closest safe border cell or capsule
        goals = [(plan[g][0], g) for g in self.border + caps if g in plan and plan[g][1]]
        if goals:
            return plan[min(goals)[1]][1]
        # no safe path. try again only with the ghosts we see (unseen ones might be wrong)
        me = gs.get_agent_position(self.index)
        seen = [o for o in self.opponents if gs.get_agent_position(o) is not None
                and not gs.get_agent_state(o).is_pacman and gs.get_agent_state(o).scared_timer <= 1]
        if not seen:
            return self.step_towards(gs, min(self.border, key=lambda b: self.get_maze_distance(me, b)))
        return self.alpha_beta_escape(gs, min(seen, key=lambda o: self.get_maze_distance(me, gs.get_agent_position(o))))

    def alpha_beta_escape(self, gs, ghost, depth=2):

        def evaluate(s):
            pos = s.get_agent_position(self.index)
            if pos == self.start:
                return -1000                          # eaten
            if not s.get_agent_state(self.index).is_pacman:
                return 500                            # home!
            return (2 * self.get_maze_distance(pos, s.get_agent_position(ghost))
                    - self.home_distance(pos) - 3 * self.depth[pos])

        def max_value(s, d, alpha, beta):
            if d == 0 or s.get_agent_position(self.index) == self.start or not s.get_agent_state(self.index).is_pacman:
                return evaluate(s)
            v = -BIG
            for a in s.get_legal_actions(self.index):
                v = max(v, min_value(s.generate_successor(self.index, a), d, alpha, beta))
                if v >= beta:
                    return v    #beta cut
                alpha = max(alpha, v)
            return v

        def min_value(s, d, alpha, beta):
            if s.get_agent_position(self.index) == self.start or not s.get_agent_state(self.index).is_pacman:
                return evaluate(s)
            v = BIG
            for a in s.get_legal_actions(ghost):
                v = min(v, max_value(s.generate_successor(ghost, a), d - 1, alpha, beta))
                if v <= alpha:
                    return v    # alpha cut
                beta = min(beta, v)
            return v

        best, best_v = None, -BIG
        for a in gs.get_legal_actions(self.index):
            v = min_value(gs.generate_successor(self.index, a), depth, best_v, BIG)
            if v > best_v:
                best, best_v = a, v
        return best

    def find_gap(self, gs, danger):
        # home and nothing safe to eat: go to the border cell where the ghosts are far
        me = gs.get_agent_position(self.index)
        side = 1 if self.red else -1
        best = max(self.border, key=lambda b: 2 * min(danger[(b[0] + side, b[1])], 15)
                   - self.get_maze_distance(me, b))
        return self.step_towards(gs, best, stay_home=True)


class DefensiveAgent(OffensiveAgent):

    def decide(self, gs):
        st = gs.get_agent_state(self.index)
        me = gs.get_agent_position(self.index)
        t = self.team
        invaders = self.invaders(gs)
        if invaders:
            if st.is_pacman:
                return self.attack(gs, go_home=True)    # we were attacking -> back home
            return self.defend(gs, invaders)

        if self.should_raid(gs):
            return self.attack(gs)
        if st.is_pacman:
            return self.attack(gs, go_home=True)
        t.targets.pop(self.index, None)

        # they just ate something -> go there
        if t.last_eaten and t.turn - t.last_eaten[1] <= 10 and self.get_maze_distance(me, t.last_eaten[0]) > 1:
            return self.step_towards(gs, t.last_eaten[0], stay_home=True)
        return self.patrol(gs)

    def should_raid(self, gs):
        # nobody crossed in 100 turns -> they only defend, so we also attack (to not tie)
        t = self.team
        return (t.turn >= 100 and t.turn - t.last_invader_turn >= 100
                and self.get_score(gs) <= 0 and gs.data.timeleft // 4 >= 30)

    def patrol(self, gs):
        # wait at the border cell close to the rival that is closest to come in, and close to our food
        me = gs.get_agent_position(self.index)
        food = self.get_food_you_are_defending(gs).as_list()
        rival = min((self.estimate(gs, o)[0] for o in self.opponents), key=self.home_distance)

        def value(b):
            v = self.get_maze_distance(b, rival)
            if food:
                v += 0.3 * min(self.get_maze_distance(b, f) for f in food)
            return v

        target = min(self.border, key=value)
        if me == target:
            return Directions.STOP
        return self.step_towards(gs, target, stay_home=True)
