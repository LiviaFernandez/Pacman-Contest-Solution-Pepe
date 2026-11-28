# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import os
import random

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions

MIN_FOOD = 2        # game is over when all food but 2 is eaten
CARRY_LIMIT = 5     # with this much food we go back home
DANGER_DIST = 5     # a ghost this close -> we think with alpha-beta
DEPTH = 2           # how deep alpha-beta looks
SCARED_SAFE = 3     # a ghost scared for more than this is not a problem
INF = 999999
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


##########
# Agents #
##########

class BaseAgent(CaptureAgent):

    # runs once at the start
    def register_initial_state(self, gs):
        CaptureAgent.register_initial_state(self, gs)   # gives self.red and get_maze_distance
        self.start = gs.get_agent_position(self.index)
        self.walls = gs.get_walls()
        self.mid = self.walls.width / 2.0               # middle line, red is x < mid

        # border = our last column, only the cells where we can cross to the other side
        if self.red:
            bx = (self.walls.width - 1) // 2
            ox = bx + 1
        else:
            bx = (self.walls.width + 1) // 2
            ox = bx - 1
        self.border = []
        for y in range(self.walls.height):
            if not self.walls[bx][y] and not self.walls[ox][y]:
                self.border.append((bx, y))

        self.dead_ends = self.find_dead_ends()

    def on_my_side(self, cell):
        if self.red:
            return cell[0] < self.mid
        else:
            return cell[0] >= self.mid

    def actions(self, cell):
        # moves that dont hit a wall
        moves = []
        for a in DIRS:
            dx, dy = DIRS[a]
            if not self.walls[cell[0] + dx][cell[1] + dy]:
                moves.append(a)
        return moves

    def result(self, cell, a):
        dx, dy = DIRS[a]
        return (cell[0] + dx, cell[1] + dy)

    def find_dead_ends(self):
        # tunnels with only one exit. we remove cells with 1 neighbour again and again
        exits = {}
        for x in range(self.walls.width):
            for y in range(self.walls.height):
                if not self.walls[x][y]:
                    exits[(x, y)] = len(self.actions((x, y)))
        dead = set()
        stack = [c for c in exits if exits[c] == 1]
        while stack:
            cell = stack.pop()
            if cell in dead:
                continue
            dead.add(cell)
            for a in self.actions(cell):
                nxt = self.result(cell, a)
                if nxt not in dead:
                    exits[nxt] -= 1
                    if exits[nxt] == 1:
                        stack.append(nxt)
        return dead

    def bfs(self, start, goals, stay_home=False):

        frontier = util.Queue()
        frontier.push((start, None))      # (cell, first action we did to get there)
        reached = set()
        reached.add(start)
        while not frontier.is_empty():
            cell, first_action = frontier.pop()
            if cell in goals:   # goal test
                return first_action
            for a in self.actions(cell):
                nxt = self.result(cell, a)
                if nxt in reached:
                    continue
                if stay_home and not self.on_my_side(nxt):
                    continue
                reached.add(nxt)
                if first_action is None:
                    frontier.push((nxt, a))
                else:
                    frontier.push((nxt, first_action))
        return None

    def heuristic(self, cell, goals):
        # manhattan to the closest goal, never more than the real distance
        best = INF
        for g in goals:
            d = util.manhattan_distance(cell, g)
            if d < best:
                best = d
        return best

    def step_cost(self, cell, ghosts):
        # normal step = 1, but cells close to a ghost are expensive
        cost = 1
        if self.on_my_side(cell):
            return cost
        for gpos in ghosts:
            d = self.get_maze_distance(cell, gpos)
            if d <= 1:
                cost += 100
            elif d == 2:
                cost += 20
            elif d == 3:
                cost += 5
        if ghosts and cell in self.dead_ends:
            cost += 5
        return cost

    def astar(self, start, goals, ghosts):
        if len(goals) == 0:
            return None
        goals = set(goals)
        frontier = util.PriorityQueue()
        frontier.push((start, None, 0), self.heuristic(start, goals))   # (cell, first action, g)
        expanded = set()
        while not frontier.is_empty():
            cell, first_action, g = frontier.pop()
            if cell in expanded:
                continue
            expanded.add(cell)
            if cell in goals:
                return first_action
            for a in self.actions(cell):
                nxt = self.result(cell, a)
                if nxt in expanded:
                    continue
                g2 = g + self.step_cost(nxt, ghosts)
                if first_action is None:
                    frontier.push((nxt, a, g2), g2 + self.heuristic(nxt, goals))
                else:
                    frontier.push((nxt, first_action, g2), g2 + self.heuristic(nxt, goals))
        return None

    def home_distance(self, cell):
        best = INF
        for b in self.border:
            d = self.get_maze_distance(cell, b)
            if d < best:
                best = d
        return best

    def visible_rivals(self, gs, pacman):
        # rivals we can see. pacman=True -> the ones in our side, False -> ghosts in their side
        rivals = []
        for o in self.get_opponents(gs):
            pos = gs.get_agent_position(o)
            if pos is not None and gs.get_agent_state(o).is_pacman == pacman:
                rivals.append(o)
        return rivals

    def choose_action(self, gs):
        legal = gs.get_legal_actions(self.index)
        try:
            action = self.decide(gs)
        except Exception:
            if _DEBUG:
                raise
            action = None
        if action not in legal:    # something went wrong -> random move so we dont lose the game
            action = random.choice(legal)
        return action

    def decide(self, gs):
        raise NotImplementedError


class OffensiveAgent(BaseAgent):
    # goes to eat their food

    def decide(self, gs):
        me = gs.get_agent_position(self.index)
        my_state = gs.get_agent_state(self.index)
        carrying = my_state.num_carrying
        food = self.get_food(gs).as_list()
        capsules = self.get_capsules(gs)
        moves_left = gs.data.timeleft // 4

        # ghosts we see that can eat us
        ghosts = []
        ghost_dist = INF
        for o in self.visible_rivals(gs, pacman=False):
            if gs.get_agent_state(o).scared_timer <= SCARED_SAFE:
                ghosts.append(o)
                d = self.get_maze_distance(me, gs.get_agent_position(o))
                ghost_dist = min(ghost_dist, d)

        # we are at home and an invader is very close -> eat him first
        if not my_state.is_pacman and my_state.scared_timer == 0:
            for o in self.visible_rivals(gs, pacman=True):
                pos = gs.get_agent_position(o)
                if self.get_maze_distance(me, pos) <= 3:
                    return self.bfs(me, [pos], stay_home=True)

        # ghost close and we are in their side -> adversarial search
        if my_state.is_pacman and ghost_dist <= DANGER_DIST:
            close = []
            for o in ghosts:
                if self.get_maze_distance(me, gs.get_agent_position(o)) <= DANGER_DIST:
                    close.append(o)
            return self.alpha_beta(gs, close)

        # if their ghosts are scared for a long time we can take more food
        scared_left = INF
        for o in self.get_opponents(gs):
            scared_left = min(scared_left, gs.get_agent_state(o).scared_timer)
        limit = CARRY_LIMIT
        if scared_left > self.home_distance(me) + 5:
            limit = CARRY_LIMIT * 2

        # time to go home?
        go_home = False
        if carrying >= limit:
            go_home = True
        if carrying > 0 and len(food) <= MIN_FOOD:
            go_home = True
        if carrying > 0 and moves_left < self.home_distance(me) + 5:     # game is ending
            go_home = True

        ghost_cells = [gs.get_agent_position(o) for o in ghosts]
        if go_home:
            goals = self.border
        elif len(ghosts) > 0 and len(capsules) > 0:
            goals = food + capsules       # someone is around, a capsule helps
        else:
            goals = food
        action = self.astar(me, goals, ghost_cells)
        if action is None:
            return Directions.STOP
        return action

    #  alpha-beta, MAX = us, MIN = the ghosts (one after the other)
    def alpha_beta(self, gs, ghosts):
        depth = DEPTH
        if len(ghosts) > 1:
            depth = 1        # two ghosts is too slow for depth 2
        best_action = Directions.STOP
        best_value = -INF
        for a in gs.get_legal_actions(self.index):
            next_state = gs.generate_successor(self.index, a)
            v = self.min_value(next_state, ghosts, 0, depth, best_value, INF)
            if v > best_value:
                best_value = v
                best_action = a
        return best_action

    def max_value(self, s, ghosts, depth, alpha, beta):
        if depth == 0 or self.game_over_for_us(s):
            return self.evaluate(s, ghosts)
        v = -INF
        for a in s.get_legal_actions(self.index):
            v = max(v, self.min_value(s.generate_successor(self.index, a), ghosts, 0, depth, alpha, beta))
            if v >= beta:
                return v       # beta cut
            alpha = max(alpha, v)
        return v

    def min_value(self, s, ghosts, i, depth, alpha, beta):
        if self.game_over_for_us(s):
            return self.evaluate(s, ghosts)
        v = INF
        g = ghosts[i]
        for a in s.get_legal_actions(g):
            nxt = s.generate_successor(g, a)
            if i == len(ghosts) - 1:
                v = min(v, self.max_value(nxt, ghosts, depth - 1, alpha, beta))
            else:
                v = min(v, self.min_value(nxt, ghosts, i + 1, depth, alpha, beta))
            if v <= alpha:
                return v       # alpha cut
            beta = min(beta, v)
        return v

    def game_over_for_us(self, s):
        # eaten (back at start) or safe at home -> no need to look further
        if s.get_agent_position(self.index) == self.start:
            return True
        if not s.get_agent_state(self.index).is_pacman:
            return True
        return False

    def evaluate(self, s, ghosts):
        """Eval function: how good is state s for us"""
        pos = s.get_agent_position(self.index)
        if pos == self.start:
            return -1000                      # eaten :(
        carrying = s.get_agent_state(self.index).num_carrying
        value = 100 * self.get_score(s)       # food already at home
        value += 10 * carrying                # food we have with us
        value -= 30 * len(self.get_capsules(s))

        for g in ghosts:
            if s.get_agent_state(g).scared_timer > SCARED_SAFE:
                continue                      # scared, cant eat us
            d_ghost = self.get_maze_distance(pos, s.get_agent_position(g))
            if d_ghost <= 1:
                value -= 500                  # he eats us next move
            value += 2 * min(d_ghost, 5)      # far is good (but more than 5 doesnt matter)
            if pos in self.dead_ends and d_ghost <= 4:
                value -= 50                   # tunnel with a ghost behind = trap

        # with food -> want to be close to home, without -> close to food
        if carrying > 0:
            value -= self.home_distance(pos)
        else:
            closest = INF
            for f in self.get_food(s).as_list():
                closest = min(closest, self.get_maze_distance(pos, f))
            if closest < INF:
                value -= closest
        return value


class DefensiveAgent(BaseAgent):
    # stays in our side and catches invaders

    def register_initial_state(self, gs):
        BaseAgent.register_initial_state(self, gs)
        self.last_eaten = None      # cell where they ate our food (we dont see them but we know they are there)

    def decide(self, gs):
        me = gs.get_agent_position(self.index)
        scared = gs.get_agent_state(self.index).scared_timer > 0

        # did they eat something since our last turn? (food or capsule)
        prev = self.get_previous_observation()
        if prev is not None:
            now = self.get_food_you_are_defending(gs).as_list() + self.get_capsules_you_are_defending(gs)
            before = self.get_food_you_are_defending(prev).as_list() + self.get_capsules_you_are_defending(prev)
            for f in before:
                if f not in now:
                    self.last_eaten = f

        # invaders we can see -> go for the closest one
        target = None
        best = INF
        for o in self.visible_rivals(gs, pacman=True):
            pos = gs.get_agent_position(o)
            d = self.get_maze_distance(me, pos)
            if d < best:
                best = d
                target = pos

        if target is not None:
            self.last_eaten = None
            if scared:
                return self.keep_distance(gs, target, 2)    # he can eat us -> follow from 2 cells
            action = self.bfs(me, [target], stay_home=True)
        elif self.last_eaten is not None:
            # we dont see him, but we know where he ate
            if me == self.last_eaten:
                self.last_eaten = None
            action = self.bfs(me, [self.last_eaten], stay_home=True)
        else:
            action = self.bfs(me, [self.patrol_cell(gs)], stay_home=True)

        if action is None:
            return Directions.STOP
        return action

    def patrol_cell(self, gs):
        # if we see one of them coming close to the border, wait in front of him
        pos = None
        closest = 6
        for o in self.visible_rivals(gs, pacman=False):
            p = gs.get_agent_position(o)
            d = self.home_distance(p)
            if d <= closest:
                closest = d
                pos = p
        if pos is not None:
            best_cell = self.border[0]
            best_d = INF
            for b in self.border:
                d = self.get_maze_distance(b, pos)
                if d < best_d:
                    best_d = d
                    best_cell = b
            return best_cell

        # nobody: wait at the border cell closest to our food
        food = self.get_food_you_are_defending(gs).as_list()
        best_cell = self.border[0]
        best_d = INF
        for b in self.border:
            for f in food:
                d = self.get_maze_distance(b, f)
                if d < best_d:
                    best_d = d
                    best_cell = b
        return best_cell

    def keep_distance(self, gs, target, k):
        # when we are scared: stay at distance k of the invader (not closer)
        me = gs.get_agent_position(self.index)
        best_action = Directions.STOP
        best_diff = INF
        for a in gs.get_legal_actions(self.index):
            if a == Directions.STOP:
                nxt = me
            else:
                nxt = self.result(me, a)
            if not self.on_my_side(nxt):
                continue
            d = self.get_maze_distance(nxt, target)
            diff = abs(d - k)
            if d < k:
                diff += 100      # too close is very bad
            if diff < best_diff:
                best_diff = diff
                best_action = a
        return best_action
