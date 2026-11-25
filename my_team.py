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

    def on_my_side(self, cell):
        if self.red:
            return cell[0] < self.mid
        else:
            return cell[0] >= self.mid

    def actions(self, cell):
        # Actions(s): moves that dont hit a wall
        moves = []
        for a in DIRS:
            dx, dy = DIRS[a]
            if not self.walls[cell[0] + dx][cell[1] + dy]:
                moves.append(a)
        return moves

    def result(self, cell, a):
        # Result(s, a): the cell where we end
        dx, dy = DIRS[a]
        return (cell[0] + dx, cell[1] + dy)

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
        moves_left = gs.data.timeleft // 4

        # closest ghost we see that can eat us (not scared)
        ghost = None
        ghost_dist = INF
        for o in self.visible_rivals(gs, pacman=False):
            if gs.get_agent_state(o).scared_timer <= 2:
                d = self.get_maze_distance(me, gs.get_agent_position(o))
                if d < ghost_dist:
                    ghost = o
                    ghost_dist = d

        # ghost close and we are in their side -> adversarial search 
        if my_state.is_pacman and ghost is not None and ghost_dist <= DANGER_DIST:
            return self.alpha_beta(gs, ghost)

        # time to go home?
        go_home = False
        if carrying >= CARRY_LIMIT:
            go_home = True
        if carrying > 0 and len(food) <= MIN_FOOD:
            go_home = True
        if carrying > 0 and moves_left < self.home_distance(me) + 5:     # game is ending
            go_home = True

        if go_home:
            action = self.bfs(me, self.border)
        else:
            action = self.bfs(me, food)        # closest food
        if action is None:
            return Directions.STOP
        return action

    #  alpha-beta, MAX = us, MIN = the ghost
    def alpha_beta(self, gs, ghost):
        best_action = Directions.STOP
        best_value = -INF
        for a in gs.get_legal_actions(self.index):
            next_state = gs.generate_successor(self.index, a)
            v = self.min_value(next_state, ghost, DEPTH, best_value, INF)
            if v > best_value:
                best_value = v
                best_action = a
        return best_action

    def max_value(self, s, ghost, depth, alpha, beta):
        if depth == 0 or self.game_over_for_us(s):
            return self.evaluate(s, ghost)
        v = -INF
        for a in s.get_legal_actions(self.index):
            v = max(v, self.min_value(s.generate_successor(self.index, a), ghost, depth, alpha, beta))
            if v >= beta:
                return v       # beta cut
            alpha = max(alpha, v)
        return v

    def min_value(self, s, ghost, depth, alpha, beta):
        if self.game_over_for_us(s):
            return self.evaluate(s, ghost)
        v = INF
        for a in s.get_legal_actions(ghost):
            v = min(v, self.max_value(s.generate_successor(ghost, a), ghost, depth - 1, alpha, beta))
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

    def evaluate(self, s, ghost):
        """Eval function: how good is state s for us"""
        pos = s.get_agent_position(self.index)
        if pos == self.start:
            return -1000                      # eaten :(
        carrying = s.get_agent_state(self.index).num_carrying
        value = 100 * self.get_score(s)       # food already at home
        value += 10 * carrying                # food we have with us

        # far from the ghost is good (but more than 5 doesnt matter)
        d_ghost = self.get_maze_distance(pos, s.get_agent_position(ghost))
        if d_ghost <= 1:
            value -= 500                      # he eats us next move
        value += 2 * min(d_ghost, 5)

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

        # did they eat something since our last turn? compare food with the previous observation
        prev = self.get_previous_observation()
        if prev is not None:
            now = self.get_food_you_are_defending(gs).as_list()
            for f in self.get_food_you_are_defending(prev).as_list():
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
            # nobody: wait at the border cell closest to our food
            action = self.bfs(me, [self.patrol_cell(gs)], stay_home=True)

        if action is None:
            return Directions.STOP
        return action

    def patrol_cell(self, gs):
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
