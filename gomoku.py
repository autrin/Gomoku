"""This program was written with the help of algorithms from AIMA book/github"""
import copy
import itertools
import random
from collections import namedtuple , Counter, defaultdict
import math
import functools
import numpy as np

cache = functools.lru_cache(10**6)


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))



# # MinMax Search
# def minmax_decision(state, game):
#     """Given a state in a game, calculate the best move by searching
#     forward all the way to the terminal states. [Figure 5.3]"""

#     player = game.to_move(state)

#     def max_value(state):
#         if game.terminal_test(state):
#             return game.utility(state, player)
#         v = -np.inf
#         for a in game.actions(state):
#             v = max(v, min_value(game.result(state, a)))
#         return v

#     def min_value(state):
#         if game.terminal_test(state):
#             return game.utility(state, player)
#         v = np.inf
#         for a in game.actions(state):
#             v = min(v, max_value(game.result(state, a)))
#         return v

#     # Body of minmax_decision:
#     return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))

# def alpha_beta_search(state, game):
#     """Search game to determine best action; use alpha-beta pruning.
#     As in [Figure 5.7], this version searches all the way to the leaves."""

#     player = game.to_move(state)

#     # Functions used by alpha_beta
#     def max_value(state, alpha, beta):
#         if game.terminal_test(state):
#             return game.utility(state, player)
#         v = -np.inf
#         for a in game.actions(state):
#             v = max(v, min_value(game.result(state, a), alpha, beta))
#             if v >= beta:
#                 return v
#             alpha = max(alpha, v)
#         return v

#     def min_value(state, alpha, beta):
#         if game.terminal_test(state):
#             return game.utility(state, player)
#         v = np.inf
#         for a in game.actions(state):
#             v = min(v, max_value(game.result(state, a), alpha, beta))
#             if v <= alpha:
#                 return v
#             beta = min(beta, v)
#         return v

#     # Body of alpha_beta_search:
#     best_score = -np.inf
#     beta = np.inf
#     best_action = None
#     for a in game.actions(state):
#         v = min_value(game.result(state, a), best_score, beta)
#         if v > best_score:
#             best_score = v
#             best_action = a
#     return best_action


def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


# def alpha_beta_player(game, state):
#     return alpha_beta_search(state, game) #!


# def expect_min_max_player(game, state):
#     return expect_minmax(state, game)


class Board(defaultdict):
    """A board has the player to move, a cached utility value, 
    and a dict of {(x, y): player} entries, where player is 'X' or 'O'."""
    empty = '.'
    off = '#'
    
    def __init__(self, width=8, height=8, to_move=None, **kwds):
        self.__dict__.update(width=width, height=height, to_move=to_move, **kwds)
        
    def new(self, changes: dict, **kwds) -> 'Board':
        "Given a dict of {(x, y): contents} changes, return a new Board with the changes."
        board = Board(width=self.width, height=self.height, **kwds)
        board.update(self)
        board.update(changes)
        return board

    def __missing__(self, loc):
        x, y = loc
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.empty
        else:
            return self.off
            
    def __hash__(self): 
        return hash(tuple(sorted(self.items()))) + hash(self.to_move)
    
    def __repr__(self):
        def row(y): return ' '.join(self[x, y] for x in range(self.width))
        return '\n'.join(map(row, range(self.height))) +  '\n'
    
"""
Transposition Tables

By treating the game tree as a tree, we can arrive at the same state through different paths, and end up duplicating effort. 
In state-space search, we kept a table of `reached` states to prevent this. For game-tree search, we can achieve the same effect 
by applying the `@cache` decorator to the `min_value` and `max_value` functions. We'll use the suffix `_tt` to indicate a function 
that uses these transisiton tables.
"""
def minimax_search_tt(game, state):
    """Search game to determine best move; return (value, move) pair."""

    player = state.to_move

    @cache
    def max_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -np.inf, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a))
            if v2 > v:
                v, move = v2, a
        return v, move

    @cache
    def min_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +np.inf, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a))
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(state)

#For alpha-beta search, we can still use a cache, but it should be based just on the state, not on whatever values alpha and beta have.
def cache1(function):
    "Like lru_cache(None), but only considers the first argument of function."
    cache = {}
    def wrapped(x, *args):
        if x not in cache:
            cache[x] = function(x, *args)
        return cache[x]
    return wrapped

def alphabeta_search_tt(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -np.inf, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
    def min_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +np.inf, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -np.inf, +np.inf)
# how to get time?
# %time play_game(TicTacToe(), {'X':player(alphabeta_search_tt), 'O':player(minimax_search_tt)})


# Heuristic Cutoffs
def cutoff_depth(d):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d

def h_alphabeta_search(game, state, cutoff=cutoff_depth(6), h=lambda s, p: 0):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = -np.inf, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
    def min_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = +np.inf, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -np.inf, +np.inf, 0)

# class CountCalls:
#     """Delegate all attribute gets to the object, and count them in ._counts"""
#     def __init__(self, obj):
#         self._object = obj
#         self._counts = Counter()
        
#     def __getattr__(self, attr):
#         "Delegate to the original object, after incrementing a counter."
#         self._counts[attr] += 1
#         return getattr(self._object, attr)
    
# def report(game, searchers):
#     for searcher in searchers:
#         game = CountCalls(game)
#         searcher(game, game.initial)
#         print('Result states: {:7,d}; Terminal tests: {:7,d}; for {}'.format(
#             game._counts['result'], game._counts['is_terminal'], searcher.__name__))
    
# report(TicTacToe(), (alphabeta_search_tt,  alphabeta_search, h_alphabeta_search, minimax_search_tt))