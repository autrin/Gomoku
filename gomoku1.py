"""This program was written with the help of algorithms from AIMA book/github"""

import copy
import itertools
import random
from collections import namedtuple, Counter, defaultdict
import math
import functools
import numpy as np

"""
Game rules: Players alternate turns placing a stone of their color (white or black) on an empty
intersection on a 15x15 Go board. Black plays first. The first player's first stone must be placed
in the center of the board. The second player's first stone may be placed anywhere on the board.
The first player's second stone must be placed at least three intersections away from the first
stone (two empty intersections in between the two stones). The winner is the first player to form
an unbroken line of five stones of their color horizontally, vertically, or diagonally. If the board is
completely filled and no one can make a line of 5 stones, then the game ends in a draw.
"""
cache = functools.lru_cache(10**6)
GameState = namedtuple("GameState", "to_move, utility, board, moves")


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    if not (a and b):
        return a or b
    if hasattr(a, "__iter__") and hasattr(b, "__iter__"):
        assert len(a) == len(b)
        return list(map(vector_add, a, b))
    else:
        try:
            return a + b
        except TypeError:
            raise Exception("Inputs must be in the same size!")


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a collection of the allowable moves from this state."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)


def play_game(game, strategies: dict, verbose=False):
    state = game.initial
    while not game.is_terminal(state):
        player = state.to_move
        if player == "B":  # Assuming 'B' is always the human player
            print("Your turn. Current board:")
            game.display(state)
            move = query_player(game, state)
        elif player in strategies:
            move = strategies[player](game, state)
        else:
            raise ValueError(f"No strategy found for player {player}")
        state = game.result(state, move)
        if verbose:
            print(f"Player {player} move: {move}")
            game.display(state)
    return state


class Gomoku(Game):
    """
    b : black, plays first
    w : w
    bStone and wStone are the numbers of the stones the players have placed
    """

    def __init__(self, to_move="B", h=15, w=15, k=5):
        self.h = h  # height of the board
        self.w = w  # width of the board
        self.k = k  # how many back to back stones we need to have a winner
        self.to_move = to_move
        self.squares = {(x, y) for x in range(w) for y in range(h)}
        self.initial = Board(
            height=h, width=w, to_move="B", utility=0, bStone=0, wStone=0
        )

    def actions(self, board):
        # Update this method to calculate allowed moves based on bStone and wStone within the board
        moves = set()
        if board.to_move == "B":
            if board.bStone == 0:
                return {(7, 7)}  # Black's first move must be at the center
            elif board.bStone == 1:
                for x in range(self.w):
                    for y in range(self.h):
                        # Ensure the move is at least three intersections away from (7, 7)
                        if abs(7 - x) >= 3 or abs(7 - y) >= 3:
                            moves.add((x, y))
            # Exclude already occupied positions
            return moves - set(board)
        elif board.to_move == "W":
            if board.wStone == 0:
                # White's first move can be anywhere except the center
                return self.squares - {(7, 7)}
        # General move calculation for all other cases
        return self.squares - set(board.keys())  # Exclude occupied positions

    def result(self, board, square):
        player = board.to_move
        new_bStone = board.bStone + 1 if player == "B" else board.bStone
        new_wStone = board.wStone + 1 if player == "W" else board.wStone
        new_board = board.new(
            {square: player},
            to_move=("B" if player == "W" else "W"),
            bStone=new_bStone,
            wStone=new_wStone,
        )
        win = k_in_row(new_board, player, square, self.k)
        new_board.utility = 0 if not win else +1 if player == "B" else -1
        return new_board

    def utility(self, board, player):
        """Return the value of this final state to player."""
        return board.utility if player == "B" else -board.utility

    def is_terminal(self, board):
        """Return True if this is a final state for the game.
        The winner is the first player to form
        an unbroken line of five stones of their color horizontally, vertically, or diagonally. If the board is
        completely filled and no one can make a line of 5 stones, then the game ends in a draw.
        """
        return board.utility != 0 or len(self.squares) == len(board)

    def to_move(self, board):
        """Return the player whose move it is in this state."""
        return board.to_move

    def display(self, board):
        """Print or otherwise display the state."""
        print(board)

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)


def k_in_row(board, player, square, k):  # ? Does this check the diagonal squares?
    """True if player has k pieces in a line through square."""

    def in_row(x, y, dx, dy):
        return 0 if board[x, y] != player else 1 + in_row(x + dx, y + dy, dx, dy)

    return any(
        in_row(*square, dx, dy) + in_row(*square, -dx, -dy) - 1 >= k
        for (dx, dy) in ((0, 1), (1, 0), (1, 1), (1, -1))
    )


def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):  # *
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
    cutoff_test = cutoff_test or (
        lambda state, depth: depth > d or game.terminal_test(state)
    )
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


def query_player(game, state):
    """Make a move by querying standard input."""
    print("Available moves: Enter the coordinates as 'x y'\nOr enter 'q' to quit")
    move = None
    while move is None:
        try:
            move_input = input("Your move? ")
            if move_input == "q":
                exit  #!
            x, y = map(int, move_input.split())
            move = (x, y)
            if move not in game.actions(state):
                print("Invalid move, please try again.")
                move = None
        except ValueError:
            print("Invalid format, please enter coordinates as 'x y'.")
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


class Board(defaultdict):
    """A board has the player to move, a cached utility value,
    and a dict of {(x, y): player} entries, where player is 'W' or 'B'."""

    empty = "."
    off = "#"

    def __init__(self, width=15, height=15, to_move=None, bStone=0, wStone=0, **kwds):
        self.__dict__.update(
            width=width,
            height=height,
            to_move=to_move,
            bStone=bStone,
            wStone=wStone,
            **kwds,
        )

    def new(
        self, changes: dict, to_move=None, bStone=None, wStone=None, **kwds
    ) -> "Board":
        "Given a dict of {(x, y): contents} changes, return a new Board with the changes."
        board = Board(
            width=self.width,
            height=self.height,
            to_move=to_move if to_move is not None else self.to_move,
            bStone=bStone if bStone is not None else self.bStone,
            wStone=wStone if wStone is not None else self.wStone,
            **kwds,
        )
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
        def row(y):
            return " ".join(self[x, y] for x in range(self.width))

        return "\n".join(map(row, range(self.height))) + "\n"


"""
Transposition Tables

By treating the game tree as a tree, we can arrive at the same state through different paths, and end up duplicating effort. 
In state-space search, we kept a table of `reached` states to prevent this. For game-tree search, we can achieve the same effect 
by applying the `@cache` decorator to the `min_value` and `max_value` functions. We'll use the suffix `_tt` to indicate a function 
that uses these transisiton tables.
"""


# For alpha-beta search, we can still use a cache, but it should be based just on the state, not on whatever values alpha and beta have.
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
            v2, _ = min_value(game.result(state, a), alpha, beta, depth + 1)
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


"""
Takes the state as input and returns its value
"""


def evaluate1(game, state):

    raise NotImplementedError


def evaluate2(game, state):
    raise NotImplementedError


def random_player(game, state):
    return random.choice(list(game.actions(state)))  # Randomly chooses a move


def player(search_algorithm):
    """A game player who uses the specified search algorithm"""
    return lambda game, state: search_algorithm(game, state)[1]


def main():
    game = Gomoku()
    play_game(game, {"W": player(h_alphabeta_search)}, verbose=True).utility


if __name__ == "__main__":
    main()