"""This program was written with the help of algorithms and classes from AIMA book/github"""

import copy
import itertools
import random
from collections import namedtuple, Counter, defaultdict
import math
import functools
import numpy as np

"""
Game rules: Players alternate turns placing a stone of their color (white or black) on an empty
intersection on a 10x10 Go board. Black plays first. The first player's first stone must be placed
in the center of the board. The second player's first stone may be placed anywhere on the board.
The first player's second stone must be placed at least three intersections away from the first
stone (two empty intersections in between the two stones). The winner is the first player to form
an unbroken line of five stones of their color horizontally, vertically, or diagonally. If the board is
completely filled and no one can make a line of 5 stones, then the game ends in a draw.
"""

class Game:

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
        if type(move) == int:
            print(f"move, {move}, was an int")
            print(f"player: {player}")
            print(f"game: {game}")
            print(f"state\n: {state}")
            print(f"strategies[player]: {strategies[player]}")

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

    def __init__(self, h=10, w=10, k=5):
        # self.h = h  # height of the board
        # self.w = w  # width of the board
        self.k = k  # how many back to back stones we need to have a winner
        self.squares = {(x, y) for x in range(w) for y in range(h)}
        self.initial = Board(
            height=h, width=w, to_move="B", utility=0, bStone=0, wStone=0
        )

    def actions(self, board):
        # Update this method to calculate allowed moves based on bStone and wStone within the board
        moves = set()
        if board.to_move == "B":
            if board.bStone == 0:
                return {
                    (board.width // 2, board.height // 2)
                }  # Black's first move must be at the center
            elif board.bStone == 1:
                for x in range(board.width):
                    for y in range(board.height):
                        # Ensure the move is at least three intersections away from center
                        if (
                            abs(board.width // 2 - x) >= 3
                            or abs(board.height // 2 - y) >= 3
                        ):
                            moves.add((x, y))
                # Exclude already occupied positions
                return moves - set(board)
        elif board.to_move == "W":
            if board.wStone == 0:
                # White's first move can be anywhere except the center
                return self.squares - {(board.width // 2, board.height // 2)}
        # General move calculation for all other cases
        return self.squares - set(board)  # Exclude occupied positions

    def result(self, board, square):
        player = board.to_move
        board.bStone = board.bStone + 1 if player == "B" else board.bStone
        board.wStone = board.wStone + 1 if player == "W" else board.wStone
        board = board.new(
            {square: player},
            to_move=("W" if player == "B" else "B"),
            bStone=board.bStone,
            wStone=board.wStone,
        )
        win = stones_in_row(board, player, square, self.k)
        board.utility = 0 if not win else +1 if player == "B" else -1
        return board

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


def stones_in_row(board, player, square, k):
    """True if player has k pieces in a line through square."""

    def in_row(x, y, dx, dy):
        return 0 if board[x, y] != player else 1 + in_row(x + dx, y + dy, dx, dy)
    return any(
        in_row(*square, dx, dy) + in_row(*square, -dx, -dy) - 1 >= k
        for (dx, dy) in ((0, 1), (1, 0), (1, 1), (1, -1))
    )


class Board(defaultdict):
    """A board has the player to move, a cached utility value,
    and a dict of {(x, y): player} entries, where player is 'W' or 'B'."""

    empty = "."
    off = "#"

    def __init__(self, width=10, height=10, to_move=None, bStone=0, wStone=0, **kwds):
        self.__dict__.update(
            width=width,
            height=height,
            to_move=to_move,
            bStone=bStone,
            wStone=wStone,
            **kwds,
        )

    def new(self, changes: dict, **kwds) -> "Board":
        "Given a dict of {(x, y): contents} changes, return a new Board with the changes."
        board = Board(
            width=self.width,
            height=self.height,
            # to_move=to_move,
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


def evaluate_board(board, player, k=5):
    # Define weights for various factors
    w1, w2, w3 = 10, 15, 20

    player_score = 0
    opponent_score = 0
    opponent_threats = 0
    opponent = "W" if player == "B" else "B"

    # Iterate through all positions on the board
    for x in range(board.width):
        for y in range(board.height):
            if board[x, y] == player:
                player_score += stones_in_row(board, player, (x, y), k) * w1
            elif board[x, y] == opponent:
                opponent_score += stones_in_row(board, opponent, (x, y), k) * w2
                # Detect immediate threat (e.g., four in a row with an open end)
                found, direction, length = stones_in_row_evaluation(
                    board, opponent, (x, y), k - 1
                )
                if found:
                    if is_open_end(board, (x, y), direction, length):
                        opponent_threats += 1 * w3

    evaluation_score = player_score - opponent_score - opponent_threats
    return evaluation_score


def stones_in_row_evaluation(board, player, square, k):
    """
    Enhanced to check sequences starting from a given square and return
    the direction and length of sequences.
    """

    def in_row(x, y, dx, dy):
        if (
            not (0 <= x < board.width and 0 <= y < board.height)
            or board[x, y] != player
        ):
            return 0
        else:
            return 1 + in_row(x + dx, y + dy, dx, dy)

    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        length = (
            in_row(square[0] + dx, square[1] + dy, dx, dy)
            + in_row(square[0] - dx, square[1] - dy, dx, dy)
            - 1
        )
        if length >= k:
            return True, (dx, dy), length
    return False, None, 0


def is_open_end(board, square, direction, length):
    """
    Check if there's an open end to the sequence of stones.
    """
    dx, dy = direction
    # Calculate the positions of the potential open ends
    end1 = (square[0] + dx * length, square[1] + dy * length)
    end2 = (square[0] - dx, square[1] - dy)
    return board[end1] == Board.empty or board[end2] == Board.empty

def evaluate_board_advanced(board, player):
    opponent = "W" if player == "B" else "B"
    score = 0
    for x, y in board:
        if board[x, y] == player:
            # Evaluate in all directions from (x, y)
            score += evaluate_position(board, x, y, player) - evaluate_position(board, x, y, opponent)
    return score

def evaluate_position(board, x, y, player):
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    position_score = 0
    for dx, dy in directions:
        count = 1
        # Check one direction
        nx, ny = x + dx, y + dy
        while (nx, ny) in board and board[nx, ny] == player:
            count += 1
            nx += dx
            ny += dy
        # Check the opposite direction
        nx, ny = x - dx, y - dy
        while (nx, ny) in board and board[nx, ny] == player:
            count += 1
            nx -= dx
            ny -= dy
        # Update position_score based on count of consecutive stones
        position_score += score_based_on_count(count)
    return position_score

def score_based_on_count(count):
    # Define scores for sequences of different lengths
    if count >= 5:
        # Winning condition
        return 100000
    elif count == 4:
        # Open four or blocked four (almost winning)
        return 10000
    elif count == 3:
        # Open three (potential to win in two moves)
        return 1000
    elif count == 2:
        # Open two (foundation for further sequences)
        return 100
    else:
        # Single stone or blocked sequences
        return 10

# Heuristic Cutoffs
def cutoff_depth(d):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d

def alpha_beta_cutoff_search(
    game, state, d=4, cutoff_test=cutoff_depth(4), eval_fn=evaluate_board):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = state.to_move

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(game, state, depth):
            return eval_fn(state, player, game.k)  # To actually use the evaluation func
            # return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(game, state, depth):
            return eval_fn(state, player, game.k)  # To actually use the evaluation func
            # return eval_fn(state)
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
        lambda state, depth: depth > d or game.is_terminal(state)
    )
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = (
        random.choice(list(game.actions(state))) if game.actions(state) else None
    )
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    print("best action:", best_action)
    return best_action


def query_player(game, state):
    """Make a move by querying standard input."""
    print("Available moves: Enter the coordinates as 'x y'\nOr enter 'q' to quit")
    move = None
    while move is None:
        try:
            move_input = input("Your move? ")
            if move_input == "q":
                quit()
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


# For alpha-beta search, we can still use a cache, but it should be based just on the state, not on whatever values alpha and beta have.
def cache1(function):
    "Like lru_cache(None), but only considers the first argument of function."
    cache = {}

    def wrapped(x, *args):
        if x not in cache:
            cache[x] = function(x, *args)
        return cache[x]

    return wrapped

def h_alphabeta_search(game, state, cutoff=cutoff_depth(3), h=evaluate_board):
#lambda s, p: 0
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

cache = functools.lru_cache(10**6)
GameState = namedtuple("GameState", "to_move, utility, board, moves")

def random_player(game, state):
    return random.choice(list(game.actions(state)))  # Randomly chooses a move

def player(search_algorithm):
    """A game player who uses the specified search algorithm"""
    return lambda game, state: search_algorithm(game, state)[1]

def main():
    game = Gomoku()
    # alpha_beta_cutoff_search uses the usual evaluation method.
    # h_alphabeta_search uses the advanced evaluation method. 
    # play_game(game, {"W": player(alpha_beta_cutoff_search)}, verbose=True).utility
    # Both of the alphabeta functions are basically the same. I just gave a different evaluation function to each.
    play_game(game, {"W": player(h_alphabeta_search)}, verbose=True).utility

if __name__ == "__main__":
    main()
