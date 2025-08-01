# Gomoku AI Game

An intelligent game-playing agent for Five-in-a-Row (Gomoku) featuring advanced AI algorithms and heuristic evaluation functions.

## Overview

This project implements a Gomoku game where players can compete against an AI opponent that uses sophisticated search algorithms including Alpha-Beta pruning with custom evaluation functions. The game follows traditional Gomoku rules on a 10x10 board.

## Game Rules

- Players alternate placing stones (Black and White) on empty intersections
- Black plays first and must place their first stone in the center
- White's first stone can be placed anywhere except the center
- Black's second stone must be at least 3 intersections away from the first
- First player to form 5 stones in a row (horizontally, vertically, or diagonally) wins
- Game ends in draw if board is full with no winner

## Features

- **Advanced AI Algorithms**: Alpha-Beta search with pruning for optimal performance
- **Dual Evaluation Functions**: 
  - Standard evaluation with basic position scoring
  - Advanced evaluation with threat detection and pattern recognition
- **Configurable Depth**: Adjustable search depth for different difficulty levels
- **Interactive Gameplay**: Human vs AI with intuitive coordinate input system
- **Performance Optimized**: Uses caching and efficient board representation

## Tech Stack

- **Python 3.x**
- **NumPy**: For mathematical operations and board evaluation
- **AIMA Framework**: Based on algorithms from "Artificial Intelligence: A Modern Approach"

## Getting Started

### Prerequisites

```bash
pip install numpy
```

### Running the Game

```bash
python gomoku.py
```

## Usage

### Making Moves

When prompted for your move:
- Enter coordinates as `x y` (e.g., `5 3`)
- Use `q` to quit the game
- Coordinates range from 0 to 9 for both x and y axes

### AI Configuration

The game offers two AI opponents with different evaluation strategies:

#### Standard AI (`alpha_beta_cutoff_search`)
- Uses basic position evaluation
- Default depth: 4
- Faster execution, good for casual play

#### Advanced AI (`h_alphabeta_search`) 
- Uses sophisticated threat detection
- Pattern recognition and sequence evaluation
- Default depth: 3
- More challenging opponent

### Customizing AI Behavior

To modify the AI settings in `main()`:

```python
# Use standard AI
play_game(game, {"W": player(alpha_beta_cutoff_search)}, verbose=True)

# Use advanced AI  
play_game(game, {"W": player(h_alphabeta_search)}, verbose=True)
```

To change search depth, modify the depth parameters:
- `alpha_beta_cutoff_search`: Change `d=4` parameter
- `h_alphabeta_search`: Change `cutoff_depth(3)` parameter

## AI Features

### Evaluation Functions

**Standard Evaluation:**
- Counts consecutive stones in all directions
- Applies positional weights
- Basic threat assessment

**Advanced Evaluation:**
- Multi-directional pattern analysis
- Winning sequence detection (5+ stones = 100,000 points)
- Threat prioritization (4 stones = 10,000 points)
- Strategic positioning (2-3 stones = 100-1,000 points)

### Search Optimization

- **Alpha-Beta Pruning**: Eliminates unnecessary search branches
- **Depth-Limited Search**: Configurable search depth for performance tuning
- **Caching**: LRU cache for previously evaluated positions
- **Move Ordering**: Prioritizes promising moves first

## Project Structure

```
Gomoku/
├── gomoku.py          # Main game implementation
├── README.md          # Project documentation
└── LICENSE           # MIT License
```

## Example Gameplay

```
Your turn. Current board:
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . B . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .

Available moves: Enter the coordinates as 'x y'
Or enter 'q' to quit
Your move? 4 4
```

## Contributing

This project was developed with algorithms and concepts from the AIMA (Artificial Intelligence: A Modern Approach) textbook. Contributions are welcome for:

- Additional evaluation heuristics
- Performance optimizations
- UI improvements
- Extended game variants

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Autrin Hakimi