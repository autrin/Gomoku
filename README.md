# Gomoku
A game-playing agent capable of playing Five-in-a-Row (Gomoku)

To run the game simply enter the command:
python gomoku.py


In the main function:
    h_alphabeta_search uses the advanced evaluation method. 
    And you can use it with:
        play_game(game, {"W": player(h_alphabeta_search)}, verbose=True).utility
    alpha_beta_cutoff_search uses the usual evaluation function. And you can use it with:
        play_game(game, {"W": player(alpha_beta_cutoff_search)}, verbose=True).utility
    
    Both of the alphabeta functions are basically the same. I just gave a different evaluation function to each.

    To change the depth, you can modify the paramters of the functions.
    Right now the depths are 3 and 4.