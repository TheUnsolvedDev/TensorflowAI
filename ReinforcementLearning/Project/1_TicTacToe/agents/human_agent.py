import numpy as np


class HumanAgent:
    def __init__(self, show_board=True):
        self.show_board = show_board

    def act(self, obs, player):
        board = obs.reshape(3, 3)
        valid_moves = np.where(obs == 0)[0]
        valid_moves = list(map(int, valid_moves))

        if self.show_board:
            self._render(board)

        while True:
            try:
                move = int(input(f"Player {player} enter move {list(valid_moves)}: "))

                if move not in valid_moves:
                    print("Invalid move")
                    continue

                return move

            except ValueError:
                print("Invalid input")

    def _render(self, board):
        symbols = {1: "X", -1: "O", 0: "."}
        for row in board:
            print(" ".join(symbols[x] for x in row))
        print()