import numpy as np


class HumanAgent:
    def __init__(self, show_board=True):
        self.show_board = show_board
        self.rows = 6
        self.cols = 7

    def act(self, obs, player):
        board = obs.reshape(self.rows, self.cols) if obs.ndim == 1 else obs

        valid_moves = np.where(board[0] == 0)[0]
        valid_moves = list(map(int, valid_moves))

        if len(valid_moves) == 0:
            return 0

        if self.show_board:
            self._render(board)

        while True:
            try:
                move = int(input(f"Player {player} enter column {valid_moves}: "))

                if move not in valid_moves:
                    print("Invalid move")
                    continue

                return move

            except ValueError:
                print("Invalid input")

    def _render(self, board):
        symbols = {1: "X", -1: "O", 0: "."}

        print("\nColumns: 0 1 2 3 4 5 6")
        for r in range(self.rows):
            print(" ".join(symbols[int(board[r, c])] for c in range(self.cols)))
        print()