import numpy as np


class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth
        self.rows = 6
        self.cols = 7

    def act(self, obs, player):
        board = obs.reshape(self.rows, self.cols) if obs.ndim == 1 else obs.copy()

        valid_moves = self._ordered_moves(board)

        if len(valid_moves) == 0:
            return 0

        best_score = -np.inf
        best_move = valid_moves[0]

        for move in valid_moves:
            row = self._get_row(board, move)
            if row is None:
                continue

            board[row, move] = player

            if self._check_winner_fast(board, row, move, player):
                board[row, move] = 0
                return move

            score = self._minimax(board, self.depth - 1, -np.inf, np.inf, False, player)

            board[row, move] = 0

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(self, board, depth, alpha, beta, maximizing, player):
        valid_moves = self._ordered_moves(board)

        if depth == 0 or len(valid_moves) == 0:
            return self._evaluate(board, player)

        if maximizing:
            value = -np.inf
            for move in valid_moves:
                row = self._get_row(board, move)
                if row is None:
                    continue

                board[row, move] = player

                if self._check_winner_fast(board, row, move, player):
                    board[row, move] = 0
                    return 1e6

                value = max(value, self._minimax(board, depth - 1, alpha, beta, False, player))
                board[row, move] = 0

                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return value

        else:
            value = np.inf
            opponent = -player

            for move in valid_moves:
                row = self._get_row(board, move)
                if row is None:
                    continue

                board[row, move] = opponent

                if self._check_winner_fast(board, row, move, opponent):
                    board[row, move] = 0
                    return -1e6

                value = min(value, self._minimax(board, depth - 1, alpha, beta, True, player))
                board[row, move] = 0

                beta = min(beta, value)
                if alpha >= beta:
                    break

            return value

    def _ordered_moves(self, board):
        valid = np.where(board[0] == 0)[0]
        valid = list(map(int, valid))

        center = self.cols // 2
        return sorted(valid, key=lambda x: abs(x - center))

    def _get_row(self, board, col):
        for r in range(self.rows - 1, -1, -1):
            if board[r, col] == 0:
                return r
        return None

    def _check_winner_fast(self, board, row, col, player):
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True

        return False

    def _evaluate(self, board, player):
        score = 0

        center_col = board[:, self.cols // 2]
        score += np.sum(center_col == player) * 3

        for r in range(self.rows):
            for c in range(self.cols - 3):
                score += self._score_window(board[r, c:c+4], player)

        for c in range(self.cols):
            for r in range(self.rows - 3):
                score += self._score_window(board[r:r+4, c], player)

        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                score += self._score_window([board[r+i, c+i] for i in range(4)], player)

        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                score += self._score_window([board[r-i, c+i] for i in range(4)], player)

        return score

    def _score_window(self, window, player):
        window = np.array(window)
        opponent = -player

        p = np.count_nonzero(window == player)
        o = np.count_nonzero(window == opponent)
        e = np.count_nonzero(window == 0)

        if p == 4:
            return 100
        if p == 3 and e == 1:
            return 5
        if p == 2 and e == 2:
            return 2
        if o == 3 and e == 1:
            return -4

        return 0