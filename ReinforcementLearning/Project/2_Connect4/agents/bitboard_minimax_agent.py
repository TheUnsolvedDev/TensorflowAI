import numpy as np


class BitboardMinimaxAgent:
    def __init__(self, depth=6):
        self.type = f"BitboardMinimax"
        self.observation_mode = "absolute"
        self.depth = depth
        self.cols = 7
        self.rows = 6

        self.bottom_mask = [(1 << (7 * c)) for c in range(self.cols)]
        self.top_mask = [(1 << (7 * c + 5)) for c in range(self.cols)]

    def act(self, obs, player):
        board = obs.reshape(self.rows, self.cols) if obs.ndim == 1 else obs

        valid_moves = np.where(board[0] == 0)[0]
        valid_moves = list(map(int, valid_moves))

        if len(valid_moves) == 0:
            return 0

        P, M = self._encode(board, player)

        ordered = self._ordered_moves(valid_moves)

        best_score = -np.inf
        best_move = ordered[0]

        for col in ordered:
            newP, newM = self._play(P, M, col)

            if self._is_win(newP):
                return col

            score = self._minimax(newP, newM, self.depth - 1, -np.inf, np.inf, False)

            if score > best_score:
                best_score = score
                best_move = col

        return best_move

    def _minimax(self, P, M, depth, alpha, beta, maximizing):
        valid_moves = [c for c in range(self.cols) if not (M & self.top_mask[c])]

        if depth == 0 or len(valid_moves) == 0:
            return 0

        ordered = self._ordered_moves(valid_moves)

        if maximizing:
            value = -np.inf
            for col in ordered:
                newP, newM = self._play(P, M, col)

                if self._is_win(newP):
                    return 1e6

                value = max(value, self._minimax(newP, newM, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        else:
            value = np.inf
            for col in ordered:
                newP, newM = self._play(P, M, col)

                if self._is_win(newP):
                    return -1e6

                value = min(value, self._minimax(newP, newM, depth - 1, alpha, beta, True))
                beta = min(beta, value)

                if alpha >= beta:
                    break

            return value

    def _play(self, P, M, col):
        move = (M + self.bottom_mask[col]) & ~M
        newM = M | move
        newP = newM ^ P
        return newP, newM

    def _ordered_moves(self, valid_moves):
        center = 3
        return sorted(valid_moves, key=lambda x: abs(x - center))

    def _is_win(self, P):
        m = P & (P >> 7)
        if m & (m >> 14):
            return True

        m = P & (P >> 6)
        if m & (m >> 12):
            return True

        m = P & (P >> 8)
        if m & (m >> 16):
            return True

        m = P & (P >> 1)
        if m & (m >> 2):
            return True

        return False

    def _encode(self, board, player):
        P = 0
        M = 0

        for c in range(self.cols):
            for r in range(self.rows):
                if board[r, c] != 0:
                    bit = 1 << (c * 7 + r)
                    M |= bit
                    if board[r, c] == player:
                        P |= bit

        return P, M