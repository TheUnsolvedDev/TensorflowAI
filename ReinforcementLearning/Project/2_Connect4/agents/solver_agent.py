import numpy as np
import time


class SolverAgent:
    def __init__(self, max_depth=10, time_limit=1.0):
        self.type = "Solver"
        self.observation_mode = "absolute"
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.cols = 7
        self.rows = 6

        self.bottom_mask = [(1 << (7 * c)) for c in range(self.cols)]
        self.top_mask = [(1 << (7 * c + 5)) for c in range(self.cols)]

        self.transposition = {}

    def act(self, obs, player):
        board = obs.reshape(self.rows, self.cols) if obs.ndim == 1 else obs

        valid_moves = np.where(board[0] == 0)[0]
        valid_moves = list(map(int, valid_moves))

        if len(valid_moves) == 0:
            return 0

        P, M = self._encode(board, player)

        start = time.time()
        best_move = valid_moves[0]

        ordered = self._ordered_moves(valid_moves)

        for depth in range(1, self.max_depth + 1):
            if time.time() - start > self.time_limit:
                break

            score, move = self._negamax(P, M, depth, -1e9, 1e9, start, ordered)

            if move is not None:
                best_move = move

        return best_move

    def _negamax(self, P, M, depth, alpha, beta, start_time, root_moves):
        if time.time() - start_time > self.time_limit:
            return 0, None

        key = (P, M)
        if key in self.transposition and self.transposition[key][0] >= depth:
            return self.transposition[key][1], None

        valid_moves = [c for c in root_moves if not (M & self.top_mask[c])]

        if len(valid_moves) == 0:
            return 0, None

        # immediate win check
        for col in valid_moves:
            newP, newM = self._play(P, M, col)
            if self._is_win(newP):
                return (1000000 + depth), col

        if depth == 0:
            return 0, None

        best_score = -1e9
        best_move = valid_moves[0]

        ordered = self._ordered_moves(valid_moves)

        for col in ordered:
            newP, newM = self._play(P, M, col)

            score, _ = self._negamax(newP, newM, depth - 1, -beta, -alpha, start_time, ordered)
            score = -score

            if score > best_score:
                best_score = score
                best_move = col

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        self.transposition[key] = (depth, best_score)
        return best_score, best_move

    def _play(self, P, M, col):
        move = (M + self.bottom_mask[col]) & ~M
        newM = M | move
        newP = newM ^ P
        return newP, newM

    def _ordered_moves(self, valid_moves):
        order = [3, 2, 4, 1, 5, 0, 6]
        return [c for c in order if c in valid_moves]

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