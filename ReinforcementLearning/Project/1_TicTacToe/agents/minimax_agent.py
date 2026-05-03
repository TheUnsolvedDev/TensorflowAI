import numpy as np


class MinimaxAgent:
    def __init__(self, max_depth=9):
        self.max_depth = max_depth

    def act(self, obs, player):
        board = obs.reshape(3, 3)
        best_score = -np.inf
        best_move = None

        for move in np.where(obs == 0)[0]:
            r, c = divmod(move, 3)
            board[r, c] = player

            score = self._minimax(board, -player, player, depth=1)

            board[r, c] = 0

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(self, board, current, root, depth):
        winner = self._check_winner(board)

        if winner is not None:
            return (10 - depth) if winner == root else (depth - 10)

        if np.all(board != 0):
            return 0

        if depth >= self.max_depth:
            return self._evaluate(board, root)

        moves = np.where(board.flatten() == 0)[0]

        if current == root:
            best = -np.inf
            for m in moves:
                r, c = divmod(m, 3)
                board[r, c] = current
                val = self._minimax(board, -current, root, depth + 1)
                board[r, c] = 0
                best = max(best, val)
            return best
        else:
            best = np.inf
            for m in moves:
                r, c = divmod(m, 3)
                board[r, c] = current
                val = self._minimax(board, -current, root, depth + 1)
                board[r, c] = 0
                best = min(best, val)
            return best

    def _evaluate(self, board, root):
        score = 0

        lines = []
        lines.extend(list(board))
        lines.extend(list(board.T))
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))

        for line in lines:
            if -root not in line:
                score += np.sum(line == root)
            if root not in line:
                score -= np.sum(line == -root)

        return score

    def _check_winner(self, b):
        for i in range(3):
            if abs(np.sum(b[i])) == 3:
                return int(np.sign(np.sum(b[i])))
            if abs(np.sum(b[:, i])) == 3:
                return int(np.sign(np.sum(b[:, i])))

        d1 = np.sum(np.diag(b))
        d2 = np.sum(np.diag(np.fliplr(b)))

        if abs(d1) == 3:
            return int(np.sign(d1))
        if abs(d2) == 3:
            return int(np.sign(d2))

        return None