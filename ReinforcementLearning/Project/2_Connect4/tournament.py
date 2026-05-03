import tqdm
import numpy as np


class TournamentRunner:
    def __init__(self, env):
        self.env = env

    def _get_valid_actions(self):
        return np.where(self.env.board[0] == 0)[0]

    def play_game(self, agentA, agentB, render=False):
        obs, _ = self.env.reset()
        done = False
        moves = 0

        while not done:
            player = self.env.current_player

            valid_actions = self._get_valid_actions()

            if player == 1:
                action = agentA.act(obs, player)
            else:
                action = agentB.act(obs, player)

            if action not in valid_actions:
                winner = -player
                return {
                    "winner": winner,
                    "draw": False,
                    "moves": moves,
                    "invalid_move": True
                }

            obs, reward, done, _, info = self.env.step(action)
            moves += 1

            if render:
                print(f"Move {moves}, Player {player}, Action {action}")
                self.env.render()
                print()

        return {
            "winner": info.get("winner", 0),
            "draw": info.get("draw", False),
            "moves": moves,
            "invalid_move": False
        }

    def run(self, agentA, agentB, num_games=100, switch_sides=True):
        results = {
            "A_wins": 0,
            "B_wins": 0,
            "draws": 0,
            "invalid_games": 0
        }

        for i in tqdm.tqdm(range(num_games)):
            # optional fairness: alternate starting player
            if switch_sides and (i % 2 == 1):
                result = self.play_game(agentB, agentA)

                if result["draw"]:
                    results["draws"] += 1
                elif result["winner"] == 1:
                    results["B_wins"] += 1
                elif result["winner"] == -1:
                    results["A_wins"] += 1
            else:
                result = self.play_game(agentA, agentB)

                if result["draw"]:
                    results["draws"] += 1
                elif result["winner"] == 1:
                    results["A_wins"] += 1
                elif result["winner"] == -1:
                    results["B_wins"] += 1

            if result.get("invalid_move", False):
                results["invalid_games"] += 1

        return results