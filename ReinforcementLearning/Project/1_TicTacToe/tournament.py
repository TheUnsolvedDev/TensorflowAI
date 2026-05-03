import tqdm

class TournamentRunner:
    def __init__(self, env):
        self.env = env

    def play_game(self, agentA, agentB, render=False):
        obs, _ = self.env.reset()
        done = False
        moves = 0

        while not done:
            player = self.env.current_player

            if player == 1:
                action = agentA.act(obs, player)
            else:
                action = agentB.act(obs, player)

            obs, reward, done, _, info = self.env.step(action)
            moves += 1

            if render:
                print(f"Move {moves}, Player {player}")
                self.env.render()
                print()

        return {
            "winner": info.get("winner", 0),
            "draw": info.get("draw", False),
            "moves": moves,
        }

    def run(self, agentA, agentB, num_games=100):
        results = {"A_wins": 0, "B_wins": 0, "draws": 0}

        for _ in tqdm.tqdm(range(num_games)):
            result = self.play_game(agentA, agentB)

            if result["draw"]:
                results["draws"] += 1
            elif result["winner"] == 1:
                results["A_wins"] += 1
            elif result["winner"] == -1:
                results["B_wins"] += 1

        return results
    