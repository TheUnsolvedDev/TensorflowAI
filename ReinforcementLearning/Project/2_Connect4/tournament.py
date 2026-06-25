import numpy as np
import tqdm

from agents.dqn_agent.wrapper import DQNWrapper

import os
import math
import numpy as np
import matplotlib.pyplot as plt


def save_boards_binary_map(boards, step, players, actions, rewards, dones, invalids, save_dir="debug_binary"):
    os.makedirs(save_dir, exist_ok=True)

    B = boards.shape[0]

    cols = int(math.ceil(np.sqrt(B)))
    rows = int(math.ceil(B / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    axes = np.array(axes).reshape(-1)

    for i in range(B):
        board = boards[i]

        vis = np.zeros_like(board, dtype=np.float32)

        vis[board == -1] = -1.0
        vis[board == 0] = 0.0
        vis[board == 1] = 1.0

        ax = axes[i]

        ax.imshow(vis, interpolation="nearest",
                  aspect="equal", vmin=-1, vmax=1)

        ax.set_xticks(np.arange(board.shape[1]))
        ax.set_yticks(np.arange(board.shape[0]))

        ax.set_title(
            f"Env={i} Step={step}\n"
            f"P={players[i]} A={actions[i]}\n"
            f"R={rewards[i]:.2f} D={dones[i]} I={invalids[i]}"
        )

        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                ax.text(c, r, str(board[r, c]),
                        ha="center", va="center", fontsize=8)

    for j in range(B, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    plt.savefig(os.path.join(
        save_dir, f"step_{step:04d}.png"), bbox_inches="tight")

    plt.close(fig)


class VectorizedTournamentRunner:
    def __init__(self, env):
        self.env = env
        self.B = env.B

    def print_boards_side_by_side(self, spacing=4):
        boards = self.env.board
        num_boards = boards.shape[0]
        rows = boards.shape[1]

        for r in range(rows):
            row_strings = []

            for b in range(num_boards):
                row_str = " ".join(map(str, boards[b, r]))
                row_strings.append(row_str)

            print((" " * spacing).join(row_strings))

    def _unwrap_agent(self, agent):
        visited = set()
        while hasattr(agent, "agent") and id(agent) not in visited:
            visited.add(id(agent))
            agent = agent.agent
        return agent

    def _act_agent(self, agent, obs_canonical, obs_plane, mask, player):
        base_agent = self._unwrap_agent(agent)
        agent_type = getattr(base_agent, "type", "")
        obs_absolute = self.env.board.copy()
        if "DQN" in agent_type or "CNN" in agent_type:
            obs = obs_plane
        else:
            obs = obs_absolute
        return agent.act(obs, mask, player)

    def play_batch(self, agentA, agentB, render=True):
        obs, _ = self.env.reset()
        done = np.zeros(self.B, dtype=bool)
        moves = np.zeros(self.B, dtype=np.int32)

        winners = np.zeros(self.B, dtype=np.int8)
        draws = np.zeros(self.B, dtype=bool)
        invalid_games = np.zeros(self.B, dtype=bool)
        step_idx = 0
        while not np.all(done):
            player = self.env.current_player.copy()
            obs_canonical = self.env.get_canonical_obs()
            mask = self.env.get_action_mask()
            obs_plane = self.env.board_to_planes(obs_canonical)
            actions_A = self._act_agent(
                agentA, obs_canonical, obs_plane, mask, player)
            actions_B = self._act_agent(
                agentB, obs_canonical, obs_plane, mask, player)

            actions = np.where(player == 1, actions_A, actions_B)
            invalid = (mask[np.arange(self.B), actions] == 0) & (~done)
            invalid_games |= invalid

            next_obs, reward, step_done, _, _ = self.env.step(actions)
            just_finished = step_done & (~done)
            win_mask = just_finished & (reward == 1.0)
            winners[win_mask] = player[win_mask]

            draw_mask = just_finished & (reward == 0.0)
            draws[draw_mask] = True
            done |= step_done
            moves += (~done).astype(np.int32)
            obs = next_obs
            if render:
                save_boards_binary_map(
                    boards=self.env.board,
                    step=step_idx,
                    players=player,
                    actions=actions,
                    rewards=reward,
                    dones=step_done,
                    invalids=invalid
                )
                step_idx += 1

        return {
            "A_wins": int(np.sum(winners == 1)),
            "B_wins": int(np.sum(winners == -1)),
            "draws": int(np.sum(draws)),
            "invalid_games": int(np.sum(invalid_games)),
            "total_games": int(self.B),
            "avg_moves": float(moves.mean())
        }

    def run(self, agentA, agentB, num_batches=100, switch_sides=True):
        results = {
            "A_wins": 0,
            "B_wins": 0,
            "draws": 0,
            "invalid_games": 0,
            "total_games": 0
        }

        for i in tqdm.tqdm(range(num_batches)):
            if switch_sides and (i % 2 == 1):
                r = self.play_batch(agentB, agentA)

                results["A_wins"] += r["B_wins"]
                results["B_wins"] += r["A_wins"]
            else:
                r = self.play_batch(agentA, agentB)

                results["A_wins"] += r["A_wins"]
                results["B_wins"] += r["B_wins"]

            results["draws"] += r["draws"]
            results["invalid_games"] += r["invalid_games"]
            results["total_games"] += r["total_games"]

        return results
