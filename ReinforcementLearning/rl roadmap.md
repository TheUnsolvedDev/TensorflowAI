# A Technical Road Map of (Deep) Reinforcement Learning

### From Policy Gradients to the Current Frontier

> Scope: the lineage of *deep* RL with its classical roots, organized by sub-domain (policy gradient, value-based, continuous control, model-based, offline, imitation/IRL, diffusion, multi-agent, exploration, representation learning, RLHF), with the canonical algorithms, the *mechanism* each introduced, the benchmark/environment it was validated on, and what it actually solved. A consolidated timeline and a benchmark map sit at the end.
> 
> Knowledge current to early 2026. Items from late-2025/2026 marked ⚠ should be re-verified against primary sources, as that frontier moves monthly.

-----

## 0. The Problem Statement (so the map has axes)

Everything below is an attempt to solve, approximate, or sidestep the **Bellman optimality** condition in a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0)$:

$$Q^*(s,a) = \mathbb{E}\big[r + \gamma \max_{a’} Q^*(s’,a’)\big], \qquad J(\pi)=\mathbb{E}_{\tau\sim\pi}\Big[\textstyle\sum_t \gamma^t r_t\Big].$$

The field splits on **how** you optimize $J$:

|Axis                                         |Question it answers                                                                                 |
|---------------------------------------------|----------------------------------------------------------------------------------------------------|
|Value-based vs. policy-based vs. actor-critic|Do you learn $Q$, learn $\pi$ directly, or both?                                                    |
|Model-free vs. model-based                   |Do you learn/use $P$ and $r$, or only sample them?                                                  |
|On-policy vs. off-policy                     |Must data come from the current $\pi$, or can you reuse a replay buffer / a fixed dataset (offline)?|
|Online vs. offline (batch)                   |Can you interact, or only learn from logged data?                                                   |
|Single- vs. multi-agent                      |One policy, or interacting policies with non-stationarity?                                          |
|Reward-given vs. reward-inferred             |Is $r$ provided, or recovered from demonstrations / preferences (IRL, RLHF)?                        |

Keep these axes in mind; each section moves along one or more of them.

-----

## 1. Classical Roots (pre-deep, but the equations never left)

|Method                                                                     |Idea                                                                                       |Why it still matters                                                                                                    |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
|**Dynamic Programming** (Bellman, 1957)                                    |Value/policy iteration with a known model                                                  |Defines optimality; every TD method is sampled DP                                                                       |
|**TD(λ)** (Sutton, 1988)                                                   |Bootstrapped value estimation from raw experience                                          |The bias–variance dial (λ) reappears as GAE                                                                             |
|**Q-learning** (Watkins, 1989)                                             |Off-policy tabular control via the max operator                                            |Direct ancestor of DQN                                                                                                  |
|**SARSA** (Rummery & Niranjan, 1994)                                       |On-policy TD control                                                                       |The on/off-policy distinction in its purest form                                                                        |
|**REINFORCE** (Williams, 1992)                                             |Monte-Carlo policy gradient via the log-likelihood trick                                   |The seed of the entire policy-gradient branch                                                                           |
|**Actor-Critic** (Barto, Sutton & Anderson, 1983; Konda & Tsitsiklis, 2000)|A critic baselines the actor’s gradient                                                    |The template for A3C, PPO, SAC, …                                                                                       |
|**MDP homomorphisms & state abstraction** (Ravindran & Barto, 2001–2004)   |Formal equivalence of MDPs under state/action maps; abstraction preserving optimal behavior|The theoretical grounding for *latent dynamics* and bisimulation — directly relevant to representation-learning RL (§11)|

The **policy gradient theorem** is the hinge:
$$\nabla_\theta J(\theta) = \mathbb{E}*{\pi*\theta}\big[\nabla_\theta \log \pi_\theta(a|s), \Psi_t\big],$$
where $\Psi_t$ can be the return, the advantage $A(s,a)$, the TD error, etc. Every choice of $\Psi$ is a different algorithm.

-----

## 2. The Policy-Gradient Lineage (on-policy backbone)

**REINFORCE (1992)** → high-variance MC gradients; needs a baseline.

**Vanilla Actor-Critic** → replace the MC return with a learned critic $V_\phi$; advantage $A = r + \gamma V(s’) - V(s)$ cuts variance at the cost of bias.

**A3C / A2C (Mnih et al., 2016)** — *Asynchronous Advantage Actor-Critic.* Parallel actors decorrelate data without a replay buffer; n-step advantages; entropy bonus for exploration. A2C is the synchronous, GPU-friendly variant.

- **Solved:** Atari ALE and continuous control (MuJoCo/TORCS) without experience replay, on CPU clusters.

**GAE (Schulman et al., 2016)** — *Generalized Advantage Estimation.* The λ-return idea applied to advantages, $\hat A^{GAE(\gamma,\lambda)}=\sum (\gamma\lambda)^l \delta_{t+l}$ — the standard variance-reduction knob in TRPO/PPO to this day.

**TRPO (Schulman et al., 2015)** — *Trust Region Policy Optimization.* Constrains each update to a KL trust region, $\max_\theta \mathbb{E}[\tfrac{\pi_\theta}{\pi_{old}}A]$ s.t. $\mathbb{E}[D_{KL}]\le\delta$. Monotonic-improvement guarantee; solved via conjugate gradient + line search.

- **Solved:** robust locomotion on MuJoCo (Hopper, Walker, Humanoid) without per-task tuning.

**PPO (Schulman et al., 2017)** — *Proximal Policy Optimization.* Replaces the hard KL constraint with a clipped surrogate $\min(r_t A_t,\ \text{clip}(r_t,1\pm\epsilon)A_t)$. First-order, trivially parallel, robust.

- **Why it dominates:** it is the default workhorse for control, robotics (Isaac), game-playing (OpenAI Five, §9), and is the policy optimizer in classic **RLHF** (§12).
- **Solved:** the whole MuJoCo/Atari suite at strong baselines with one hyperparameter set; scaled to Dota 2.

**Phasic Policy Gradient / V-MPO / IMPALA** — refinements: IMPALA (Espeholt et al., 2018) added the **V-trace** off-policy correction for massively distributed actor-learner setups; MPO/V-MPO (Abdolmaleki et al., 2018) recast policy improvement as an EM / KL-regularized inference problem.

-----

## 3. The Value-Based (Deep Q) Lineage

**DQN (Mnih et al., 2013/2015, Nature)** — the spark of deep RL. CNN over raw pixels + **experience replay** (breaks correlation) + **target network** (stabilizes the bootstrap). Minimizes $(r+\gamma\max_{a’}Q_{\bar\theta}(s’,a’) - Q_\theta(s,a))^2$.

- **Solved:** human-level play on 49 Atari games (ALE) from pixels — the result that launched the modern field.

Then a decade of fixes, each patching a specific pathology:

|Algorithm                                              |Pathology fixed                                       |Mechanism                                                                             |
|-------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------------|
|**Double DQN** (van Hasselt, 2016)                     |Max-operator overestimation                           |Decouple action selection from evaluation                                             |
|**Dueling DQN** (Wang et al., 2016)                    |Wasted capacity on $Q$ where action doesn’t matter    |Split into $V(s)+A(s,a)$ streams                                                      |
|**Prioritized Experience Replay** (Schaul et al., 2016)|Uniform replay wastes samples                         |Sample by TD-error magnitude                                                          |
|**C51 / Distributional RL** (Bellemare et al., 2017)   |A scalar $Q$ throws away risk info                    |Learn the *distribution* of returns over fixed atoms                                  |
|**QR-DQN / IQN** (Dabney et al., 2018)                 |C51’s fixed support                                   |Quantile regression / implicit quantile networks                                      |
|**Noisy Nets** (Fortunato et al., 2018)                |ε-greedy is dumb exploration                          |Learnable parametric noise in weights                                                 |
|**Rainbow** (Hessel et al., 2018)                      |Each fix in isolation                                 |Combine all six — the canonical “kitchen-sink” Atari agent                            |
|**R2D2** (Kapturowski et al., 2019)                    |Partial observability, short memory                   |Recurrent (LSTM) Q-net + distributed replay                                           |
|**NGU / Agent57** (Badia et al., 2020)                 |The last few unbeaten Atari games (Montezuma, Pitfall)|Episodic + life-long intrinsic motivation; a meta-controller over exploration policies|

**Agent57** was the first agent to beat the human baseline on **all 57 Atari games** — closing the ALE chapter for raw score (sample-efficiency remained open, see §6/§11).

-----

## 4. Continuous Control (off-policy actor-critic)

Discrete $\max_a$ doesn’t work in $\mathbb{R}^n$ action spaces; this branch makes Q-learning continuous.

**DDPG (Lillicrap et al., 2016)** — *Deep Deterministic Policy Gradient.* Deterministic actor $\mu_\theta(s)$ trained by the **deterministic policy gradient** $\nabla_\theta \mathbb{E}[Q(s,\mu_\theta(s))]$; off-policy with replay + target nets. Powerful but brittle (overestimation, hyperparameter sensitivity).

**TD3 (Fujimoto et al., 2018)** — *Twin Delayed DDPG.* Three targeted fixes: (1) **clipped double-Q** (min of two critics) kills overestimation; (2) **delayed** policy updates; (3) **target-policy smoothing**. The reliable deterministic baseline.

**SAC (Haarnoja et al., 2018)** — *Soft Actor-Critic.* **Maximum-entropy RL**: maximize $\mathbb{E}[\sum r + \alpha\mathcal H(\pi(\cdot|s))]$. Stochastic actor, automatic temperature tuning, off-policy sample efficiency, and far less sensitive than DDPG/TD3.

- **Solved / dominates:** MuJoCo and DMC continuous control as the default model-free baseline; the standard real-robot model-free choice.

These three (TD3 + SAC especially) are the off-policy critics that later **offline** and **diffusion** methods build directly on top of.

-----

## 5. Model-Based RL (learn $P$, plan or imagine)

The sample-efficiency branch: instead of millions of env steps, learn a dynamics model and plan/dream inside it.

**Dyna (Sutton, 1991)** — interleave real experience with planning on a learned model. The conceptual ancestor.

**PETS (Chua et al., 2018)** — *Probabilistic Ensembles with Trajectory Sampling.* Ensemble of probabilistic dynamics models + **MPC/CEM** planning. Strong sample efficiency on MuJoCo from pixels-free states.

**World Models (Ha & Schmidhuber, 2018)** — VAE encodes observations into a latent $z$; an MDN-RNN predicts latent dynamics; a tiny controller is even evolved *entirely inside the dream*. The first vivid demonstration of training in a learned latent imagination.

**PlaNet (Hafner et al., 2019)** — *Deep Planning Network.* The **RSSM** (Recurrent State-Space Model) with combined deterministic+stochastic latent state; latent MPC planning from pixels.

**Dreamer v1→v2→v3 (Hafner et al., 2020/2021/2023)** — learn an actor-critic *purely from latent rollouts* of the RSSM world model; backprop value gradients through imagined trajectories.

- **DreamerV2:** first world-model agent to reach human-level on Atari from a world model.
- **DreamerV3:** *one fixed hyperparameter set* across 150+ tasks (Atari, DMC, Crafter, Minecraft) — and famously **collected diamonds in Minecraft from scratch** without human data. The current general model-based reference.

**The MuZero family** — model-based *without* reconstructing observations; learn only what matters for value/policy/reward (a *value-equivalent* latent model), then plan with MCTS in latent space.

|Algorithm                                      |Adds                                                                                                                                         |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
|**MuZero** (Schrittwieser et al., 2020)        |Learned latent dynamics + MCTS; mastered Go/Chess/Shogi **and** Atari with no given rules                                                    |
|**Sampled MuZero** (Hubert et al., 2021)       |Sampled-action MCTS → continuous / large action spaces                                                                                       |
|**EfficientZero** (Ye et al., 2021)            |Self-supervised consistency (SPR-style), value-prefix, off-policy correction → **median human-level Atari in 100k frames** (~2 hours of play)|
|**Stochastic MuZero** (Antonoglou et al., 2022)|Chance nodes for stochastic environments (2048, backgammon)                                                                                  |


> **Research note (your lane):** the value-equivalent / latent-dynamics principle here — predict only quantities relevant to the optimal policy, not the full observation — is exactly the territory where **MDP-homomorphism / bisimulation** abstraction (§11) meets MuZero-style planning. This is the conceptual seam your latent-dynamics work sits in.

-----

## 6. Offline (Batch) RL — learn from a fixed dataset, no interaction

The central pathology: **distributional shift**. Bootstrapping on out-of-distribution actions makes $Q$ explode. Every method below is a way to stay near the data.

|Algorithm                                       |Anti-OOD mechanism                                                                                                                                                                                                            |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**BCQ** (Fujimoto et al., 2019)                 |Only consider actions a generative model says are in-distribution                                                                                                                                                             |
|**BEAR** (Kumar et al., 2019)                   |Constrain policy to dataset support via MMD                                                                                                                                                                                   |
|**CQL** (Kumar et al., 2020)                    |**Conservative Q-learning**: penalize Q on OOD actions → a value *lower bound*. The most-cited offline baseline                                                                                                               |
|**TD3+BC** (Fujimoto & Gu, 2021)                |Add a behavior-cloning term to TD3; minimalist and shockingly strong                                                                                                                                                          |
|**IQL** (Kostrikov et al., 2022)                |**Implicit Q-Learning**: expectile regression learns $Q$ *without ever querying OOD actions* — no explicit constraint needed                                                                                                  |
|**XQL / Extreme Q-Learning** (Garg et al., 2023)|Models the Bellman-optimal **max** via **Gumbel/extreme-value regression**, recovering MaxEnt RL *without sampling actions for the entropy term*; tightens IQL’s implicit-max into a principled loss. Works online and offline|

**Sequence-modeling view of offline RL** (treat RL as conditional sequence prediction):

- **Decision Transformer (Chen et al., 2021)** — a GPT conditioned on (return-to-go, state, action); generate actions by *prompting with desired return*. RL as autoregressive modeling, no TD bootstrapping.
- **Trajectory Transformer (Janner et al., 2021)** — model full trajectories with a Transformer + beam search as planning.

**Benchmark:** the whole offline branch is measured on **D4RL** (MuJoCo locomotion mixtures, AntMaze, Adroit, Kitchen, CARLA) and **RL Unplugged**.

-----

## 7. Imitation Learning & Inverse RL (no reward, only demonstrations)

**Behavioral Cloning (Pomerleau, 1988 — ALVINN)** — supervised regression $\min \mathbb{E}_{(s,a)\sim D}[-\log\pi(a|s)]$. Simple, but suffers **covariate shift / compounding error**: one mistake takes you off the demo manifold, errors $\sim O(\epsilon T^2)$.

**DAgger (Ross et al., 2011)** — *Dataset Aggregation.* Iteratively roll out the learner, query the expert on visited states, aggregate. Reduces error to $O(\epsilon T)$ — fixes covariate shift, but needs an interactive expert.

**Inverse RL** — recover $r$, then plan:

- **GAIL (Ho & Ermon, 2016)** — *Generative Adversarial Imitation Learning.* A discriminator distinguishes expert vs. policy trajectories; the policy (PPO/TRPO) is rewarded for fooling it. Matches occupancy measures without recovering $r$ explicitly.
- **AIRL (Fu et al., 2018)** — recovers a *transferable, disentangled* reward, robust to dynamics change.
- **SQIL (Reddy et al., 2019)** — reward = +1 on demo transitions, 0 elsewhere; soft-Q learning. Simpler than adversarial training.
- **IQ-Learn (Garg et al., 2021)** — learn a single **Q implicitly representing both reward and policy**; non-adversarial, SOTA sample efficiency on imitation benchmarks.

**Benchmarks:** MuJoCo (locomotion from few demos), Atari from demos, and robotics manipulation.

-----

## 8. Diffusion (and Flow/Consistency) RL

The newest model class: use **diffusion models** as expressive multimodal policies or planners, fixing BC’s unimodality and offline RL’s restrictive policy classes.

**Planning-as-generation:**

- **Diffuser (Janner et al., 2022)** — diffuse over *entire trajectories*; guide sampling with a learned return gradient (classifier guidance) → planning = conditional sampling.
- **Decision Diffuser (Ajay et al., 2023)** — **classifier-free** conditioning on return/constraints/skills; cleaner, composable conditioning.

**Diffusion as the policy (offline RL):**

- **Diffusion-QL (Wang et al., 2023)** — a diffusion policy regularized by a Q-learning term; the diffusion model expresses multimodal behavior, Q pushes toward high value.
- **IDQL (Hansen-Estruch et al., 2023)** — reinterprets IQL with a diffusion **behavior** policy + importance reweighting at sampling time.
- **QGPO (Lu et al., 2023)** — *Q-Guided Policy Optimization*: exact energy-guided sampling with a learned intermediate guidance, avoiding unstable backprop-through-sampling.
- **SRPO** — score-regularized policy that distills the diffusion behavior model into a fast deterministic policy.

**Diffusion for visuomotor robotics:**

- **Diffusion Policy (Chi et al., 2023)** — denoise action *sequences* conditioned on visual observations; receding-horizon control. Now a dominant real-robot manipulation paradigm (precise, multimodal, stable training).
- **Consistency / flow-matching policies (2023–2025 ⚠)** — distill the multi-step denoiser into 1–few step generation for real-time control; flow-matching variants for speed.

**Benchmarks:** D4RL (offline), and real/ sim manipulation — **robomimic, ManiSkill, RoboCasa, Push-T**.

-----

## 9. Multi-Agent RL (MARL)

The new difficulty: **non-stationarity** (other agents’ policies change), credit assignment, and partial observability. Standard frame: **CTDE** — *Centralized Training, Decentralized Execution*.

|Algorithm                       |Setting                           |Idea                                                                             |
|--------------------------------|----------------------------------|---------------------------------------------------------------------------------|
|**Independent Q / IPPO**        |baseline                          |Treat others as part of the environment (non-stationary, but surprisingly strong)|
|**MADDPG** (Lowe et al., 2017)  |mixed coop/competitive, continuous|Centralized critic sees all agents; decentralized actors                         |
|**VDN** (Sunehag et al., 2017)  |cooperative                       |Factor team value as a **sum** of per-agent values                               |
|**QMIX** (Rashid et al., 2018)  |cooperative                       |Monotonic **mixing network** generalizes VDN; the SMAC workhorse                 |
|**QTRAN** (Son et al., 2019)    |cooperative                       |Removes QMIX’s monotonicity restriction (at cost of complexity)                  |
|**COMA** (Foerster et al., 2018)|cooperative                       |**Counterfactual** baseline for multi-agent credit assignment                    |
|**MAPPO** (Yu et al., 2021)     |coop                              |PPO + centralized value; the strong, simple modern default                       |

**Self-play / emergent (population-based) — the headline results:**

- **AlphaGo (2016)** → **AlphaGo Zero (2017)** → **AlphaZero (2018)**: MCTS + self-play + a single value/policy net; tabula-rasa superhuman Go/Chess/Shogi.
- **OpenAI Five (2019):** scaled **PPO** + LSTM + self-play to **Dota 2 5v5** — beat the world champions; the proof that PPO scales to enormous horizons.
- **AlphaStar (2019):** league/population self-play → Grandmaster **StarCraft II** under pro-like constraints.

**Benchmarks:** **PettingZoo** (the Gym of MARL), **SMAC / SMACv2** (StarCraft micro), **Google Research Football**, **Hanabi** (theory-of-mind), **Melting Pot** (social generalization), **Overcooked** (human-AI coordination).

-----

## 10. Hierarchical RL (HRL) — temporal abstraction for long horizons

The problem HRL attacks: when the horizon is tens of thousands of steps and reward is sparse (diamonds, Montezuma’s Revenge), a flat policy cannot assign credit across the gap. HRL introduces **temporal abstraction** — a high-level policy picks subgoals/options; low-level policies execute primitive actions over extended time. Formally this lifts the MDP to a **semi-MDP (SMDP)**, where decisions persist for variable durations.

|Method                                              |Idea                                                                                                                    |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
|**Options framework** (Sutton, Precup & Singh, 1999)|An *option* = (initiation set, intra-option policy, termination condition); the foundational SMDP formalism for HRL     |
|**MAXQ** (Dietterich, 2000)                         |Recursive value-function decomposition over a hand-given task hierarchy                                                 |
|**h-DQN** (Kulkarni et al., 2016)                   |Meta-controller proposes intrinsic goals, a controller achieves them — an early sparse-reward (Montezuma-style) result  |
|**Option-Critic** (Bacon et al., 2017)              |Learn the intra-option policies *and* their termination functions end-to-end via the option policy-gradient             |
|**FeUdal Networks (FuN)** (Vezhnevets et al., 2017) |A *Manager* sets directional goals in a **learned latent space**; a *Worker* acts to follow them — latent-space subgoals|
|**HIRO** (Nachum et al., 2018)                      |Off-policy, data-efficient goal-conditioned HRL with **goal relabeling** to cope with the non-stationary lower level    |
|**HAC** (Levy et al., 2019)                         |Hindsight at multiple levels of the hierarchy, training all levels in parallel                                          |

**Connection to abstraction (your lane):** options/SMDPs have their own homomorphism theory — Ravindran & Barto extended **MDP homomorphisms to SMDPs/options**, so HRL is, formally, *state-and-temporal* abstraction. HRL and the representation-sufficiency story (§11) are two faces of the same question: *which structure can be collapsed without losing optimal behavior?* FeUdal’s latent goals are the bridge between them.

**Benchmarks:** long-horizon navigation (AntMaze, point-maze), Montezuma’s Revenge, and — the headline stress test — **MineRL / Minecraft** (§14), whose tech tree is almost a textbook options hierarchy.

-----

## 11. Exploration & Representation Learning (the sample-efficiency engines)

### Exploration (beating sparse reward, e.g. Montezuma’s Revenge)

- **Pseudo-counts / count-based** (Bellemare et al., 2016) — density model → exploration bonus.
- **ICM** (Pathak et al., 2017) — *Intrinsic Curiosity*: reward prediction error in a learned feature space.
- **RND** (Burda et al., 2018) — *Random Network Distillation*: bonus = error predicting a fixed random net; simple, strong.
- **Go-Explore** (Ecoffet et al., 2021) — remember and return to promising states, then explore — **solved Montezuma’s Revenge and Pitfall**, the long-standing hard-exploration Atari games.

### Representation learning (the branch closest to your thesis)

Sample efficiency from pixels comes from *good latent states*:

- **CURL** (Laskin et al., 2020) — contrastive auxiliary loss alongside RL.
- **RAD / DrQ / DrQ-v2** (Laskin/Kostrikov/Yarats, 2020–2021) — data augmentation is most of the win; DrQ-v2 is a strong model-free pixel baseline on DMC.
- **SPR** (Schwarzer et al., 2021) — self-predictive latent dynamics as an auxiliary task; a key ingredient inside EfficientZero.
- **DeepMDP** (Gelada et al., 2019) — learn a latent MDP whose dynamics/reward match the real one; bounds tie latent quality to value error.
- **Deep Bisimulation for Control / DBC** (Zhang et al., 2021) — **bisimulation metrics**: states are close iff they yield the same future rewards under all action sequences — invariant to task-irrelevant detail (e.g. backgrounds).
- **MDP homomorphisms** (Ravindran & Barto; revived in deep settings by van der Pol et al., 2020, plan-symmetric / equivariant nets) — the formal abstraction theory underneath all of the above, and the principled statement of *what a latent state should preserve*.

> This cluster — value-equivalence (MuZero), self-prediction (SPR/Dreamer), bisimulation (DBC), and homomorphisms — is precisely the convergence point your latent-space RL research operates in: different routes to the same question of *which information a latent representation must keep to be control-sufficient*.

**Benchmarks:** **Atari 100k** and **DMC-100k** (sample-efficiency regime), **Distracting Control Suite** (representation robustness), **Procgen** (generalization across procedurally generated levels), **Crafter** (open-ended achievements).

-----

## 12. RL for Sequence Models / RLHF (arguably today’s most-deployed RL)

The branch that put RL into every chatbot. Reward is *inferred from human preferences*.

- **Deep RL from Human Preferences (Christiano et al., 2017)** — learn a reward model from pairwise comparisons, then optimize with RL. The template.
- **InstructGPT / RLHF (Ouyang et al., 2022)** — SFT → reward model → **PPO** against the RM with a KL-to-reference penalty. The recipe behind instruction-tuned LLMs.
- **DPO (Rafailov et al., 2023)** — *Direct Preference Optimization*: a closed-form loss that optimizes the same objective **without an explicit reward model or RL loop** — recasts the RLHF objective as classification on preference pairs. Hugely popular for its simplicity.
- **RLAIF / Constitutional AI (Bai et al., 2022)** — replace human labels with AI feedback against a set of principles.
- **GRPO (DeepSeek, 2024)** — *Group Relative Policy Optimization*: drop the value critic, normalize advantages within a group of sampled responses; the optimizer behind recent **reasoning** models (verifiable-reward RL on math/code).
- **2025–2026 frontier ⚠** — RL with verifiable rewards (RLVR) for reasoning, process-reward models, and large-scale on-policy RL on LLMs are evolving fast; verify specifics before citing.

**Benchmarks:** preference win-rates, MT-Bench/Arena-style human eval, and (for reasoning RL) math/code suites (GSM8K, MATH, competition benchmarks).

-----

## 13. Environments & Benchmarks — what each one *tests* and who solved it

|Environment / Suite                             |What it stresses                                   |Landmark “solver”                                                                  |
|------------------------------------------------|---------------------------------------------------|-----------------------------------------------------------------------------------|
|**OpenAI Gym → Gymnasium (Farama)**             |Standard API; the lingua franca of RL              |— (the interface itself)                                                           |
|**Arcade Learning Environment (ALE / Atari)**   |Pixels, discrete control, diverse tasks            |DQN (human-level), Rainbow, **Agent57** (all 57), EfficientZero (100k)             |
|**MuJoCo** (now open-source)                    |Continuous locomotion control                      |TRPO/PPO/TD3/**SAC**                                                               |
|**DeepMind Control Suite (DMC)**                |Continuous control from states & pixels            |SAC, DrQ-v2, **DreamerV3**                                                         |
|**D4RL**                                        |Offline RL with mixed-quality data                 |CQL, IQL, **XQL**, Diffusion-QL                                                    |
|**Meta-World (ML1/ML10/ML45)**                  |Multi-task & meta-RL manipulation                  |multi-task SAC, PEARL, MT/MAML variants                                            |
|**Procgen**                                     |Generalization over procedural levels              |PPO + data aug, PPG                                                                |
|**Crafter**                                     |Open-ended achievement breadth, sample efficiency  |**DreamerV3**                                                                      |
|**NetHack Learning Environment (NLE)**          |Extreme long-horizon, sparse, procedural           |open (still largely unsolved without symbolic help)                                |
|**MineRL / MineDojo / Minecraft**               |Open-world, long-horizon, sparse                   |**DreamerV3** (diamonds, from scratch); VPT (BC from video) — full breakdown in §14|
|**ManiSkill / robosuite / robomimic / RoboCasa**|Visuomotor manipulation                            |**Diffusion Policy**, BC-Transformer                                               |
|**PettingZoo**                                  |Multi-agent API standard                           |— (interface)                                                                      |
|**SMAC / SMACv2**                               |Cooperative MARL micro-management                  |**QMIX**, MAPPO                                                                    |
|**Hanabi / Overcooked / Melting Pot**           |Coordination, theory-of-mind, social generalization|open / human-AI coordination research                                              |
|**Isaac Gym / Isaac Lab, Brax, EnvPool**        |Massively parallel GPU/TPU simulation              |sim-to-real PPO (e.g., ANYmal locomotion)                                          |
|**Go / Chess / Shogi / Dota 2 / StarCraft II**  |Planning, self-play, huge horizons                 |**AlphaZero / MuZero**, **OpenAI Five**, **AlphaStar**                             |

-----

## 14. Case Study: Solving MineRL / Minecraft — where HRL, IL, IRL, and world models collide

This is the cleanest demonstration of the whole map converging on one MDP — and exactly where your “HRL vs. IRL solve the same thing” instinct gets tested.

**The problem (MineRL `ObtainDiamond`, NeurIPS 2019–2021).** From raw pixels, obtain a diamond: a ~10-stage tech tree (log → planks → crafting table → wooden pickaxe → cobblestone → stone pickaxe → iron ore → furnace → iron pickaxe → diamond), tens of thousands of timesteps, brutally sparse terminal reward, a huge structured action space, partial observability. Critically, the competition imposed a **tight sample/compute budget** *and* shipped a **human demonstration dataset** (MineRL-v0, ~60M frames). By design, pure online RL was infeasible — you *had* to exploit demonstrations. That single design choice is **why IL, IRL, and HRL all appear here**.

**Your two named families, and why they’re complements, not rivals:**

- **Imitation / Inverse RL attacks the *sparsity*** — it imports dense guidance from demonstrations, so you aren’t waiting for a one-in-a-billion accidental diamond. BC copies the action directly; IRL recovers the *reward/intent* behind it.
- **HRL attacks the *horizon*** — it factors the tech tree into subgoals (get-wood, get-stone, …), turning one impossible 20k-step credit-assignment problem into ten tractable short ones.

The strongest classical entries did **both**: a hierarchy of subtask policies, each *trained from demonstrations*. So your intuition is right in spirit — they target the two halves of the difficulty (reward sparsity vs. horizon length), and the best agents fuse them.

**The strategies, in the order the field tried them:**

|Strategy                                       |Representative work                                       |Mechanism                                                                                                                            |Outcome                                                                                   |
|-----------------------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
|**IL from demos (DQfD-style)**                 |DQfD (Hester et al., 2018) as the base technique          |Pretrain Q with combined TD + supervised-margin loss on demos, then continue with RL                                                 |The backbone of most 2019 entries                                                         |
|**Hierarchical IL + RL**                       |**ForgER** (Skrynnik et al., 2019) — top MineRL-2019 entry|Hierarchy of subtask policies + *forgetful experience replay* over demonstrations                                                    |Top placement on the 2019 `ObtainDiamond` track                                           |
|**Sample-efficiency-forced IL**                |MineRL-2020 rules                                         |Hard caps on env interaction made demonstrations mandatory                                                                           |Pushed the field fully toward IL + hierarchy                                              |
|**Learning from human feedback (no reward fn)**|**MineRL BASALT (2021)**                                  |Tasks (FindCave, MakeWaterfall, BuildHouse) judged by *humans*; no coded reward → preference/IRL-style learning                      |The explicit IRL/RLHF-flavored track                                                      |
|**Internet-scale IL via inverse dynamics**     |**VPT** (Baker et al., OpenAI, 2022)                      |Train an Inverse Dynamics Model on a little labeled data → pseudo-label ~70k hrs of web video → BC a foundation policy → RL fine-tune|**First agent to craft a diamond pickaxe**                                                |
|**Learned reward from language/video**         |**MineDojo + MineCLIP** (Fan et al., 2022)                |CLIP-style video↔text alignment yields a dense shaped reward from natural-language goals                                             |Open-ended, goal-conditioned agents                                                       |
|**Model-based, from scratch**                  |**DreamerV3** (Hafner et al., 2023)                       |World model + actor-critic trained in latent imagination; **no human data, no curriculum**                                           |**First to collect diamonds from scratch**                                                |
|**LLM-as-planner (hierarchical)**              |**Voyager**, **DEPS**, **Plan4MC**, **GITM** (2023)       |An LLM is the high-level planner / skill-composer; RL or scripted skills form the low level                                          |Open-ended lifelong skill acquisition; Voyager grows a reusable skill library autonomously|

**The synthesis worth remembering:**

1. **Two routes to the diamond.** *Import human knowledge* (IL/IRL → VPT) **or** *solve the horizon with a strong imaginer* (model-based → DreamerV3). VPT says “demonstrations make it tractable”; DreamerV3 says “a good enough world model can do it cold.” Both are legitimate answers to the same MDP — which is the whole point of having a map with multiple axes.
1. **HRL is the connective tissue.** Whether explicit (ForgER’s subtask hierarchy, Plan4MC’s skill graph, Voyager’s LLM planner) or implicit (DreamerV3’s long-horizon latent rollouts), *some* temporal abstraction carries the horizon.
1. **IL and IRL are the same impulse at different abstraction levels.** BC copies the *action*; IRL / BASALT / MineCLIP recover the *intent* (reward), which transfers better and composes with planning. HRL and IRL aren’t competitors — they’re orthogonal fixes that the winning systems stack.

-----

## 15. Consolidated Timeline (one glance)

```
1992  REINFORCE                         policy gradient seed
1999  Options framework                 temporal abstraction (HRL) seed
2013  DQN                               deep RL begins (Atari from pixels)
2015  TRPO · DDPG                       trust-region PG · deep continuous control
2016  A3C/A2C · GAE · Double/Dueling DQN · PER · h-DQN(HRL) · AlphaGo
2017  PPO · C51 · MADDPG · VDN · GAIL · Option-Critic · FeUdal · Christiano RLHF
2018  TD3 · SAC · IQN · Rainbow · QMIX · COMA · AIRL · ICM/RND · World Models · PETS · HIRO · DQfD · AlphaZero
2019  R2D2 · BCQ · BEAR · OpenAI Five · AlphaStar · PlaNet · DeepMDP · MuZero(preprint) · ForgER(MineRL)
2020  Agent57 · Dreamer · CQL · CURL/RAD/DrQ · MuZero(Nature) · Decision Transformer(seed)
2021  DreamerV2 · IQL · TD3+BC · Decision/Trajectory Transformer · EfficientZero · DBC · SPR · MAPPO · Go-Explore(Nature) · MineRL-BASALT
2022  Diffuser · InstructGPT(RLHF) · Stochastic MuZero · Decision Diffuser · VPT(diamond pickaxe) · MineDojo/MineCLIP
2023  DreamerV3(diamonds from scratch) · XQL · Diffusion-QL · IDQL · QGPO · Diffusion Policy · DPO · Voyager/DEPS/Plan4MC(LLM planners)
2024  GRPO · RL-with-verifiable-rewards for reasoning
2025–26 ⚠ flow/consistency policies · large-scale reasoning RL (verify primary sources)
```

-----

## 16. How to Read the Map as a Practitioner (decision guide)

- **Online, continuous control, just want a strong baseline?** → **SAC** (or TD3).
- **On-policy, robust, scalable, parallel?** → **PPO**.
- **Sample efficiency is the constraint (few env steps)?** → model-based: **DreamerV3** (general) or **EfficientZero** (discrete/Atari).
- **Discrete planning with a learnable model / search?** → **MuZero** family.
- **Only logged data, no interaction?** → offline: **IQL / CQL / XQL**, or **Diffusion-QL** for multimodal behavior; **Decision Transformer** if you prefer the sequence-modeling frame.
- **Only demonstrations, no reward?** → **BC**+**DAgger** if you can query an expert; **GAIL/IQ-Learn** if not.
- **Expressive, multimodal action distributions (esp. robot manipulation)?** → **Diffusion Policy**.
- **Multiple agents, cooperative?** → **QMIX / MAPPO** under CTDE.
- **Long horizon + sparse reward (Minecraft-like)?** → temporal abstraction (**HRL**/options) and/or demonstrations (**IL/IRL**, VPT-style), and/or a strong world model (**DreamerV3**). The strongest systems stack all three — see §14.
- **Reward defined only by human/AI preference (LLMs)?** → **PPO-RLHF** or **DPO**; **GRPO** for verifiable-reward reasoning.

-----

## 17. Open Frontiers (where the map runs out)

1. **Sample efficiency at scale** — closing the gap between 100k-frame agents and the data appetite of large policies.
1. **Generalization & out-of-distribution robustness** — Procgen/NetHack remain humbling.
1. **Representation sufficiency** — *what minimal latent is control-complete?* (value-equivalence vs. bisimulation vs. homomorphism — still no unified, scalable answer). **← your territory.**
1. **Long-horizon credit assignment & hierarchy** — options/HRL (§10) remain unsolved at scale.
1. **Offline-to-online fine-tuning** without unlearning.
1. **Real-time diffusion/flow policies** — quality vs. inference latency.
1. **RL for reasoning** — stable, sample-efficient verifiable-reward RL on large sequence models.
1. **World models as foundation models** — general, transferable learned simulators.

-----

*Compiled as a study/reference map. Equations are stated in standard notation; for any algorithm you want to implement, go to the primary paper — this map is the index, not the source.*