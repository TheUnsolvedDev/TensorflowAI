# TensorflowAI Working Context

## Repository shape

- Root: `/home/shuvrajeet/Documents/TensorflowAI`
- Main areas: `ComputerVision`, `NaturalLanguageProcessing`, `ReinforcementLearning`.
- TensorFlow-first research code organized as numbered learning/project families.
- Many model families use standalone sibling folders.

## Usual project structure

Most standalone implementations contain some combination of:

- `config.py` — paths, hyperparameters, runtime settings
- `dataset.py` — local loading and preprocessing
- `model.py` — architecture and model-specific learning logic
- `train_and_test.py` or `train.py` — execution entrypoint
- `run.sh` or `train.sh` — repeatable command-line workflow
- `logs/`, `checkpoints/`, `artifacts/`, `samples/` — experiment outputs

## Working conventions

- Inspect the target folder and neighboring sibling before editing.
- Preserve the existing folder-local style and public CLI/run behavior.
- Do not introduce shared helpers or broad refactors unless explicitly requested.
- Prefer resumable workflows and reuse valid artifacts; use explicit force/reset options for rebuilds.
- Keep README, config, CLI, and path documentation synchronized with interface changes.
- Put progress reporting at the slowest meaningful operation without flooding logs.
- Use syntax/static checks as limited evidence; verify runtime, dependency, device, GPU, and multi-GPU behavior separately.
- Keep TensorFlow as the default stack where practical.
- For experiments, use the requested fixed environment/algorithm/support pairing and report aggregate results honestly.

## Repository cautions

- This workspace contains nested Git repositories and substantial pre-existing generated/cache changes.
- Preserve unrelated user changes; inspect status and diffs narrowly around the requested target.
- Generated logs, images, checkpoints, caches, and datasets should not be treated as source intent.
- Exact paths, entrypoints, tracebacks, and current checkout contents take precedence over assumptions or older notes.

## Project families observed

- Computer vision: diffusion, GANs, image classification, and object detection.
- NLP: attention, sequence models, transformers, pretrained models, and RedditStory.
- Reinforcement learning: standalone environments, policy-gradient/REINFORCE variants, vectorized agents, and game projects.

## How to use this file

Add project-specific decisions, conventions, commands, paths, known failures, and validation boundaries below. Keep entries short and dated so future prompts can provide this file as context.

## Project trail

<!-- Add dated project notes here. -->
