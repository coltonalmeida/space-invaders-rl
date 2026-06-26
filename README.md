# space-invaders-rl

End-to-end deep reinforcement learning pipeline that trains an agent to play
**Atari Space Invaders** (`ALE/SpaceInvaders-v5`) and more than doubles the score
of a random-play baseline. A **PPO** agent is trained, evaluated, and compared against a
random-play baseline with tracked experiments, saved checkpoints, and recorded gameplay.

<video src="https://github.com/user-attachments/assets/b0d9951d-6dbe-4e43-b6e3-461de39d4caa" controls width="600">
  <!-- Fallback for viewers that don't render the video tag (npm, some IDEs) -->
  <a href="https://www.youtube.com/watch?v=yJduFZuogFA">Watch the gameplay demo on YouTube</a>
</video>

## Results

Mean episode reward from evaluation — PPO is the best checkpoint (9.6M steps,
5-episode eval); random is a 10-episode baseline:

| Agent  | Mean episode reward | Std | Improvement vs random |
|--------|---------------------|-----|-----------------------|
| Random | 465                 | 278 | 1.0×                  |
| PPO    | 983                 | 342 | 2.1×                  |

The PPO agent was trained for 10M steps on a cloud GPU and more than doubles
the score of random play.

## Setup

```bash
uv venv
uv sync
```

ROMs are installed automatically via AutoROM (`accept-rom-license` extra).

## Usage

```bash
# Random-agent baseline (sanity check + baseline reward)
python -m atari_rl.evaluate --random --episodes 30 --out reports/random_baseline.json

# Train PPO (Hydra config; override any value on the CLI)
python -m atari_rl.train --config-name=ppo

# Evaluate a trained model (prints improvement factor vs the random baseline)
python -m atari_rl.evaluate --model models/<run>/best/best_model.zip --episodes 30

# Record gameplay video (add --wandb to upload the clip to Weights & Biases)
python -m atari_rl.record --model models/<run>/best/best_model.zip --step 10000000 --wandb
```

### Running on a cloud GPU

Training to 10M steps is GPU-heavy. You can run it on your own machine, or rent a cloud GPU
(e.g. [RunPod](https://www.runpod.io/), Vast.ai, or Lambda Labs): clone the repo, `uv sync`,
and run the same `train` command. The 10M-step run in the results above was trained this way.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
```

## Stack

| Layer | Tools |
|-------|-------|
| **Deep RL** | ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-792EE5) ![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5) ![ALE-py](https://img.shields.io/badge/ALE--py-CC0000) |
| **Tracking & config** | ![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black) ![Hydra](https://img.shields.io/badge/Hydra-89B8CD) ![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white) |
| **Tooling** | ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white) ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black) ![Black](https://img.shields.io/badge/Black-000000) |
