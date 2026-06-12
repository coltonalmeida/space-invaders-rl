# asteroids-rl

Training deep-RL agents (PPO and DQN, via Stable-Baselines3) to play **Atari Asteroids**
(`ALE/Asteroids-v5` from Gymnasium / the Arcade Learning Environment).

> 🚧 Work in progress — results table, training curves, and gameplay GIF land here
> once training runs complete.

## Results

| Agent  | Mean episode reward (30 eps) | Std |
|--------|------------------------------|-----|
| Random | 465.0 (10 eps)               | 278 |
| PPO    | _TBD_                        | —   |
| DQN    | _TBD_                        | —   |

## Setup

```bash
uv venv
uv sync
```

ROMs are installed automatically via AutoROM (`accept-rom-license` extra).

## Usage

```bash
# Random-agent baseline (sanity check + baseline reward)
python -m asteroids_rl.evaluate --random --episodes 30

# Train PPO
python -m asteroids_rl.train --config configs/ppo.yaml

# Train DQN
python -m asteroids_rl.train --config configs/dqn.yaml

# Evaluate a trained model
python -m asteroids_rl.evaluate --model models/ppo_best/best_model.zip --episodes 30

# Record gameplay video
python -m asteroids_rl.record --model models/ppo_best/best_model.zip
```

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
```

## Stack

Python 3.10+ · Gymnasium + ALE-py · Stable-Baselines3 (PyTorch) · wandb ·
YAML configs · pytest · ruff/black · GitHub Actions · Docker
