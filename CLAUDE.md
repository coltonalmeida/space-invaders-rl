# CLAUDE.md — Asteroids RL Agent

> Project context file for Claude Code. Train a reinforcement-learning agent to
> play Atari Asteroids. Optimized to be a portfolio piece employers will respect.

---

## 1. Project Goal

Train an AI agent to play **Atari Asteroids** well above random-play baseline,
using a standard, reproducible deep-RL pipeline. Ship it with clean code,
tracked experiments, evaluation metrics, and a recorded gameplay video.

**Success criteria**
- Agent mean episode reward significantly beats a random-action baseline.
- Training is reproducible from a single command + config file.
- Repo includes metrics, plots, a saved model, and a gameplay video.

## 2. Key Decision: Don't Build the Game

You do **not** need to find or build an Asteroids game on GitHub. The
**Arcade Learning Environment (ALE)**, exposed through **Farama Gymnasium**,
ships Asteroids as a ready, standardized environment: `ALE/Asteroids-v5`.
This is the exact environment used in RL research and industry, which is why
it's the right call for a portfolio. Your only job is the *training pipeline*.

```python
import gymnasium as gym
env = gym.make("ALE/Asteroids-v5", render_mode="rgb_array")
```

ROMs install via AutoROM (handled by the `[extra]` / `[accept-rom-license]`
extras — see setup below).

## 3. Tech Stack

Chosen so each item is something employers actively look for on a resume.

| Layer | Tool | Why it signals well |
|---|---|---|
| Language | **Python 3.11+** | RL/ML standard |
| Env | **Gymnasium + ALE-py** (Farama) | The current industry standard (Gym is deprecated) |
| RL algorithms | **Stable-Baselines3** (PyTorch) | Trusted, reliable implementations; shows you know not to reinvent wheels |
| DL backend | **PyTorch 2.3+** | Most in-demand DL framework |
| Experiment tracking | **Weights & Biases (wandb)** | Widely used at ML companies; shows MLOps maturity |
| Config | **Hydra** or YAML + `argparse` | Reproducible, parameterized experiments |
| Packaging/env | **uv** (or venv + pip) | Modern, fast Python dependency management |
| Testing | **pytest** | Demonstrates engineering discipline |
| Lint/format | **ruff** + **black** | Clean-code signal |
| CI | **GitHub Actions** | Shows you ship production-grade repos |
| Containerization | **Docker** | Reproducibility; a strong hireability signal |
| Video/eval | Gymnasium `RecordVideo` wrapper | Tangible demo of results |

**Algorithm choice:** Start with **PPO** (stable, parallelizable, strong on
Atari). Then train **DQN** as a second baseline so you can *compare* — comparison
tables and plots are what make a portfolio project look rigorous.

## 4. Repository Structure

```
asteroids-rl/
├── CLAUDE.md                 # this file
├── README.md                 # results, plots, gif, how-to-run
├── pyproject.toml            # deps (uv/pip), ruff/black config
├── Dockerfile
├── .github/workflows/ci.yml  # ruff + pytest on push
├── configs/
│   ├── ppo.yaml
│   └── dqn.yaml
├── src/asteroids_rl/
│   ├── __init__.py
│   ├── env.py                # make_env(): wrappers, frame-stack, preprocessing
│   ├── train.py              # CLI entry: reads config, trains, logs to wandb
│   ├── evaluate.py           # load model, run N episodes, report mean/std
│   ├── record.py             # save gameplay video of trained agent
│   └── callbacks.py          # eval + checkpoint callbacks
├── tests/
│   └── test_env.py           # env builds, step/reset shapes correct
├── models/                   # saved checkpoints (gitignored if large)
├── videos/                   # recorded gameplay
└── reports/                  # plots, final metrics
```

## 5. Environment Setup

```bash
# Using uv (preferred)
uv venv && source .venv/bin/activate
uv pip install "stable-baselines3[extra]" "gymnasium[atari,accept-rom-license]" \
               torch wandb hydra-core pytest ruff black tensorboard
```

`stable-baselines3[extra]` pulls in the Atari preprocessing helper
`make_atari_env` plus ROM handling. Requires Python >= 3.10.

## 6. Implementation Plan (phased)

**Phase 0 — Scaffolding**
- Create GitHub repo `asteroids-rl`, init with README + .gitignore (Python).
- Create the project folder under the user's home dir, set up venv, commit skeleton.
- Add ruff/black config and a passing pytest stub. Wire GitHub Actions CI.

**Phase 1 — Environment**
- Implement `make_env()` using SB3's `make_atari_env` (handles grayscale,
  resize to 84×84, frame-stacking of 4, frame-skip, sticky actions).
- Add a `random_agent` sanity check + record baseline mean reward.

**Phase 2 — Train PPO**
- `train.py` reads `configs/ppo.yaml`, runs vectorized envs (`n_envs=8`),
  logs to wandb + TensorBoard, checkpoints best model.
- Start ~1M steps to validate the loop, then scale to 10M.

**Phase 3 — Train DQN (comparison baseline)**
- Same pipeline, `configs/dqn.yaml`. Reuse callbacks.

**Phase 4 — Evaluate & visualize**
- `evaluate.py`: 30 episodes, report mean ± std vs random baseline.
- `record.py`: save an MP4/GIF of the best agent for the README.
- Generate reward-curve plots into `reports/`.

**Phase 5 — Polish**
- README with results table (Random vs PPO vs DQN), training curves, gameplay
  gif, and exact reproduce commands.
- Dockerfile so anyone can run it. Confirm CI is green.

## 7. Core Hyperparameters (starting points)

**PPO (Atari defaults):** `n_envs=8`, `n_steps=128`, `batch_size=256`,
`n_epochs=4`, `gamma=0.99`, `learning_rate=2.5e-4` (linearly decayed),
`clip_range=0.1`, `ent_coef=0.01`, `vf_coef=0.5`, policy=`CnnPolicy`.

**DQN (Atari defaults):** `buffer_size=100_000`, `learning_starts=100_000`,
`target_update_interval=1000`, `train_freq=4`, `gamma=0.99`,
`exploration_fraction=0.1`, `learning_rate=1e-4`, policy=`CnnPolicy`.

> Note: `CnnPolicy` is required — observations are raw pixels.

## 8. Compute Notes

Atari runs to ~10M steps. On CPU this is slow; a single GPU (e.g. T4) trains a
comparable Atari game in ~8 hours. Validate the full loop at 1M steps first, then
commit to a long run. Consider Colab/cloud GPU if no local GPU.

## 9. Conventions for Claude Code

- Keep env-building logic in **one** place (`env.py`) — never duplicate wrappers.
- Every training run must be driven by a **config file**, never hardcoded params.
- All runs log to **wandb** with the config attached, so experiments are
  comparable and reproducible.
- Write a test before wiring a new module into the pipeline.
- Prefer SB3's built-in helpers over custom implementations unless there's a
  concrete reason; document any deviation.

## 10. Resume / Portfolio Framing

When this is done, it demonstrates: deep RL (PPO + DQN), PyTorch, the modern
Gymnasium/ALE stack, experiment tracking (wandb), reproducible configs,
testing, CI/CD, and Docker — i.e. an *end-to-end ML engineering* project, not
just a notebook. Lead the README with the results table and the gameplay gif.
