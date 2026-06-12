"""Config-driven training entry point. Every run is driven by a YAML config.

Usage:
    python -m asteroids_rl.train --config configs/ppo.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from stable_baselines3 import DQN, PPO

from asteroids_rl.callbacks import build_callbacks
from asteroids_rl.env import make_env

ALGOS = {"ppo": PPO, "dqn": DQN}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def linear_schedule(initial: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return schedule


def load_model(model_path: str, env):
    """Load a saved SB3 model, inferring the algorithm from the zip."""
    for algo_cls in ALGOS.values():
        try:
            return algo_cls.load(model_path, env=env)
        except (ValueError, KeyError, AssertionError):
            continue
    raise ValueError(f"Could not load {model_path} as any of {list(ALGOS)}")


def train(config: dict) -> Path:
    algo_name = config["algo"].lower()
    run_name = f"{algo_name}_{int(time.time())}"

    env = make_env(
        config["env_id"],
        n_envs=config["n_envs"],
        seed=config["seed"],
        frame_stack=config.get("frame_stack", 4),
    )

    hyperparams = dict(config["hyperparams"])
    if algo_name == "ppo" and "learning_rate" in hyperparams:
        hyperparams["learning_rate"] = linear_schedule(float(hyperparams["learning_rate"]))

    wandb_run = None
    callbacks = [build_callbacks(config, run_name)]
    if config.get("wandb", {}).get("enabled", False):
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            name=run_name,
            config=config,
            sync_tensorboard=True,
        )
        callbacks.append(WandbCallback())

    model = ALGOS[algo_name](
        config["policy"],
        env,
        seed=config["seed"],
        tensorboard_log=f"runs/{run_name}",
        verbose=1,
        **hyperparams,
    )
    model.learn(total_timesteps=int(config["total_timesteps"]), callback=callbacks)

    final_path = Path("models") / run_name / "final_model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)

    env.close()
    if wandb_run is not None:
        wandb_run.finish()
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    path = train(load_config(args.config))
    print(f"Saved final model to {path}.zip")


if __name__ == "__main__":
    main()
