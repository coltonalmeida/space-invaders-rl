"""Evaluate a trained model or a random-action baseline over N episodes.

Usage:
    python -m asteroids_rl.evaluate --random --episodes 30
    python -m asteroids_rl.evaluate --model models/ppo_best/best_model.zip --episodes 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asteroids_rl.env import ENV_ID, make_env


def run_random_baseline(env_id: str, episodes: int, seed: int = 42) -> tuple[float, float]:
    """Random-agent sanity check: env builds, steps, and yields a baseline reward."""
    env = make_env(env_id, n_envs=1, seed=seed, eval_mode=True)
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            action = np.array([rng.integers(env.action_space.n)])
            _, reward, dones, _ = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def run_model(model_path: str, env_id: str, episodes: int, seed: int = 42) -> tuple[float, float]:
    from stable_baselines3.common.evaluation import evaluate_policy

    from asteroids_rl.train import load_model

    env = make_env(env_id, n_envs=1, seed=seed, eval_mode=True)
    model = load_model(model_path, env)
    mean, std = evaluate_policy(model, env, n_eval_episodes=episodes)
    env.close()
    return float(mean), float(std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="Path to a saved SB3 model .zip")
    parser.add_argument("--random", action="store_true", help="Evaluate a random-action baseline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.random == (args.model is not None):
        parser.error("Provide exactly one of --random or --model")

    if args.random:
        label = "random"
        mean, std = run_random_baseline(args.env_id, args.episodes, args.seed)
    else:
        label = args.model
        mean, std = run_model(args.model, args.env_id, args.episodes, args.seed)

    result = {"agent": label, "episodes": args.episodes, "mean_reward": mean, "std_reward": std}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
