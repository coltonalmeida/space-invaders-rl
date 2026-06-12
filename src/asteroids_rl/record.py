"""Record a gameplay video of a trained agent.

Usage:
    python -m asteroids_rl.record --model models/ppo_best/best_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from asteroids_rl.env import ENV_ID
from asteroids_rl.train import load_model


def _make_recorded_env(env_id: str, out_dir: str):
    """Raw env wrapped with RecordVideo (full-color frames), then Atari preprocessing
    so the model sees the same observations as during training."""

    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda _: True)
        return AtariWrapper(env)

    return _thunk


def record(model_path: str, env_id: str, out_dir: str, episodes: int, seed: int) -> None:
    vec = DummyVecEnv([_make_recorded_env(env_id, out_dir)])
    vec.seed(seed)
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    model = load_model(model_path, vec)

    for _ in range(episodes):
        obs = vec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec.step(np.asarray(action))
            done = bool(dones[0])
    vec.close()
    print(f"Videos saved to {Path(out_dir).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    record(args.model, args.env_id, args.out_dir, args.episodes, args.seed)


if __name__ == "__main__":
    main()
