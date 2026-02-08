"""Gymnasium wrapper for automatic recording of environment interactions.

Usage:
    import gymnasium as gym
    import roboreplay

    env = roboreplay.wrap(gym.make("CartPole-v1"), name="cartpole_run")
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()  # Saves recording
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roboreplay.recorder import Recorder


class RecordingWrapper:
    """Wraps a Gymnasium-compatible environment to auto-record interactions.

    Records observation, action, and reward channels automatically.
    Marks episode resets as events. Saves the recording on close().

    Uses duck-typing — works with any env that has step(), reset(), close().
    """

    def __init__(
        self,
        env: Any,
        name: str = "gym_recording",
        path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._env = env
        meta = metadata or {}
        meta.setdefault("task", "gymnasium")
        self._recorder = Recorder(name, path=path, metadata=meta)
        self._last_obs: np.ndarray | None = None
        self._episode_count = 0
        self._step_count = 0
        self._closed = False

    def reset(self, **kwargs: Any) -> Any:
        """Reset the environment and record the event."""
        result = self._env.reset(**kwargs)

        # Gymnasium returns (obs, info), older gym returns just obs
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result

        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._episode_count += 1

        if self._recorder._started:
            self._recorder.mark_event("episode_reset", {
                "episode": self._episode_count,
                "step": self._step_count,
            })

        return result

    def step(self, action: Any) -> Any:
        """Step the environment and record the interaction."""
        result = self._env.step(action)

        # Gymnasium returns (obs, reward, terminated, truncated, info)
        # Older gym returns (obs, reward, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        obs_arr = np.asarray(obs, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)

        channels: dict[str, Any] = {
            "observation": obs_arr,
            "action": action_arr,
            "reward": float(reward),
        }

        self._recorder.step(**channels)
        self._step_count += 1
        self._last_obs = obs_arr

        if terminated:
            self._recorder.mark_event("episode_terminated", {
                "episode": self._episode_count,
                "step": self._step_count,
            })

        if truncated:
            self._recorder.mark_event("episode_truncated", {
                "episode": self._episode_count,
                "step": self._step_count,
            })

        return result

    def close(self) -> None:
        """Close the environment and save the recording."""
        if not self._closed:
            if self._recorder._started:
                self._recorder.mark_event("recording_end", {
                    "total_steps": self._step_count,
                    "total_episodes": self._episode_count,
                })
                self._recorder.save()
            self._env.close()
            self._closed = True

    @property
    def recording_path(self) -> Path:
        """Path to the recording file."""
        return self._recorder.path

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def __enter__(self) -> RecordingWrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def wrap(
    env: Any,
    name: str = "gym_recording",
    path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> RecordingWrapper:
    """Wrap a Gymnasium environment for automatic recording.

    Args:
        env: A Gymnasium-compatible environment.
        name: Name for the recording.
        path: Explicit output path. If None, saves as {name}.rrp.
        metadata: Additional metadata dict.

    Returns:
        A RecordingWrapper that auto-records step/reset interactions.
    """
    return RecordingWrapper(env, name=name, path=path, metadata=metadata)
