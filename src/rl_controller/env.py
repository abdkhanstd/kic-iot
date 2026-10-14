# Author: abdkhan
"""Gymnasium environment wrapping the IoT cluster SCM for the causal-RL
QoS controller (component 3 of CausalTinyIoT).

Observation: current telemetry vector (13 continuous features, battery
included) after the previous action was applied.
Action: continuous (duty_cycle, tx_power) in [0,1]^2 -- the two
controllable nodes of the SCM.
Reward: weighted QoS objective (lower latency/energy/packet_loss, higher
throughput), i.e. the multi-objective tradeoff an autonomous IoT node must
balance without a human in the loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.scm_iot import IoTClusterSCM, NODES

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = object  # type: ignore

OBS_FIELDS = [n for n in NODES if n != "t"]

REWARD_WEIGHTS = dict(latency=1.0, energy=0.6, packet_loss=1.0, throughput=0.8)


def compute_reward(state: dict) -> float:
    return (
        - REWARD_WEIGHTS["latency"] * state["latency"]
        - REWARD_WEIGHTS["energy"] * state["energy"]
        - REWARD_WEIGHTS["packet_loss"] * state["packet_loss"]
        + REWARD_WEIGHTS["throughput"] * state["throughput"]
        + 0.2 * state["battery"]
    )


class _Base:
    def __init__(self, seed: int = 0, episode_len: int = 500,
                 poisoned_scm_kwargs: dict | None = None):
        self.episode_len = episode_len
        self.seed_val = seed
        self._poisoned_kwargs = poisoned_scm_kwargs or {}
        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        seed = seed if seed is not None else self.seed_val
        self.scm = IoTClusterSCM(seed=seed, **self._poisoned_kwargs)
        self._step_count = 0
        state = self.scm.step(dict(duty_cycle=0.5, tx_power=0.5))
        obs = self._obs(state)
        return obs, {}

    def _obs(self, state: dict) -> np.ndarray:
        return np.array([state[f] for f in OBS_FIELDS], dtype=np.float32)

    def step(self, action):
        duty_cycle = float(np.clip(action[0], 0.02, 0.98))
        tx_power = float(np.clip(action[1], 0.02, 0.98))
        state = self.scm.step(dict(duty_cycle=duty_cycle, tx_power=tx_power))
        reward = compute_reward(state)
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.episode_len
        return self._obs(state), reward, terminated, truncated, dict(state=state)


if _HAS_GYM:
    class IoTQoSEnv(_Base, gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, seed: int = 0, episode_len: int = 500,
                     poisoned_scm_kwargs: dict | None = None):
            self.observation_space = spaces.Box(low=0.0, high=1.0,
                                                 shape=(len(OBS_FIELDS),), dtype=np.float32)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
            _Base.__init__(self, seed=seed, episode_len=episode_len,
                            poisoned_scm_kwargs=poisoned_scm_kwargs)
else:
    class IoTQoSEnv(_Base):
        pass
