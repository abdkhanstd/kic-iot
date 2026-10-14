# Author: abdkhan
"""Non-learned baseline controllers for the IoT QoS control loop, used as
comparison points against the causal-RL controller and the tiny-LLM +
verifier loop.
"""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, obs_fields):
        return self.rng.uniform(0.1, 0.9, size=2)


class StaticPolicy:
    """Fixed mid-range action -- the 'do nothing adaptive' baseline."""
    def act(self, obs, obs_fields):
        return np.array([0.5, 0.5])


class ReactivePID:
    """A hand-tuned reactive controller: if latency/packet_loss is high,
    raise tx_power (fights collisions) and lower duty_cycle (fights queueing);
    if energy/battery is a concern, do the opposite. This is the kind of
    non-causal heuristic an engineer would hand-write without an SCM, used
    to show the value added by causal-informed control.
    """
    def __init__(self, kp=0.6):
        self.kp = kp

    def act(self, obs, obs_fields):
        idx = {f: i for i, f in enumerate(obs_fields)}
        latency = obs[idx["latency"]]
        packet_loss = obs[idx["packet_loss"]]
        energy = obs[idx["energy"]]
        battery = obs[idx["battery"]]

        qos_pressure = 0.5 * latency + 0.5 * packet_loss  # 0..1, want low
        energy_pressure = energy * (1.2 - battery)         # want low, weighted by battery scarcity

        tx_power = 0.5 + self.kp * (qos_pressure - 0.5) - 0.3 * (energy_pressure - 0.3)
        duty_cycle = 0.5 - self.kp * (qos_pressure - 0.5) * 0.5

        return np.clip([duty_cycle, tx_power], 0.05, 0.95)
