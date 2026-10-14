# Author: abdkhan
"""Plant variants for the robustness sweep.

VariantSCM keeps the IoT SCM's structure but adds (i) queue MEMORY
(q_t = rho * q_{t-1} + (1 - rho) * f(...)), so actions have delayed effects on
queue, CPU load, latency, loss and energy, and (ii) a noise multiplier. With
rho = 0 and noise_mult = 1 it reproduces IoTClusterSCM exactly. The static
oracle in oracle.py does not model memory, so memory variants are scored on
the realized (unforged) cost averaged over seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.scm_iot import IoTClusterSCM, RegimeParams, EXOGENOUS, ACTIONS


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class VariantSCM(IoTClusterSCM):
    def __init__(self, seed: int = 0, regime_shift_prob: float = 0.002,
                 memory: float = 0.0, noise_mult: float = 1.0,
                 drift_amp: float = 0.0, drift_period: float = 1600.0):
        reg = RegimeParams()
        reg.noise_scale *= noise_mult
        super().__init__(seed=seed, regime=reg, regime_shift_prob=regime_shift_prob)
        self.memory = memory
        self.q_prev = None
        self._in_step = False
        # continuous drift: the interference mean follows a slow sinusoid on
        # top of the regime's level, so the local effects drift between
        # regime breaks (the drift envelope V of Assumption 2(c))
        self.drift_amp, self.drift_period = drift_amp, drift_period
        self._off = 0.0
        self._k = 0

    def structural_equations(self, exo, actions, noise_scale, do=None):
        if self.memory == 0.0 or not self._in_step:
            return super().structural_equations(exo, actions, noise_scale, do)
        n = lambda: self.rng.normal(0, noise_scale)
        v = dict(exo); v.update(actions)
        c = lambda z: float(np.clip(z, 0, 1))
        v["collision_rate"] = c(_sig(3.0 * v["interference"] + 2.5 * v["duty_cycle"] - 1.5 * v["tx_power"]
                                     - 2.0 * v["link_quality"] - 1.0 + n()))
        v["retransmissions"] = c(_sig(3.5 * v["collision_rate"] - 1.2 * v["tx_power"] - 0.5 + n()))
        q_now = c(_sig(2.5 * v["retransmissions"] + 1.5 * v["arrival_rate"] - 1.0 * v["duty_cycle"] - 0.8 + n()))
        q = q_now if self.q_prev is None else self.memory * self.q_prev + (1 - self.memory) * q_now
        self.q_prev = q
        v["queue_length"] = q
        v["cpu_load"] = c(_sig(2.0 * q + 1.0 * v["duty_cycle"] - 1.2 + n()))
        v["latency"] = c(_sig(2.2 * v["retransmissions"] + 2.0 * q + 1.0 * v["cpu_load"] - 1.8 + n()))
        v["throughput"] = c(_sig(2.0 * v["duty_cycle"] - 1.8 * v["collision_rate"]
                                 - 1.5 * v["retransmissions"] + 0.3 + n()))
        v["packet_loss"] = c(_sig(2.5 * v["collision_rate"] + 1.8 * v["retransmissions"] + 1.0 * q - 2.5 + n()))
        v["energy"] = c(_sig(2.0 * v["tx_power"] + 1.8 * v["duty_cycle"] + 1.0 * v["cpu_load"]
                             + 0.8 * v["retransmissions"] - 1.5 + n()))
        return v

    def step(self, actions, do=None):
        if self.drift_amp:
            self._k += 1
            off = self.drift_amp * np.sin(2 * np.pi * self._k / self.drift_period)
            self.regime.interference_mean = float(np.clip(
                self.regime.interference_mean + off - self._off, 0.02, 0.95))
            self._off = off
        self._in_step = True
        try:
            return super().step(actions, do)
        finally:
            self._in_step = False
