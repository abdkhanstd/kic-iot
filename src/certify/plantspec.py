# Author: abdkhan
"""Plant specifications for the closed-loop harness (loop.py).

A PlantSpec bundles what the harness needs to know about a plant: the
actuator, exogenous-driver and measured-channel names, the operator cost,
how to instantiate the stochastic plant, and a vectorised ground-truth
simulator used ONLY to score decisions (never to make them).

Three plants are registered:
  scm  the sigmoidal IoT edge-cluster SCM of the original study (oracle.py)
  mec  a physics-based mobile-edge-computing offloading node (mec.py):
       M/M/1/K queues, a Shannon-rate uplink with Rayleigh fading and a
       cubic DVFS power law -- structurally unrelated to the SCM.
  grid smart-inverter Volt/VAR control on the IEEE 33-bus feeder (grid.py):
       AC power flow on pandapower's case33bw, two PV inverters.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LO, HI = 0.02, 0.98
_EXP_CACHE: dict = {}


@dataclasses.dataclass
class PlantSpec:
    name: str
    actuators: list
    exo: list
    objectives: list                  # measured channels that enter the cost
    bounds: dict                      # residual clip bound per certificate target
    cost: Callable                    # dict of values (arrays ok) -> operator cost
    make: Callable                    # (cfg) -> plant with .step(applied) and .regime
    simulate_vec: Callable            # (regime, actions, n, rng, exo=None) -> dict of arrays
    reward: Optional[Callable] = None # legacy reward of the original study (scm only)
    spoof_actuator: str = ""
    spoof_targets: tuple = ()
    mc_seed: int = 12345

    @property
    def targets(self):
        return list(self.objectives) + ["cost"]

    def value(self, v: dict, objective: str):
        return self.cost(v) if objective == "cost" else v[objective]

    def expected(self, regime, actions: dict, objective: str, n: int = 2000,
                 seed: Optional[int] = None) -> float:
        """E[objective | do(actions)] by Monte Carlo with a fixed seed, so two
        calls share their exogenous and noise draws (common random numbers).
        Deterministic, hence memoized (the harness re-scores the same points)."""
        try:
            key = (self.name, dataclasses.astuple(regime),
                   tuple((a, round(float(actions[a]), 12)) for a in sorted(actions)), objective, n, seed)
        except TypeError:
            key = None
        if key is not None and key in _EXP_CACHE:
            return _EXP_CACHE[key]
        v = self.simulate_vec(regime, actions, n, np.random.default_rng(self.mc_seed if seed is None else seed))
        v.setdefault("battery", 1.0)
        out = float(np.mean(self.value(v, objective)))
        if key is not None:
            if len(_EXP_CACHE) > 400_000:
                _EXP_CACHE.clear()
            _EXP_CACHE[key] = out
        return out

    def op_cost(self, regime, nominal: dict, delta: float, objective: str = "cost",
                n: int = 2000) -> float:
        """Operating cost M_delta(c) = E_s[m(c + delta s)]: the expected cost the
        plant pays when the gate holds nominal c and adds its +-delta dither
        (s uniform on the corners). delta = 0 gives m(c). The actuator is
        centred at least delta inside its range, as in kic.dither()."""
        if delta <= 0:
            return self.expected(regime, nominal, objective, n)
        c = {a: float(np.clip(nominal[a], LO + delta, HI - delta)) for a in self.actuators}
        k = len(self.actuators)
        tot = 0.0
        for m in range(2 ** k):
            u = {a: c[a] + delta * (1 if (m >> j) & 1 else -1) for j, a in enumerate(self.actuators)}
            tot += self.expected(regime, u, objective, n)
        return tot / 2 ** k

    def local_effect(self, regime, centre: dict, actuator: str, objective: str, delta: float,
                     n: int = 2000, smoothed: bool = True) -> float:
        """The keyed estimand beta_{t,j} of Lemma 1: the symmetric difference
        along `actuator`, averaged over the OTHER actuators' dither corners
        (smoothed=True), or at the nominal of the others (smoothed=False)."""
        others = [a for a in self.actuators if a != actuator]
        offs = [dict()]
        if smoothed and delta > 0:
            offs = []
            for m in range(2 ** len(others)):
                offs.append({a: delta * (1 if (m >> j) & 1 else -1) for j, a in enumerate(others)})
        tot = 0.0
        for o in offs:
            base = dict(centre)
            for a, d in o.items():
                base[a] = centre[a] + d
            up = dict(base); up[actuator] = centre[actuator] + delta
            dn = dict(base); dn[actuator] = centre[actuator] - delta
            tot += (self.expected(regime, up, objective, n) - self.expected(regime, dn, objective, n)) / (2 * delta)
        return tot / len(offs)


# ---------------------------------------------------------------------------
# the IoT edge-cluster SCM of the original study
# ---------------------------------------------------------------------------

def _scm_spec() -> PlantSpec:
    from certify import oracle as O
    from rl_controller.env import compute_reward

    def make(cfg):
        from simulator.scm_iot import IoTClusterSCM
        if cfg.plant_memory or cfg.noise_mult != 1.0 or cfg.drift_amp:
            from certify.plants import VariantSCM
            return VariantSCM(seed=cfg.seed, regime_shift_prob=cfg.regime_shift_prob,
                              memory=cfg.plant_memory, noise_mult=cfg.noise_mult,
                              drift_amp=cfg.drift_amp, drift_period=cfg.drift_period)
        return IoTClusterSCM(seed=cfg.seed, regime_shift_prob=cfg.regime_shift_prob)

    return PlantSpec(
        name="scm", actuators=["duty_cycle", "tx_power"],
        exo=["interference", "link_quality", "arrival_rate"],
        objectives=["latency", "energy", "packet_loss"],
        bounds={"cost": 0.2, "latency": 0.1, "energy": 0.1, "packet_loss": 0.1},
        cost=O.operator_cost, make=make, simulate_vec=O.simulate_vec, reward=compute_reward,
        spoof_actuator="duty_cycle", spoof_targets=("latency", "energy", "packet_loss"))


def _mec_spec() -> PlantSpec:
    from certify import mec as M
    return PlantSpec(
        name="mec", actuators=list(M.ACTUATORS), exo=list(M.EXO), objectives=list(M.OBJECTIVES),
        bounds={"cost": 0.2, **{k: 0.1 for k in M.OBJECTIVES}},
        cost=M.operator_cost, make=lambda cfg: M.MECPlant(seed=cfg.seed,
                                                           regime_shift_prob=cfg.regime_shift_prob,
                                                           noise_mult=cfg.noise_mult),
        simulate_vec=M.simulate_vec, reward=None,
        spoof_actuator="offload", spoof_targets=("latency", "drop"))


def _grid_spec() -> PlantSpec:
    from certify import grid as GR
    return PlantSpec(
        name="grid", actuators=list(GR.ACTUATORS), exo=list(GR.EXO), objectives=list(GR.OBJECTIVES),
        bounds={"cost": 0.2, **{k: 0.1 for k in GR.OBJECTIVES}},
        cost=GR.operator_cost, make=lambda cfg: GR.GridPlant(seed=cfg.seed,
                                                              regime_shift_prob=cfg.regime_shift_prob,
                                                              noise_mult=cfg.noise_mult),
        simulate_vec=GR.simulate_vec, reward=None,
        spoof_actuator="q18", spoof_targets=("loss",))


_CACHE: dict = {}


def get(name: str = "scm") -> PlantSpec:
    if name not in _CACHE:
        _CACHE[name] = {"scm": _scm_spec, "mec": _mec_spec, "grid": _grid_spec}[name]()
    return _CACHE[name]
