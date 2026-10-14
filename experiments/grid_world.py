# Author: abdkhan
"""Scenario builder for the third plant (smart-inverter Volt/VAR control on the
IEEE 33-bus feeder, src/certify/grid.py). Mirrors mec_world.py:

  oracle           one-step planner with the true feeder model
  learned          a DMS planner on a linear surrogate fitted to a SCADA log whose
                   operator injected more reactive power under heavy demand
                   (so the surrogate inherits confounding)
  backdoor_lie     the learned planner with a 10% backdoor proposing the most
                   harmful in-range move, with a lying effect claim
  a1_telemetry     A1: 60% of LOGGED rows shifted so that injecting reactive power
                   at bus 18 appears to cut the metered losses; the surrogate and the
                   observational histories are refitted on the poisoned log
  spoof_blind      key-blind false-data injection on the loss telemetry
                   (-0.6 (x - 0.5) + 0.3, x the nominal q18 setpoint the DMS publishes)
                   while a backdoor pushes q18 up
  a1_spoof         A1 on both paths
  spoof_aware / spoof_raw   the key-aware and output-adaptive attackers
  spoof_meter:<bus>:<sigma>:<n>
                   the same injection, but the attacker guesses the q18 dither sign
                   from its own smart meter at <bus> (voltage, noise sigma p.u., n
                   readings per tick): a physical side channel through the feeder
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify import grid as GR
from certify import plantspec
from certify.loop import PlannerProposer, BackdoorProposer, Spoofer, OracleProposer

P = plantspec.get("grid")
TG = P.objectives
ENV = dict(G=2.2, L_hat=4.5, V=2e-4, B_cost=2 * 0.05 * 2.2 + 0.19)     # characterized: G = 2.02, L = 3.95


def behaviour_log(n=6000, seed=0):
    plant = GR.GridPlant(seed=seed + 5000, regime_shift_prob=0.004)
    rng = np.random.default_rng(seed + 17)
    rows = []
    ex = plant.step({"q18": 0.5, "q33": 0.5})
    for _ in range(n):
        a = float(np.clip(0.2 + 0.6 * ex["demand"] + rng.normal(0, 0.15), 0.02, 0.98))
        b = float(np.clip(0.2 + 0.5 * ex["demand"] + 0.2 * ex["qshare"] + rng.normal(0, 0.15), 0.02, 0.98))
        ex = plant.step({"q18": a, "q33": b})
        rows.append(ex)
    return {k: np.array([r[k] for r in rows]) for k in P.actuators + P.exo + TG}


def poison(H, frac=0.6, kappa=0.8, seed=0):
    rng = np.random.default_rng(seed)
    H = {k: v.copy() for k, v in H.items()}
    m = rng.random(len(H["q18"])) < frac
    H["loss"][m] = np.clip(H["loss"][m] - kappa * 0.2 * (H["q18"][m] - 0.5), 0, 1)
    return H


class LinSurrogate:
    """Ridge regression of each measured channel on actions and exogenous drivers."""

    def __init__(self, H, ridge=1e-3):
        X = self._feat(H)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.coef = {k: np.linalg.solve(A, X.T @ H[k]) for k in TG}

    def _feat(self, D):
        cols = [np.asarray(D[k], dtype=float) for k in P.actuators + P.exo]
        n = len(np.atleast_1d(cols[0]))
        cols = [np.broadcast_to(np.atleast_1d(c), (n,)) for c in cols]
        return np.column_stack(cols + [np.ones(n)])

    def predict(self, x):
        F = self._feat(x)
        return {k: float(np.clip((F @ self.coef[k])[0], 0, 1)) for k in TG}


def hist_cost(H):
    H = dict(H)
    H["cost"] = P.cost(H)
    return H


def build_world():
    H = behaviour_log()
    Hp = poison(H)
    return dict(H_clean=hist_cost(H), H_pois=hist_cost(Hp), sur_clean=LinSurrogate(H), sur_pois=LinSurrogate(Hp))


def planner(sur, name):
    def f(actions, objective, obs):
        x = {k: obs[k] for k in P.exo}
        x.update(actions)
        p = sur.predict(x)
        p.update(actions)
        return float(P.cost(p)) if objective == "cost" else p[objective]
    return PlannerProposer(f, name)


def meter_spoofer(bus, sigma, n_obs, seed):
    return Spoofer(actuator="q18", targets=("loss",), kappa=0.6, knowledge="meter", actuators=P.actuators,
                   offset=0.3, seed=seed, meter=dict(bus=int(bus), sigma=float(sigma), n=int(n_obs),
                                                      exo=list(P.exo)))


def scenario(name, seed, W):
    """-> (proposer, gstar, spoofer, hist, excitation)"""
    hist, spoof, exc = W["H_clean"], None, "keyed"
    base = planner(W["sur_clean"], "learned")
    acts = P.actuators
    if name == "oracle":
        prop = OracleProposer()
    elif name == "oracle_backdoor":
        prop = BackdoorProposer(OracleProposer(), 0.1, "lie", seed=seed)
    elif name == "learned":
        prop = base
    elif name == "backdoor_lie":
        prop = BackdoorProposer(base, 0.1, "lie", seed=seed)
    elif name == "a1_telemetry":
        prop = planner(W["sur_pois"], "learned_pois")
        hist = W["H_pois"]
    elif name.startswith("spoof_meter"):
        _, bus, sigma, n_obs = name.split(":")
        prop = BackdoorProposer(base, 0.2, "lie", seed=seed, restrict=("q18", 1))
        spoof = meter_spoofer(bus, sigma, n_obs, seed)
    elif name.startswith("spoof_") or name == "a1_spoof":
        know = "blind" if name == "a1_spoof" else name.split("_", 1)[1]
        prop = BackdoorProposer(base, 0.2, "lie", seed=seed, restrict=("q18", 1))
        kn = {"blind": "nominal", "aware": "applied", "raw": "raw"}[know]
        spoof = Spoofer(actuator="q18", targets=("loss",), kappa=0.6, knowledge=kn,
                        actuators=acts, offset=0.0 if kn == "raw" else 0.3)
        if name == "a1_spoof":
            prop = BackdoorProposer(planner(W["sur_pois"], "learned_pois"), 0.2, "lie", seed=seed,
                                    restrict=("q18", 1))
            hist = W["H_pois"]
    else:
        raise ValueError(name)
    return prop, None, spoof, hist, exc
