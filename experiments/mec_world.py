# Author: abdkhan
"""Scenario builder for the second plant (MEC offloading node, src/certify/mec.py).

Mirrors kic_attacks.scenario() for the SCM:
  oracle           one-step planner with the true plant model
  learned          a planner on a linear surrogate fitted to a behaviour log
                   (the log's operator offloaded more on good channels, so the
                   surrogate inherits confounding)
  backdoor_lie     the learned planner with a 10% backdoor proposing the most
                   harmful in-range move, with a lying effect claim
  a1_telemetry     A1: 60% of LOGGED rows shifted so that offloading appears to
                   cut latency and drops; the surrogate and the observational
                   histories are refitted on the poisoned log
  spoof_blind      key-blind live spoofing of latency and drops (-0.6 (x - 0.5) + 0.3,
                   x the nominal offload share; the offset keeps forged reports
                   in range, so the forgery stays additive) while a backdoor
                   pushes offload up
  a1_spoof         A1 on both paths: the poisoned log and key-blind live spoofing
  spoof_aware / spoof_raw   the key-aware and output-adaptive attackers
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify import mec as M
from certify import plantspec
from certify.loop import PlannerProposer, BackdoorProposer, Spoofer, OracleProposer

P = plantspec.get("mec")
TG = P.objectives


def behaviour_log(n=6000, seed=0):
    plant = M.MECPlant(seed=seed + 5000, regime_shift_prob=0.004)
    rng = np.random.default_rng(seed + 17)
    rows = []
    ex = plant.step({"offload": 0.5, "cpu_freq": 0.5})
    for _ in range(n):
        x = float(np.clip(0.15 + 0.7 * ex["channel"] + rng.normal(0, 0.15), 0.02, 0.98))
        f = float(np.clip(0.25 + 0.5 * ex["arrival"] + rng.normal(0, 0.15), 0.02, 0.98))
        ex = plant.step({"offload": x, "cpu_freq": f})
        rows.append(ex)
    return {k: np.array([r[k] for r in rows]) for k in P.actuators + P.exo + TG}


def poison(H, frac=0.6, kappa=0.8, seed=0):
    rng = np.random.default_rng(seed)
    H = {k: v.copy() for k, v in H.items()}
    m = rng.random(len(H["offload"])) < frac
    for k in ("latency", "drop"):
        H[k][m] = np.clip(H[k][m] - kappa * 0.5 * (H["offload"][m] - 0.5), 0, 1)
    return H


class QuadSurrogate:
    """Ridge regression of each measured channel on actions and exogenous
    drivers: linear (the default learned model, as the SCM's linear surrogate
    of the discovered graph), or with squares and pairwise products."""

    def __init__(self, H, ridge=1e-3, quad=False):
        self.quad = quad
        X = self._feat(H)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.coef = {k: np.linalg.solve(A, X.T @ H[k]) for k in TG}

    def _feat(self, D):
        cols = [np.asarray(D[k], dtype=float) for k in P.actuators + P.exo]
        n = len(np.atleast_1d(cols[0]))
        cols = [np.broadcast_to(np.atleast_1d(c), (n,)) for c in cols]
        f = list(cols)
        if self.quad:
            f += [c * c for c in cols]
            f += [cols[i] * cols[j] for i in range(len(cols)) for j in range(i + 1, len(cols))]
        return np.column_stack(f + [np.ones(n)])

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
    return dict(H_clean=hist_cost(H), H_pois=hist_cost(Hp), sur_clean=QuadSurrogate(H), sur_pois=QuadSurrogate(Hp))


def planner(sur, name):
    def f(actions, objective, obs):
        x = {k: obs[k] for k in P.exo}
        x.update(actions)
        p = sur.predict(x)
        return float(P.cost(p)) if objective == "cost" else p[objective]
    return PlannerProposer(f, name)


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
    elif name.startswith("spoof_") or name == "a1_spoof":
        know = "blind" if name == "a1_spoof" else name.split("_", 1)[1]
        prop = BackdoorProposer(base, 0.2, "lie", seed=seed, restrict=("offload", 1))
        kn = {"blind": "nominal", "aware": "applied", "raw": "raw"}[know]
        spoof = Spoofer(actuator="offload", targets=("latency", "drop"), kappa=0.6, knowledge=kn,
                        actuators=acts, offset=0.0 if kn == "raw" else 0.3)
        if name == "a1_spoof":          # poisoned log AND forged live telemetry
            prop = BackdoorProposer(planner(W["sur_pois"], "learned_pois"), 0.2, "lie", seed=seed,
                                    restrict=("offload", 1))
            hist = W["H_pois"]
    else:
        raise ValueError(name)
    return prop, None, spoof, hist, exc
