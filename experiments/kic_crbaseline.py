# Author: abdkhan
"""A same-target baseline: corruption-robust least-squares certificates.

Target (shared by every gate): authorize a proposed full step of 0.05 along one
actuator only if it lowers the operating cost M, with a deployment-level error
budget. Plant: the analytically bounded coupled cubic family (p = 2, exact M
labels, exact G and L), nominal following a slow reflected random walk so that
observational evidence exists.

Gates
  KIC        keyed certificates (this paper), anchors of radius 0.1.
  CR-LS(C)   least squares of the reported cost on the applied action over the
             last 400 ticks, following the corruption-budget principle of
             corruption-robust bandits: a declared total corruption C in the
             window moves the slope estimate by at most C max_i |w_ji| (w: the
             least-squares weights), which is added to a finite-horizon
             (Bonferroni) one-sided t bound, together with the curvature terms
             L (max distance + step/2) that any finite-step certificate needs.
             C = 0 is ordinary least squares.
  Oracle     authorizes exactly the beneficial proposals.
Attacks
  none; C1(a): committed offset -a (c1 - 0.5) on the reported cost (depends on
  the published nominal only); C2(b): offset -b (u1 - c1) that reads the
  applied action (total corruption per window about 400 b delta).

    python kic_crbaseline.py --procs 30
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify.kic import CertificateBank, KICGate, KeyedRademacher
from kic_controlled import CubicPlant

RES = ROOT / "experiments" / "results"
DELTA, ALPHA, STEP, TOL, T, WIN = 0.05, 0.05, 0.05, 1e-12, 1600, 400
ATTACKS = [("none", 0.0), ("C1", 0.4), ("C1", 1.6), ("C1", 6.4), ("C2", 1.6)]
BUDGETS = [0.0, 1.0, 10.0, 100.0]


def metric():
    return dict(benef=0, harm=0, auth_benef=0, auth_harm=0, ever_harm=False)


def rec(m, dm, acc):
    m["benef"] += dm < -TOL
    m["harm"] += dm > TOL
    if acc:
        m["auth_benef"] += dm < -TOL
        m["auth_harm"] += dm > TOL
        m["ever_harm"] |= bool(dm > TOL)


def one(args):
    kind, mag, seed = args
    rng = np.random.default_rng(60000 + seed)
    plant = CubicPlant(np.array([0.8, -0.8]))
    G, L = plant.G, plant.L
    bank = CertificateBank(["a0", "a1"], ["cost"], DELTA, ALPHA, grid=1, bounds={"cost": 2 * DELTA * G + 0.2},
                           window=None, anchor_radius=0.1, robust=True, G=G, L=L, V=0.0, family_wise=True,
                           decision_targets=["cost"], bound_control_variate=True)
    gate = KICGate(bank, mode="hold", fallback=False, L_hat=L, cert_cap=STEP)
    exc = KeyedRademacher(f"crbase-{seed}".encode())
    c = rng.uniform(0.4, 0.6, 2)
    U, Y, Cs = [], [], []
    crit = student_t.ppf(1 - ALPHA / T, WIN - 3)
    names = ["KIC"] + [f"CR-LS({int(C)})" for C in BUDGETS] + ["Oracle"]
    R = {g: metric() for g in names}
    fit = None
    for t in range(1, T + 1):
        # proposal and exact label
        j, d = int(rng.integers(2)), int(rng.choice([-1, 1]))
        cp = c.copy(); cp[j] += d * STEP
        dm = plant.cost(cp, DELTA) - plant.cost(c, DELTA)
        centre = {"a0": float(c[0]), "a1": float(c[1])}
        cert = bank.query(f"a{j}", "cost", centre)
        rec(R["KIC"], dm, gate.certified_step(cert, d, -1, STEP) >= STEP - 1e-14)
        if len(U) >= 50 and (fit is None or t % 10 == 0):
            Uw = np.array(U[-WIN:]); Yw = np.array(Y[-WIN:])
            X = np.column_stack([np.ones(len(Uw)), Uw - Uw.mean(0)])
            XtX = X.T @ X
            inv = np.linalg.inv(XtX)
            W = inv @ X.T
            beta = W @ Yw
            res = Yw - X @ beta
            s2 = float(res @ res) / max(len(Yw) - 3, 1)
            se = np.sqrt(np.diag(inv) * s2)
            wmax = np.abs(W).max(1)
            dist = float(np.max(np.linalg.norm(Uw - c, axis=1)))
            fit = (beta, se, wmax, dist)
        for C in BUDGETS:
            acc = False
            if fit is not None:
                beta, se, wmax, dist = fit
                ub = d * beta[1 + j] + crit * se[1 + j] + C * wmax[1 + j] + L * (dist + STEP / 2)
                acc = ub < 0
            rec(R[f"CR-LS({int(C)})"], dm, acc)
        rec(R["Oracle"], dm, dm < -TOL)
        # actuation, measurement, attack
        signs = {f"a{k}": exc.sign(t, k) for k in range(2)}
        u = c + DELTA * np.array([signs["a0"], signs["a1"]])
        y = plant.cost(u) + rng.normal(0, 0.02)
        if kind == "C1":
            y += -mag * (c[0] - 0.5)
        elif kind == "C2":
            y += -mag * (u[0] - c[0])
        bank.update(centre, signs, {"cost": y})
        U.append(u.copy()); Y.append(y)
        # nominal: slow reflected random walk in [0.35, 0.65]
        c = c + rng.normal(0, 0.004, 2)
        c = np.where(c < 0.35, 0.7 - c, c); c = np.where(c > 0.65, 1.3 - c, c)
    return dict(kind=kind, mag=mag, seed=seed, R=R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=30)
    a = ap.parse_args()
    jobs = [(k, m, s) for k, m in ATTACKS for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    agg = {}
    for k, m in ATTACKS:
        rr = [r for r in rows if r["kind"] == k and r["mag"] == m]
        for g in rr[0]["R"]:
            tot = {f: sum(r["R"][g][f] for r in rr) for f in ("benef", "harm", "auth_benef", "auth_harm")}
            agg[f"{k}|{m}|{g}"] = dict(bar=tot["auth_benef"] / max(tot["benef"], 1), har=tot["auth_harm"] / max(tot["harm"], 1),
                                       runs_harm=int(sum(r["R"][g]["ever_harm"] for r in rr)), n=len(rr))
            print(k, m, g, agg[f"{k}|{m}|{g}"], flush=True)
    json.dump(dict(agg=agg, config=dict(vars(a), T=T, window=WIN, attacks=ATTACKS, budgets=BUDGETS)),
              open(RES / "kic_crbaseline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
