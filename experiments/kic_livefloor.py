# Author: abdkhan
"""Deployed liveness (Theorem 5): the gate's observable liveness floor.

At every tick the gate can compute, from its own certificate state,
    kappa_t(c) = 2 w_t(c) + (2B/delta) (W_t + W_t^beta),
where W_t is the half-width of the active confidence sequence (level alpha_1)
and W_t^beta the half-width of the same sequence at level beta (same lambdas).
Theorem 5: with probability >= 1 - beta - alpha_2, at every tick at which the
true slope satisfies |d_j M| > kappa_t, KIC licenses a step in the improving
direction. This script checks that implication on the deployed configuration
(finite windows W = 0.08/V, alpha-spending with the deployment's k-th
certificate, the monitor), with and without a key-blind (C1) attacker that
inflates the variance and the near-clip frequency, and records when the floor
first falls below the slope against when the gate first licenses.

    python kic_livefloor.py --procs 32
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify.kic import CertificateBank, KeyedRademacher
from certify.loop import window_for

RES = ROOT / "experiments" / "results"
DELTA, ALPHA, NOISE, T, BETA = 0.05, 0.05, 0.03, 1600, 0.05
CONFIGS = {"declared": dict(G=1.6, L=3.5, V=2e-4, B=0.35), "certified": dict(G=2.0, L=6.3, V=2e-4, B=0.39)}
SLOPES = [0.5, 1.0, 1.5, 2.0]
KS = [1, 10, 100]
ATTACKS = ["none", "c1"]


def floor(bank, key, centre, env):
    cs = bank._cs.get(key)
    if cs is None or not cs.sum_lam or cs.near is None:
        return math.inf
    W = (cs._log + cs.sum_v_psi) / cs.sum_lam
    Wb = (math.log(2.0 / BETA) + cs.sum_v_psi) / cs.sum_lam
    x = np.array([centre["a"]])
    w = (bank.G * cs.near.upper() + bank.L * (cs.rms_dist(x) + 0.5 * DELTA)
         + bank.V * (cs.mean_age(bank.tick) + 1))
    return 2.0 * w + 2.0 * env["B"] / DELTA * (W + Wb)


def one(args):
    cfg, beta, k, attack, seed = args
    env = CONFIGS[cfg]
    rng = np.random.default_rng(seed)
    exc = KeyedRademacher(f"floor{seed}".encode())
    bank = CertificateBank(["a"], ["y"], DELTA, ALPHA, grid=1, bounds={"y": env["B"]},
                           window=window_for(env["V"]), robust=True, G=env["G"], V=env["V"], L=env["L"],
                           family_wise=True)
    bank.k_created = k - 1
    centre = {"a": 0.5}
    first = cross = None
    wrong = False
    viol = 0            # ticks with |slope| > floor at which no improving step was licensed
    below = 0           # ticks with |slope| > floor
    floors = []
    for t in range(1, T + 1):
        s = exc.sign(t, 0)
        e = 0.0
        if attack == "c1":
            # committed before s_t: a slow offset plus rare large spikes (key-blind)
            e = 0.15 * math.sin(t / 17.0 + seed) + (rng.choice([-1.0, 1.0]) * 0.4 if rng.random() < 0.03 else 0.0)
        y = s * DELTA * beta + rng.normal(0, NOISE) + e
        bank.update(centre, {"a": s}, {"y": y}, {"y": 0.0})
        iv = bank.current_slope_intervals("a", "y", centre)
        if not iv:
            continue
        if len(iv) > 1 and max(x[0] for x in iv) > min(x[1] for x in iv):
            bank.reset()
            continue
        key = ("a", "y", bank.cell(centre))
        kap = floor(bank, key, centre, env)
        c = bank.query("a", "y", centre)
        w = c.bias + bank.L * (c.eta + 0.5 * DELTA) + bank.V * (c.age + 1)
        lo, hi = c.lo - w, c.hi + w
        improving = -1 if beta > 0 else 1
        lic = None
        if lo > 0:
            lic = -1
        elif hi < 0:
            lic = 1
        if lic is not None and lic != improving:
            wrong = True
        if lic == improving and first is None:
            first = t
        if abs(beta) > kap:
            below += 1
            if cross is None:
                cross = t
            if lic != improving:
                viol += 1
        if t % 100 == 0:
            floors.append(kap)
    return dict(cfg=cfg, beta=beta, k=k, attack=attack, seed=seed, first=first, cross=cross, wrong=wrong,
                viol=viol, below=below, floors=floors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    jobs = [(c, b, k, at, 90_000 + i) for c in CONFIGS for b in SLOPES for k in KS for at in ATTACKS
            for i in range(a.runs)]
    with Pool(a.procs) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    agg = {}
    for c in CONFIGS:
        for b in SLOPES:
            for k in KS:
                for at in ATTACKS:
                    R = [r for r in rows if (r["cfg"], r["beta"], r["k"], r["attack"]) == (c, b, k, at)]
                    f = [r["first"] for r in R if r["first"] is not None]
                    x = [r["cross"] for r in R if r["cross"] is not None]
                    agg[f"{c}|{b}|{k}|{at}"] = dict(
                        n=len(R), licensed=len(f), median_first=float(np.median(f)) if f else None,
                        crossed=len(x), median_cross=float(np.median(x)) if x else None,
                        runs_violating=int(sum(r["viol"] > 0 for r in R)),
                        ticks_violating=int(sum(r["viol"] for r in R)), ticks_below=int(sum(r["below"] for r in R)),
                        wrong=int(sum(r["wrong"] for r in R)),
                        floor_end=float(np.median([r["floors"][-1] for r in R if r["floors"]])),
                        first_le_cross=int(sum(1 for r in R if r["cross"] is not None and r["first"] is not None
                                               and r["first"] <= r["cross"])))
                    print(c, b, k, at, agg[f"{c}|{b}|{k}|{at}"], flush=True)
    json.dump(dict(agg=agg, config=dict(vars(a), T=T, noise=NOISE, delta=DELTA, beta=BETA, configs=CONFIGS,
                                        slopes=SLOPES, ks=KS, attacks=ATTACKS)),
              open(RES / "kic_livefloor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
