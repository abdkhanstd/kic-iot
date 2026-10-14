# Author: abdkhan
"""Liveness of the DEPLOYED certificate configuration (not the stylized setting
of the proof of Theorem 5): finite windows W = 0.08/V = 400, alpha-spending over every
certificate the gate creates, the envelope monitor, and the SCM's declared
envelopes (G = 1.6, L = 3.5, V = 2e-4, B = 0.35) or the certified ones
(G = 2.0, L = 6.3, B = 0.39).

A scalar keyed experiment at a fixed operating point: y_t = s_t delta beta + noise
(noise sd 0.03, the SCM's residual scale). For each true slope beta we record
the first tick at which Proposition 2 licenses a step in the improving
direction, and whether any step is ever licensed in the wrong direction
(it never should be). The certificate index k (how many certificates the
deployment has already spent) sets alpha_k = 6 alpha / (pi^2 k^2).

    python kic_liveness.py --procs 16
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify.kic import CertificateBank, KeyedRademacher
from certify.loop import window_for

RES = ROOT / "experiments" / "results"
DELTA, ALPHA, NOISE, T = 0.05, 0.05, 0.03, 1600
CONFIGS = {"declared": dict(G=1.6, L=3.5, V=2e-4, B=0.35), "certified": dict(G=2.0, L=6.3, V=2e-4, B=0.39)}
SLOPES = [0.25, 0.5, 1.0, 1.5, 2.0]
KS = [1, 10, 100]


def one(args):
    cfg, beta, k, seed = args
    env = CONFIGS[cfg]
    rng = np.random.default_rng(seed)
    exc = KeyedRademacher(f"live{seed}".encode())
    bank = CertificateBank(["a"], ["y"], DELTA, ALPHA, grid=1, bounds={"y": env["B"]},
                           window=window_for(env["V"]), robust=True, G=env["G"], V=env["V"], L=env["L"],
                           family_wise=True)
    bank.k_created = k - 1                      # the deployment has already spent k - 1 certificates
    first, wrong = None, False
    for t in range(1, T + 1):
        s = exc.sign(t, 0)
        y = s * DELTA * beta + rng.normal(0, NOISE)
        bank.update({"a": 0.5}, {"a": s}, {"y": y}, {"y": 0.0})
        iv = bank.current_slope_intervals("a", "y", {"a": 0.5})
        if not iv:
            continue
        if len(iv) > 1 and max(x[0] for x in iv) > min(x[1] for x in iv):
            bank.reset()                        # the monitor (Proposition 5); never fires here
            continue
        c = bank.query("a", "y", {"a": 0.5})    # the certificate the gate acts on (Theorem 3 interval)
        w = c.bias + bank.L * (c.eta + 0.5 * DELTA) + bank.V * (c.age + 1)
        lo, hi = c.lo - w, c.hi + w
        if hi < 0 or lo > 0:
            gamma = -hi if hi < 0 else lo
            if gamma > 0:
                improving = -1 if beta > 0 else 1
                licensed_dir = -1 if lo > 0 else 1
                if licensed_dir != improving:
                    wrong = True
                if first is None and licensed_dir == improving:
                    first = t
    return dict(cfg=cfg, beta=beta, k=k, seed=seed, first=first, wrong=wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    jobs = [(c, b, k, 70_000 + i) for c in CONFIGS for b in SLOPES for k in KS for i in range(a.runs)]
    with Pool(a.procs) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    agg = {}
    for c in CONFIGS:
        for b in SLOPES:
            for k in KS:
                R = [r for r in rows if r["cfg"] == c and r["beta"] == b and r["k"] == k]
                f = [r["first"] for r in R if r["first"] is not None]
                agg[f"{c}|{b}|{k}"] = dict(
                    frac=len(f) / len(R), median=float(np.median(f)) if f else None,
                    p90=float(np.quantile(f, 0.9)) if len(f) >= 0.9 * len(R) else None,
                    by={str(T_): float(np.mean([x is not None and x <= T_ for x in [r["first"] for r in R]]))
                        for T_ in (200, 400, 800, 1600)},
                    wrong=int(sum(r["wrong"] for r in R)), n=len(R))
                print(c, b, k, agg[f"{c}|{b}|{k}"], flush=True)
    json.dump(dict(agg=agg, config=dict(vars(a), T=T, noise=NOISE, delta=DELTA, configs=CONFIGS, slopes=SLOPES,
                                        ks=KS)), open(RES / "kic_liveness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
