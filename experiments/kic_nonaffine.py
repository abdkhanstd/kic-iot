# Author: abdkhan
"""Non-affine costs (Proposition 1): a constant, committed, key-blind offset
on a channel that enters the cost nonlinearly reverses the certified slope;
a magnitude bound E on the offset, priced by the widening kappa_h * E * G_Y,
restores soundness.

Scalar plant: Y = y0 + g (u - 0.5) + noise, cost J = h(Y) = Y^2 (kappa_h = 2),
g = 1, dither delta = 0.05, nominal held at 0.5, so d M / d c = 2 g y0.
The attacker adds a constant b to the reported Y (known before deployment,
hence independent of every secret sign).

    python kic_nonaffine.py --procs 16
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

RES = ROOT / "experiments" / "results"
DELTA, ALPHA, NOISE, T, GY, KH = 0.05, 0.05, 0.03, 1600, 1.0, 2.0
G_ENV, L_ENV = 2.0 * 1.0 * GY, KH * GY ** 2          # |h'| <= 2 on |Y| <= 1; h'' g^2
# (label, y0, b, E): E = declared magnitude bound (None: unbounded, no widening)
CASES = [("unbounded", 0.30, -0.60, None), ("bounded, no widening", 0.15, -0.35, None),
         ("bounded, widened", 0.15, -0.35, 0.35), ("bounded, widened, live", 0.30, -0.10, 0.10),
         ("no attack", 0.30, 0.0, None)]


def one(args):
    ci, seed = args
    label, y0, b, E = CASES[ci]
    rng = np.random.default_rng(seed)
    exc = KeyedRademacher(f"nonaffine{seed}".encode())
    extra = 0.0 if E is None else KH * E * GY
    bank = CertificateBank(["a"], ["y"], DELTA, ALPHA, grid=1, bounds={"y": 2 * DELTA * (G_ENV + extra) + 0.19},
                           window=None, robust=True, G=G_ENV + extra, V=0.0, L=L_ENV, family_wise=True,
                           w_extra=extra)
    improving = -1 if y0 > 0 else 1
    first, wrong = None, False
    for t in range(1, T + 1):
        s = exc.sign(t, 0)
        Y = y0 + GY * DELTA * s + rng.normal(0, NOISE) + b
        bank.update({"a": 0.5}, {"a": s}, {"y": Y * Y}, None)
        c = bank.query("a", "y", {"a": 0.5})
        if c.sign == 0:
            continue
        w = c.bias + bank.L * (c.eta + 0.5 * DELTA)
        lo, hi = c.lo - w, c.hi + w
        lic = -1 if lo > 0 else (1 if hi < 0 else None)
        if lic is not None and lic != improving:
            wrong = True
        if lic == improving and first is None:
            first = t
    return dict(case=label, seed=seed, first=first, wrong=wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    with Pool(a.procs) as pool:
        rows = pool.map(one, [(ci, 50_000 + i) for ci in range(len(CASES)) for i in range(a.runs)], chunksize=8)
    agg = {}
    for label, y0, b, E in CASES:
        R = [r for r in rows if r["case"] == label]
        f = [r["first"] for r in R if r["first"] is not None]
        agg[label] = dict(y0=y0, b=b, E=E, true_slope=2 * GY * y0, attacked_slope=2 * GY * (y0 + b),
                          widening=0.0 if E is None else KH * E * GY, n=len(R),
                          runs_wrong=int(sum(r["wrong"] for r in R)), runs_licensed=len(f),
                          median_first=float(np.median(f)) if f else None)
        print(label, agg[label], flush=True)
    json.dump(dict(agg=agg, config=dict(vars(a), T=T, noise=NOISE, delta=DELTA, kappa_h=KH, g=GY)),
              open(RES / "kic_nonaffine.json", "w"), indent=1)


if __name__ == "__main__":
    main()
