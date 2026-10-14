# Author: abdkhan
"""Corruption actually injected per CR-LS window in kic_crbaseline.py.

CR-LS declares a total absolute corruption C per 400-tick window. This script replays the random
stream of kic_crbaseline.one() for every seed (initial nominal, then per tick the proposal draws,
the measurement noise and the nominal's reflected random walk; the plant and the gates draw
nothing from it) and sums |offset| over every full 400-tick window, for each attack:
  C1(a): -a (c_1 - 0.5)      C2(b): -b (u_1 - c_1), |u_1 - c_1| = delta

    python experiments/kic_crbaseline_budget.py      -> results/kic_crbaseline_budget.json
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
T, WIN, DELTA, SEEDS = 1600, 400, 0.05, 30
ATTACKS = [("C1", 0.4), ("C1", 1.6), ("C1", 6.4), ("C2", 1.6)]
BUDGETS = [1.0, 10.0, 100.0]


def window_totals(kind, mag, seed):
    rng = np.random.default_rng(60000 + seed)
    c = rng.uniform(0.4, 0.6, 2)
    e = []
    for _ in range(T):
        rng.integers(2); rng.choice([-1, 1])          # proposal: actuator and direction
        rng.normal(0, 0.02)                           # measurement noise
        e.append(abs(mag * (c[0] - 0.5)) if kind == "C1" else mag * DELTA)
        c = c + rng.normal(0, 0.004, 2)
        c = np.where(c < 0.35, 0.7 - c, c); c = np.where(c > 0.65, 1.3 - c, c)
    cs = np.concatenate([[0.0], np.cumsum(e)])
    return cs[WIN:] - cs[:-WIN]


def main():
    out = {}
    for kind, mag in ATTACKS:
        w = np.concatenate([window_totals(kind, mag, s) for s in range(SEEDS)])
        out[f"{kind}|{mag}"] = dict(min=float(w.min()), median=float(np.median(w)), max=float(w.max()),
                                    **{f"share_above_{int(C)}": float(np.mean(w > C)) for C in BUDGETS})
        print(kind, mag, out[f"{kind}|{mag}"], flush=True)
    json.dump(dict(agg=out, config=dict(T=T, window=WIN, delta=DELTA, seeds=SEEDS)),
              open(RES / "kic_crbaseline_budget.json", "w"), indent=1)


if __name__ == "__main__":
    main()
