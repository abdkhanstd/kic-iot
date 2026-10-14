# Author: abdkhan
"""Actuator budget of the keyed dither on the three application plants.

For 40 random regimes and nominal actions per plant: the dither cost
M(c) - m(c) (operating minus undithered cost, common random numbers,
4000 draws per corner), relative to m(c); on the feeder also the extra
feeder loss in kW and the reactive-power swing in MVAr. Actuator activity
of a Rademacher dither: 0.5 sign reversals per tick and mean travel delta
per tick, independent of the plant.

    python kic_dither_budget.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify import plantspec as PS
from certify.envelopes import regimes_for

RES = ROOT / "experiments" / "results"
DELTA, N, NPTS = 0.05, 4000, 40


def main():
    out = {}
    for name in ("scm", "mec", "grid"):
        P = PS.get(name)
        rng = np.random.default_rng(777)
        rows = []
        for i in range(NPTS):
            reg = regimes_for(name)(rng)
            c = {a: float(rng.uniform(0.15, 0.85)) for a in P.actuators}
            m = P.op_cost(reg, c, 0.0, n=N)
            M = P.op_cost(reg, c, DELTA, n=N)
            row = dict(m=m, M=M, d=M - m, rel=(M - m) / abs(m))
            if name == "grid":
                from certify import grid as GR
                loss = lambda acts: float(np.mean(GR.LOSS_LO + (GR.LOSS_HI - GR.LOSS_LO)
                                                  * P.simulate_vec(reg, acts, N, np.random.default_rng(99))["loss"]))
                l0 = loss(c)
                lc = np.mean([loss({a: c[a] + DELTA * (1 if (k >> j) & 1 else -1) for j, a in enumerate(P.actuators)})
                              for k in range(4)])
                row.update(loss_mw=l0, dloss_kw=1000 * (lc - l0))
            rows.append(row)
            print(name, i, {k: round(v, 5) for k, v in row.items()}, flush=True)
        d = np.array([r["d"] for r in rows]); rel = np.array([r["rel"] for r in rows])
        agg = dict(mean=float(d.mean()), median=float(np.median(d)), max=float(d.max()),
                   rel_median=float(np.median(rel)), rel_max=float(rel.max()))
        if name == "grid":
            from certify import grid as GR
            dl = np.array([r["dloss_kw"] for r in rows]); l0 = np.array([r["loss_mw"] for r in rows])
            agg.update(dloss_kw_median=float(np.median(dl)), dloss_kw_max=float(dl.max()),
                       loss_kw_median=float(1000 * np.median(l0)),
                       q_swing_mvar=float(DELTA * (GR.QMAX - GR.QMIN)))
        out[name] = dict(agg=agg, rows=rows)
    out["activity"] = dict(reversals_per_tick=0.5, travel_per_tick=DELTA, delta=DELTA)
    json.dump(out, open(RES / "kic_dither_budget.json", "w"), indent=1)
    print({k: v["agg"] for k, v in out.items() if k != "activity"})


if __name__ == "__main__":
    main()
