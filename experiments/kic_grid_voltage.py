# Author: abdkhan
"""Feeder voltages and inverter loading over the operating domains.

The feeder cost's voltage term 4 * sum(1 - V) is signed (affine, as the theory
requires), so it rewards voltage support and does not penalize overvoltage.
This records how often the inverter voltages exceed the 1.05 p.u. service
limit, and the apparent power the inverters need, over 150 sampled regimes
with uniformly drawn actions (2000 draws each), on the full domain (level C)
and on the certified domain D_f (level T).

    python kic_grid_voltage.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify import grid as GR
from certify.envelopes import grid_regimes, grid_regimes_domain

RES = ROOT / "experiments" / "results"


def main():
    rng = np.random.default_rng(0)
    out = {}
    for name, regimes in (("full", grid_regimes), ("D_f", grid_regimes_domain)):
        vs = []
        for k in range(150):
            reg = regimes(rng)
            u = {a: float(rng.uniform(0.07, 0.93)) for a in GR.ACTUATORS}   # nominal range with the dither
            v = GR.simulate_vec(reg, u, 2000, np.random.default_rng(k))
            vs.append(np.concatenate([np.asarray(v["v18"]), np.asarray(v["v33"])]))
        V = GR.V_LO + (GR.V_HI - GR.V_LO) * np.concatenate(vs)                  # back to p.u.
        out[name] = dict(v_min=float(V.min()), v_p1=float(np.percentile(V, 1)), v_median=float(np.median(V)),
                         v_p99=float(np.percentile(V, 99)), v_max=float(V.max()),
                         frac_above_1=float(np.mean(V > 1.0)), frac_above_105=float(np.mean(V > 1.05)),
                         frac_below_095=float(np.mean(V < 0.95)))
        print(name, {k_: round(x, 4) for k_, x in out[name].items()}, flush=True)
    p_max = GR.PV_RATED * (1 + 5 * GR.NOISE["pv"])                              # irradiance 1, 5 sigma of PV noise
    q_max = max(abs(GR.QMIN), abs(GR.QMAX))
    out["inverter"] = dict(pv_rated_mw=GR.PV_RATED, pv_max_mw=p_max, q_range_mvar=[GR.QMIN, GR.QMAX],
                           s_needed_rated_mva=float(np.hypot(GR.PV_RATED, q_max)),
                           s_needed_max_mva=float(np.hypot(p_max, q_max)))
    print("inverter", out["inverter"])
    json.dump(out, open(RES / "kic_grid_voltage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
