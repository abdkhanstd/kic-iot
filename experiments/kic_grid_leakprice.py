# Author: abdkhan
"""The liveness price of pricing a side channel on the feeder.

An operator who must assume that an attacker may own a meter at bus b widens
every certificate of the bus-18 inverter by eta_bound(b) F / delta, whether or
not the attack happens. Here: no attacker, the learned DMS planner and the
oracle, KIC with the widening for each exposure level of Table (gridleak).

    python kic_grid_leakprice.py --procs 60
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

RES = ROOT / "experiments" / "results"
_W = {}


def _init(w):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.update(w)


def job(a):
    scen, bus, seed = a
    import grid_world as GW
    from certify.loop import RunConfig, run
    prop, _, spoof, hist, exc = GW.scenario(scen, seed, _W["world"])
    kw = dict(GW.ENV)
    if bus is not None:
        kw.update(leak=_W["eta"][bus], leak_F=_W["F"])
    r = run(RunConfig(gate="kic_sound", plant="grid", n_steps=800, seed=seed, excitation=exc, **kw), prop,
            spoofer=spoof, hist=hist)
    return dict(scen=scen, bus=bus, seed=seed, ecost=r["ecost"], harmful=r["exec_harmful_moves"],
                moves=r["exec_moves"], licensing=int(r["exec_moves"] > 0), first=r["first_licensed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=60)
    a = ap.parse_args()
    import grid_world as GW
    sc = json.load(open(RES / "kic_grid_sidechannel.json"))
    sig, n = sc["config"]["loop_meter"]
    buses = [19, 24, 5, 16]
    eta = {b: sc["channel"][str(b)][f"bound|{sig}|{n}"] for b in buses}
    world = GW.build_world()
    jobs = [(s, b, sd) for s in ("learned",) for b in [None] + buses for sd in range(a.seeds)]
    with Pool(a.procs, initializer=_init, initargs=({"world": world, "eta": eta, "F": sc["config"]["F"]},)) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    agg = {}
    for s in ("learned",):
        for b in [None] + buses:
            R = [r for r in rows if r["scen"] == s and r["bus"] == b]
            fl = [r["first"] for r in R if r["first"]]
            agg[f"{s}|{b}"] = dict(ecost=float(np.mean([r["ecost"] for r in R])), moves=int(sum(r["moves"] for r in R)),
                                  harmful=int(sum(r["harmful"] for r in R)), runs_licensing=int(sum(r["licensing"] for r in R)),
                                  first_median=float(np.median(fl)) if fl else None,
                                  eta=None if b is None else eta[b],
                                  widening=0.0 if b is None else eta[b] * sc["config"]["F"] / sc["config"]["delta"])
            print(s, b, agg[f"{s}|{b}"], flush=True)
    json.dump(dict(agg=agg, config=dict(vars(a), buses=buses)), open(RES / "kic_grid_leakprice.json", "w"), indent=1)


if __name__ == "__main__":
    main()
