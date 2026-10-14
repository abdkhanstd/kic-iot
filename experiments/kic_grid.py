# Author: abdkhan
"""The third plant: smart-inverter Volt/VAR control on the IEEE 33-bus
feeder (src/certify/grid.py, pandapower's case33bw; scenarios in
experiments/grid_world.py). Same gates and protocol as the other plants
(common random numbers per seed, 800 ticks), plus KIC without its fallback
and its fallback alone. Envelopes characterized by the paired-input
procedure: G = 2.02, L = 3.95; declared G = 2.2, L = 4.5, V = 2e-4, B = 0.41.
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
from certify.loop import RunConfig, run
import grid_world as MW

RES = ROOT / "experiments" / "results"
ENV = MW.ENV                           # characterized: G = 2.02, L = 3.95; declared 2.2 / 4.5
GATES = ["obsci", "wm", "kic_sound", "kic_sound_gate", "kic_sound_fbonly"]
SCENARIOS = ["oracle", "learned", "a1_telemetry", "spoof_blind", "a1_spoof", "spoof_aware"]
_W = {}


def _init(w):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.update(w)


def job(a):
    scen, gate, seed = a
    prop, _, spoof, hist, exc = MW.scenario(scen, seed, _W)
    kw = dict(ENV) if gate.startswith("kic") else {}
    kw["harm_n"] = _W.get("harm_n", 4000)
    r = run(RunConfig(gate=gate, plant="grid", n_steps=800, seed=seed, excitation=exc, **kw), prop,
            spoofer=spoof, hist=hist)
    r.update(scenario=scen, gate=gate, seed=seed)
    r.pop("cert_pt_abs", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=62)
    ap.add_argument("--harm_n", type=int, default=4000)
    args = ap.parse_args()
    W = MW.build_world()
    jobs = [(s, g, sd) for s in SCENARIOS for g in GATES for sd in range(args.seeds)]
    W = dict(W, harm_n=args.harm_n)
    with Pool(args.procs, initializer=_init, initargs=(W,)) as p:
        rows = p.map(job, jobs, chunksize=1)
    agg = {}
    rng = np.random.default_rng(0)
    for s in SCENARIOS:
        k = {r["seed"]: r["ecost"] for r in rows if r["scenario"] == s and r["gate"] == "kic_sound"}
        for g in GATES:
            R = [r for r in rows if r["scenario"] == s and r["gate"] == g]
            b = {r["seed"]: r["ecost"] for r in R}
            d = np.array([b[i] - k[i] for i in sorted(k)])
            bs = rng.choice(d, (4000, len(d))).mean(1)
            agg[f"{s}|{g}"] = dict(ecost=float(np.mean(list(b.values()))),
                                   paired=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                   kic_better=int((d > 0).sum()),
                                   harmful=sum(r["exec_harmful_moves"] for r in R), moves=sum(r["exec_moves"] for r in R),
                                   lic_dir_wrong=sum(r["lic_dir_wrong"] for r in R),
                                   lic_dir_checked=sum(r["lic_dir_checked"] for r in R),
                                   max_harm=max(r["max_harm"] for r in R),
                                   first_licensed=[r["first_licensed"] for r in R],
                                   monitor_alarms=sum(r["monitor_alarms"] for r in R))
            v = agg[f"{s}|{g}"]
            print(f"{s:16s} {g:16s} J={v['ecost']:.3f} d={v['paired']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
                  f"harm={v['harmful']}/{v['moves']} dirwrong={v['lic_dir_wrong']}/{v['lic_dir_checked']}", flush=True)
    json.dump(dict(agg=agg, rows=rows, config=dict(vars(args), env=ENV)), open(RES / "kic_grid.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
