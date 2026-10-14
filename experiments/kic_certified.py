# Author: abdkhan
"""Theorem-covered runs: every assumption of Theorem 4 holds.

The main closed-loop matrix (kic_attacks.py, kic_mec.py) uses envelopes that
were *characterized* from 300 sampled regimes and actions, and plants whose
regime shifts abruptly (probability 0.002 per tick), which violates
Assumption 2(c). Those runs are stress tests. Here:

  * the plant is stationary: one regime per seed, drawn from the envelope
    sampler's box, and no regime shifts, so V = 0 exactly (declared 2e-4);
  * the envelopes are the *certified* bounds of certify_env.py (interval
    branch-and-bound over every action, exogenous value and noise value
    within 6 sigma; the tail adds < 1e-7 to the width). For the MEC node
    the envelopes are characterized (see the paper), so its runs are
    stationary but not certified.

    python kic_certified.py --plants scm --procs 62
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

RES = ROOT / "experiments" / "results"
# SCM: certified G <= 2.0, L <= 6.3 within 6 sigma; the noise tail adds Gbar * eps = 3.5e-8 to the width
ENV = {"scm": dict(G=2.0, L_hat=6.3, V=2e-4, B_cost=2 * 0.05 * 2.0 + 0.19, w_extra=3.5e-8),     # certified
       "mec": dict(G=2.0, L_hat=5.5, V=2e-4, B_cost=2 * 0.05 * 2.0 + 0.19),     # characterized
       "grid": dict(G=2.2, L_hat=4.5, V=2e-4, B_cost=2 * 0.05 * 2.2 + 0.19)}    # characterized
if os.environ.get("KIC_GRID_DOMAIN") == "1":
    # certified operating domain D of the feeder (see kic_shadow.py)
    ENV["grid"] = dict(G=3.35, L_hat=7.5, V=2e-4, B_cost=2 * 0.05 * 3.35 + 0.19, w_extra=0.016,
                       leak=0.047, leak_F=0.27)
SCEN = {"scm": ["oracle", "oracle_backdoor", "clean", "backdoor_lie", "backdoor_omit", "backdoor_gstar",
                "a1_telemetry", "spoof_blind"],
        "mec": ["oracle", "oracle_backdoor", "learned", "backdoor_lie", "a1_telemetry", "spoof_blind",
                "a1_spoof"],
        "grid": ["oracle", "oracle_backdoor", "learned", "backdoor_lie", "a1_telemetry", "spoof_blind",
                 "a1_spoof"]}
GATES = ["kic_sound", "obsci"]
_W = {}


def _init(w):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.update(w)


def job(a):
    plant, scen, gate, seed = a
    base = dict(plant=plant, gate=gate, n_steps=800, seed=seed, regime_shift_prob=0.0, random_regime=True)
    kw = dict(ENV[plant]) if gate.startswith("kic") else {}
    if plant == "scm":
        import kic_attacks as KA
        W = _W["scm"]
        prop, gstar, spoof, hist, exc = KA.scenario(scen, seed, W)
        r = run(RunConfig(excitation=exc, **base, **kw), prop, gstar=gstar, spoofer=spoof,
                surrogate=W["sur_clean"], hist=hist)
    else:
        if plant == "mec":
            import mec_world as MW
        else:
            import grid_world as MW
        prop, _, spoof, hist, exc = MW.scenario(scen, seed, _W[plant])
        r = run(RunConfig(excitation=exc, **base, **kw), prop, spoofer=spoof, hist=hist)
    return dict(plant=plant, scenario=scen, gate=gate, seed=seed, ecost=r["ecost"],
                harmful=r["exec_harmful_moves"], moves=r["exec_moves"], max_harm=r["max_harm"],
                first_licensed=r["first_licensed"], monitor_alarms=r["monitor_alarms"],
                lic_dir_wrong=r["lic_dir_wrong"], lic_dir_checked=r["lic_dir_checked"],
                harmful_prop=r["harmful_prop"], harmful_exec=r["harmful_exec"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=62)
    ap.add_argument("--plants", nargs="*", default=["scm", "mec"])
    ap.add_argument("--out", default="kic_certified.json")
    args = ap.parse_args()
    worlds = {}
    if "scm" in args.plants:
        import kic_attacks as KA
        worlds["scm"] = KA.build_world()
    if "mec" in args.plants:
        import mec_world as MW
        worlds["mec"] = MW.build_world()
    if "grid" in args.plants:
        import grid_world as GW
        worlds["grid"] = GW.build_world()
    jobs = [(p, s, g, sd) for p in args.plants for s in SCEN[p] for g in GATES for sd in range(args.seeds)]
    with Pool(args.procs, initializer=_init, initargs=(worlds,)) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    agg = {}
    rng = np.random.default_rng(0)
    for p in args.plants:
        for s in SCEN[p]:
            k = {r["seed"]: r["ecost"] for r in rows if r["plant"] == p and r["scenario"] == s
                 and r["gate"] == "kic_sound"}
            for g in GATES:
                R = [r for r in rows if r["plant"] == p and r["scenario"] == s and r["gate"] == g]
                b = {r["seed"]: r["ecost"] for r in R}
                d = np.array([b[i] - k[i] for i in sorted(k)])
                bs = rng.choice(d, (4000, len(d))).mean(1)
                fl = [r["first_licensed"] for r in R if r["first_licensed"]]
                agg[f"{p}|{s}|{g}"] = dict(
                    ecost=float(np.mean(list(b.values()))), paired=float(d.mean()),
                    ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    harmful=int(sum(r["harmful"] for r in R)), moves=int(sum(r["moves"] for r in R)),
                    max_harm=float(max(r["max_harm"] for r in R)),
                    first_median=float(np.median(fl)) if fl else None, runs_licensing=len(fl),
                    monitor_alarms=int(sum(r["monitor_alarms"] for r in R)),
                    harmful_prop=int(sum(r["harmful_prop"] for r in R)),
                    harmful_exec=int(sum(r["harmful_exec"] for r in R)))
                v = agg[f"{p}|{s}|{g}"]
                print(f"{p} {s:16s} {g:10s} J={v['ecost']:.3f} d={v['paired']:+.3f} harm={v['harmful']}/{v['moves']} "
                      f"max={v['max_harm']:.2e} t1={v['first_median']} ({v['runs_licensing']}) "
                      f"alarms={v['monitor_alarms']}", flush=True)
    json.dump(dict(agg=agg, rows=rows, config=dict(vars(args), env=ENV)), open(RES / args.out, "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
