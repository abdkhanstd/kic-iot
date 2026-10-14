# Author: abdkhan
"""Does the ordering of gates survive changes to the plant?

Plant variants: noise x0.5 / x2, faster regime drift, and queue MEMORY
rho in {0.5, 0.8} (actions then have delayed effects on queue, CPU, latency,
loss and energy, which a one-tick certificate does not model). Scenarios: A1
(poisoned log, rediscovered) and the lying backdoor. Scored on the realized,
unforged operating cost averaged over the episode and seeds, because the
static oracle does not model plant memory.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify.loop import RunConfig, run
from kic_attacks import build_world, scenario, _init, _W

RES = ROOT / "experiments" / "results"
VARIANTS = {"base": {}, "noise x0.5": dict(noise_mult=0.5), "noise x2": dict(noise_mult=2.0),
            "drift x5": dict(regime_shift_prob=0.01), "memory 0.5": dict(plant_memory=0.5),
            "memory 0.8": dict(plant_memory=0.8)}
GATES = ["none", "gstar_claims", "obsci", "kic"]
SCEN = ["a1_telemetry", "backdoor_lie"]


def job(a):
    var, scen, gate, seed = a
    prop, gstar, spoof, hist, exc = scenario(scen, seed, _W)
    cfg = RunConfig(gate=gate, n_steps=800, seed=seed, excitation=exc, **VARIANTS[var])
    r = run(cfg, prop, gstar=gstar, spoofer=spoof, surrogate=_W["sur_clean"], hist=hist)
    return dict(var=var, scen=scen, gate=gate, seed=seed, cost=r["cost"], ecost=r["ecost"],
                cert_wrong=r["cert_wrong"], cert_checked=r["cert_checked"])


def main():
    world = build_world()
    jobs = [(v, s, g, sd) for v in VARIANTS for s in SCEN for g in GATES for sd in range(10)]
    with Pool(54, initializer=_init, initargs=(world,)) as p:
        rows = p.map(job, jobs, chunksize=1)
    out = {}
    for v in VARIANTS:
        for s in SCEN:
            k = {r["seed"]: r["cost"] for r in rows if r["var"] == v and r["scen"] == s and r["gate"] == "kic"}
            for g in GATES:
                R = [r for r in rows if r["var"] == v and r["scen"] == s and r["gate"] == g]
                b = {r["seed"]: r["cost"] for r in R}
                diff = np.array([b[i] - k[i] for i in sorted(k)])
                bs = np.random.default_rng(0).choice(diff, (4000, len(diff))).mean(1)
                out[f"{v}|{s}|{g}"] = dict(cost=float(np.mean([r["cost"] for r in R])),
                                          paired_vs_kic=float(diff.mean()),
                                          paired_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                          cert_wrong=sum(r["cert_wrong"] for r in R),
                                          cert_checked=sum(r["cert_checked"] for r in R))
                print(v, s, g, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                for kk, vv in out[f"{v}|{s}|{g}"].items()}, flush=True)
    json.dump(dict(agg=out, rows=rows), open(RES / "kic_robustness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
