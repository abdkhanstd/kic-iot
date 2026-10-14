# Author: abdkhan
"""The efficiency/safety frontier of the trust budget.

KIC's earned-trust budget rho lets uncertified proposals move the plant by up
to rho between confirmations (not covered by Theorem 4). rho = 0 is the improvement-
sound gate; rho -> infinity approaches executing every proposal in 0.05
steps. For each rho we measure the operating cost with a good proposer (the
oracle planner) and with bad ones (the discovered-SCM planner with a
backdoor, and A1), and the uncertified harmful moves. Certificates are
alpha-spent as in the main gate.
"""
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
from kic_attacks import build_world, scenario, _init, _W

RES = ROOT / "experiments" / "results"
RHOS = [0.0, 0.05, 0.1, 0.15, 0.3, 0.6, 1.2]
SCEN = ["oracle", "oracle_backdoor", "backdoor_lie", "a1_telemetry"]


def job(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rho, scen, seed = a
    prop, gstar, spoof, hist, exc = scenario(scen, seed, _W)
    cfg = RunConfig(gate="kic" if rho > 0 else "kic_sound", n_steps=800, seed=seed, excitation=exc,
                    family_wise=True, monitor=True)
    import certify.loop as LP
    if rho > 0:
        # KICGate reads trust_radius at construction; pass it through a subclass hook
        orig = LP.KICGate.__init__

        def init(self, *a_, **k_):
            k_["trust_radius"] = rho
            orig(self, *a_, **k_)
        LP.KICGate.__init__ = init
    r = run(cfg, prop, gstar=gstar, spoofer=spoof, surrogate=_W["sur_clean"], hist=hist)
    if rho > 0:
        LP.KICGate.__init__ = orig
    hb = r["harm_by_src"].get("earned_trust", [0, 0])
    return dict(rho=rho, scen=scen, seed=seed, ecost=r["ecost"], trust_harm=hb[0], trust_moves=hb[1],
                harmful=r["exec_harmful_moves"], moves=r["exec_moves"])


def main():
    world = build_world()
    jobs = [(rho, s, sd) for rho in RHOS for s in SCEN for sd in range(20)]
    with Pool(62, initializer=_init, initargs=(world,)) as p:
        rows = p.map(job, jobs, chunksize=1)
    out = {}
    for rho in RHOS:
        for s in SCEN:
            R = [r for r in rows if r["rho"] == rho and r["scen"] == s]
            e = np.array([r["ecost"] for r in R])
            out[f"{rho}|{s}"] = dict(ecost=float(e.mean()), se=float(e.std(ddof=1) / np.sqrt(len(e))),
                                     trust_harm=sum(r["trust_harm"] for r in R),
                                     trust_moves=sum(r["trust_moves"] for r in R),
                                     harmful=sum(r["harmful"] for r in R), moves=sum(r["moves"] for r in R))
            print(rho, s, out[f"{rho}|{s}"], flush=True)
    json.dump(dict(agg=out, rows=rows), open(RES / "kic_frontier.json", "w"), indent=1)


if __name__ == "__main__":
    main()
