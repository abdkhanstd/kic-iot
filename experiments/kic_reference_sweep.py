# Author: abdkhan
"""Misspecified reference graph G*.

The published verifier's guarantees are conditional on G* being right. We
corrupt G* at increasing rates -- flip the sign of a fraction f of edges,
drop f/2, add f/2*|E| spurious edges -- and replay the same proposals (an
oracle planner with a 10% backdoor) through
  gstar_target  the graph-sign rule (claim-free form of the published gate)
  kic_prior     KIC that falls back to G* while a direction is uncertified
  kic           KIC that never consults G*
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.scm_iot import ground_truth_graph
from certify.oracle import TRUE_EDGE_SIGNS, perturb_reference, cost_sign_from_graph
from certify.loop import RunConfig, run, OracleProposer, BackdoorProposer

RES = ROOT / "experiments" / "results"
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
GATES = ["gstar_target", "kic_prior", "kic", "kic_sound"]


def job(a):
    f, draw, seed, gate = a
    g, es = perturb_reference(TRUE_EDGE_SIGNS, ground_truth_graph(), flip=f, drop=f / 2,
                              spurious=f / 2, seed=draw)
    signs = {x: cost_sign_from_graph(g, es, x) for x in ["duty_cycle", "tx_power"]}
    prop = BackdoorProposer(OracleProposer(), 0.1, "lie", seed=seed)
    r = run(RunConfig(gate=gate, n_steps=600, seed=seed), prop, gstar=(g, es))
    return dict(level=f, draw=draw, seed=seed, gate=gate, ecost=r["ecost"],
                harmful_prop=r["harmful_prop"], harmful_exec=r["harmful_exec"],
                benef_prop=r["benef_prop"], benef_exec=r["benef_exec"],
                ref_signs=signs, at_limit=r["at_limit"])


def main():
    jobs = [(f, d, 500 + 10 * d + s, g) for f in LEVELS for d in range(4) for s in range(5) for g in GATES]
    with Pool(40) as p:
        rows = p.map(job, jobs, chunksize=1)
    out = {}
    for f in LEVELS:
        for g in GATES:
            R = [r for r in rows if r["level"] == f and r["gate"] == g]
            hp = sum(r["harmful_prop"] for r in R); he = sum(r["harmful_exec"] for r in R)
            bp = sum(r["benef_prop"] for r in R); be = sum(r["benef_exec"] for r in R)
            ec = np.array([r["ecost"] for r in R])
            wrong = np.mean([sum(r["ref_signs"][x] != s for x, s in [("duty_cycle", 1), ("tx_power", -1)])
                             for r in R])
            out[f"{f}|{g}"] = dict(ecost=float(ec.mean()), ecost_se=float(ec.std() / np.sqrt(len(ec))),
                                   false_accept=he / hp if hp else None,
                                   false_reject=1 - be / bp if bp else None,
                                   ref_sign_errors=float(wrong), at_limit=float(np.mean([r["at_limit"] for r in R])),
                                   n=len(R))
            print(f, g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f"{f}|{g}"].items()},
                  flush=True)
    json.dump(dict(agg=out, rows=rows), open(RES / "kic_reference_sweep.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
