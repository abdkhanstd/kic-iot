# Author: abdkhan
"""What does certification cost, and how fast is it?

(a) Excitation cost: an oracle planner with no gate, run with and without
    the dither, at several amplitudes delta -- the pure price of probing.
(b) Time to first certificate and final operating cost of the certified
    reversionary controller alone (kic_fb_only), as a function of delta,
    with and without the online control variate. The clip bound scales with
    delta so that the near-clip threshold B - 2 delta G of Theorem 3 stays at
    0.19 (B = 0.35 at delta = 0.05).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify.loop import RunConfig, run, OracleProposer

RES = ROOT / "experiments" / "results"
DELTAS = [0.02, 0.035, 0.05, 0.08]


def job(a):
    kind, delta, cv, seed = a
    if kind == "excite":
        r = run(RunConfig(gate="none", n_steps=600, seed=seed, delta=delta, dither_always=delta > 0),
                OracleProposer())
    else:
        r = run(RunConfig(gate="kic_fb_only", n_steps=800, seed=seed, delta=delta, cv=cv,
                          B_cost=0.19 + 2 * delta * 1.6, family_wise=True, monitor=True), OracleProposer(),
                record=True)
        moves = [x["t"] for i, x in enumerate(r["trace"][1:], 1)
                 if (x["n_duty_cycle"], x["n_tx_power"]) != (r["trace"][i - 1]["n_duty_cycle"],
                                                             r["trace"][i - 1]["n_tx_power"])]
        r["t_first_move"] = moves[0] if moves else None
        del r["trace"]
    r.update(kind=kind, delta=delta, cv=cv, seed=seed)
    return r


def main():
    jobs = [("excite", d, "online", s) for d in [0.0] + DELTAS for s in range(10)]
    jobs += [("fb", d, cv, s) for d in DELTAS for cv in ["online", "cellmean"] for s in range(10)]
    with Pool(40) as p:
        rows = p.map(job, jobs, chunksize=1)
    out = {}
    base = np.mean([r["ecost"] for r in rows if r["kind"] == "excite" and r["delta"] == 0.0])
    for d in DELTAS:
        e = [r["ecost"] for r in rows if r["kind"] == "excite" and r["delta"] == d]
        out[f"excite|{d}"] = dict(ecost=float(np.mean(e)), overhead=float(np.mean(e) - base))
        for cv in ["online", "cellmean"]:
            R = [r for r in rows if r["kind"] == "fb" and r["delta"] == d and r["cv"] == cv]
            tf = [r["t_first_move"] for r in R if r["t_first_move"] is not None]
            out[f"fb|{d}|{cv}"] = dict(ecost=float(np.mean([r["ecost"] for r in R])),
                                       ecost_tail=float(np.mean([r["ecost_tail"] for r in R])),
                                       median_t_first_cert=float(np.median(tf)) if tf else None,
                                       frac_certified=len(tf) / len(R),
                                       exec_harmful_moves=float(np.mean([r["exec_harmful_moves"] for r in R])),
                                       exec_moves=float(np.mean([r["exec_moves"] for r in R])))
    out["excite_base"] = float(base)
    for k, v in out.items():
        print(k, v)
    json.dump(out, open(RES / "kic_cost_of_certification.json", "w"), indent=1)


if __name__ == "__main__":
    main()
