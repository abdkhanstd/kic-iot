# Author: abdkhan
"""Deconfounding authorization from the certified fallback.

In closed loop, KIC moves the plant with its own licensed fallback when it
refuses a proposal; the comparators simply hold. Here KIC runs *without* the
fallback (gate 'kic_sound_gate'): it executes licensed parts of proposals and
otherwise holds, exactly like every comparator. Same scenarios, seeds and
envelopes as kic_attacks.py / kic_mec.py, so the per-seed pairing with the
comparators' rows in kic_attacks.json / kic_mec.json is exact.

    python kic_gateonly.py --procs 62
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
SCEN = {"scm": ["oracle", "oracle_backdoor", "clean", "backdoor_lie", "backdoor_omit", "backdoor_gstar",
                "a1_telemetry", "spoof_blind"],
        "mec": ["oracle", "oracle_backdoor", "learned", "backdoor_lie", "a1_telemetry", "spoof_blind", "a1_spoof"]}
MEC_ENV = dict(G=2.0, L_hat=5.5, V=2e-4, B_cost=2 * 0.05 * 2.0 + 0.19)
_W = {}


def _init(w):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.update(w)


GATES = ["kic_sound_gate", "kic_sound_fbonly"]


def job(a):
    plant, scen, seed, gate = a
    if plant == "scm":
        import kic_attacks as KA
        W = _W["scm"]
        prop, gstar, spoof, hist, exc = KA.scenario(scen, seed, W)
        r = run(RunConfig(gate=gate, n_steps=800, seed=seed, excitation=exc), prop, gstar=gstar,
                spoofer=spoof, surrogate=W["sur_clean"], hist=hist)
    else:
        import mec_world as MW
        prop, _, spoof, hist, exc = MW.scenario(scen, seed, _W["mec"])
        r = run(RunConfig(gate=gate, plant="mec", n_steps=800, seed=seed, excitation=exc, **MEC_ENV),
                prop, spoofer=spoof, hist=hist)
    return dict(plant=plant, scenario=scen, gate=gate, seed=seed, ecost=r["ecost"],
                exec_harmful_moves=r["exec_harmful_moves"], exec_moves=r["exec_moves"], max_harm=r["max_harm"],
                harmful_prop=r["harmful_prop"], harmful_exec=r["harmful_exec"], benef_prop=r["benef_prop"],
                benef_exec=r["benef_exec"], trig_harm_prop=r["trig_harm_prop"], trig_harm_exec=r["trig_harm_exec"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=62)
    args = ap.parse_args()
    import kic_attacks as KA
    import mec_world as MW
    worlds = {"scm": KA.build_world(), "mec": MW.build_world()}
    jobs = [(p, s, sd, g) for p in SCEN for s in SCEN[p] for sd in range(args.seeds) for g in GATES]
    with Pool(args.procs, initializer=_init, initargs=(worlds,)) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    for p in SCEN:
        for s in SCEN[p]:
          for g in GATES:
            R = [r for r in rows if r["plant"] == p and r["scenario"] == s and r["gate"] == g]
            print(p, s, g, f"J={np.mean([r['ecost'] for r in R]):.3f}",
                  f"harm={sum(r['exec_harmful_moves'] for r in R)}/{sum(r['exec_moves'] for r in R)}",
                  f"benef_exec={sum(r['benef_exec'] for r in R)}/{sum(r['benef_prop'] for r in R)}", flush=True)
    json.dump(dict(rows=rows, config=vars(args)), open(RES / "kic_gateonly.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
