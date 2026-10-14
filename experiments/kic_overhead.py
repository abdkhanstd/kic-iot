# Author: abdkhan
"""Implementation cost of the gate itself (not the SLM).

Per-tick wall time of each KIC component on one CPU core, in the reference
Python implementation: keyed PRF (HMAC-SHA256, one call per actuator), the
certificate-bank update (all targets x actuators), the gate decision, and
the online control variate. Also the number of live confidence sequences and
their state size after a full 800-tick episode. These are workstation
numbers for the reference code, not embedded measurements.
"""
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.scm_iot import IoTClusterSCM
from certify.kic import KeyedRademacher, CertificateBank, OnlineCV, KICGate, dither
from certify.loop import with_cost, ACTUATORS, TARGETS, EXO, BOUNDS

RES = ROOT / "experiments" / "results"


def main(T=800, reps=5):
    tims = {k: [] for k in ["prf", "bank_update", "decide", "cv"]}
    n_cs, state_kb = [], []
    for rep in range(reps):
        scm = IoTClusterSCM(seed=rep)
        exc = KeyedRademacher(b"bench")
        bank = CertificateBank(ACTUATORS, TARGETS, 0.05, 0.05, bounds=BOUNDS, anchor_radius=0.1,
                               decision_targets=["cost"])
        gate = KICGate(bank, mode="trust", L_hat=3.0)
        cv = OnlineCV(TARGETS, EXO, ACTUATORS)
        nom = dict(duty_cycle=0.5, tx_power=0.5)
        rng = np.random.default_rng(rep)
        tracemalloc.start()
        for t in range(1, T + 1):
            t0 = time.perf_counter()
            signs = [exc.sign(t, j) for j in range(len(ACTUATORS))]
            tims["prf"].append(time.perf_counter() - t0)
            c, s, u = dither(nom, t, exc, 0.05, ACTUATORS)
            y = with_cost(scm.step(u))
            t0 = time.perf_counter(); yh = cv.predict(y, c); cv.update(y, c, y)
            tims["cv"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); bank.update(c, s, y, yh)
            tims["bank_update"].append(time.perf_counter() - t0)
            prop = {"duty_cycle": float(np.clip(nom["duty_cycle"] + rng.choice([-0.1, 0.1]), 0.07, 0.93))}
            t0 = time.perf_counter(); d = gate.decide(prop, nom, "cost")
            tims["decide"].append(time.perf_counter() - t0)
            nom = {k: float(np.clip(v, 0.07, 0.93)) for k, v in d.action.items()}
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        n_cs.append(len(bank._cs) + len(bank._shadow))
        state_kb.append(peak / 1024)
    out = {k: dict(median_us=float(np.median(v) * 1e6), p99_us=float(np.quantile(v, .99) * 1e6))
           for k, v in tims.items()}
    out["total_median_us"] = float(sum(out[k]["median_us"] for k in tims))
    out["live_confidence_sequences"] = float(np.mean(n_cs))
    out["peak_traced_memory_kb"] = float(np.mean(state_kb))
    print(json.dumps(out, indent=1))
    json.dump(out, open(RES / "kic_overhead.json", "w"), indent=1)


if __name__ == "__main__":
    main()
