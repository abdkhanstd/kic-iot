# Author: abdkhan
"""Proposition 4: the price of leakage.

A side channel lets the attacker guess the dither sign s_{t,j} correctly with
probability (1 + eta)/2 (total-variation advantage eta; eta = 0 is C1, eta = 1
is C2). Proposition 4 widens the certificate by eta * min(F, B) / delta, with
F the budget of the leak-dependent part of the forgery (F = B: no budget).

(1) Scalar keyed experiment with known slope beta = +0.5 (G = 1.2, B = 0.35):
    the attacker writes -a * guess with a = 0.09, the largest amplitude that
    stays below the near-clip threshold, so the near-clip term cannot see it.
    Its bias a*eta/delta is the tightness witness of Proposition 4.
    Certificates: Theorem 3 (no leakage term), widened with F = B, widened
    with the declared budget F = a.
(2) Closed loop on the SCM (C1 spoofer + backdoor, but the spoofer reads a
    leaky side channel on the duty-cycle dither): KIC without the leakage
    term, with F = B, and with the spoofer's true budget F = 0.08.

    python kic_leakage.py --procs 62
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify.kic import CertificateBank, KeyedRademacher

RES = ROOT / "experiments" / "results"
DELTA, B, ALPHA, NOISE, G, BETA, A_SUB = 0.05, 0.35, 0.05, 0.03, 1.2, 0.5, 0.09
ETAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def one_scalar(args):
    gate, eta, seed, T = args
    rng = np.random.default_rng(seed)
    exc = KeyedRademacher(f"leak{seed}".encode())
    widen = {"thm3": 0.0, "widenB": eta * B / DELTA, "widenF": eta * A_SUB / DELTA}[gate]
    bank = CertificateBank(["a"], ["y"], DELTA, ALPHA, grid=1, bounds={"y": B}, window=None,
                           robust=True, G=G, V=0.0, L=0.0)
    wrong, right_t = False, None
    for t in range(1, T + 1):
        s = exc.sign(t, 0)
        guess = s if rng.random() < (1 + eta) / 2 else -s
        y = s * DELTA * BETA + rng.normal(0, NOISE) - A_SUB * guess
        bank.update({"a": 0.5}, {"a": s}, {"y": y}, {"y": 0.0})
        c = bank.query("a", "y", {"a": 0.5})
        if c.n == 0:
            continue
        w = c.bias + bank.L * (c.eta + 0.5 * DELTA) + widen
        lo, hi = c.lo - w, c.hi + w
        if hi < 0:
            wrong = True
        if lo > 0 and right_t is None:
            right_t = t
    return dict(kind="scalar", gate=gate, eta=eta, seed=seed, wrong=wrong, right_t=right_t)


def loop_job(args):
    gate, eta, F, seed, n_steps = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import kic_attacks as KA
    from certify.loop import RunConfig, run, Spoofer, BackdoorProposer, PlannerProposer, \
        surrogate_predict_factory
    W = KA._W
    base = PlannerProposer(surrogate_predict_factory(W["sur_clean"]), "sur")
    prop = BackdoorProposer(base, 0.2, "lie", seed=seed, restrict=("duty_cycle", 1))
    spoof = Spoofer(kappa=0.6, knowledge="leak", leak=eta, seed=seed)
    widen = gate != "plain"
    cfg = RunConfig(gate="kic_sound", n_steps=n_steps, seed=seed,
                    leak=eta if widen else 0.0, leak_F=(F if gate == "widenF" else None))
    r = run(cfg, prop, gstar=W["gstar_true"], spoofer=spoof, surrogate=W["sur_clean"], hist=W["H_clean"])
    return dict(kind="loop", gate=gate, eta=eta, seed=seed, ecost=float(np.mean(r["ecost"])),
                harmful=r["exec_harmful_moves"], moves=r["exec_moves"], max_harm=r["max_harm"],
                licensing=int(r["exec_moves"] > 0))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n_steps", type=int, default=800)
    ap.add_argument("--procs", type=int, default=62)
    ap.add_argument("--loop_etas", nargs="*", type=float, default=[0.1, 0.25, 0.5, 1.0])
    a = ap.parse_args()
    out = {"config": dict(runs=a.runs, T=a.T, seeds=a.seeds, delta=DELTA, B=B, G=G, beta=BETA,
                          a_sub=A_SUB, noise=NOISE, loop_etas=a.loop_etas, loop_F=0.08),
           "scalar": {}, "loop": {}}
    jobs = [(g, e, 50_000 + i, a.T) for g in ("thm3", "widenB", "widenF") for e in ETAS
            for i in range(a.runs)]
    with Pool(a.procs) as pool:
        rs = pool.map(one_scalar, jobs, chunksize=8)
    for g in ("thm3", "widenB", "widenF"):
        for e in ETAS:
            rr = [r for r in rs if r["gate"] == g and r["eta"] == e]
            k = sum(r["wrong"] for r in rr)
            ts = [r["right_t"] for r in rr if r["right_t"] is not None]
            out["scalar"][f"{g}|{e}"] = dict(n=len(rr), wrong=k / len(rr), wrong_ci=wilson(k, len(rr)),
                                           power=len(ts) / len(rr),
                                           t_right=float(np.median(ts)) if ts else None)
            print("scalar", g, e, out["scalar"][f"{g}|{e}"], flush=True)
    import kic_attacks as KA
    world = KA.build_world()
    jobs = [(g, e, 0.08, s, a.n_steps) for g in ("plain", "widenB", "widenF") for e in a.loop_etas
            for s in range(a.seeds)]
    with Pool(a.procs, initializer=KA._init, initargs=(world,)) as pool:
        rl = pool.map(loop_job, jobs, chunksize=1)
    for g in ("plain", "widenB", "widenF"):
        for e in a.loop_etas:
            rr = [r for r in rl if r["gate"] == g and r["eta"] == e]
            out["loop"][f"{g}|{e}"] = dict(n=len(rr), ecost=float(np.mean([r["ecost"] for r in rr])),
                                         harmful=int(sum(r["harmful"] for r in rr)),
                                         moves=int(sum(r["moves"] for r in rr)),
                                         max_harm=float(max(r["max_harm"] for r in rr)),
                                         runs_licensing=int(sum(r["licensing"] for r in rr)))
            print("loop", g, e, out["loop"][f"{g}|{e}"], flush=True)
    json.dump(out, open(RES / "kic_leakage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
