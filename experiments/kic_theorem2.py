# Author: abdkhan
"""Theorem 3: does the certificate bound the CURRENT, UNCLIPPED slope?

Two failure modes of the naive certificate (sign of a lambda-weighted
average of CLIPPED finite differences) that we identified, each on a
scalar keyed experiment where the truth is known exactly.

(A) Clipping inside the threat model. The attacker is key-blind but sees the
    tick's exogenous draw, hence the per-realization response D_t. On a rare
    realization class (probability p) the response is large and positive;
    otherwise it is small and negative, so the mean response beta is
    POSITIVE. The attacker adds A_t = B + D_t on the rare class only, which
    saturates both arms (omega_t = 0) and deletes exactly the positive mass:
    beta^B < 0 < beta. The clipping rate is p/2 < 1%, below the guard.
    v1 (1% guard) vs Theorem 3 (near-clip correction with the true G).

(B) Drift and breaks, at a fixed nominal action (so locality plays no role):
    'ramp'  the slope moves linearly from +1.2 to -1.2 over the middle T/2 ticks,
    'jump'  the slope flips from +1.2 to -1.2 at a random tick.
    We record, per run, whether any certificate the gate could act on
    disagrees in sign with the CURRENT slope (|beta_t| > 0.05), and the
    number of such ticks. v1: raw interval, windowed. Theorem 3: interval
    widened by V*(weighted age + 1) with the declared V (ramp: the true rate;
    jump: V = 0, i.e. a violated envelope). For the jump we also record the
    last tick after the break at which a certificate still contradicts the
    new slope: window rotation bounds it by 2W = 300.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify.kic import CertificateBank, KeyedRademacher

RES = ROOT / "experiments" / "results"
DELTA, B, ALPHA, NOISE = 0.05, 0.35, 0.05, 0.03
S = 1.2                                          # slope magnitude in the drift cases


def corrected(bank, cert, V):
    w = cert.bias + bank.L * (cert.eta + 0.5 * bank.delta) + V * (cert.age + 1)
    return cert.lo - w, cert.hi + w


def one_clip(args):
    mode, seed, T, p, big, small = args
    rng = np.random.default_rng(seed)
    exc = KeyedRademacher(f"c{seed}".encode())
    G = big                                     # true envelope: |D| <= delta*big
    rob = mode == "thm2"
    bank = CertificateBank(["a"], ["y"], DELTA, ALPHA, grid=1, bounds={"y": B}, window=None,
                           robust=rob, G=G, V=0.0, L=0.0,
                           clip_guard=0.01 if not rob else None)
    beta = p * big - (1 - p) * small
    wrong, first = 0, None
    for t in range(1, T + 1):
        s = exc.sign(t, 0)
        rare = rng.random() < p
        D = DELTA * (big if rare else -small)
        A = (B + D) if (rare and mode != "honest") else 0.0   # key-blind: uses D, not s
        y = A + s * D + rng.normal(0, NOISE)
        bank.update({"a": 0.5}, {"a": s}, {"y": y}, {"y": 0.0})
        c = bank.query("a", "y", {"a": 0.5})
        if rob:
            lo, hi = corrected(bank, c, 0.0)
            sg = 1 if lo > 0 else (-1 if hi < 0 else 0)
        else:
            sg = c.sign
        if sg != 0 and first is None:
            first = (t, sg)
        if sg == -np.sign(beta):
            wrong += 1
    return dict(kind="clip", mode=mode, seed=seed, beta=beta, wrong_ever=wrong > 0,
                certified_right=first is not None and first[1] == np.sign(beta),
                clip_rate=bank.n_clip / bank.n_obs)


def one_drift(args):
    kind, mode, seed, T = args
    rng = np.random.default_rng(seed)
    exc = KeyedRademacher(f"d{seed}".encode())
    if kind == "ramp":
        # +S until T/4, linear to -S at 3T/4, then -S
        slope = lambda t: float(np.clip(S - 2 * S * (t - T / 4) / (T / 2), -S, S))
        V = 2 * S / (T / 2)
    else:
        tj = int(rng.integers(T // 4, 3 * T // 4))
        slope = lambda t: S if t < tj else -S
        V = 0.0                                  # declared envelope is violated by the break
    rob = mode != "v1"
    bank = CertificateBank(["a"], ["y"], DELTA, ALPHA, grid=1, bounds={"y": B}, window=150,
                           robust=rob, G=1.0, V=V, L=0.0)
    bad_ticks, bad_ever, cov_fail = 0, False, False
    stale = 0       # ticks after a break until no certificate contradicts the new slope
    alarms = 0
    for t in range(1, T + 1):
        s = exc.sign(t, 0)
        b = slope(t)
        y = s * DELTA * b + rng.normal(0, NOISE)
        bank.update({"a": 0.5}, {"a": s}, {"y": y}, {"y": 0.0})
        if mode == "thm2mon":
            # Proposition 5: current-slope intervals of the live certificates
            # (main and shadow window) must intersect; if not, reset
            iv = bank.current_slope_intervals("a", "y", {"a": 0.5})
            if len(iv) > 1 and max(x[0] for x in iv) > min(x[1] for x in iv):
                bank.reset(); alarms += 1
                continue
        c = bank.query("a", "y", {"a": 0.5})
        if c.n == 0:
            continue
        if rob:
            lo, hi = corrected(bank, c, V)
        else:
            lo, hi = c.lo, c.hi
        bnext = slope(t + 1)                     # the slope the next step will meet
        sg = 1 if lo > 0 else (-1 if hi < 0 else 0)
        if not (lo <= bnext <= hi):
            cov_fail = True
        if sg != 0 and abs(bnext) > 0.05 and sg != np.sign(bnext):
            bad_ticks += 1
            bad_ever = True
            stale = t - (tj if kind == "jump" else 0)
    return dict(kind=kind, mode=mode, seed=seed, bad_ever=bad_ever, bad_ticks=bad_ticks,
                cov_fail=cov_fail, stale=stale, alarms=alarms)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=60)
    a = ap.parse_args()
    jobs_c = [(m, 20_000 + i, a.T, 0.012, 30.0, 0.3) for m in ["honest", "v1", "thm2"]
              for i in range(a.runs)]
    jobs_d = [(k, m, 30_000 + i, a.T) for k in ["ramp", "jump"] for m in ["v1", "thm2", "thm2mon"]
              for i in range(a.runs)]
    with Pool(a.procs) as pool:
        rc = pool.map(one_clip, jobs_c, chunksize=4)
        rd = pool.map(one_drift, jobs_d, chunksize=4)
    out = {}
    for m in ["honest", "v1", "thm2"]:
        R = [r for r in rc if r["mode"] == m]
        n = len(R); k = sum(r["wrong_ever"] for r in R); kr = sum(r["certified_right"] for r in R)
        out[f"clip|{m}"] = dict(n=n, wrong=k / n, wrong_ci=wilson(k, n), right=kr / n,
                                clip_rate=float(np.mean([r["clip_rate"] for r in R])),
                                beta=R[0]["beta"])
    for kd in ["ramp", "jump"]:
        for m in ["v1", "thm2", "thm2mon"]:
            R = [r for r in rd if r["kind"] == kd and r["mode"] == m]
            n = len(R); k = sum(r["bad_ever"] for r in R); kc = sum(r["cov_fail"] for r in R)
            out[f"{kd}|{m}"] = dict(n=n, wrong_ever=k / n, wrong_ci=wilson(k, n),
                                    cov_fail=kc / n, cov_ci=wilson(kc, n),
                                    wrong_ticks=float(np.mean([r["bad_ticks"] for r in R])),
                                    stale_max=float(np.max([r["stale"] for r in R])),
                                    stale_mean=float(np.mean([r["stale"] for r in R])),
                                    alarms=float(np.mean([r["alarms"] for r in R])))
    for k, v in out.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    json.dump(dict(summary=out, config=vars(a)), open(RES / "kic_theorem2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
