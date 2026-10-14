# Author: abdkhan
"""Does the certificate keep its promise, and whom can an attacker fool?

Null target: the measured `interference` channel. It is exogenous, so the
true effect of either actuator on it is exactly 0, and any certificate on it
is a false certification. An attacker forges a strong relation between duty
cycle and the MEASURED interference; the nominal action follows an
adversarial random walk (the proposer is attacker-controlled).

For each condition we run many independent loops and record whether a
certificate on (duty_cycle -> interference) EVER appears in T ticks, i.e.
the anytime false-certification rate, which Theorem 2 bounds by alpha.

Observational baseline: OLS slope of the measured channel on the APPLIED
duty cycle, with a 95% t-interval, looked at every 25 ticks (a continuously
monitored observational verifier). It is fooled by any forgery correlated
with the action.

Conditions
  honest       no spoofing
  blind        forged term kappa*(nominal-0.5): attacker sees commands, not K
  blind_cv     blind + an ADVERSARIAL control variate supplied by the attacker
  aware        forged term uses the APPLIED actuator (attacker reads it: K leaked)
  public       public sinusoidal dither, attacker knows the schedule
  raw          OUTPUT-ADAPTIVE: the attacker sees the genuine tick-t latency
               before replacing it and reflects it about its own running mean
               (key-blind, but its forgery depends on s_t through the plant)
  raw_delayed  the same attacker with a one-tick-late view of the genuine output

Factorial (evidence x inference), on the null target:
  obs_fixed    observational OLS slope, fixed-sample t-test, looked at every 25 ticks
  obs_cs       observational sign of (u_t - running mean) as the "treatment",
               fed to the same empirical-Bernstein confidence sequence as KIC
  key_fixed    keyed dither sign as regressor, fixed-sample t-test, every 25 ticks
  key_cs       KIC
Power: the same loops on the true target `latency` (duty -> latency is positive in 98% of states), reporting the fraction of
runs certified with the correct sign, and the median time to certificate.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.scm_iot import IoTClusterSCM
from certify.kic import KeyedRademacher, PublicSinusoid, CertificateBank, OnlineCV, dither
from certify.loop import with_cost, ACTUATORS, EXO

RES = ROOT / "experiments" / "results"
TARGETS = ["interference", "latency"]
BOUNDS = {"interference": 0.7, "latency": 0.2}


def one(args):
    cond, seed, T, alpha, delta, kappa = args
    rng = np.random.default_rng(seed)
    scm = IoTClusterSCM(seed=seed, regime_shift_prob=0.0)
    exc = PublicSinusoid() if cond == "public" else KeyedRademacher(f"k{seed}".encode())
    bank = CertificateBank(ACTUATORS, TARGETS, delta, alpha, grid=1, bounds=BOUNDS)
    # CV on exogenous drivers would trivially explain `interference`; use the
    # others only, so the null channel keeps its full noise
    cv = OnlineCV(TARGETS, ["link_quality", "arrival_rate"], ACTUATORS)
    nominal = dict(duty_cycle=float(rng.uniform(0.2, 0.8)), tx_power=float(rng.uniform(0.2, 0.8)))
    xs, ys, ss = [], [], []
    first_null, obs_fooled, first_cost, cost_sign = None, None, None, 0
    key_fixed_fooled, obs_cs_fooled = None, None
    from certify.kic import EBConfSeq
    obs_cs = EBConfSeq(alpha)
    mu_u, n_u = 0.0, 0
    mu_y, t_y = 0.0, 0
    lat_m, lat_n, prev_true_lat = 0.0, 0, None
    for t in range(1, T + 1):
        # adversarial random walk of the nominal action (attacker-chosen)
        a = "duty_cycle" if rng.random() < 0.5 else "tx_power"
        nominal[a] = float(np.clip(nominal[a] + rng.choice([-0.05, 0.05]), 0.1, 0.9))
        centre, signs, applied = dither(nominal, t, exc, delta, ACTUATORS)
        true = with_cost(scm.step(applied))
        obs = dict(true)
        if cond in ("blind", "blind_cv"):
            x = nominal["duty_cycle"]
        elif cond == "aware":
            x = applied["duty_cycle"]
        elif cond == "public":
            x = nominal["duty_cycle"] + delta * exc.sign(t, 0)
        else:
            x = None
        if x is not None:
            obs["interference"] = float(np.clip(true["interference"] + kappa * (x - 0.5), 0, 1))
        if cond in ("raw", "raw_delayed"):
            src = true["latency"] if cond == "raw" else (prev_true_lat if prev_true_lat is not None else true["latency"])
            mu = lat_m / lat_n if lat_n else src
            obs["latency"] = float(np.clip(2 * mu - src, 0, 1))
            lat_m += true["latency"]; lat_n += 1; prev_true_lat = true["latency"]
        yh = cv.predict(obs, centre)
        if cond == "blind_cv" and yh is not None:
            # attacker-supplied predictor: anything computable before s_t
            yh = dict(yh)
            yh["interference"] = 0.5 - 3 * kappa * (nominal["duty_cycle"] - 0.5) + 0.2 * np.sin(t)
        if yh is not None:
            bank.update(centre, signs, obs, yh)
        cv.update(obs, centre, obs)
        c_null = bank.query("duty_cycle", "interference", centre)
        if first_null is None and c_null.sign != 0:
            first_null = t
        c_cost = bank.query("duty_cycle", "latency", centre)
        if first_cost is None and c_cost.sign != 0:
            first_cost, cost_sign = t, c_cost.sign
        # observational + anytime-valid: the action's own deviation direction
        # plays the role of the dither in the same EB confidence sequence
        # Purely observational: the "treatment" is the nominal action's
        # deviation from its running mean (no dither), and the baseline is the
        # running mean of the measurement (a CV containing the action would
        # regress the tested effect away).
        if t > 1:
            xdir = 1 if nominal["duty_cycle"] >= mu_u / n_u else -1
            rr = float(np.clip(obs["interference"] - mu_y / t_y, -0.7, 0.7)) / 0.7
            obs_cs.update((xdir * rr + 1) / 2)
            lo, hi = obs_cs.interval()
            if obs_cs_fooled is None and (lo > 0.5 or hi < 0.5):
                obs_cs_fooled = t
        mu_u += nominal["duty_cycle"]; n_u += 1
        mu_y += obs["interference"]; t_y += 1
        xs.append(applied["duty_cycle"]); ys.append(obs["interference"]); ss.append(signs["duty_cycle"])
        if key_fixed_fooled is None and t % 25 == 0 and t >= 50:
            Xs = np.column_stack([ss, np.ones(len(ss))])
            bk, rk, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
            s2k = float(rk[0]) / (len(ss) - 2) if len(rk) else 0.0
            sek = np.sqrt(s2k * np.linalg.inv(Xs.T @ Xs)[0, 0])
            if abs(bk[0]) > 1.96 * sek:
                key_fixed_fooled = t
        if obs_fooled is None and t % 25 == 0 and t >= 50:
            X = np.column_stack([xs, np.ones(len(xs))])
            b, res, *_ = np.linalg.lstsq(X, ys, rcond=None)
            s2 = float(res[0]) / (len(xs) - 2) if len(res) else 0.0
            se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
            if abs(b[0]) > 1.96 * se:
                obs_fooled = t
    clip_rate = bank.n_clip / bank.n_obs if bank.n_obs else 0.0
    return dict(cond=cond, seed=seed, null_certified=first_null is not None, t_null=first_null,
                clip_rate=clip_rate,
                key_fixed_fooled=key_fixed_fooled is not None, obs_cs_fooled=obs_cs_fooled is not None,
                obs_fooled=obs_fooled is not None, t_obs=obs_fooled,
                cost_certified=first_cost is not None, t_cost=first_cost, cost_sign=cost_sign)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=0.4)
    ap.add_argument("--procs", type=int, default=56)
    args = ap.parse_args()
    conds = ["honest", "blind", "blind_cv", "aware", "public", "raw", "raw_delayed"]
    jobs = [(c, 10_000 + i, args.T, args.alpha, args.delta, args.kappa)
            for c in conds for i in range(args.runs)]
    with Pool(args.procs) as p:
        rows = p.map(one, jobs, chunksize=4)
    out = {}
    for c in conds:
        R = [r for r in rows if r["cond"] == c]
        n = len(R)
        kn = sum(r["null_certified"] for r in R)
        ko = sum(r["obs_fooled"] for r in R)
        kc = sum(r["cost_certified"] and r["cost_sign"] == 1 for r in R)
        kw = sum(r["cost_certified"] and r["cost_sign"] == -1 for r in R)
        tc = [r["t_cost"] for r in R if r["cost_certified"]]
        kf = sum(r["key_fixed_fooled"] for r in R); oc = sum(r["obs_cs_fooled"] for r in R)
        out[c] = dict(n=n, kic_false_cert=kn / n, kic_false_cert_ci=wilson(kn, n),
                      key_fixed_false_cert=kf / n, key_fixed_ci=wilson(kf, n),
                      obs_cs_false_cert=oc / n, obs_cs_ci=wilson(oc, n),
                      obs_false_cert=ko / n, obs_false_cert_ci=wilson(ko, n),
                      cost_power=kc / n, cost_wrong_sign=kw / n,
                      median_t_cost=float(np.median(tc)) if tc else None,
                      clip_rate=float(np.mean([r["clip_rate"] for r in R])))
        print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[c].items()})
    json.dump(dict(summary=out, config=vars(args)), open(RES / "kic_validity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
