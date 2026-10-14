# Author: abdkhan
"""Operational certification of leakage (eta) and committedness on the feeder.

A reproducible procedure (paper, Sec. IV-G) that converts a physical model or
measurements, with their uncertainty, into a conservative upper bound on the
attacker's advantage eta for any coalition of side channels, and checks the
timing race of committedness:

  1. Gain. For every channel b (a smart meter at bus b), an upper bound on the
     voltage shift per dither half-step, kappa_b^+ delta. Here it is PROVED by
     interval analysis over the certified operating domain D (grid_cert.py);
     on hardware it would be the upper confidence bound of a keyed calibration
     (regressing the reading on the secret sign), at level alpha_kappa.
  2. Noise floor. A lower bound sigma_b^- on the channel's noise at the
     attacker's best position: from the device specification (as here, the
     revenue-grade class 0.002 p.u.) or the chi-square lower confidence bound
     of N_cal calibration readings, at level alpha_sigma.
  3. Coalitions. Given everything but the channels' own noise, independent
     Gaussian channels with n_b readings per tick combine into one statistic:
         eta <= 2 Phi( sqrt(sum_b n_b (kappa_b^+ delta / sigma_b^-)^2) ) - 1.
     Correlated channels must be calibrated jointly (whitened); non-Gaussian
     channels use Pinsker, eta <= sqrt(sum_b n_b KL_b / 2).
  4. Uncertainty. The calibration failure probabilities add to alpha.
  5. Timing race. Committedness holds by timing alone iff
     t_m^+ < t_a^- + d_o^- + d_i^- (worst-case sampling instant, earliest
     actuation, attacker's fastest observation and injection); otherwise the
     channel's eta must be priced.

The script also measures, by simulation, the advantage an optimal (likelihood
ratio) attacker achieves with each coalition, which must lie below eta^+.

    python kic_eta_procedure.py --procs 40
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "experiments" / "results"
DELTA = 0.05
SIGMA_METER = 0.002          # revenue-grade voltage meter, p.u. (device class = specification lower bound)
DOMAIN = [(0.1, 0.7), (0.0, 0.8), (0.0, 1.0), (0.02, 0.98), (0.02, 0.98)]
COALITIONS = {
    "bus 17 (next to inverter 18)": [16],
    "bus 6 (main feeder)": [5],
    "bus 25 (lateral from bus 3)": [24],
    "lateral from bus 2 (19-22)": [18, 19, 20, 21],
    "laterals from buses 2 and 3 (19-25)": [18, 19, 20, 21, 22, 23, 24],
    "all 30 non-inverter meters": [b for b in range(1, 33) if b not in (17, 32)],
}


ACTS = ("q18", "q33")          # both inverters are licensed, so both dithers must be priced


def _kappa_chunk(args):
    lo, hi = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from certify import grid_cert as GC
    GC.K_SIGMA = 5.0
    o = GC.sensitivities(lo, hi, second=False)
    return GC.rabsmax(*o["dr18"]), GC.rabsmax(*o["dr33"]), o["ok"]


def certified_kappa(procs, grid=(6, 8, 8, 12, 12)):
    """kappa_b^+(j) = sup over D (and +-5 sigma noise) of |d|V_b|/du_j| for each
    inverter j, by interval evaluation on a fine partition of D (which covers every
    value of the other inverter's action); invalid boxes are bisected."""
    from certify import grid_cert as GC
    los, his = GC.grid_boxes(DOMAIN, grid)
    scale = np.array([b - a for a, b in DOMAIN])
    kap = [None, None]
    with Pool(procs) as pool:
        for _ in range(8):
            chunks = [(los[i:i + 512], his[i:i + 512]) for i in range(0, len(los), 512)]
            res = pool.map(_kappa_chunk, chunks)
            ok = np.concatenate([r[2] for r in res])
            for j in range(2):
                k = np.concatenate([r[j] for r in res])
                if ok.any():
                    m = k[ok].max(0)
                    kap[j] = m if kap[j] is None else np.maximum(kap[j], m)
            if ok.all():
                break
            los, his = GC.bisect(los[~ok], his[~ok], scale)
    m = GC._M()
    # indices in bus numbering (0-based buses 1..32 are the non-root buses)
    return {a: {int(b): float(kap[j][m["pos"][b]]) for b in m["idx"]} for j, a in enumerate(ACTS)}


def eta_bound(kappas, sigmas, ns):
    snr = math.sqrt(sum(n * (k * DELTA / s) ** 2 for k, s, n in zip(kappas, sigmas, ns)))
    return 2 * norm.cdf(snr) - 1, snr


def _attack_chunk(args):
    buses, seed, n, act = args          # act: index of the inverter whose sign the attacker guesses
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from certify import grid as GR
    rng = np.random.default_rng(seed)
    F = GR.feeder()
    x = rng.uniform([a for a, _ in DOMAIN], [b for _, b in DOMAIN], (n, 5))
    lvl = 0.5 + 0.8 * x[:, 0]; qm = 0.6 + 0.8 * x[:, 1]; irr = x[:, 2]
    eP = 1 + 0.03 * rng.standard_normal((n, F["nb"])); eQ = 1 + 0.03 * rng.standard_normal((n, F["nb"]))
    epv = 1 + 0.03 * rng.standard_normal((n, 2))
    s = rng.choice([-1.0, 1.0], n)
    out = []
    for sign in (+1, -1):
        # the guessed inverter is dithered; the other sits at a known action (its sign is in the view)
        u = [x[:, 3], x[:, 4]]
        u[act] = np.clip(u[act], 0.02 + DELTA, 0.98 - DELTA) + sign * DELTA
        P = F["p"][None] * lvl[:, None] * eP; Q = F["q"][None] * (lvl * qm)[:, None] * eQ
        for j, (b, uj) in enumerate(zip(GR.INV_BUS, u)):
            P[:, b] -= GR.PV_RATED * irr * epv[:, j]; Q[:, b] -= GR.QMIN + (GR.QMAX - GR.QMIN) * uj
        V, _ = GR.bfs(P + 1j * Q, F["vm_ext"])
        out.append(np.abs(V)[:, buses])
    vp, vm = out
    # the attacker knows everything but its meters' noise and the sign: optimal test
    mid, dif = 0.5 * (vp + vm), 0.5 * (vp - vm)
    obs = np.where(s[:, None] > 0, vp, vm) + SIGMA_METER * rng.standard_normal(vp.shape)
    stat = ((obs - mid) * dif).sum(1)
    guess = np.where(stat >= 0, 1.0, -1.0)
    return float(np.mean(guess == s)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--n_cal", type=int, default=3600, help="calibration readings per meter (1 h at 1 Hz)")
    ap.add_argument("--alpha_cal", type=float, default=0.001)
    a = ap.parse_args()
    kap = certified_kappa(a.procs)
    # noise floor: specification, and, for comparison, the chi-square lower bound of a calibration
    rng = np.random.default_rng(5)
    cal = SIGMA_METER * rng.standard_normal(a.n_cal)
    s_hat = float(np.std(cal, ddof=1))
    s_lo = float(math.sqrt((a.n_cal - 1) * s_hat ** 2 / chi2.ppf(1 - a.alpha_cal, a.n_cal - 1)))
    rows = {}
    with Pool(a.procs) as pool:
        for name, buses in COALITIONS.items():
            per = {}
            for j, act in enumerate(ACTS):
                ks = [kap[act][b] for b in buses]
                eta_spec, snr = eta_bound(ks, [SIGMA_METER] * len(ks), [1] * len(ks))
                eta_cal, _ = eta_bound(ks, [s_lo] * len(ks), [1] * len(ks))
                res = pool.map(_attack_chunk, [(buses, 1000 * (j + 1) + i, 4000, j) for i in range(20)])
                acc = sum(r[0] * r[1] for r in res) / sum(r[1] for r in res)
                per[act] = dict(kappa_max=max(ks), snr=snr, eta_spec=eta_spec, eta_cal=eta_cal,
                                eta_emp=2 * acc - 1, n_emp=sum(r[1] for r in res))
            # the gate prices every certificate with one eta: the binding (largest) actuator's
            bind = max(ACTS, key=lambda a_: per[a_]["eta_spec"])
            rows[name] = dict(buses=[b + 1 for b in buses], binding=bind,
                              kappa_max=max(p_["kappa_max"] for p_ in per.values()),
                              snr=per[bind]["snr"], eta_spec=per[bind]["eta_spec"], eta_cal=per[bind]["eta_cal"],
                              eta_emp=max(p_["eta_emp"] for p_ in per.values()), n_emp=per[bind]["n_emp"],
                              by_actuator=per)
            print(name, {k_: v_ for k_, v_ in rows[name].items() if k_ != "by_actuator"}, flush=True)
            for act in ACTS:
                print("   ", act, per[act], flush=True)
    timing = dict(t_sample_after_actuation_s=1.0, d_observe_min_s=0.02, d_inject_min_s=0.001,
                  committed_by_timing=bool(1.0 < 0.02 + 0.001))
    json.dump(dict(kappa=kap, sigma_spec=SIGMA_METER, sigma_hat=s_hat, sigma_lo=s_lo, n_cal=a.n_cal,
                   alpha_cal=a.alpha_cal, coalitions=rows, timing=timing, domain=DOMAIN, delta=DELTA),
              open(RES / "kic_eta_procedure.json", "w"), indent=1)


if __name__ == "__main__":
    main()
