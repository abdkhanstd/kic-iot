# Author: abdkhan
"""A physical side channel on the IEEE 33-bus feeder (Proposition 4, Corollary 3).

The dither of the inverter at bus 18 moves every bus voltage a little. An
attacker who owns a smart meter somewhere on the feeder can try to read the
dither sign from its own voltage and then forge the loss telemetry with it.

(1) Channel parameters per bus, from the power-flow model: the voltage gain
    kappa_b = dV_b / du_18 (so a dither half-step moves V_b by kappa_b * delta),
    and the residual noise a realistic attacker faces after regressing its
    meter on the published drivers and nominal commands.
(2) Two leakage figures per bus and meter class (noise sigma p.u., n readings
    per tick):
      eta_bound  Corollary 3 conditional on everything but the meter's own
                 noise: 2 Phi(sqrt(n) kappa delta / sigma) - 1. This is the
                 upper bound Proposition 4 needs (valid for any processing).
      eta_real   the advantage of the regression attacker, measured in closed loop.
(3) Closed loop (30 seeds) with the attacker at chosen buses: KIC without the
    leakage term, and KIC widened by eta_bound * F / delta with the attacker's
    forgery budget F.

    python kic_grid_sidechannel.py --procs 60
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify import grid as GR
from certify import envelopes as E

RES = ROOT / "experiments" / "results"
DELTA = 0.05
# the spoofer forges the normalized loss channel by -0.6 (x - 0.5) with x = c + delta s_hat:
# its leak-dependent part is +-0.6 delta on [0, 1], i.e. +-W_loss (LOSS_HI - LOSS_LO) 0.6 delta in cost
F_ATT = GR.COST_W["loss"] * (GR.LOSS_HI - GR.LOSS_LO) * 0.6 * DELTA
METERS = [(0.002, 1), (0.002, 10), (0.01, 1)]          # (sigma p.u., readings per tick)
LOOP_BUSES = [16, 5, 24]                              # 0-based: next to inverter 18, mid-feeder, other lateral
LOOP_METER = (0.002, 1)
_W = {}


def channel_params(n_reg=40, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    nb = GR.feeder()["nb"]
    kap = np.zeros((n_reg, nb))
    res_sd = np.zeros((n_reg, nb))
    for i in range(n_reg):
        reg = E.grid_regimes(rng)
        c = {"q18": float(rng.uniform(0.2, 0.8)), "q33": float(rng.uniform(0.2, 0.8))}
        up, dn = dict(c), dict(c)
        up["q18"] += DELTA
        dn["q18"] -= DELTA
        vu = GR.simulate_vec(reg, up, n, np.random.default_rng(100 + i))["_vm"]
        vd = GR.simulate_vec(reg, dn, n, np.random.default_rng(100 + i))["_vm"]
        kap[i] = np.mean(vu - vd, 0) / (2 * DELTA)
        # what the regression attacker cannot explain: plant noise given the drivers,
        # plus the other inverter's dither, at a fixed nominal
        v = GR.simulate_vec(reg, c, n, np.random.default_rng(900 + i))
        q33 = np.where(np.random.default_rng(7 + i).random(n) < 0.5, 1, -1)
        v2 = GR.simulate_vec(reg, {"q18": c["q18"], "q33": c["q33"] + DELTA * q33}, n, np.random.default_rng(900 + i))
        X = np.column_stack([v["demand"], v["irradiance"], v["qshare"], np.ones(n)])
        for b in range(nb):
            y = v2["_vm"][:, b]
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            res_sd[i, b] = np.std(y - X @ coef)
    return np.median(np.abs(kap), 0), np.median(res_sd, 0)


def loop_job(a):
    gate, bus, seed = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import grid_world as GW
    from certify.loop import RunConfig, run
    sigma, n_obs = LOOP_METER
    scen = f"spoof_meter:{bus}:{sigma}:{n_obs}"
    prop, _, spoof, hist, exc = GW.scenario(scen, seed, _W["grid"])
    kw = dict(GW.ENV)
    if gate == "widened":
        kw.update(leak=_W["eta_bound"][bus], leak_F=F_ATT)
    r = run(RunConfig(gate="kic_sound", plant="grid", n_steps=800, seed=seed, excitation=exc, **kw), prop,
            spoofer=spoof, hist=hist)
    return dict(gate=gate, bus=bus, seed=seed, ecost=r["ecost"], harmful=r["exec_harmful_moves"],
                moves=r["exec_moves"], max_harm=r["max_harm"], licensing=int(r["exec_moves"] > 0),
                n_guess=spoof.n_guess, n_correct=spoof.n_correct)


def _init(w):
    _W.update(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=60)
    a = ap.parse_args()
    kap, rsd = channel_params()
    nb = len(kap)
    table = {}
    for b in range(nb):
        row = dict(kappa=float(kap[b]), dv=float(kap[b] * DELTA), plant_sd=float(rsd[b]))
        for sig, n in METERS:
            eb = 2 * norm.cdf(np.sqrt(n) * kap[b] * DELTA / sig) - 1
            er = 2 * norm.cdf(kap[b] * DELTA / np.sqrt(rsd[b] ** 2 + sig ** 2 / n)) - 1
            row[f"bound|{sig}|{n}"] = float(eb)
            row[f"gauss_real|{sig}|{n}"] = float(er)
            row[f"widening|{sig}|{n}"] = float(eb * F_ATT / DELTA)
        table[b] = row
        print(b + 1, {k: round(v, 5) for k, v in row.items()}, flush=True)
    import grid_world as GW
    world = GW.build_world()
    sig, n = LOOP_METER
    eta_bound = {b: table[b][f"bound|{sig}|{n}"] for b in LOOP_BUSES}
    jobs = [(g, b, s) for g in ("plain", "widened") for b in LOOP_BUSES for s in range(a.seeds)]
    with Pool(a.procs, initializer=_init, initargs=({"grid": world, "eta_bound": eta_bound},)) as pool:
        rows = pool.map(loop_job, jobs, chunksize=1)
    loop = {}
    for g in ("plain", "widened"):
        for b in LOOP_BUSES:
            R = [r for r in rows if r["gate"] == g and r["bus"] == b]
            ng, nc = sum(r["n_guess"] for r in R), sum(r["n_correct"] for r in R)
            loop[f"{g}|{b}"] = dict(ecost=float(np.mean([r["ecost"] for r in R])), harmful=int(sum(r["harmful"] for r in R)),
                                  moves=int(sum(r["moves"] for r in R)), max_harm=float(max(r["max_harm"] for r in R)),
                                  runs_licensing=int(sum(r["licensing"] for r in R)),
                                  eta_emp=float(2 * nc / max(ng, 1) - 1), eta_bound=float(eta_bound[b]),
                                  widening=float(eta_bound[b] * F_ATT / DELTA))
            print("loop", g, b + 1, loop[f"{g}|{b}"], flush=True)
    json.dump(dict(channel={str(b): v for b, v in table.items()}, loop=loop,
                   config=dict(delta=DELTA, F=F_ATT, meters=METERS, loop_buses=LOOP_BUSES, loop_meter=LOOP_METER,
                               seeds=a.seeds)),
              open(RES / "kic_grid_sidechannel.json", "w"), indent=1)


if __name__ == "__main__":
    main()
