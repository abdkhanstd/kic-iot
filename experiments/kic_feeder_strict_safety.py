# Author: abdkhan
"""Feeder (level T, certified domain D_f): strict noise-tail margin and safety-filter composition.

Re-runs the level-T feeder shadow benchmark (kic_shadow_grid_cert.json: certified envelopes,
stationary regimes, 15 seeds x 800 ticks, the four in-model scenarios) with KIC only, and records
for every proposal and every KIC authorization:

  strict  the authorization also satisfies the noise-tail margin of Corollary cor:tail(ii): the
          certified decrease of the tail-free cost M^c, s*Delta - L Delta^2/2 with s the slack
          recomputed with 2*Gbar*eps in place of w_extra, exceeds eps*Rbar, so the move lowers M
          itself (exact soundness, not "up to 5.3e-4");
  filter  a model-based safety filter (Remark rem:filter (iii)) accepts the move: at the regime's
          mean drivers, every dither corner of the new nominal keeps every bus at or below
          V_MAX = 1.05 p.u., and each inverter's apparent power sqrt(P_pv^2 + Q^2) stays within
          S_RATED = 1.2 MVA (the commanded reactive range).

The composed gate executes a proposal only if the filter accepts the proposal, KIC licenses it and
the filter accepts the licensed (possibly shortened) move. KIC's own decisions are checked against
the stored run, which they must reproduce exactly (shadow decisions do not affect the plant).

    python experiments/kic_feeder_strict_safety.py --procs 12   -> results/kic_feeder_strict_safety.json
"""
import json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
if os.name == "nt":
    import ctypes
    _k = ctypes.windll.kernel32
    _k.GetCurrentProcess.restype = ctypes.c_void_p
    _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k.SetPriorityClass(_k.GetCurrentProcess(), 0x4000)          # BELOW_NORMAL_PRIORITY_CLASS
os.environ["KIC_GRID_DOMAIN"] = "1"
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
from scipy.stats import norm
import kic_shadow as KS
from certify import kic as K
from certify import grid as GR
from certify import plantspec

DWELL, SEEDS = 800, 15
SCEN = ["none", "a1_telemetry", "spoof_blind", "a1_spoof"]
V_MAX, S_RATED = 1.05, 1.2
# Corollary cor:tail(ii) on the feeder (kic_tail_check.py): eps, the printed Gbar and Rbar, w_extra
EPS = 69 * 2 * norm.cdf(-5)
GBAR, RBAR, W_EXTRA = 131.0, 13.4, 0.016
MARGIN, SPARE = EPS * RBAR, W_EXTRA - 2 * GBAR * EPS

_cs, _q = K.KICGate.certified_step, K.CertificateBank.query


def _query(self, actuator, target, centre):
    c = _q(self, actuator, target, centre)
    c._act = actuator
    return c


def _certified_step(self, cert, d, sense, cap):
    """Unchanged decision; additionally remembers the robust slack of the certificate used."""
    r = _cs(self, cert, d, sense, cap)
    if r > 0 and cert.robust:
        m = -cert.hi if d > 0 else cert.lo
        if sense > 0:
            m = cert.lo if d > 0 else -cert.hi
        self._slack = (getattr(cert, "_act", None), d,
                       m - cert.bias - self.L_hat * (cert.eta + 0.5 * self.bank.delta) - self.bank.V * (cert.age + 1))
    return r


K.CertificateBank.query, K.KICGate.certified_step = _query, _certified_step


def safe(regime, nominal):
    """Model-based filter: every dither corner of `nominal` within V_MAX on every bus and within
    the inverters' apparent-power rating, at the regime's mean drivers."""
    F = GR.feeder()
    lvl = 0.5 + 0.8 * regime.demand_mean
    qm = 0.6 + 0.8 * regime.qshare_mean
    pv = GR.PV_RATED * regime.irr_mean
    loads = []
    for s18 in (-1, 1):
        for s33 in (-1, 1):
            u = {"q18": nominal["q18"] + 0.05 * s18, "q33": nominal["q33"] + 0.05 * s33}
            P = F["p"] * lvl
            Q = F["q"] * lvl * qm
            for j, (a, b) in enumerate(zip(GR.ACTUATORS, GR.INV_BUS)):
                q = float(GR._q_of(u[a]))
                if np.hypot(pv, q) > S_RATED:
                    return False
                P = P.copy(); Q = Q.copy()
                P[b] -= pv; Q[b] -= q
            loads.append(P + 1j * Q)
    V, _ = GR.bfs(np.array(loads), F["vm_ext"])
    return bool(np.abs(V).max() <= V_MAX)


def job(args):
    scen, seed = args
    KS.DWELL, KS.N_DWELL, KS.STATIONARY, KS.ENV = DWELL, 1, True, KS.ENV_CERT
    P = plantspec.get("grid")

    class C:
        pass
    cfg = C(); cfg.seed = seed; cfg.regime_shift_prob = 0.0; cfg.plant_memory = 0.0
    cfg.noise_mult = 1.0; cfg.drift_amp = 0.0; cfg.drift_period = 1600.0
    plant = P.make(cfg)
    from certify import envelopes
    plant.regime = envelopes.regimes_for("grid")(np.random.default_rng(seed + 7))
    K.EBConfSeq.CAP = 0.75
    G_ = KS.KICShadow(P, seed, True, KS.ENV["grid"])
    spoof = KS.spoofer_for("grid", scen)
    exc = KS.KeyedRademacher(f"key-{seed}".encode())
    traj = KS.trajectory(seed, P.actuators)
    props = KS.proposals(seed, P.actuators, len(traj))
    wc = lambda v: {**v, "cost": float(P.cost(v))}
    names = ("kic", "strict", "filter", "strict_filter")
    rec = {g: dict(auth=0, auth_harm=0, auth_benef=0, gain=0.0) for g in names}
    tot = dict(harm=0, benef=0, benef_safe=0, harm_safe=0, prop_unsafe=0, kic_auth_unsafe=0, strict_auth_unsafe=0)
    reg = plant.regime
    for t in range(1, len(traj) + 1):
        nominal = traj[t - 1]
        a, step = props[t - 1]
        c_in = {k: float(np.clip(nominal[k], K.LO + 0.05, K.HI - 0.05)) for k in P.actuators}
        m0 = P.op_cost(reg, c_in, 0.05, n=2000)
        full = dict(c_in); full[a] = float(np.clip(c_in[a] + step, K.LO + 0.05, K.HI - 0.05))
        dm_full = P.op_cost(reg, full, 0.05, n=2000) - m0
        lab = 1 if dm_full > KS.TOL else (-1 if dm_full < -KS.TOL else 0)
        prop_safe = safe(reg, full)
        tot["harm"] += lab == 1; tot["benef"] += lab == -1
        tot["benef_safe"] += lab == -1 and prop_safe; tot["harm_safe"] += lab == 1 and prop_safe
        tot["prop_unsafe"] += not prop_safe
        G_.gate._slack = None
        verdict, st = G_.decide(a, step, c_in)
        if verdict == "auth" and abs(st) > 1e-12:
            if abs(st - (full[a] - c_in[a])) < 1e-9:
                dm, tgt = dm_full, full
            else:
                tgt = dict(c_in); tgt[a] = c_in[a] + st
                dm = P.op_cost(reg, tgt, 0.05, n=2000) - m0
            sl = G_.gate._slack
            d = 1 if st > 0 else -1
            if sl is None or sl[0] != a or sl[1] != d:
                raise RuntimeError("slack of the licensing certificate not recorded")
            D = abs(st)
            strict_ok = (sl[2] + SPARE) * D - 0.5 * G_.gate.L_hat * D * D > MARGIN
            move_safe = safe(reg, tgt)
            tot["kic_auth_unsafe"] += not move_safe
            tot["strict_auth_unsafe"] += strict_ok and not move_safe
            for g, ok in (("kic", True), ("strict", strict_ok), ("filter", prop_safe and move_safe),
                          ("strict_filter", strict_ok and prop_safe and move_safe)):
                if ok:
                    r = rec[g]
                    r["auth"] += 1; r["auth_harm"] += dm > KS.TOL; r["auth_benef"] += dm < -KS.TOL
                    r["gain"] += max(-dm, 0.0)
        centre, signs, applied = KS.dither(nominal, t, exc, 0.05, P.actuators)
        true_state = wc(plant.step(applied))
        obs = wc(spoof(true_state, t, centre, applied)) if spoof else true_state
        G_.observe(centre, signs, obs)
    return dict(scen=scen, seed=seed, rec=rec, tot=tot)


def main():
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    jobs = [(s, sd) for s in SCEN for sd in range(SEEDS)]
    with Pool(args.procs) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    stored = json.load(open(RES / "kic_shadow_grid_cert.json"))["agg"]
    agg, match = {}, True
    for s in SCEN:
        R = [r for r in rows if r["scen"] == s]
        T = {k: sum(r["tot"][k] for r in R) for k in R[0]["tot"]}
        for g in R[0]["rec"]:
            x = {k: sum(r["rec"][g][k] for r in R) for k in ("auth", "auth_harm", "auth_benef", "gain")}
            den = T["benef_safe"] if "filter" in g else T["benef"]
            agg[f"{s}|{g}"] = dict(x, bar=x["auth_benef"] / max(den, 1), benef_den=den,
                                   har=x["auth_harm"] / max(T["harm_safe"] if "filter" in g else T["harm"], 1))
        agg[f"{s}|totals"] = T
        st = stored[f"grid|{s}|KIC"]
        match &= all(agg[f"{s}|kic"][k] == st[k] for k in ("auth", "auth_harm", "auth_benef")) \
            and T["harm"] == st["harm"] and T["benef"] == st["benef"]
        print(s, {g: (agg[f'{s}|{g}']['auth'], f"{100 * agg[f'{s}|{g}']['bar']:.1f}%") for g in R[0]["rec"]}, T, flush=True)
    out = dict(agg=agg, rows=rows, decisions_match_stored=bool(match),
               config=dict(seeds=SEEDS, dwell=DWELL, scenarios=SCEN, v_max=V_MAX, s_rated=S_RATED, eps=EPS,
                           Gbar=GBAR, Rbar=RBAR, margin=MARGIN, spare=SPARE))
    json.dump(out, open(RES / "kic_feeder_strict_safety.json", "w"), indent=1, default=float)
    print("decisions match stored run:", match)


if __name__ == "__main__":
    main()
