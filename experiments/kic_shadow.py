# Author: abdkhan
"""Shadow-mode authorization benchmark: gate quality in isolation.

In closed loop each gate drives its own trajectory, and KIC's certified
fallback moves the plant when proposals are refused, so cost differences mix
verification quality with fallback control. Here every gate sees the SAME
states, the SAME telemetry and the SAME proposals, and its decisions are
scored but never executed:

  * the nominal action follows a fixed operating trajectory per seed (four
    dwell points of 200 ticks, each with a slow reflected random walk);
  * the actuator adds the keyed +-delta dither on every tick, so all gates
    receive identical records (u_t, Y_t); only KIC knows the key;
  * on every tick a proposal (actuator, step in {+-0.05, +-0.10}) is drawn
    from a fixed stream and presented to every gate;
  * ground truth is the change in the operating cost M_delta (dither
    included), by common-random-number Monte Carlo; harmful if it raises M
    by more than 1e-3, beneficial if it lowers it by more than 1e-3.

KIC authorizes only the licensed part of a step (Proposition 2); the
observational gates authorize the whole step. No gate has a fallback.

Metrics per gate: harmful-authorization rate (share of harmful proposals
authorized), beneficial-authorization rate (power), abstention, the harmful
share of authorizations, the latency to the first authorized beneficial
proposal after each dwell change, and the realized improvement if every
authorized step were executed.
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
from certify import plantspec
from certify.kic import (KeyedRademacher, CertificateBank, KICGate, OnlineCV, ExoChangeDetector,
                         dither, EBConfSeq, LO, HI)
from certify.loop import ObsAdjGate, ObsRLSGate, WatermarkDetector, Spoofer, MAX_STEP

RES = ROOT / "experiments" / "results"
DWELL, N_DWELL = 200, 4          # --dwell 800 --n_dwell 1 holds one operating point per seed
NBIN = 4                          # authorization rates are also recorded per quarter of a dwell
STEPS = [-0.10, -0.05, 0.05, 0.10]
TOL = 1e-3
ENV = {"scm": dict(G=1.6, L=3.5, V=2e-4), "mec": dict(G=2.0, L=5.5, V=2e-4), "grid": dict(G=2.2, L=4.5, V=2e-4)}
# certified envelopes (certify_env.py: interval branch-and-bound over the whole input space)
ENV_CERT = {"scm": dict(G=2.0, L=6.3, V=2e-4, w_extra=3.5e-8), "mec": dict(G=2.0, L=5.5, V=2e-4),
            "grid": dict(G=2.2, L=4.5, V=2e-4)}
import os as _os
if _os.environ.get("KIC_GRID_DOMAIN") == "1":
    # feeder on its certified operating domain D (grid_certify_run.py): certified G, L; the
    # +-5 sigma noise tail adds 3 Gfree eps = 0.016; leakage premise: attacker meters only on
    # the lateral from bus 2, certified eta <= 0.047 (kic_eta_procedure.py), forgery budget F = 0.27
    ENV_CERT["grid"] = dict(G=3.35, L=7.5, V=2e-4, w_extra=0.016, leak=0.047, leak_F=0.27)
STATIONARY = False                # --stationary: no regime shifts, one random regime per seed
_W = {}


def trajectory(seed, acts):
    rng = np.random.default_rng(seed + 4242)
    out = []
    for k in range(N_DWELL):
        c = rng.uniform(0.15, 0.85, len(acts))
        for _ in range(DWELL):
            c = c + rng.normal(0, 0.002, len(acts))
            c = np.where(c < 0.1, 0.2 - c, c)
            c = np.where(c > 0.9, 1.8 - c, c)
            out.append({a: float(c[j]) for j, a in enumerate(acts)})
    return out


def proposals(seed, acts, n):
    rng = np.random.default_rng(seed + 777)
    return [(acts[int(rng.integers(len(acts)))], float(rng.choice(STEPS))) for _ in range(n)]


class KICShadow:
    def __init__(self, P, seed, sound, env, delta=0.05, alpha=0.05):
        self.P, self.sound = P, sound
        B = 2 * delta * env["G"] + 0.19
        from certify.loop import window_for
        self.bank = CertificateBank(P.actuators, ["cost"], delta=delta, alpha=alpha, bounds={"cost": B},
                                    anchor_radius=0.1, family_wise=sound, decision_targets=["cost"],
                                    robust=True, G=env["G"], V=env["V"], L=env["L"], window=window_for(env["V"]),
                                    w_extra=env.get("w_extra", 0.0), leak=env.get("leak", 0.0),
                                    leak_F=env.get("leak_F"))
        self.gate = KICGate(self.bank, max_step=MAX_STEP, mode="hold", fallback=False, L_hat=env["L"])
        self.cv = OnlineCV(["cost"], P.exo, P.actuators)
        self.det = ExoChangeDetector(P.exo)
        self.alarms = 0

    def decide(self, a, step, nominal):
        if self.sound:
            for a_ in self.P.actuators:
                iv = self.bank.current_slope_intervals(a_, "cost", nominal)
                if len(iv) > 1 and max(x[0] for x in iv) > min(x[1] for x in iv):
                    self.bank.reset(); self.alarms += 1
                    break
        v = float(np.clip(nominal[a] + step, LO, HI))
        dec = self.gate.decide({a: v}, nominal, "cost", sense=-1)
        if dec.verdict == "ACCEPT":
            return "auth", dec.action[a] - nominal[a]
        return ("reject" if dec.reason == "CERTIFIED_HARMFUL" else "abstain"), 0.0

    def observe(self, centre, signs, obs):
        yh = self.cv.predict(obs, centre)
        self.cv.update(obs, centre, obs)
        if yh is not None:
            self.bank.update(centre, signs, obs, yh)
        if self.det.update(obs):
            self.bank.reset(); self.cv.reset()


class ObsShadow:
    def __init__(self, P, hist, kind):
        self.P, self.kind = P, kind
        self.g = ObsRLSGate(hist=hist, P=P) if kind == "rls" else ObsAdjGate(live=True, hist=hist, ci_z=1.645, P=P)

    def decide(self, a, step, nominal):
        v = float(np.clip(nominal[a] + step, LO, HI))
        verdict, nxt = self.g.decide({a: v}, nominal, "cost")
        if verdict == "ACCEPT":
            return "auth", nxt[a] - nominal[a]
        return ("reject" if verdict == "REJECT" else "abstain"), 0.0

    def observe(self, applied, obs, true_state):
        self.g.observe(applied, obs)


class WMShadow(ObsShadow):
    """Watermark test on the dither with the TRUE plant model; alarm -> abstain
    (hold) or, for the resilient variant, switch Obs.-CI to a trusted sensor."""

    def __init__(self, P, hist, plant, switch):
        super().__init__(P, hist, "ci")
        self.switch = switch

        def pred(act, ob):
            ex = {k: np.full(1000, float(ob[k])) for k in P.exo}
            v_ = P.simulate_vec(plant.regime, act, 1000, np.random.default_rng(), exo=ex)
            return float(np.mean(P.cost(v_)))
        self.wm = WatermarkDetector(pred, 0.05)
        self.flushed = False

    def decide(self, a, step, nominal):
        if self.wm.alarm and not self.switch:
            return "abstain", 0.0
        return super().decide(a, step, nominal)

    def observe(self, applied, obs, true_state, sign0=None):
        self.wm.update(applied, sign0, obs)
        if self.switch and self.wm.alarm:
            if not self.flushed:
                self.g.buf, self.flushed = [], True
            self.g.observe(applied, true_state)
        else:
            self.g.observe(applied, obs)


def spoofer_for(plant_name, scen):
    if plant_name == "scm":
        kn = {"spoof_blind": "nominal", "a1_spoof": "nominal", "spoof_aware": "applied", "spoof_raw": "raw"}.get(scen)
        return None if kn is None else Spoofer(kappa=0.6, knowledge=kn)
    kn = {"spoof_blind": "nominal", "a1_spoof": "nominal", "spoof_aware": "applied", "spoof_raw": "raw"}.get(scen)
    if kn is None:
        return None
    if plant_name == "grid":
        P = plantspec.get("grid")
        return Spoofer(actuator="q18", targets=("loss",), kappa=0.6, knowledge=kn,
                       actuators=P.actuators, offset=0.0 if kn == "raw" else 0.3)
    P = plantspec.get("mec")
    return Spoofer(actuator="offload", targets=("latency", "drop"), kappa=0.6, knowledge=kn,
                   actuators=P.actuators, offset=0.0 if kn == "raw" else 0.3)


def job(args):
    plant_name, scen, seed = args
    P = plantspec.get(plant_name)
    W = _W[plant_name]
    hist = W["H_pois"] if scen in ("a1_telemetry", "a1_spoof") else W["H_clean"]

    class C:  # minimal config for P.make
        pass
    cfg = C(); cfg.seed = seed; cfg.regime_shift_prob = 0.0 if STATIONARY else 0.002; cfg.plant_memory = 0.0
    cfg.noise_mult = 1.0; cfg.drift_amp = 0.0; cfg.drift_period = 1600.0
    plant = P.make(cfg)
    if STATIONARY:
        from certify import envelopes
        sampler = envelopes.regimes_for(plant_name)
        plant.regime = sampler(np.random.default_rng(seed + 7))
    EBConfSeq.CAP = 0.75
    env = ENV[plant_name]
    gates = {"KIC": KICShadow(P, seed, True, env), "KIC-pc": KICShadow(P, seed, False, env),
             "Obs.-CI": ObsShadow(P, hist, "ci"), "Obs.-RLS": ObsShadow(P, hist, "rls"),
             "WM+hold": WMShadow(P, hist, plant, False), "WM+switch": WMShadow(P, hist, plant, True)}
    spoof = spoofer_for(plant_name, scen)
    exc = KeyedRademacher(f"key-{seed}".encode())
    traj = trajectory(seed, P.actuators)
    props = proposals(seed, P.actuators, len(traj))
    wc = lambda v: {**v, "cost": float(P.cost(v))}
    rec = {g: dict(auth=0, auth_harm=0, auth_benef=0, harm=0, benef=0, neutral=0, abstain=0, reject=0,
                   gain=0.0, loss=0.0, first=[], bins=[[0, 0, 0, 0] for _ in range(NBIN)]) for g in gates}
    dwell_first = {g: None for g in gates}
    for t in range(1, len(traj) + 1):
        nominal = traj[t - 1]
        if (t - 1) % DWELL == 0:
            for g in gates:
                if dwell_first[g] is not None or t > 1:
                    rec[g]["first"].append(dwell_first[g])
                dwell_first[g] = None
        a, step = props[t - 1]
        reg = plant.regime
        c_in = {k: float(np.clip(nominal[k], LO + 0.05, HI - 0.05)) for k in P.actuators}
        m0 = P.op_cost(reg, c_in, 0.05, n=2000)
        full = dict(c_in); full[a] = float(np.clip(c_in[a] + step, LO + 0.05, HI - 0.05))
        dm_full = P.op_cost(reg, full, 0.05, n=2000) - m0
        lab = 1 if dm_full > TOL else (-1 if dm_full < -TOL else 0)
        b_ = ((t - 1) % DWELL) * NBIN // DWELL
        for g, G_ in gates.items():
            r = rec[g]
            r["harm"] += lab == 1; r["benef"] += lab == -1; r["neutral"] += lab == 0
            r["bins"][b_][1] += lab == 1; r["bins"][b_][3] += lab == -1
            verdict, st = G_.decide(a, step, c_in)
            if verdict == "auth" and abs(st) > 1e-12:
                r["auth"] += 1
                if abs(st - (full[a] - c_in[a])) < 1e-9:
                    dm = dm_full
                else:
                    part = dict(c_in); part[a] = c_in[a] + st
                    dm = P.op_cost(reg, part, 0.05, n=2000) - m0
                r["auth_harm"] += dm > TOL
                r["auth_benef"] += dm < -TOL
                r["bins"][b_][0] += dm > TOL; r["bins"][b_][2] += dm < -TOL
                r["gain"] += max(-dm, 0.0); r["loss"] += max(dm, 0.0)
                if dm < -TOL and dwell_first[g] is None:
                    dwell_first[g] = (t - 1) % DWELL + 1
            elif verdict == "reject":
                r["reject"] += 1
            else:
                r["abstain"] += 1
        # actuate with the keyed dither and measure (identical for every gate)
        centre, signs, applied = dither(nominal, t, exc, 0.05, P.actuators)
        true_state = wc(plant.step(applied))
        obs = wc(spoof(true_state, t, centre, applied)) if spoof else true_state
        for g, G_ in gates.items():
            if isinstance(G_, KICShadow):
                G_.observe(centre, signs, obs)
            elif isinstance(G_, WMShadow):
                G_.observe(applied, obs, true_state, signs[P.actuators[0]])
            else:
                G_.observe(applied, obs, true_state)
    for g in gates:
        rec[g]["first"].append(dwell_first[g])
        rec[g]["monitor_alarms"] = getattr(gates[g], "alarms", 0)
    return dict(plant=plant_name, scen=scen, seed=seed, rec=rec)


def _init(worlds, cfg=None):
    # main() sets these globals; spawned workers (Windows) re-import the module and
    # would see the defaults, so pass them explicitly (a no-op under fork)
    global DWELL, N_DWELL, STATIONARY, ENV
    _W.update(worlds)
    if cfg:
        DWELL, N_DWELL, STATIONARY, ENV = cfg["DWELL"], cfg["N_DWELL"], cfg["STATIONARY"], cfg["ENV"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=60)
    ap.add_argument("--plants", nargs="*", default=["scm", "mec"])
    ap.add_argument("--out", default="kic_shadow.json")
    ap.add_argument("--dwell", type=int, default=200)
    ap.add_argument("--n_dwell", type=int, default=4)
    ap.add_argument("--stationary", action="store_true")
    ap.add_argument("--certified", action="store_true", help="certified envelopes (ENV_CERT)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--no_leak", action="store_true",
                    help="drop the leakage price (eta, F) from the envelopes: isolates its effect on liveness")
    args = ap.parse_args()
    global DWELL, N_DWELL, STATIONARY, ENV
    DWELL, N_DWELL, STATIONARY = args.dwell, args.n_dwell, args.stationary
    if args.certified:
        ENV = ENV_CERT
    if args.no_leak:
        ENV = {p: {k: v for k, v in e.items() if k not in ("leak", "leak_F")} for p, e in ENV.items()}
    worlds = {}
    if "scm" in args.plants:
        from kic_attacks import build_world
        w = build_world()
        worlds["scm"] = dict(H_clean=w["H_clean"], H_pois=w["H_pois"])
    if "mec" in args.plants:
        import mec_world
        w = mec_world.build_world()
        worlds["mec"] = dict(H_clean=w["H_clean"], H_pois=w["H_pois"])
    if "grid" in args.plants:
        import grid_world
        w = grid_world.build_world()
        worlds["grid"] = dict(H_clean=w["H_clean"], H_pois=w["H_pois"])
    SC = args.scenarios or ["none", "a1_telemetry", "spoof_blind", "a1_spoof", "spoof_aware", "spoof_raw"]
    jobs = [(p, s, sd) for p in args.plants for s in SC for sd in range(args.seeds)]
    cfg = dict(DWELL=DWELL, N_DWELL=N_DWELL, STATIONARY=STATIONARY, ENV=ENV)
    with Pool(args.procs, initializer=_init, initargs=(worlds, cfg)) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    json.dump(dict(rows=rows, config=vars(args)), open(RES / ("raw_" + args.out), "w"), default=float)
    agg = {}
    for p in args.plants:
        for s in SC:
            R = [r for r in rows if r["plant"] == p and r["scen"] == s]
            for g in R[0]["rec"]:
                tot = {k: sum(r["rec"][g][k] for r in R) for k in
                       ("auth", "auth_harm", "auth_benef", "harm", "benef", "neutral", "abstain", "reject",
                        "gain", "loss", "monitor_alarms")}
                bins = [[sum(r["rec"][g]["bins"][b][k] for r in R) for k in range(4)] for b in range(NBIN)]
                firsts = [f for r in R for f in r["rec"][g]["first"]]
                got = [f for f in firsts if f is not None]
                n_ticks = len(R) * DWELL * N_DWELL
                agg[f"{p}|{s}|{g}"] = dict(
                    har=tot["auth_harm"] / max(tot["harm"], 1), bar=tot["auth_benef"] / max(tot["benef"], 1),
                    abstain_rate=tot["abstain"] / n_ticks, reject_rate=tot["reject"] / n_ticks,
                    auth_rate=tot["auth"] / n_ticks,
                    harm_share=tot["auth_harm"] / max(tot["auth"], 1),
                    gain_per_tick=tot["gain"] / n_ticks, loss_per_tick=tot["loss"] / n_ticks,
                    net_per_tick=(tot["gain"] - tot["loss"]) / n_ticks,
                    first_median=float(np.median(got)) if got else None, first_frac=len(got) / max(len(firsts), 1),
                    har_bins=[b[0] / max(b[1], 1) for b in bins], bar_bins=[b[2] / max(b[3], 1) for b in bins],
                    **tot)
                v = agg[f"{p}|{s}|{g}"]
                print(f"{p} {s:13s} {g:9s} HAR={v['har']:.3f} BAR={v['bar']:.3f} harmshare={v['harm_share']:.3f} "
                      f"abst={v['abstain_rate']:.2f} net={1e3 * v['net_per_tick']:+.2f}e-3 loss={1e3 * v['loss_per_tick']:.2f}e-3 "
                      f"first={v['first_median']} ({v['first_frac']:.2f}) alarms={v['monitor_alarms']}", flush=True)
    json.dump(dict(agg=agg, rows=rows, config=vars(args)), open(RES / args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
