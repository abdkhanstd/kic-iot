# Author: abdkhan
"""Declared envelopes and plant variants.

(a) Plant variants (as E8) plus CONTINUOUS drift: the interference mean
    follows a sinusoid of amplitude 0.15 and period 1600 ticks on top of the
    regime level, so local effects drift between regime breaks. The drift's
    true slope-variation rate is V = 1.5e-3 per tick (max |d beta / d I| = 2.54
    times the maximal rate of the mean); KIC declares it in that variant.
(b) Envelope misspecification: KIC with the declared L, G or V scaled below
    and above the plant's true value, one at a time, reporting cost and the
    share of LICENSED moves that raised the current expected cost.
(c) Harm audit of every executed move by source (licensed proposal,
    licensed fallback, earned trust), across the in-model attack scenarios.
"""
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certify.loop import RunConfig, run
from kic_attacks import build_world, scenario, _init, _W

RES = ROOT / "experiments" / "results"
V_DRIFT = 1.5e-3
VARIANTS = {"base": {}, "noise x0.5": dict(noise_mult=0.5), "noise x2": dict(noise_mult=2.0),
            "breaks x5": dict(regime_shift_prob=0.01), "cont. drift": dict(drift_amp=0.15),
            "memory 0.5": dict(plant_memory=0.5), "memory 0.8": dict(plant_memory=0.8)}
GATES = ["none", "gstar_claims", "obsci", "obsrls", "wm", "wm_switch", "kic_sound", "kic_hold", "kic"]
SCEN = ["a1_telemetry", "backdoor_lie"]
SWEEP = {"L": [0.875, 1.75, 3.5, 7.0], "G": [0.4, 0.8, 1.6, 3.2], "V": [0.0, 2e-4, 2e-3]}
AUDIT_GATES = ["kic_sound", "kic_hold", "kic", "kic_sign"]
AUDIT_SCEN = ["oracle", "oracle_backdoor", "clean", "backdoor_lie", "backdoor_omit", "backdoor_gstar",
              "a1_telemetry", "spoof_blind", "spoof_rawdelayed"]


def job(a):
    part, var, scen, gate, seed, extra = a
    prop, gstar, spoof, hist, exc = scenario(scen, seed, _W)
    kw = dict(VARIANTS.get(var, {}))
    if var == "cont. drift" and gate.startswith("kic") and gate != "kic_v1":
        kw["V"] = V_DRIFT
    kw.update({("L_hat" if k == "L" else k): v for k, v in extra.items()})
    cfg = RunConfig(gate=gate, n_steps=800, seed=seed, excitation=exc, **kw)
    r = run(cfg, prop, gstar=gstar, spoofer=spoof, surrogate=_W["sur_clean"], hist=hist)
    return dict(part=part, var=var, scen=scen, gate=gate, seed=seed, extra=extra, cost=r["cost"],
                ecost=r["ecost"], harm_by_src=r["harm_by_src"], motion=r["motion"],
                tick_src=r["tick_src"], first_cert=r["first_cert"], licensed_ticks=r["licensed_ticks"],
                cert_wrong=r["cert_wrong"], cert_checked=r["cert_checked"],
                cert_pt_wrong=r["cert_pt_wrong"], cert_pt_checked=r["cert_pt_checked"],
                resets=r["resets"], monitor_alarms=r["monitor_alarms"], max_harm=r["max_harm"],
                first_licensed=r["first_licensed"])


def paired(rows, ref):
    k = {r["seed"]: r["cost"] for r in ref}
    d = np.array([r["cost"] - k[r["seed"]] for r in rows if r["seed"] in k])
    bs = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
    return float(d.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    world = build_world()
    jobs = [("var", v, s, g, sd, {}) for v in VARIANTS for s in SCEN for g in GATES for sd in range(20)]
    jobs += [("sweep", f"{p}={x}", s, g, sd, {p: x}) for p, xs in SWEEP.items() for x in xs
             for s in SCEN for sd in range(20) for g in ("kic_sound", "kic_holdfw")]
    jobs += [("audit", "base", s, g, sd, {}) for s in AUDIT_SCEN for g in AUDIT_GATES
             for sd in range(30)]
    with Pool(60, initializer=_init, initargs=(world,)) as p:
        rows = p.map(job, jobs, chunksize=1)
    out = {}
    for v in VARIANTS:
        for s in SCEN:
            ref = [r for r in rows if r["part"] == "var" and r["var"] == v and r["scen"] == s and r["gate"] == "kic_sound"]
            for g in GATES:
                R = [r for r in rows if r["part"] == "var" and r["var"] == v and r["scen"] == s and r["gate"] == g]
                m, ci = paired(R, ref)
                out[f"var|{v}|{s}|{g}"] = dict(cost=float(np.mean([r["cost"] for r in R])),
                                              ecost=float(np.mean([r["ecost"] for r in R])),
                                              paired_vs_kic=m, paired_ci=ci,
                                              monitor_alarms=sum(r["monitor_alarms"] for r in R),
                                              cert_pt_wrong=sum(r["cert_pt_wrong"] for r in R),
                                              cert_pt_checked=sum(r["cert_pt_checked"] for r in R))
    for pn, xs in SWEEP.items():
        for x in xs:
            for s in SCEN:
                for g in ("kic_sound", "kic_holdfw"):
                    R = [r for r in rows if r["part"] == "sweep" and r["var"] == f"{pn}={x}" and r["scen"] == s
                         and r["gate"] == g]
                    hl = sum(r["harm_by_src"].get(k, [0, 0])[0] for r in R
                             for k in ("licensed_proposal", "licensed_fallback"))
                    nl = sum(r["harm_by_src"].get(k, [0, 0])[1] for r in R
                             for k in ("licensed_proposal", "licensed_fallback"))
                    out[f"sweep|{pn}|{x}|{s}|{g}"] = dict(ecost=float(np.mean([r["ecost"] for r in R])),
                                                         licensed_harm=hl, licensed_n=nl,
                                                         max_harm=max(r["max_harm"] for r in R),
                                                         monitor_alarms=sum(r["monitor_alarms"] for r in R))
    for s in AUDIT_SCEN:
        for g in AUDIT_GATES:
            R = [r for r in rows if r["part"] == "audit" and r["scen"] == s and r["gate"] == g]
            hb, mo, ts = defaultdict(lambda: [0, 0]), defaultdict(float), defaultdict(int)
            for r in R:
                for k, (h, n) in r["harm_by_src"].items():
                    hb[k][0] += h; hb[k][1] += n
                for k, x in r["motion"].items():
                    mo[k] += x / len(R)
                for k, x in r["tick_src"].items():
                    ts[k] += x
            fc = [r["first_cert"] for r in R if r["first_cert"] is not None]
            out[f"audit|{s}|{g}"] = dict(harm_by_src=dict(hb), motion=dict(mo),
                                        tick_share={k: x / sum(ts.values()) for k, x in ts.items()},
                                        first_cert_median=float(np.median(fc)) if fc else None,
                                        first_cert_frac=len(fc) / len(R),
                                        licensed_ticks=float(np.mean([r["licensed_ticks"] for r in R])),
                                        ecost=float(np.mean([r["ecost"] for r in R])))
    for k, v in out.items():
        print(k, v, flush=True)
    json.dump(dict(agg=out, rows=rows), open(RES / "kic_envelopes.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
