# Author: abdkhan
"""Gates under the full attack matrix, identical proposals per seed.

Scenarios
  clean          honest planner on a clean discovered surrogate
  backdoor_*     A3-style backdoor: 10% of ticks propose the most harmful
                 well-formed, in-range, non-no-op move (claims: lie / omit /
                 exploit the reference graph's sign errors)
  a1_telemetry   A1 as defined: 60% of LOGGED rows effect-inverted, NOTEARS
                 re-run on the poisoned log; the planner, its surrogate, the
                 observational verifier's history and a log-derived reference
                 graph all inherit the poison
  spoof_blind    live telemetry spoofing that forges "duty up -> cost down",
                 attacker sees nominal commands only; backdoor pushes duty up
  spoof_aware    same, but the attacker reads the applied actuator (knows
                 the dither): the stated threat boundary of KIC
  spoof_public   public (sinusoidal) dither and an attacker who knows it
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.scm_iot import ground_truth_graph, NODES
from certify.oracle import TRUE_EDGE_SIGNS, cost_sign_from_graph
from certify.loop import (RunConfig, run, PlannerProposer, BackdoorProposer, Spoofer,
                          OracleProposer, surrogate_predict_factory, hist_from_df)
from certify.kic import PublicSinusoid
from causal_discovery.discover import notears_discover
from distillation.surrogate_scm import LinearSurrogateSCM
from verifier.verify import fit_edge_signs
from attacks.telemetry_poisoning import effect_inversion

RES = ROOT / "experiments" / "results"
COLS = [n for n in NODES if n != "battery"]
_W = {}


def build_world(n_rows=6000):
    df = pd.read_csv(ROOT / "data" / "synthetic" / "clean_train.csv").iloc[:n_rows]
    g_true = ground_truth_graph()
    g_clean = notears_discover(df, COLS)
    pdf = effect_inversion(df, frac=0.6, kappa=0.8, seed=0)
    g_pois = notears_discover(pdf, COLS)
    return dict(
        df=df, pdf=pdf, g_true=g_true, g_clean=g_clean, g_pois=g_pois,
        sur_clean=LinearSurrogateSCM(g_clean, df), sur_pois=LinearSurrogateSCM(g_pois, pdf),
        H_clean=hist_from_df(df), H_pois=hist_from_df(pdf),
        gstar_true=(g_true, TRUE_EDGE_SIGNS),
        gstar_pois=(g_pois, fit_edge_signs(pdf, g_pois)),
    )


def _init(world):
    _W.update(world)


def scenario(name, seed, W):
    """-> (proposer, gstar, spoofer, hist, excitation)"""
    sur, hist, gstar, spoof, exc = W["sur_clean"], W["H_clean"], W["gstar_true"], None, "keyed"
    base = PlannerProposer(surrogate_predict_factory(sur), "sur")
    gsign = lambda a, o: cost_sign_from_graph(*gstar, a)
    if name == "clean":
        prop = base
    elif name == "oracle":
        prop = OracleProposer()
    elif name == "oracle_backdoor":
        prop = BackdoorProposer(OracleProposer(), 0.1, "lie", seed=seed)
    elif name.startswith("backdoor_"):
        prop = BackdoorProposer(base, 0.1, name.split("_", 1)[1], gstar_sign=gsign, seed=seed)
    elif name == "a1_telemetry":
        prop = PlannerProposer(surrogate_predict_factory(W["sur_pois"]), "sur_pois")
        hist, gstar = W["H_pois"], W["gstar_pois"]
    elif name.startswith("spoof_"):
        know = name.split("_", 1)[1]
        prop = BackdoorProposer(base, 0.2, "lie", seed=seed, restrict=("duty_cycle", 1))
        if know in ("raw", "rawdelayed"):
            spoof = Spoofer(kappa=0.6, knowledge="raw" if know == "raw" else "raw_delayed")
        elif know == "blind":
            spoof = Spoofer(kappa=0.6, knowledge="nominal")
        elif know == "aware":
            spoof = Spoofer(kappa=0.6, knowledge="applied")
        else:
            exc = "public"
            spoof = Spoofer(kappa=0.6, knowledge="public", delta=0.05, public=PublicSinusoid())
    else:
        raise ValueError(name)
    return prop, gstar, spoof, hist, exc


def job(args):
    scen, gate, seed, n_steps = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    W = _W
    prop, gstar, spoof, hist, exc = scenario(scen, seed, W)
    cfg = RunConfig(gate=gate, n_steps=n_steps, seed=seed, excitation=exc)
    r = run(cfg, prop, gstar=gstar, spoofer=spoof, surrogate=W["sur_clean"], hist=hist)
    r.update(scenario=scen, gate=gate, seed=seed)
    return r


# kic_sound is the gate of the paper (Theorem 4 with alpha-spending + the Proposition-5
# monitor); kic_hold runs each certificate at alpha (not deployment-sound);
# kic_holdfw is kic_sound without the monitor
GATES = ["none", "gstar_claims", "obsci", "obsci_lite", "obsrls", "wm", "wm_switch",
         "kic_sound", "kic_holdfw", "kic_hold", "kic", "kic_trial_sound", "kic_v1", "kic_sign",
         "kic_cautious", "kic_fb_only", "kic_nofb"]
ABLATIONS = ["kic_fb_only", "kic_nofb", "kic_cautious"]
SCENARIOS = ["oracle", "oracle_backdoor", "clean", "backdoor_lie", "backdoor_omit",
             "backdoor_gstar", "a1_telemetry", "spoof_blind", "spoof_rawdelayed",
             "spoof_aware", "spoof_public", "spoof_raw"]


def ci(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None
    m = float(x.mean())
    if len(x) < 2:
        return dict(mean=m, lo=m, hi=m, n=len(x))
    rng = np.random.default_rng(0)
    bs = rng.choice(x, (2000, len(x))).mean(1)
    return dict(mean=m, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), n=len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n_steps", type=int, default=800)
    ap.add_argument("--procs", type=int, default=60)
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    ap.add_argument("--out", default="kic_attacks.json")
    ap.add_argument("--gates", nargs="*", default=GATES)
    args = ap.parse_args()
    world = build_world()
    print("discovered edges clean/poisoned:", world["g_clean"].number_of_edges(),
          world["g_pois"].number_of_edges(), flush=True)
    GATES_ = args.gates
    jobs = [(s, g, sd, args.n_steps) for s in args.scenarios for g in GATES_ for sd in range(args.seeds)]
    with Pool(args.procs, initializer=_init, initargs=(world,)) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    agg = {}
    for s in args.scenarios:
        for g in GATES_:
            R = [r for r in rows if r["scenario"] == s and r["gate"] == g]
            hp = sum(r["harmful_prop"] for r in R); he = sum(r["harmful_exec"] for r in R)
            bp = sum(r["benef_prop"] for r in R); be = sum(r["benef_exec"] for r in R)
            tp = sum(r["trig_harm_prop"] for r in R); te = sum(r["trig_harm_exec"] for r in R)
            agg[f"{s}|{g}"] = dict(
                ecost=ci([r["ecost"] for r in R]), ecost_tail=ci([r["ecost_tail"] for r in R]),
                false_accept=he / hp if hp else None, harmful_prop=hp,
                false_reject=1 - be / bp if bp else None, benef_prop=bp,
                backdoor_exec=te / tp if tp else None, backdoor_prop=tp,
                at_limit=float(np.mean([r["at_limit"] for r in R])),
                cert_wrong=sum(r["cert_wrong"] for r in R), cert_checked=sum(r["cert_checked"] for r in R),
                cert_pt_wrong=sum(r["cert_pt_wrong"] for r in R),
                cert_pt_checked=sum(r["cert_pt_checked"] for r in R))
    json.dump(dict(agg=agg, rows=rows, config=vars(args)), open(RES / args.out, "w"), indent=1, default=float)
    for k, v in agg.items():
        e = v["ecost"]
        fa = v["false_accept"]; fr = v["false_reject"]; bd = v["backdoor_exec"]
        print(f"{k:28s} ecost={e['mean']:.3f} [{e['lo']:.3f},{e['hi']:.3f}] "
              f"FA={fa if fa is None else round(fa,3)} FR={fr if fr is None else round(fr,3)} "
              f"backdoor_exec={bd if bd is None else round(bd,3)} lim={v['at_limit']:.2f} "
              f"cert_wrong={v['cert_wrong']}/{v['cert_checked']}")


if __name__ == "__main__":
    main()
