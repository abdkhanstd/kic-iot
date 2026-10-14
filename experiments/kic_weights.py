# Author: abdkhan
"""Sensitivity to the operator's cost weights.

The main comparison is repeated under three alternative weightings of the
operator cost J = w_l*latency + w_e*energy + w_p*loss - w_th*throughput
+ w_tx*tx^2. The whole world (surrogates, observational histories, graph
signs) is rebuilt under each weighting, and KIC's declared envelopes G and
L are recomputed for it from the simulator (the same procedure that gave
G = 1.6 and L = 3 for the base weights), rounded up; the clip bound follows
B = 0.19 + 2 delta G, which gives B = 0.35 for the base weights.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import certify.oracle as O

RES = ROOT / "experiments" / "results"
WEIGHTS = {
    "base": dict(latency=1.0, energy=0.6, packet_loss=1.0, throughput=2.5, tx2=0.7),
    "energy-heavy": dict(latency=1.0, energy=1.5, packet_loss=1.0, throughput=2.5, tx2=0.7),
    "QoS-heavy": dict(latency=2.0, energy=0.6, packet_loss=2.0, throughput=2.5, tx2=0.7),
    "throughput-light": dict(latency=1.0, energy=0.6, packet_loss=1.0, throughput=1.5, tx2=0.7),
}
GATES = ["none", "gstar_claims", "obsci", "obsrls", "wm", "kic_sound", "kic_hold"]
SCEN = ["a1_telemetry", "backdoor_lie", "spoof_blind"]
_S = {}


def envelopes(w, n_states=150):
    """G and L for the weighting w, by the same paired-input procedure as the
    main experiments (certify/envelopes.py): L is the norm of the Hessian row."""
    O.COST_W.clear(); O.COST_W.update(w)
    from certify import plantspec, envelopes as E
    return E.derive(plantspec.get("scm"), E.scm_regimes, n_states=n_states)


def _prepare(wname):
    """Set the weights and build the world in the parent; forked workers inherit both."""
    O.COST_W.clear(); O.COST_W.update(WEIGHTS[wname])
    import kic_attacks as K
    _S["K"] = K
    _S["W"] = K.build_world()


def job(a):
    wname, scen, gate, seed, G, L = a
    from certify.loop import RunConfig, run
    K, W = _S["K"], _S["W"]
    prop, gstar, spoof, hist, exc = K.scenario(scen, seed, W)
    # clip bound from the same rule as the main experiments: near-clip
    # threshold B - 2 delta G = 0.19 (B = 0.35 for the base weights)
    cfg = RunConfig(gate=gate, n_steps=800, seed=seed, excitation=exc, G=G, L_hat=L,
                    B_cost=0.19 + 2 * 0.05 * G)
    r = run(cfg, prop, gstar=gstar, spoofer=spoof, surrogate=W["sur_clean"], hist=hist)
    return dict(w=wname, scen=scen, gate=gate, seed=seed, ecost=r["ecost"],
                exec_harmful_moves=r["exec_harmful_moves"], exec_moves=r["exec_moves"])


def main():
    out, rows = {}, []
    for wname, w in WEIGHTS.items():
        G, L = envelopes(w)
        Gd, Ld = float(np.ceil(G * 10.5) / 10), float(np.ceil(L * 1.05 * 10) / 10)
        if wname == "base":             # the main experiments' declared envelopes
            Gd, Ld = 1.6, 3.5
        print(wname, "measured G, L:", round(G, 3), round(L, 3), "declared:", Gd, Ld, flush=True)
        jobs = [(wname, s, g, sd, Gd, Ld) for s in SCEN for g in GATES for sd in range(20)]
        _prepare(wname)
        with Pool(60) as p:
            R = p.map(job, jobs, chunksize=1)
        rows += R
        out[f"{wname}|envelopes"] = dict(G=G, L=L, G_declared=Gd, L_declared=Ld)
        for s in SCEN:
            k = {r["seed"]: r["ecost"] for r in R if r["scen"] == s and r["gate"] == "kic_sound"}
            for g in GATES:
                b = {r["seed"]: r["ecost"] for r in R if r["scen"] == s and r["gate"] == g}
                diff = np.array([b[i] - k[i] for i in sorted(k)])
                bs = np.random.default_rng(0).choice(diff, (4000, len(diff))).mean(1)
                out[f"{wname}|{s}|{g}"] = dict(
                    ecost=float(np.mean(list(b.values()))), paired=float(diff.mean()),
                    ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    harmful=sum(r["exec_harmful_moves"] for r in R if r["scen"] == s and r["gate"] == g),
                    moves=sum(r["exec_moves"] for r in R if r["scen"] == s and r["gate"] == g))
                print(wname, s, g, out[f"{wname}|{s}|{g}"], flush=True)
    json.dump(dict(agg=out, rows=rows), open(RES / "kic_weights.json", "w"), indent=1)


if __name__ == "__main__":
    main()
