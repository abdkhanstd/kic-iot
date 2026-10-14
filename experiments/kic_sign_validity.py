# Author: abdkhan
"""How often is the reference-graph sign rule wrong about the plant?

Even a PERFECT reference graph with the TRUE edge signs can misjudge the
total effect of an actuator, because the majority-path rule ignores path
magnitudes and state dependence. We measure disagreement between
  (a) majority-path sign on the true graph with true edge signs,
  (b) majority-path sign with correlation-fitted signs (verify.py as used),
and the true local total effect  dE[y|do(a)]/da  over a grid of operating
points and exogenous regimes, on the IoT SCM and on random SCM families
with random-sign coefficients.
"""
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.scm_iot import ground_truth_graph, RegimeParams
from certify.oracle import (TRUE_EDGE_SIGNS, majority_path_sign, true_local_effect,
                            reward_sign_from_graph, cost_sign_from_graph)
from verifier.verify import fit_edge_signs

OUT = ROOT / "experiments" / "results" / "kic_sign_validity.json"
TARGETS = ["latency", "energy", "packet_loss", "throughput", "reward", "cost"]
ACT = ["duty_cycle", "tx_power"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def iot_part():
    g = ground_truth_graph()
    df = pd.read_csv(ROOT / "data" / "synthetic" / "clean_train.csv")
    corr_signs = fit_edge_signs(df, g)
    rng = np.random.default_rng(0)
    regimes = [RegimeParams()]
    for _ in range(19):
        regimes.append(RegimeParams(
            interference_mean=float(rng.uniform(0.05, 0.9)),
            link_quality_mean=float(rng.uniform(0.2, 0.95)),
            arrival_rate_mean=float(rng.uniform(0.1, 0.9))))
    grid = np.linspace(0.1, 0.9, 9)
    rows = []
    for ri, reg in enumerate(regimes):
        for d in grid:
            for p in grid:
                c = dict(duty_cycle=float(d), tx_power=float(p))
                for a in ACT:
                    for k in TARGETS:
                        eff = true_local_effect(reg, c, a, k, delta=0.05, n=1500)
                        if k == "cost":
                            s_true = cost_sign_from_graph(g, TRUE_EDGE_SIGNS, a)
                            s_corr = cost_sign_from_graph(g, corr_signs, a)
                        elif k == "reward":
                            s_true = reward_sign_from_graph(g, TRUE_EDGE_SIGNS, a)
                            s_corr = reward_sign_from_graph(g, corr_signs, a)
                        else:
                            s_true = majority_path_sign(g, TRUE_EDGE_SIGNS, a, k)
                            s_corr = majority_path_sign(g, corr_signs, a, k)
                        rows.append(dict(regime=ri, duty=d, tx=p, actuator=a, target=k,
                                         effect=eff, s_true_graph=s_true, s_corr_graph=s_corr))
    R = pd.DataFrame(rows)
    R = R[R.effect.abs() > 1e-3]          # ignore numerically-flat points
    out = {}
    for (a, k), sub in R.groupby(["actuator", "target"]):
        ts = np.sign(sub.effect)
        n = len(sub)
        w1 = int((sub.s_true_graph.fillna(0) != ts).sum())
        w2 = int((sub.s_corr_graph.fillna(0) != ts).sum())
        out[f"{a}->{k}"] = dict(
            n=n, frac_positive=float((ts > 0).mean()),
            graph_true_signs_rule=sub.s_true_graph.iloc[0],
            wrong_true_signs=w1 / n, wrong_true_signs_ci=wilson(w1, n),
            wrong_corr_signs=w2 / n, wrong_corr_signs_ci=wilson(w2, n))
    Q = R[R.target.isin(["latency", "energy", "packet_loss", "throughput"])]
    tot = len(Q)
    w1 = int((Q.s_true_graph.fillna(0) != np.sign(Q.effect)).sum())
    w2 = int((Q.s_corr_graph.fillna(0) != np.sign(Q.effect)).sum())
    out["_overall"] = dict(n=tot, wrong_true_signs=w1 / tot, ci=wilson(w1, tot),
                           wrong_corr_signs=w2 / tot, ci_corr=wilson(w2, tot),
                           sign_reversing_pairs=int(sum(
                               1 for (a, k), s in Q.groupby(["actuator", "target"])
                               if 0 < (s.effect > 0).mean() < 1)))
    C = R[R.target == "cost"]
    wc = int((C.s_true_graph.fillna(0) != np.sign(C.effect)).sum())
    out["_cost"] = dict(n=len(C), wrong_true_signs=wc / len(C), ci=wilson(wc, len(C)))
    return out


def random_part(n_graphs=300, n_nodes=12, epn=2.0, seed=0):
    """Random DAGs with random-sign coefficients and tanh/sigmoid links; the
    'actuator' is a random root with >= 2 directed paths to a random sink."""
    from simulator.random_scm import sample_dag
    rng = np.random.default_rng(seed)
    res = dict(n=0, wrong=0, by_mech={})
    for gi in range(n_graphs):
        g = sample_dag(n_nodes, epn, rng)
        W = {e: rng.uniform(0.5, 2.0) * rng.choice([-1, 1]) for e in g.edges()}
        mech = ["linear", "tanh", "sigmoid"][gi % 3]
        order = list(nx.topological_sort(g))
        roots = [v for v in order if g.in_degree(v) == 0]
        pairs = [(r, s) for r in roots for s in order
                 if r != s and len(list(nx.all_simple_paths(g, r, s, cutoff=6))) >= 2]
        if not pairs:
            continue
        r, s = pairs[rng.integers(len(pairs))]
        signs = {e: int(np.sign(w)) for e, w in W.items()}
        rule = majority_path_sign(g, signs, r, s)

        def sim(xr, noise, base):
            val = {}
            for v in order:
                if v == r:
                    val[v] = xr
                elif g.in_degree(v) == 0:
                    val[v] = base[v]
                else:
                    z = sum(W[(u, v)] * val[u] for u in g.predecessors(v)) + noise[v]
                    val[v] = z if mech == "linear" else (np.tanh(z) if mech == "tanh" else 1 / (1 + np.exp(-z)))
            return val[s]
        for _ in range(5):                      # 5 operating points per graph
            x0 = rng.normal(0, 1.0)
            m = 400
            noise = {v: rng.normal(0, 0.3, m) for v in order}
            base = {v: rng.normal(0, 1, m) for v in order}
            eff = np.mean(sim(x0 + 0.05, noise, base) - sim(x0 - 0.05, noise, base)) / 0.1
            if abs(eff) < 1e-3:
                continue
            res["n"] += 1
            bad = int((rule or 0) != np.sign(eff))
            res["wrong"] += bad
            bm = res["by_mech"].setdefault(mech, [0, 0])
            bm[0] += bad; bm[1] += 1
    res["frac_wrong"] = res["wrong"] / res["n"]
    res["ci"] = wilson(res["wrong"], res["n"])
    res["by_mech"] = {k: dict(wrong=v[0] / v[1], n=v[1], ci=wilson(*v)) for k, v in res["by_mech"].items()}
    return res


def main():
    out = dict(iot=iot_part(), random_scm=random_part())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    for k, v in out["iot"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("random:", out["random_scm"]["frac_wrong"], out["random_scm"]["ci"], out["random_scm"]["by_mech"])


if __name__ == "__main__":
    main()
