# Author: abdkhan
"""Experiment: does the discovery ranking survive outside our one graph?

Every structural result elsewhere in this paper uses the single hand-built
14-node IoT SCM. That invites the objection that the ranking is an artifact of
that topology, mechanism and noise law. Here the same five algorithms are run
across a grid of randomly sampled SCM families spanning size, density,
mechanism (linear / sigmoid / tanh) and noise (gaussian / uniform /
exponential), with several graph draws per cell.

The grid is chosen so that each method gets a regime built for it: linear +
gaussian suits PC and GES; non-gaussian noise suits LiNGAM; the nonlinear
mechanisms suit none of them exactly. A ranking that holds only in one corner
should be reported as such.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.random_scm import sample_family  # noqa: E402
from causal_discovery.discover import notears_linear, evaluate_discovery, _break_cycles  # noqa: E402

OUT = ROOT / "experiments" / "results" / "scm_families.json"


def _postprocess(g: nx.DiGraph) -> nx.DiGraph:
    return g if nx.is_directed_acyclic_graph(g) else _break_cycles(g)


def run_notears(df, cols):
    X = df[cols].to_numpy(float)
    W = notears_linear(X, lambda1=0.08, w_threshold=0.1)
    g = nx.DiGraph(); g.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i != j and W[i, j] != 0:
                g.add_edge(a, b)
    return _postprocess(g)


def run_pc(df, cols):
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(df[cols].to_numpy(float), alpha=0.05, indep_test="fisherz",
            show_progress=False, node_names=cols)
    from causallearn.graph.Endpoint import Endpoint
    g = nx.DiGraph(); g.add_nodes_from(cols)
    for e in cg.G.get_graph_edges():
        a, b = e.get_node1().get_name(), e.get_node2().get_name()
        if e.get_endpoint1() == Endpoint.TAIL and e.get_endpoint2() == Endpoint.ARROW:
            g.add_edge(a, b)
        elif e.get_endpoint1() == Endpoint.ARROW and e.get_endpoint2() == Endpoint.TAIL:
            g.add_edge(b, a)
        else:
            g.add_edge(a, b)   # unoriented: take one direction, as elsewhere
    return _postprocess(g)


def run_ges(df, cols):
    from causallearn.search.ScoreBased.GES import ges
    G = ges(df[cols].to_numpy(float))["G"].graph
    g = nx.DiGraph(); g.add_nodes_from(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and G[j, i] == 1 and G[i, j] == -1:
                g.add_edge(cols[i], cols[j])
    return _postprocess(g)


def run_lingam(df, cols, variant="direct"):
    from causallearn.search.FCMBased import lingam
    m = lingam.DirectLiNGAM() if variant == "direct" else lingam.ICALiNGAM()
    m.fit(df[cols].to_numpy(float))
    B = np.asarray(m.adjacency_matrix_)
    g = nx.DiGraph(); g.add_nodes_from(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(B[i, j]) > 0.05:
                g.add_edge(cols[j], cols[i])
    return _postprocess(g)


METHODS = {
    "PC": run_pc,
    "GES": run_ges,
    "DirectLiNGAM": lambda d, c: run_lingam(d, c, "direct"),
    "ICA-LiNGAM": lambda d, c: run_lingam(d, c, "ica"),
    "NOTEARS (ours)": run_notears,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_graphs", type=int, default=3, help="graph draws per cell")
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--n_nodes", type=int, nargs="+", default=[10, 16])
    ap.add_argument("--density", type=float, nargs="+", default=[1.5])
    ap.add_argument("--mechanisms", nargs="+", default=["linear", "sigmoid", "tanh"])
    ap.add_argument("--noises", nargs="+", default=["gaussian", "uniform", "exponential"])
    args = ap.parse_args()

    rows = []
    cells = [(n, d, m, z) for n in args.n_nodes for d in args.density
             for m in args.mechanisms for z in args.noises]
    print(f"{len(cells)} cells x {args.n_graphs} graphs x {len(METHODS)} methods", flush=True)

    for ci, (n_nodes, dens, mech, noise) in enumerate(cells):
        for gi in range(args.n_graphs):
            seed = 10_000 + ci * 100 + gi
            g_true, df = sample_family(n_nodes, dens, mech, noise, args.n_samples, seed)
            cols = list(g_true.nodes())
            for name, fn in METHODS.items():
                t0 = time.time()
                try:
                    g_est = fn(df, cols)
                    m = evaluate_discovery(g_true, g_est)
                    m.update(runtime_sec=round(time.time() - t0, 2), failed=False)
                except Exception as e:
                    m = dict(shd=np.nan, sid=np.nan, precision=np.nan, recall=np.nan,
                             f1=np.nan, failed=True, error=str(e)[:120])
                rows.append(dict(method=name, n_nodes=n_nodes, density=dens,
                                  mechanism=mech, noise=noise, graph=gi, **m))
        print(f"  cell {ci+1}/{len(cells)}  n={n_nodes} {mech}/{noise}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "experiments" / "results" / "scm_families_raw.csv", index=False)

    summary = {}
    for name in METHODS:
        sub = df[(df.method == name) & (~df.failed)]
        summary[name] = dict(
            n_runs=int(len(sub)),
            shd_mean=float(sub.shd.mean()), shd_std=float(sub.shd.std()),
            f1_mean=float(sub.f1.mean()), f1_std=float(sub.f1.std()),
            sid_mean=float(sub.sid.mean()),
            precision_mean=float(sub.precision.mean()), recall_mean=float(sub.recall.mean()),
            by_mechanism={mech: float(sub[sub.mechanism == mech].f1.mean())
                          for mech in args.mechanisms},
            by_noise={z: float(sub[sub.noise == z].f1.mean()) for z in args.noises},
        )
    with open(OUT, "w") as f:
        json.dump(dict(config=vars(args), n_cells=len(cells), methods=summary), f, indent=2)

    print(f"\n{'method':16s}{'SHD':>14}{'F1':>14}{'lin':>7}{'sig':>7}{'tanh':>7}")
    for name, s in summary.items():
        bm = s["by_mechanism"]
        print(f"{name:16s}{s['shd_mean']:7.1f}+-{s['shd_std']:<5.1f}"
              f"{s['f1_mean']:7.3f}+-{s['f1_std']:<5.3f}"
              f"{bm.get('linear', float('nan')):7.3f}{bm.get('sigmoid', float('nan')):7.3f}"
              f"{bm.get('tanh', float('nan')):7.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
