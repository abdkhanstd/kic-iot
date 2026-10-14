# Author: abdkhan
"""Head-to-head causal-discovery comparison on OUR ground-truth benchmark.

Why re-run the baselines instead of quoting published numbers: recent IoT
causal-analysis work reports classification accuracy on attack datasets
(e.g. LiNGAM-based pipelines reporting ~100% F1 for attack classification on
CICIoT2023). That is a different task from structural recovery and a
different metric; placing those numbers beside our SHD/SID/F1 would be a
category error. The scientifically meaningful comparison is to run the same
*algorithms* on the same ground-truth graph, which is what this does.

Baselines:
  PC            constraint-based (Fisher-z)
  GES           score-based (BIC)
  DirectLiNGAM  functional causal model, non-Gaussian noise -- the family
                used by recent IoT causal-analysis pipelines
  ICA-LiNGAM    ICA-based variant of the same family
  NOTEARS       continuous optimisation (ours, as configured in the paper)
All are scored identically against the known 25-edge graph, with the same
domain-knowledge root constraint applied to every method so no baseline is
disadvantaged by a prior only our method receives.
"""
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import ground_truth_graph, NODES
from causal_discovery.discover import (notears_discover, pc_discover,
                                        evaluate_discovery, ROOT_VARS, _break_cycles)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
COLS = [c for c in NODES if c != "t"]


def _postprocess(g: nx.DiGraph) -> nx.DiGraph:
    """Same domain-knowledge cleanup every method gets: exogenous/actuator
    variables cannot have causal parents, and the result must be acyclic."""
    for u, v in list(g.edges()):
        if v in ROOT_VARS:
            g.remove_edge(u, v)
    if not nx.is_directed_acyclic_graph(g):
        g = _break_cycles(g)
    return g


def _from_adj(B: np.ndarray, cols, thresh=0.05) -> nx.DiGraph:
    """LiNGAM returns B with B[i,j] != 0 meaning j -> i."""
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(B[i, j]) > thresh:
                g.add_edge(cols[j], cols[i])
    return _postprocess(g)


def run_lingam(df, cols, variant="direct"):
    from causallearn.search.FCMBased import lingam
    X = df[cols].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    model = lingam.DirectLiNGAM() if variant == "direct" else lingam.ICALiNGAM()
    model.fit(X)
    return _from_adj(np.asarray(model.adjacency_matrix_), cols)


def run_ges(df, cols):
    from causallearn.search.ScoreBased.GES import ges
    X = df[cols].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    rec = ges(X)
    G = rec["G"].graph  # -1/1 endpoint encoding
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            # i -> j when endpoint at i is tail (-1) and at j is arrow (1)
            if G[j, i] == 1 and G[i, j] == -1:
                g.add_edge(cols[i], cols[j])
            elif G[i, j] == -1 and G[j, i] == -1 and not g.has_edge(cols[j], cols[i]):
                # undirected: orient with the same root prior other methods get
                if cols[i] in ROOT_VARS and cols[j] not in ROOT_VARS:
                    g.add_edge(cols[i], cols[j])
    return _postprocess(g)


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    g_true = ground_truth_graph()
    methods = {
        "PC": lambda: pc_discover(df, COLS, alpha=0.05),
        "GES": lambda: run_ges(df, COLS),
        "DirectLiNGAM": lambda: run_lingam(df, COLS, "direct"),
        "ICA-LiNGAM": lambda: run_lingam(df, COLS, "ica"),
        "NOTEARS (ours)": lambda: notears_discover(df, COLS, lambda1=0.08, w_threshold=0.1),
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        try:
            g = fn()
            m = evaluate_discovery(g_true, g)
            m["runtime_sec"] = round(time.time() - t0, 2)
            print(f"{name:16s} SHD={m['shd']:3d} SID={m['sid']:3d} "
                  f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
                  f"({m['runtime_sec']}s)", flush=True)
        except Exception as e:
            m = {"error": str(e)[:200]}
            print(f"{name:16s} FAILED: {e}", flush=True)
        results[name] = m

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "discovery_baselines.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {RESULTS_DIR/'discovery_baselines.json'}")


if __name__ == "__main__":
    main()
