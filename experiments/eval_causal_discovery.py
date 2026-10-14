# Author: abdkhan
"""Experiment 1: causal discovery quality on clean synthetic data.
Compares PC (causal-learn) vs NOTEARS against the ground-truth SCM graph,
using SHD / SID / precision / recall / F1 (paper Section 5, first bullet).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import ground_truth_graph, NODES
from causal_discovery.discover import pc_discover, notears_discover, evaluate_discovery

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    df = pd.read_csv(data_dir / "clean_train.csv")
    cols = [c for c in NODES if c != "t"]
    g_true = ground_truth_graph()

    results = {}

    print("Running NOTEARS...")
    t0 = time.time()
    g_notears = notears_discover(df, cols, lambda1=0.08, w_threshold=0.1)
    results["notears"] = dict(evaluate_discovery(g_true, g_notears),
                               runtime_sec=round(time.time() - t0, 2),
                               edges=list(g_notears.edges()))

    print("Running PC...")
    t0 = time.time()
    try:
        g_pc = pc_discover(df, cols, alpha=0.05)
        results["pc"] = dict(evaluate_discovery(g_true, g_pc),
                              runtime_sec=round(time.time() - t0, 2),
                              edges=list(g_pc.edges()))
    except Exception as e:
        results["pc"] = dict(error=str(e))

    results["ground_truth_edges"] = list(g_true.edges())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "causal_discovery_clean.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for method in ["notears", "pc"]:
        r = results[method]
        if "error" in r:
            print(f"{method}: FAILED ({r['error']})")
        else:
            print(f"{method}: SHD={r['shd']} SID={r['sid']} "
                  f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
                  f"({r['n_est_edges']} edges vs {r['n_true_edges']} true, "
                  f"{r['runtime_sec']}s)")


if __name__ == "__main__":
    main()
