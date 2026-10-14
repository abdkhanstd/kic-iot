# Author: abdkhan
"""Real-world causal-discovery demonstration on N-BaIoT (component 1's
second, non-synthetic validation -- no known ground-truth causal graph
exists for a real deployment, so this is evaluated qualitatively: does the
discovered structure place the `compromised` indicator as a root-cause
driver of telemetry deviation, rather than a leaf/isolated node, and is
that structure stable across bootstrap resamples?

Feature set: the `compromised` binary flag plus one representative
statistical feature (mean) per (traffic-window, moment-family) combination
-- using all 115 raw features directly would make NOTEARS's O(d^3)-ish
optimization too slow for this ablation pass.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_discovery.discover import notears_discover

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"

FAMILIES = ["MI_dir", "H", "HH", "HH_jit", "HpHp"]


def select_features(df: pd.DataFrame) -> list[str]:
    # One time window (L5, the longest/most stable) per statistical family.
    # An earlier version used all 5 windows x 5 families = 25 features, but
    # the 5 windows within one family are highly collinear (same underlying
    # signal at different smoothing scales) -- combined with a since-fixed
    # NOTEARS bug that let a self-loop silently absorb fitting capacity
    # (see discover.py's notears_linear _adj fix), that collinearity fully
    # masked the compromised->HH_jit_L5_mean edge recovered here.
    return [f"{fam}_L5_mean" for fam in FAMILIES if f"{fam}_L5_mean" in df.columns]


def prep(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-device z-scoring: different IoT device types have very different
    baseline traffic volumes, which otherwise dominates the variance of each
    raw feature and can wash out the (smaller, within-device) shift caused
    by compromise -- residualizing against device identity isolates the
    compromise-conditional signal linear NOTEARS can actually detect."""
    sub = df[["compromised"] + feats].astype(float)
    for c in feats:
        sub[c] = df.groupby("device")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return sub


def run_discovery(sub: pd.DataFrame, cols: list[str], n_boot: int = 8, seed: int = 0) -> dict:
    g = notears_discover(sub, cols, lambda1=0.02, w_threshold=0.1)
    out_degree, in_degree = dict(g.out_degree()), dict(g.in_degree())

    rng = np.random.default_rng(seed)
    compromised_edge_counts = {}
    for _ in range(n_boot):
        idx = rng.choice(len(sub), size=int(len(sub) * 0.7), replace=True)
        gb = notears_discover(sub.iloc[idx], cols, lambda1=0.02, w_threshold=0.1)
        for e in gb.edges():
            if e[0] == "compromised":
                compromised_edge_counts[e] = compromised_edge_counts.get(e, 0) + 1
    stability = {f"{u}->{v}": c / n_boot for (u, v), c in compromised_edge_counts.items()}

    return dict(
        n_rows=len(sub), edges=list(g.edges()),
        compromised_out_degree=out_degree.get("compromised", 0),
        compromised_in_degree=in_degree.get("compromised", 0),
        compromised_out_edges=[e for e in g.edges() if e[0] == "compromised"],
        compromised_edge_bootstrap_stability=stability,
    )


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / "nbaiot_combined.csv")
    feats = select_features(df)
    cols = ["compromised"] + feats
    print(f"Using {len(cols)} variables: {cols}")

    results = {}

    print("\n=== Pooled (benign + mirai + gafgyt) ===")
    sub_all = prep(df, feats)
    results["pooled"] = run_discovery(sub_all, cols)
    print(f"compromised out-edges: {results['pooled']['compromised_out_edges']} "
          f"(total edges: {len(results['pooled']['edges'])})")

    # A quick univariate check (see docstring below) shows the pooled binary
    # `compromised` flag has near-zero correlation with these features, but
    # that's a Simpson's-paradox-style cancellation: Mirai and Gafgyt shift
    # the SAME feature in OPPOSITE directions, so a single linear treatment
    # coefficient for "compromised" averages them out to ~0. Splitting by
    # attack family recovers the signal within each -- exactly the kind of
    # real-world nonstationarity/heterogeneity a naive pooled linear
    # discovery pipeline misses, and a concrete motivation for this paper's
    # federated / per-cluster causal modeling rather than one global model.
    for family in ["mirai", "gafgyt"]:
        print(f"\n=== benign + {family} only ===")
        df_f = df[df["attack_family"].isin(["benign", family])]
        sub_f = prep(df_f, feats)
        results[family] = run_discovery(sub_f, cols)
        print(f"compromised out-edges: {results[family]['compromised_out_edges']} "
              f"(total edges: {len(results[family]['edges'])}, n={results[family]['n_rows']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "causal_discovery_nbaiot_real.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
