# Author: abdkhan
"""Controlled comparison of causal-feature-selection pipelines for IoT
attack detection, all evaluated under IDENTICAL conditions.

Motivation. Recent IoT causal-analysis systems report attack-classification
accuracy (e.g. a LiNGAM-based pipeline reporting ~100% F1 on CICIoT2023),
whereas our pipeline is scored on structural recovery. Those numbers cannot
be placed in one table honestly. The fix is not to omit the comparison but
to remove the confound: we re-implement the competing pipeline as described
and run every method on the same dataset, the same split, the same
downstream classifier, and the same metrics, so that the ONLY thing that
differs is the causal stage under test.

These are our re-implementations following each method's published
description, not the original authors' code; absolute numbers will differ
from their papers because the dataset and protocol are held fixed here.

Pipelines compared (identical except for the feature-selection stage):
  all-features      no causal stage at all -- the control
  correlation-topk  non-causal filter: top-k |Pearson r| with the label
  mutual-info-topk  non-causal filter: top-k mutual information
  LiNGAM-causal     LiNGAM-based causal parents of the label (TOCA-IoT style)
  NOTEARS-causal    our discovery stage, same use of the recovered parents
Downstream classifier, split, seed and metrics are shared by construction.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from causal_discovery.discover import notears_discover  # noqa: E402

OUT = ROOT / "experiments" / "results" / "comparison_ids.json"
LABEL = "compromised"
K = 8  # features kept by every selector that selects, fixed across methods


def load(n_per_class: int, seed: int):
    df = pd.read_csv(ROOT / "data" / "processed" / "nbaiot_combined.csv")
    feats = [c for c in df.columns
             if c not in ("device", "attack_family", "attack_type", LABEL)]
    # balanced subsample keeps the comparison from being dominated by the
    # attack-heavy class distribution of the raw archive
    parts = [g.sample(n=min(n_per_class, len(g)), random_state=seed)
             for _, g in df.groupby(LABEL)]
    df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # per-device standardisation, as used throughout the paper
    for c in feats:
        df[c] = df.groupby("device")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return df.fillna(0.0), feats


def sel_all(df, feats, seed):
    return feats


def sel_corr(df, feats, seed):
    r = {c: abs(np.corrcoef(df[c], df[LABEL])[0, 1]) for c in feats}
    r = {c: (v if v == v else 0.0) for c, v in r.items()}
    return sorted(r, key=r.get, reverse=True)[:K]


def sel_mi(df, feats, seed):
    mi = mutual_info_classif(df[feats].to_numpy(), df[LABEL].to_numpy(),
                              random_state=seed)
    return [feats[i] for i in np.argsort(mi)[::-1][:K]]


def _parents_of_label(g: nx.DiGraph, feats):
    """Features the recovered graph links to the label, preferring true
    parents and falling back to any adjacency if the label is isolated."""
    if LABEL not in g:
        return []
    picked = [p for p in g.predecessors(LABEL)]
    if len(picked) < K:
        picked += [c for c in g.successors(LABEL) if c not in picked]
    return [p for p in picked if p in feats][:K]


def sel_lingam(df, feats, seed):
    """TOCA-IoT-style: LiNGAM causal structure, then take the label's causal
    neighbourhood as the feature set."""
    from causallearn.search.FCMBased import lingam
    cols = feats + [LABEL]
    X = df[cols].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    model = lingam.DirectLiNGAM()
    model.fit(X)
    B = np.asarray(model.adjacency_matrix_)
    g = nx.DiGraph(); g.add_nodes_from(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(B[i, j]) > 0.05:
                g.add_edge(cols[j], cols[i])
    picked = _parents_of_label(g, feats)
    return picked or sel_corr(df, feats, seed)


def sel_notears(df, feats, seed):
    """Our discovery stage, used the same way as the LiNGAM arm."""
    cols = feats + [LABEL]
    g = notears_discover(df[cols], cols, lambda1=0.02, w_threshold=0.1)
    picked = _parents_of_label(g, feats)
    return picked or sel_corr(df, feats, seed)


SELECTORS = {
    "all-features": sel_all,
    "correlation-topk": sel_corr,
    "mutual-info-topk": sel_mi,
    "LiNGAM-causal (TOCA-IoT style)": sel_lingam,
    "NOTEARS-causal (ours)": sel_notears,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    results = {}
    for name, fn in SELECTORS.items():
        per_seed = []
        for seed in args.seeds:
            df, feats = load(args.n_per_class, seed)
            t0 = time.time()
            picked = fn(df, feats, seed)
            sel_time = time.time() - t0

            Xtr, Xte, ytr, yte = train_test_split(
                df[picked].to_numpy(), df[LABEL].to_numpy(),
                test_size=0.3, random_state=seed, stratify=df[LABEL])
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            clf.fit(Xtr, ytr)
            p = clf.predict(Xte)
            prob = clf.predict_proba(Xte)[:, 1]
            per_seed.append(dict(
                accuracy=accuracy_score(yte, p), precision=precision_score(yte, p, zero_division=0),
                recall=recall_score(yte, p, zero_division=0), f1=f1_score(yte, p, zero_division=0),
                auc=roc_auc_score(yte, prob), n_features=len(picked),
                selection_sec=round(sel_time, 2)))
            print(f"  {name:34s} seed={seed} f1={per_seed[-1]['f1']:.4f} "
                  f"acc={per_seed[-1]['accuracy']:.4f} k={len(picked)}", flush=True)

        agg = {k: dict(mean=float(np.mean([r[k] for r in per_seed])),
                        std=float(np.std([r[k] for r in per_seed])))
               for k in ("accuracy", "precision", "recall", "f1", "auc",
                          "n_features", "selection_sec")}
        agg["per_seed"] = per_seed
        results[name] = agg
        print(f"{name:34s} F1={agg['f1']['mean']:.4f}+-{agg['f1']['std']:.4f}  "
              f"ACC={agg['accuracy']['mean']:.4f}  k={agg['n_features']['mean']:.0f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(dataset="N-BaIoT (balanced subsample)", label=LABEL,
                        n_per_class=args.n_per_class, seeds=args.seeds,
                        classifier="RandomForest(200)", top_k=K, methods=results), f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
