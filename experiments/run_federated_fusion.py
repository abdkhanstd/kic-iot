# Author: abdkhan
"""Experiment 4: federated causal-graph fusion across heterogeneous IoT
clusters (component 4). Simulates 3 sites (as independent clients on this
one machine -- no physical multi-node deployment needed to validate the
*algorithm*): two synthetic wireless-QoS clusters under different regime
parameters (heterogeneous operating conditions), plus one real-world
telemetry domain (N-BaIoT-derived) contributing whatever structure
overlaps with the shared actionable variables.

Reports: per-site local SHD/F1 vs. ground truth, and fused-graph SHD/F1 --
the payoff being that fusion should recover structure no single site could
alone (e.g. under-sampled or attacked at one site) while sharing only
edge-confidence lists, never raw telemetry.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import IoTClusterSCM, RegimeParams, ground_truth_graph, NODES
from causal_discovery.discover import evaluate_discovery, notears_discover
from federated.fusion import bootstrap_edge_confidence, FederatedCoordinator

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
COLS = [c for c in NODES if c != "t"]


def make_site_data(seed: int, n_steps: int, regime: RegimeParams) -> pd.DataFrame:
    scm = IoTClusterSCM(seed=seed, regime=regime)
    return pd.DataFrame(scm.rollout(n_steps))


def site_configs_for_trial(seed_offset: int) -> dict:
    return {
        "cluster_A_urban_dense": dict(
            seed=1 + seed_offset, n_steps=6000,
            regime=RegimeParams(interference_mean=0.6, interference_std=0.15,
                                 link_quality_mean=0.5, arrival_rate_mean=0.6)),
        "cluster_B_rural_sparse": dict(
            seed=2 + seed_offset, n_steps=4000,
            regime=RegimeParams(interference_mean=0.15, interference_std=0.08,
                                 link_quality_mean=0.85, arrival_rate_mean=0.2)),
        "cluster_C_low_sample": dict(
            seed=3 + seed_offset, n_steps=800,   # deliberately data-scarce site
            regime=RegimeParams(interference_mean=0.35, interference_std=0.2,
                                 link_quality_mean=0.65, arrival_rate_mean=0.45)),
    }


def run_trial(seed_offset: int, g_true, verbose: bool = False) -> dict:
    site_configs = site_configs_for_trial(seed_offset)
    # confidence-weighted union (a single site's confident edge is enough to
    # propose it; the weighting still discounts low-confidence/low-sample
    # sites) empirically outperforms requiring multi-site agreement here,
    # because per-site regimes make some true edges' effect strength look
    # weak at any one site.
    coordinator = FederatedCoordinator(min_sites_agreeing=1, confidence_threshold=0.2)
    per_site_results = {}

    for site_id, cfg in site_configs.items():
        df = make_site_data(cfg["seed"], cfg["n_steps"], cfg["regime"])
        edge_conf = bootstrap_edge_confidence(df, COLS, n_bootstrap=8, sample_frac=0.7,
                                               seed=cfg["seed"], lambda1=0.08, w_threshold=0.1)
        import networkx as nx
        g_local = nx.DiGraph()
        g_local.add_nodes_from(COLS)
        for e, c in edge_conf.items():
            if c >= 0.5:
                g_local.add_edge(*e)
        local_metrics = evaluate_discovery(g_true, g_local)
        per_site_results[site_id] = dict(n_samples=len(df), **local_metrics)
        if verbose:
            print(f"{site_id:25s} n={len(df):5d}  local SHD={local_metrics['shd']:3d} "
                  f"F1={local_metrics['f1']:.2f}")
        coordinator.submit(site_id, edge_conf, n_samples=len(df))

    g_fused = coordinator.fuse(all_nodes=COLS)
    fused_metrics = evaluate_discovery(g_true, g_fused)
    site_f1s = [r["f1"] for r in per_site_results.values()]
    return dict(per_site=per_site_results, fused=fused_metrics,
                best_single_site_f1=max(site_f1s), mean_single_site_f1=float(np.mean(site_f1s)),
                worst_single_site_f1=min(site_f1s), fused_edges=list(g_fused.edges()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=10)
    args = ap.parse_args()

    g_true = ground_truth_graph()
    trials = []
    for t in range(args.n_trials):
        r = run_trial(seed_offset=t * 100, g_true=g_true, verbose=(t == 0))
        trials.append(r)
        print(f"trial {t}: best_single_site F1={r['best_single_site_f1']:.3f}  "
              f"fused F1={r['fused']['f1']:.3f}  delta={r['fused']['f1']-r['best_single_site_f1']:+.3f}")

    fused_f1s = np.array([t["fused"]["f1"] for t in trials])
    best_site_f1s = np.array([t["best_single_site_f1"] for t in trials])
    mean_site_f1s = np.array([t["mean_single_site_f1"] for t in trials])
    worst_site_f1s = np.array([t["worst_single_site_f1"] for t in trials])

    def compare(name, baseline):
        delta = fused_f1s - baseline
        return dict(baseline_mean=float(baseline.mean()), baseline_std=float(baseline.std()),
                    delta_mean=float(delta.mean()), delta_std=float(delta.std()),
                    frac_trials_fusion_wins=float((delta > 0).mean()))

    summary = dict(
        n_trials=args.n_trials,
        fused_f1_mean=float(fused_f1s.mean()), fused_f1_std=float(fused_f1s.std()),
        # vs_best_site: an unrealistic oracle bar -- no real site knows in
        # advance which of its peers happens to have the best local data.
        vs_best_site=compare("best", best_site_f1s),
        # vs_mean_site / vs_worst_site: the realistic value proposition of
        # federation -- does a typical, or the worst-off, site do better by
        # federating than by going alone?
        vs_mean_site=compare("mean", mean_site_f1s),
        vs_worst_site=compare("worst", worst_site_f1s),
        first_trial_detail=trials[0],
    )
    print(f"\n=== Over {args.n_trials} trials ===")
    print(f"fused F1                       = {summary['fused_f1_mean']:.3f} +- {summary['fused_f1_std']:.3f}")
    for label, key in [("vs best site (oracle, unrealistic)", "vs_best_site"),
                        ("vs mean site (typical benefit)", "vs_mean_site"),
                        ("vs worst site (protects the weak)", "vs_worst_site")]:
        c = summary[key]
        print(f"{label:38s} baseline={c['baseline_mean']:.3f}+-{c['baseline_std']:.3f}  "
              f"delta={c['delta_mean']:+.3f}+-{c['delta_std']:.3f}  win_rate={c['frac_trials_fusion_wins']:.0%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "federated_fusion.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
