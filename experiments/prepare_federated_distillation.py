# Author: abdkhan
"""The full-stack claim from the original pitch, tested for real: federated
causal fusion re-seeds a site's tiny-LLM distillation corpus. This script
prepares the two corpora being compared (run separately from LoRA training
since that step needs the GPU and takes ~20-25 min per corpus):

  local-only : cluster_C_low_sample's OWN discovered graph (n=800, the
               deliberately data-scarce site -- the "worst-off" site the
               federated_fusion.json experiment already showed benefits
               most from fusion at the graph-F1 level) + a linear
               surrogate SCM fit from cluster_C's OWN local data along
               those edges.
  federated  : the FUSED graph (structure only, from all 3 sites) + a
               linear surrogate fit from -- critically -- cluster_C's OWN
               local data along the FUSED edges. This matches the privacy
               model everywhere else in this project: federation shares
               graph STRUCTURE/confidence, never raw data or fitted
               coefficients, so each site still fits its own parameters
               locally, just along a (hopefully more complete) fused
               topology.

Neither corpus ever touches the true nonlinear SCM's do-calculus -- this
is deliberately more realistic than every earlier distillation experiment
in this project, which used the ground-truth SCM as an oracle. The
question this answers: does the graph-level federation benefit already
shown in run_federated_fusion.py (protects the worst-off site, +0.066 F1)
actually propagate through distillation into a better-controlling LLM, or
does it wash out? As far as this project's literature check found, no
adjacent paper tests this.
"""
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import NODES, ground_truth_graph
from causal_discovery.discover import evaluate_discovery
from federated.fusion import bootstrap_edge_confidence, FederatedCoordinator
from distillation.surrogate_scm import LinearSurrogateSCM
from distillation.build_corpus import build_from_predictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_federated_fusion import site_configs_for_trial, make_site_data

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
COLS = [c for c in NODES if c != "t"]
TARGET_SITE = "cluster_C_low_sample"


def graph_from_confidence(edge_conf: dict, threshold: float = 0.5) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(COLS)
    for e, c in edge_conf.items():
        if c >= threshold:
            g.add_edge(*e)
    return g


def main():
    seed_offset = 0
    site_configs = site_configs_for_trial(seed_offset)
    g_true = ground_truth_graph()

    print("Running local discovery at every site (needed for fusion)...")
    all_reports, all_dfs = {}, {}
    for site_id, cfg in site_configs.items():
        df = make_site_data(cfg["seed"], cfg["n_steps"], cfg["regime"])
        all_dfs[site_id] = df
        edge_conf = bootstrap_edge_confidence(df, COLS, n_bootstrap=8, sample_frac=0.7,
                                               seed=cfg["seed"], lambda1=0.08, w_threshold=0.1)
        all_reports[site_id] = dict(edges=edge_conf, n=len(df))

    local_graph = graph_from_confidence(all_reports[TARGET_SITE]["edges"])
    local_metrics = evaluate_discovery(g_true, local_graph)
    print(f"{TARGET_SITE} LOCAL graph: {local_graph.number_of_edges()} edges, "
          f"F1={local_metrics['f1']:.3f} vs ground truth")

    coordinator = FederatedCoordinator(min_sites_agreeing=1, confidence_threshold=0.2)
    for site_id, r in all_reports.items():
        coordinator.submit(site_id, r["edges"], n_samples=r["n"])
    fused_graph = coordinator.fuse(all_nodes=COLS)
    fused_metrics = evaluate_discovery(g_true, fused_graph)
    print(f"FEDERATED FUSED graph: {fused_graph.number_of_edges()} edges, "
          f"F1={fused_metrics['f1']:.3f} vs ground truth")

    target_df = all_dfs[TARGET_SITE]  # cluster_C's OWN data -- never shared

    print(f"\nFitting local-graph surrogate SCM on {TARGET_SITE}'s own {len(target_df)} rows...")
    local_surrogate = LinearSurrogateSCM(local_graph, target_df)
    print(f"Fitting fused-graph surrogate SCM on {TARGET_SITE}'s own {len(target_df)} rows...")
    fused_surrogate = LinearSurrogateSCM(fused_graph, target_df)

    print("\nGenerating corpora from each surrogate...")
    n_train_interventional, n_train_proposal = 1500, 750
    n_val_interventional, n_val_proposal = 150, 75

    for tag, predictor in [("local", local_surrogate), ("fused", fused_surrogate)]:
        train = build_from_predictor(predictor, n_train_interventional, n_train_proposal, seed=1)
        val = build_from_predictor(predictor, n_val_interventional, n_val_proposal, seed=999)
        for name, data in [(f"distill_train_{tag}graph.jsonl", train),
                            (f"distill_val_{tag}graph.jsonl", val)]:
            with open(DATA_DIR / name, "w") as f:
                for ex in data:
                    f.write(json.dumps(ex) + "\n")
            print(f"  wrote {len(data)} examples -> {DATA_DIR/name}")

    summary = dict(local_graph_f1=local_metrics["f1"], local_graph_edges=local_graph.number_of_edges(),
                    fused_graph_f1=fused_metrics["f1"], fused_graph_edges=fused_graph.number_of_edges(),
                    target_site=TARGET_SITE, target_site_n_samples=len(target_df))
    RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "federated_distillation_prep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
