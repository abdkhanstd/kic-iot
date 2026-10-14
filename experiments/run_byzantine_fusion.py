# Author: abdkhan
"""Byzantine-robust federated causal fusion (novel angle #2): none of the
adjacent work found in this project's literature check tests federated
causal discovery against a malicious/compromised PARTICIPATING SITE -- the
existing poisoning ablation (run_poisoning_ablation.py) corrupts the
TELEMETRY at a single, non-federated site; the federated-causal-discovery
survey cited in the original idea doc does not stress-test any concrete
aggregator against an adversarial client. Because a federated coordinator
only ever sees a site's reported (edge -> confidence) dict -- never raw
data -- a malicious site can lie about its report directly, with no need
to even corrupt real telemetry: a fabrication attack pastes fake
high-confidence edges into an otherwise-plausible report.

Compares the naive aggregators already used elsewhere in this project
(sample_weighted, mean, max -- all vulnerable by construction, since a
single site's report enters the fused confidence directly or via a
sample-count weight the site does not control the attacker's cost of)
against two standard Byzantine-robust aggregators from the federated-
learning literature applied per-edge (coordinate-wise median; trimmed
mean), across a sweep of how many of N sites are malicious.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import ground_truth_graph, NODES
from causal_discovery.discover import evaluate_discovery
from federated.fusion import bootstrap_edge_confidence, FederatedCoordinator
from run_federated_fusion import site_configs_for_trial, make_site_data

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
COLS = [c for c in NODES if c != "t"]


def malicious_fabrication_report(true_graph: nx.DiGraph, columns: list[str], seed: int,
                                  n_fake_edges: int = 6, fake_confidence: float = 0.95) -> dict:
    """A malicious site's report: NO real discovery is run (it doesn't need
    real data to lie) -- it fabricates high-confidence edges for
    variable pairs with NO true causal relationship (neither direction),
    trying to inject spurious structure into every other site's fused
    graph. This is the realistic threat model for federated causal
    fusion: the coordinator only ever sees a (edge -> confidence) report,
    never the underlying telemetry, so nothing stops a compromised or
    dishonest participant from reporting fiction.
    """
    rng = np.random.default_rng(seed)
    true_edges = set(true_graph.edges())
    non_edges = [(u, v) for u in columns for v in columns
                 if u != v and (u, v) not in true_edges and (v, u) not in true_edges]
    chosen = rng.choice(len(non_edges), size=min(n_fake_edges, len(non_edges)), replace=False)
    return {non_edges[i]: fake_confidence for i in chosen}


def compute_honest_reports(seed_offset: int) -> dict:
    """The expensive part (real bootstrap causal discovery per honest site)
    -- computed once per trial and reused across every aggregation method
    and malicious-site count, since honest sites' reports don't depend on
    either."""
    site_configs = site_configs_for_trial(seed_offset)
    reports = {}
    for site_id, cfg in site_configs.items():
        df = make_site_data(cfg["seed"], cfg["n_steps"], cfg["regime"])
        edge_conf = bootstrap_edge_confidence(df, COLS, n_bootstrap=8, sample_frac=0.7,
                                               seed=cfg["seed"], lambda1=0.08, w_threshold=0.1)
        reports[site_id] = dict(edges=edge_conf, n=len(df))
    return reports


def fuse_with_attackers(honest_reports: dict, g_true: nx.DiGraph, seed_offset: int,
                         n_malicious: int, aggregation: str, trim_k: int = 1,
                         collude: bool = False) -> dict:
    """collude=True: every malicious site submits the SAME fabricated edge
    set (a coordinated botnet of compromised gateways), instead of each
    picking its own random fake edges. This is the strictly stronger threat
    model and it separates genuine Byzantine robustness from a threshold
    artifact: with independent attackers, each fake edge is backed by only
    one site, so plain `mean` dilutes it below the acceptance threshold and
    looks deceptively robust; under collusion the same fake edge is backed
    by every attacker at once, so only aggregators with a real breakdown
    point (median, trimmed mean with trim_k >= n_malicious) still reject it.
    """
    coordinator = FederatedCoordinator(min_sites_agreeing=1, confidence_threshold=0.2)
    for site_id, r in honest_reports.items():
        coordinator.submit(site_id, r["edges"], n_samples=r["n"])

    all_fabricated_edges = set()
    for m in range(n_malicious):
        mal_id = f"malicious_site_{m}"
        # colluding attackers share one seed -> identical fabricated edge set
        mal_seed = seed_offset * 10 if collude else seed_offset * 10 + m
        report = malicious_fabrication_report(g_true, COLS, seed=mal_seed)
        all_fabricated_edges |= set(report.keys())
        # a malicious site can also lie about its sample count to gain
        # more weight under sample_weighted aggregation -- model the
        # stronger adversary that claims to be the largest honest site.
        coordinator.submit(mal_id, report, n_samples=6000)

    g_fused = coordinator.fuse(all_nodes=COLS, aggregation=aggregation, trim_k=trim_k)
    metrics = evaluate_discovery(g_true, g_fused)
    # specifically: how many of THIS attack's fabricated edges made it into
    # the fused graph (not honest-discovery false positives in general).
    fabricated_survived = sum(1 for e in all_fabricated_edges if g_fused.has_edge(*e))
    return dict(f1=metrics["f1"], shd=metrics["shd"], n_edges=metrics["n_est_edges"],
                n_fabricated_edges_submitted=len(all_fabricated_edges),
                n_fabricated_edges_survived=fabricated_survived)


def main():
    import pickle
    g_true = ground_truth_graph()
    n_trials = 10
    # trim_k=1 tolerates 1 malicious site; trim_k=2 tolerates 2 (the
    # Byzantine guarantee needs trim_k < n_sites/2, so with 3 honest sites
    # trim_k=2 is the largest usable setting at 5 total participants).
    aggregations = [("sample_weighted", 0), ("mean", 0), ("max", 0),
                     ("median", 0), ("trimmed_mean", 1), ("trimmed_mean_k2", 2)]

    cache = RESULTS_DIR / "_honest_reports_cache.pkl"
    if cache.exists():
        print(f"Loading cached honest-site reports from {cache}")
        honest_reports_per_trial = pickle.load(open(cache, "rb"))
    else:
        print("Precomputing honest-site reports (real bootstrap discovery, once per trial)...")
        honest_reports_per_trial = [compute_honest_reports(t * 100) for t in range(n_trials)]
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(honest_reports_per_trial, open(cache, "wb"))
    print(f"Ready: {n_trials} trials.\n")

    results = {}
    for collude in [False, True]:
        for n_malicious in [0, 1, 2]:
            if collude and n_malicious < 2:
                continue  # collusion is only meaningful with >= 2 attackers
            for agg_label, trim_k in aggregations:
                agg = "trimmed_mean" if agg_label.startswith("trimmed_mean") else agg_label
                key = f"{agg_label}_malicious{n_malicious}{'_collude' if collude else ''}"
                trials = [fuse_with_attackers(honest_reports_per_trial[t], g_true, t * 100,
                                               n_malicious, agg, trim_k, collude)
                          for t in range(n_trials)]
                f1s = np.array([t["f1"] for t in trials])
                fakes = np.array([t["n_fabricated_edges_survived"] for t in trials])
                n_submitted = float(np.mean([t["n_fabricated_edges_submitted"] for t in trials]))
                results[key] = dict(
                    aggregation=agg_label, trim_k=trim_k, n_malicious=n_malicious,
                    collude=collude,
                    f1_mean=float(f1s.mean()), f1_std=float(f1s.std()),
                    mean_fabricated_edges_submitted=n_submitted,
                    mean_fabricated_edges_survived=float(fakes.mean()),
                    leak_fraction=float(fakes.mean() / n_submitted) if n_submitted else 0.0,
                    frac_trials_with_any_fabricated_edge=float((fakes > 0).mean()),
                )
                print(f"{agg_label:17s} n_mal={n_malicious}{' COLLUDING' if collude else '':10s}: "
                      f"F1={f1s.mean():.3f}+-{f1s.std():.3f}  "
                      f"fabricated edges surviving: {fakes.mean():.2f}/{n_submitted:.0f} "
                      f"({(fakes>0).mean():.0%} of trials have >=1)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "byzantine_fusion.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Fabricated-edge leakage as attackers are added (lower is better) ===")
    print(f"{'aggregator':18s} {'0 mal':>12s} {'1 mal':>12s} {'2 mal':>12s} {'2 COLLUDING':>14s}")
    for agg_label, _ in aggregations:
        cells = []
        for key in [f"{agg_label}_malicious0", f"{agg_label}_malicious1",
                     f"{agg_label}_malicious2", f"{agg_label}_malicious2_collude"]:
            r = results[key]
            cells.append(f"{r['mean_fabricated_edges_survived']:.2f}/"
                         f"{r['mean_fabricated_edges_submitted']:.0f}")
        print(f"{agg_label:18s} {cells[0]:>12s} {cells[1]:>12s} {cells[2]:>12s} {cells[3]:>14s}")


if __name__ == "__main__":
    main()
