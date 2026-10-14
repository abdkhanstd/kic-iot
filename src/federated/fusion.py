# Author: abdkhan
"""Federated causal-graph fusion across heterogeneous IoT edge clusters
(component 4). Each site runs causal discovery locally (with bootstrap
resampling for edge-confidence estimates) and reports ONLY an edge list +
confidence scores to a coordinator -- no raw telemetry ever leaves a site.
The coordinator fuses per-site graphs via confidence-weighted voting, which
is then used to re-seed each site's tiny-LLM distillation corpus.

This targets the "distributed causal learning / causal representation
learning for heterogeneous IoT" CFP bullets, and is implemented as a
software simulation of multiple clients on one machine (no physical
multi-node deployment required to validate the fusion algorithm itself).
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_discovery.discover import notears_discover, ROOT_VARS


def bootstrap_edge_confidence(df: pd.DataFrame, columns: list[str],
                               n_bootstrap: int = 10, sample_frac: float = 0.7,
                               seed: int = 0, **notears_kwargs) -> dict:
    """Run NOTEARS on `n_bootstrap` resamples of a site's local data;
    return {(u,v): confidence in [0,1]} = fraction of runs containing edge."""
    rng = np.random.default_rng(seed)
    edge_counts = defaultdict(int)
    n = len(df)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=int(n * sample_frac), replace=True)
        sample = df.iloc[idx]
        g = notears_discover(sample, columns, **notears_kwargs)
        for e in g.edges():
            edge_counts[e] += 1
    return {e: c / n_bootstrap for e, c in edge_counts.items()}


class FederatedCoordinator:
    """Aggregates per-site edge-confidence dicts into a shared causal
    backbone. Sites are NOT required to share the same variable set --
    fusion happens over the union of variables, so heterogeneous domains
    (e.g. a wireless-QoS cluster and a smart-energy cluster) can still share
    whatever overlapping structure exists (e.g. duty_cycle/energy links)."""

    def __init__(self, min_sites_agreeing: int = 2, confidence_threshold: float = 0.4):
        self.min_sites_agreeing = min_sites_agreeing
        self.confidence_threshold = confidence_threshold
        self.site_reports: dict[str, dict] = {}

    def submit(self, site_id: str, edge_confidence: dict, n_samples: int):
        """A site sends ONLY (edge -> confidence) plus its sample count for
        weighting -- never raw telemetry rows."""
        self.site_reports[site_id] = dict(edges=edge_confidence, n=n_samples)

    def fuse(self, all_nodes: list | None = None, aggregation: str = "sample_weighted",
             trim_k: int = 1) -> nx.DiGraph:
        """aggregation: 'sample_weighted' (confidence weighted by each site's
        sample count -- can let a large, noisy site dilute a smaller site's
        clean detection of a real edge), 'max' (an edge counts if ANY site
        detects it with high confidence -- appropriate when sites differ in
        statistical power rather than in whether the edge is real), 'mean'
        (unweighted average across sites), or two Byzantine-robust
        aggregators from the federated-learning literature (Blanchard et al.
        2017 Krum lineage; Yin et al. 2018 trimmed mean) applied per-edge to
        the vector of site-reported confidences instead of full model
        updates: 'median' (coordinate-wise median -- a single malicious
        site's report can shift the median by at most one rank), or
        'trimmed_mean' (drop the `trim_k` highest and `trim_k` lowest
        reports per edge, then average what remains -- tolerates up to
        `trim_k` colluding/malicious sites as long as trim_k < n_sites/2).
        None of the sample-weighted/max/mean aggregators are robust: a
        single site reporting a fabricated high-confidence edge, or
        suppressing a real one, moves the fused result directly.
        """
        all_edges = set()
        for r in self.site_reports.values():
            all_edges |= set(r["edges"].keys())

        total_n = sum(r["n"] for r in self.site_reports.values())
        n_sites = len(self.site_reports)
        fused = nx.DiGraph()
        if all_nodes is not None:
            fused.add_nodes_from(all_nodes)
        for e in all_edges:
            confs = [r["edges"].get(e, 0.0) for r in self.site_reports.values()]
            n_agree = sum(1 for c in confs if c >= self.confidence_threshold)

            if aggregation == "max":
                agg_conf = max(confs)
            elif aggregation == "mean":
                agg_conf = float(np.mean(confs))
            elif aggregation == "median":
                agg_conf = float(np.median(confs))
            elif aggregation == "trimmed_mean":
                sorted_confs = sorted(confs)
                k = min(trim_k, (n_sites - 1) // 2)  # never trim away every report
                trimmed = sorted_confs[k:n_sites - k] if n_sites - 2 * k > 0 else sorted_confs
                agg_conf = float(np.mean(trimmed))
            else:
                agg_conf = sum(c * (r["n"] / total_n) for c, r in zip(confs, self.site_reports.values()))

            if agg_conf >= self.confidence_threshold and n_agree >= min(
                    self.min_sites_agreeing, len(self.site_reports)):
                fused.add_edge(*e, weight=round(agg_conf, 3))

        for u, v in list(fused.edges()):
            fused.add_node(u); fused.add_node(v)
            if v in ROOT_VARS:
                fused.remove_edge(u, v)
        if not nx.is_directed_acyclic_graph(fused):
            while not nx.is_directed_acyclic_graph(fused):
                cyc = nx.find_cycle(fused)
                weakest = min(cyc, key=lambda e: fused.get_edge_data(e[0], e[1]).get("weight", 0))
                fused.remove_edge(weakest[0], weakest[1])
        return fused
