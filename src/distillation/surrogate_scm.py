# Author: abdkhan
"""A linear surrogate SCM fit from a DISCOVERED graph (not oracle access
to the true nonlinear SCM) -- this is what a real deployment actually has
available to generate distillation-corpus labels: a site's own (possibly
incomplete or federated-fused) causal graph, plus a linear regression per
node on its discovered parents, fit from local observational data.

Used for the federation-feeds-distillation experiment: does re-seeding a
data-scarce site's distillation corpus from the FEDERATED fused graph
(instead of that site's own noisy local graph) produce a tiny LLM that
controls QoS better? Every earlier corpus-generation experiment in this
project used exact do-calculus from the true SCM as an oracle upper bound;
this is deliberately more realistic and more novel, since it tests whether
the *federation* benefit (already shown at the graph-F1 level in
run_federated_fusion.py) actually propagates through distillation into
better real control -- something neither this project's earlier
experiments nor the related literature found in the search checked.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


class LinearSurrogateSCM:
    def __init__(self, graph: nx.DiGraph, df: pd.DataFrame):
        self.graph = graph
        self.node_means = {c: float(df[c].mean()) for c in df.columns}
        self.coefs: dict[str, dict] = {}
        for node in nx.topological_sort(graph):
            parents = [p for p in graph.predecessors(node) if p in df.columns]
            if not parents or node not in df.columns:
                continue
            X = df[parents].to_numpy(dtype=float)
            y = df[node].to_numpy(dtype=float)
            X1 = np.column_stack([X, np.ones(len(X))])
            coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
            self.coefs[node] = dict(zip(parents, coef[:-1].tolist()))
            self.coefs[node]["_intercept"] = float(coef[-1])

    def predict(self, actions: dict, do: dict | None = None) -> dict:
        """Forward-propagate through the discovered DAG in topological
        order to get a point prediction for every node (deterministic --
        this is a point-estimate surrogate, not a Monte-Carlo do-calculus
        estimate like IoTClusterSCM.counterfactual)."""
        do = do or {}
        values = dict(self.node_means)
        values.update(actions)
        values.update(do)
        for node in nx.topological_sort(self.graph):
            if node in do:
                continue
            if node not in self.coefs:
                continue
            c = self.coefs[node]
            v = c["_intercept"]
            for p, w in c.items():
                if p == "_intercept":
                    continue
                v += w * values.get(p, self.node_means.get(p, 0.5))
            values[node] = float(np.clip(v, 0.0, 1.0))
        return values

    def counterfactual(self, actions: dict, do: dict | None = None, n_samples: int = 1) -> dict:
        """Same call signature as IoTClusterSCM.counterfactual for drop-in
        compatibility with the corpus builder -- this surrogate is a point
        estimate (no exogenous noise model), so n_samples is accepted and
        ignored."""
        return self.predict(actions, do)
