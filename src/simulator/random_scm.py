# Author: abdkhan
"""Random SCM families, so structural claims do not rest on one hand-built graph.

The IoT SCM in scm_iot.py is a single, deliberately-designed 14-node graph.
Any conclusion drawn from it alone ("NOTEARS beats PC here") is vulnerable to
the objection that it is an artifact of that one topology, functional form and
noise model. This module samples whole *families* of SCMs so the same
comparison can be repeated across them and reported with a spread.

A family is defined by four axes that plausibly matter for discovery:
  size        number of variables
  density     expected edges per node
  mechanism   how a child depends on its parents (linear / sigmoid / tanh)
  noise       exogenous noise law (gaussian / uniform / exponential)

The linear+gaussian corner is the regime PC and GES were designed for; the
sigmoid/tanh corners violate their assumptions; non-gaussian noise is the
regime LiNGAM was designed for. Sampling across the grid therefore tests
whether a ranking survives outside the conditions that favour any one method.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

MECHANISMS = ("linear", "sigmoid", "tanh")
NOISES = ("gaussian", "uniform", "exponential")


def sample_dag(n_nodes: int, edges_per_node: float, rng: np.random.Generator) -> nx.DiGraph:
    """Erdos-Renyi DAG: sample upper-triangular adjacency under a random
    topological order, which makes acyclicity automatic."""
    order = rng.permutation(n_nodes)
    p = min(1.0, edges_per_node * 2.0 / max(n_nodes - 1, 1))
    g = nx.DiGraph()
    g.add_nodes_from(f"x{i}" for i in range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                g.add_edge(f"x{order[i]}", f"x{order[j]}")
    return g


def _noise(kind: str, size: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(0.0, scale, size)
    if kind == "uniform":
        return rng.uniform(-scale * 1.7, scale * 1.7, size)
    if kind == "exponential":
        e = rng.exponential(scale, size)
        return e - e.mean()          # centre it; the skew is the point
    raise ValueError(kind)


def _apply(kind: str, z: np.ndarray) -> np.ndarray:
    if kind == "linear":
        return z
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "tanh":
        return np.tanh(z)
    raise ValueError(kind)


def simulate(g: nx.DiGraph, n_samples: int, mechanism: str, noise: str,
             rng: np.random.Generator, noise_scale: float = 0.4) -> pd.DataFrame:
    """Forward-simulate the DAG in topological order."""
    cols = list(g.nodes())
    data = {c: None for c in cols}
    for node in nx.topological_sort(g):
        parents = list(g.predecessors(node))
        e = _noise(noise, n_samples, noise_scale, rng)
        if not parents:
            data[node] = e
            continue
        z = np.zeros(n_samples)
        for p in parents:
            w = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
            z = z + w * data[p]
        data[node] = _apply(mechanism, z) + e
    df = pd.DataFrame({c: data[c] for c in cols})
    return (df - df.mean()) / (df.std() + 1e-9)


def sample_family(n_nodes: int, edges_per_node: float, mechanism: str, noise: str,
                  n_samples: int, seed: int):
    """Return (ground-truth DAG, simulated data) for one draw from a family."""
    rng = np.random.default_rng(seed)
    g = sample_dag(n_nodes, edges_per_node, rng)
    df = simulate(g, n_samples, mechanism, noise, rng)
    return g, df
