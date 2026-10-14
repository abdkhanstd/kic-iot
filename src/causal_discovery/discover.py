# Author: abdkhan
"""Causal discovery on IoT telemetry streams.

Two families, both operating on the *endogenous+action* variable set
(exogenous drivers are included as candidate parents but the ground-truth
DAG constrains them to be roots):

  1. PC algorithm (constraint-based, via causal-learn) -- fast, gives a
     CPDAG; we orient remaining undirected edges using the known
     action/exogenous vs. outcome variable ordering (a mild domain-knowledge
     prior any real IoT deployment would also have: "you know which
     telemetry fields are actions vs. sensed outcomes").
  2. NOTEARS (continuous score-based optimization) -- reimplemented here
     directly (no external dependency) since it is a small, well-known
     algorithm; gives a weighted adjacency matrix thresholded into a DAG.

Both are evaluated against the ground-truth graph with:
  - SHD  (Structural Hamming Distance): count of edge insertions/deletions/
    reversals needed to turn the estimated graph into the true graph.
  - SID  (Structural Intervention Distance, approximated): counts pairs
    (i, j) for which the estimated graph would give the *wrong* answer to
    "does intervening on i change j", used as a downstream-correctness proxy
    which SHD does not capture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.scm_iot import NODES, EXOGENOUS, ACTIONS, ENDOGENOUS


ROOT_VARS = set(EXOGENOUS) | set(ACTIONS)  # domain-knowledge: these have no parents


# --------------------------------------------------------------------------
# NOTEARS (Zheng et al., 2018) -- minimal numpy/scipy reimplementation
# --------------------------------------------------------------------------

def notears_linear(X: np.ndarray, lambda1: float = 0.05, max_iter: int = 100,
                    h_tol: float = 1e-8, rho_max: float = 1e16,
                    w_threshold: float = 0.15) -> np.ndarray:
    """Fit a linear SEM DAG via the augmented-Lagrangian NOTEARS formulation.
    Returns a (d,d) weighted adjacency matrix W where W[i,j] != 0 means i->j.
    """
    from scipy.optimize import minimize

    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)

    def _loss(W):
        M = Xc @ W
        R = Xc - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * Xc.T @ R
        return loss, G_loss

    def _h(W):
        # h(W) = tr(e^{W∘W}) - d  (acyclicity constraint, exact matrix exponential)
        E = np.linalg.matrix_power(np.eye(d) + (W * W) / d, d - 1)
        h_val = (E.T * (W * W)).sum() - d
        G_h = E.T * W * 2
        return h_val, G_h

    def _adj(w):
        W = (w[:d * d] - w[d * d:]).reshape(d, d)
        np.fill_diagonal(W, 0.0)  # a variable cannot be its own parent; without
        # this the acyclicity penalty alone does not reliably suppress
        # self-loops within max_iter, letting a self-loop silently absorb
        # fitting capacity that should go to real edges (found via a raw,
        # unthresholded weight matrix showing W[i,i]=0.99 on real data).
        return W

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h_val, G_h = _h(W)
        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h_val + alpha) * G_h
        g_obj = np.concatenate((G_smooth, -G_smooth), axis=None) + lambda1
        return obj, g_obj.reshape(-1)

    bnds = [(0, None)] * (2 * d * d)
    for it in range(max_iter):
        while rho < rho_max:
            sol = minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est = w_new
        h = h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W


def notears_discover(df: pd.DataFrame, columns: list[str], lambda1: float = 0.05,
                      w_threshold: float = 0.15) -> nx.DiGraph:
    X = df[columns].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    W = notears_linear(X, lambda1=lambda1, w_threshold=w_threshold)
    g = nx.DiGraph()
    g.add_nodes_from(columns)
    d = len(columns)
    for i in range(d):
        for j in range(d):
            if W[i, j] != 0:
                g.add_edge(columns[i], columns[j])
    # enforce domain-knowledge root constraint: drop any edge pointing INTO
    # a known exogenous/action root (these cannot have causal parents)
    for u, v in list(g.edges()):
        if v in ROOT_VARS:
            g.remove_edge(u, v)
    if not nx.is_directed_acyclic_graph(g):
        g = _break_cycles(g)
    return g


def _break_cycles(g: nx.DiGraph) -> nx.DiGraph:
    g = g.copy()
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        g.remove_edge(*cycle[-1][:2])
    return g


# --------------------------------------------------------------------------
# PC algorithm via causal-learn, with root-variable orientation prior
# --------------------------------------------------------------------------

def pc_discover(df: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> nx.DiGraph:
    from causallearn.search.ConstraintBased.PC import pc as pc_search

    X = df[columns].to_numpy(dtype=float)
    cg = pc_search(X, alpha=alpha, indep_test="fisherz", show_progress=False,
                    node_names=columns)
    g = nx.DiGraph()
    g.add_nodes_from(columns)
    for edge in cg.G.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        e1, e2 = edge.get_endpoint1(), edge.get_endpoint2()
        # ARROW at n2 side & TAIL at n1 side => n1 -> n2
        from causallearn.graph.Endpoint import Endpoint
        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            _add_oriented(g, n1, n2)
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            _add_oriented(g, n2, n1)
        else:
            # undirected/bidirected: use domain prior (root vars can only be sources)
            if n1 in ROOT_VARS and n2 not in ROOT_VARS:
                g.add_edge(n1, n2)
            elif n2 in ROOT_VARS and n1 not in ROOT_VARS:
                g.add_edge(n2, n1)
            else:
                g.add_edge(n1, n2)
    for u, v in list(g.edges()):
        if v in ROOT_VARS:
            g.remove_edge(u, v)
    if not nx.is_directed_acyclic_graph(g):
        g = _break_cycles(g)
    return g


def _add_oriented(g, src, dst):
    if dst in ROOT_VARS and src not in ROOT_VARS:
        g.add_edge(dst, src)
    else:
        g.add_edge(src, dst)


# --------------------------------------------------------------------------
# Metrics: SHD and approximate SID
# --------------------------------------------------------------------------

def shd(g_true: nx.DiGraph, g_est: nx.DiGraph) -> int:
    nodes = list(g_true.nodes())
    true_edges = set(g_true.edges())
    est_edges = set(g_est.edges())
    dist = 0
    checked = set()
    for u in nodes:
        for v in nodes:
            if u == v or (v, u) in checked:
                continue
            checked.add((u, v))
            t_fwd, t_bwd = (u, v) in true_edges, (v, u) in true_edges
            e_fwd, e_bwd = (u, v) in est_edges, (v, u) in est_edges
            if (t_fwd, t_bwd) == (e_fwd, e_bwd):
                continue
            dist += 1
    return dist


def approx_sid(g_true: nx.DiGraph, g_est: nx.DiGraph) -> int:
    """Approximate SID: for each ordered pair (i,j), i != j, check whether
    the descendant-set-based test for 'does do(i) affect j' agrees between
    true and estimated graphs. This mirrors the definition of SID for DAGs
    (an intervention on i affects j iff j is a descendant of i), without
    requiring the R `SID` package.
    """
    nodes = list(g_true.nodes())
    desc_true = {n: nx.descendants(g_true, n) for n in nodes}
    desc_est = {n: nx.descendants(g_est, n) for n in nodes}
    mismatches = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if (j in desc_true[i]) != (j in desc_est[i]):
                mismatches += 1
    return mismatches


def evaluate_discovery(g_true: nx.DiGraph, g_est: nx.DiGraph) -> dict:
    true_edges = set(g_true.edges())
    est_edges = set(g_est.edges())
    tp = len(true_edges & est_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        shd=shd(g_true, g_est),
        sid=approx_sid(g_true, g_est),
        precision=precision, recall=recall, f1=f1,
        n_true_edges=len(true_edges), n_est_edges=len(est_edges),
    )
