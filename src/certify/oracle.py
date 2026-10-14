# Author: abdkhan
"""Ground truth for scoring gates, and the reference-graph sign rule.

Everything here reads the TRUE simulator and is used only to score
decisions, never to make them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.scm_iot import IoTClusterSCM, RegimeParams, GROUND_TRUTH_EDGES
from rl_controller.env import compute_reward

# Signs of the structural coefficients in scm_iot.structural_equations. The
# sigmoid link is monotone, so these are the true per-edge signs.
TRUE_EDGE_SIGNS = {
    ("interference", "collision_rate"): 1, ("duty_cycle", "collision_rate"): 1,
    ("tx_power", "collision_rate"): -1, ("link_quality", "collision_rate"): -1,
    ("collision_rate", "retransmissions"): 1, ("tx_power", "retransmissions"): -1,
    ("retransmissions", "queue_length"): 1, ("duty_cycle", "queue_length"): -1,
    ("arrival_rate", "queue_length"): 1, ("queue_length", "cpu_load"): 1,
    ("duty_cycle", "cpu_load"): 1, ("retransmissions", "latency"): 1,
    ("queue_length", "latency"): 1, ("cpu_load", "latency"): 1,
    ("duty_cycle", "throughput"): 1, ("collision_rate", "throughput"): -1,
    ("retransmissions", "throughput"): -1, ("collision_rate", "packet_loss"): 1,
    ("retransmissions", "packet_loss"): 1, ("queue_length", "packet_loss"): 1,
    ("tx_power", "energy"): 1, ("duty_cycle", "energy"): 1,
    ("cpu_load", "energy"): 1, ("retransmissions", "energy"): 1,
    ("energy", "battery"): -1,
}
assert set(TRUE_EDGE_SIGNS) == set(GROUND_TRUTH_EDGES)


def objective_value(v: dict, objective: str) -> float:
    if objective == "reward":
        return compute_reward(v)
    if objective == "cost":            # operator's scalar objective, minimised
        return operator_cost(v)
    return v[objective]


# Operator cost for the certification experiments. The original reward is
# optimised with tx_power pinned at its upper limit in every regime, which is
# the saturation regime the old verifier was dominated by. A spectrum-
# emission penalty on tx_power and a stronger throughput term put the
# optimum in the interior for most regimes and make it MOVE with the
# regime, so accepting or rejecting a proposal changes the trajectory.
COST_W = dict(latency=1.0, energy=0.6, packet_loss=1.0, throughput=2.5, tx2=0.7)


def operator_cost(v: dict):
    w = COST_W
    return (w["latency"] * v["latency"] + w["energy"] * v["energy"] +
            w["packet_loss"] * v["packet_loss"] - w["throughput"] * v["throughput"] +
            w["tx2"] * v["tx_power"] ** 2)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_vec(regime: RegimeParams, actions: dict, n: int, rng: np.random.Generator,
                 exo: dict | None = None) -> dict:
    """Vectorised copy of IoTClusterSCM.structural_equations (checked against
    it in tests); `actions` values may be scalars or length-n arrays."""
    r = regime
    if exo is None:
        exo = dict(
            interference=np.clip(rng.normal(r.interference_mean, r.interference_std, n), 0, 1),
            link_quality=np.clip(rng.normal(r.link_quality_mean, r.link_quality_std, n), 0, 1),
            arrival_rate=np.clip(rng.normal(r.arrival_rate_mean, r.arrival_rate_std, n), 0, 1))
    ns = r.noise_scale
    nz = rng.normal(0, ns, (8, n))
    d, p = actions["duty_cycle"], actions["tx_power"]
    I, L, A = exo["interference"], exo["link_quality"], exo["arrival_rate"]
    c = lambda z: np.clip(z, 0, 1)
    col = c(_sig(3.0 * I + 2.5 * d - 1.5 * p - 2.0 * L - 1.0 + nz[0]))
    ret = c(_sig(3.5 * col - 1.2 * p - 0.5 + nz[1]))
    q = c(_sig(2.5 * ret + 1.5 * A - 1.0 * d - 0.8 + nz[2]))
    cpu = c(_sig(2.0 * q + 1.0 * d - 1.2 + nz[3]))
    lat = c(_sig(2.2 * ret + 2.0 * q + 1.0 * cpu - 1.8 + nz[4]))
    thr = c(_sig(2.0 * d - 1.8 * col - 1.5 * ret + 0.3 + nz[5]))
    pl = c(_sig(2.5 * col + 1.8 * ret + 1.0 * q - 2.5 + nz[6]))
    en = c(_sig(2.0 * p + 1.8 * d + 1.0 * cpu + 0.8 * ret - 1.5 + nz[7]))
    return dict(exo, duty_cycle=d, tx_power=p, collision_rate=col, retransmissions=ret,
                queue_length=q, cpu_load=cpu, latency=lat, throughput=thr,
                packet_loss=pl, energy=en)


def expected_outcome(regime: RegimeParams, actions: dict, objective: str,
                     n: int = 2000, seed: int = 12345, battery: float = 1.0) -> float:
    """E[objective | do(actions)] under `regime`, by Monte Carlo with a fixed
    seed so that two calls share exogenous and noise draws (common random
    numbers, which makes finite differences low-variance)."""
    v = simulate_vec(regime, actions, n, np.random.default_rng(seed))
    v["battery"] = battery
    return float(np.mean(objective_value(v, objective)))


def true_local_effect(regime: RegimeParams, centre: dict, actuator: str, objective: str,
                      delta: float, n: int = 2000) -> float:
    """(m(c+delta) - m(c-delta)) / (2 delta): the KIC estimand, exactly."""
    up = dict(centre); up[actuator] = centre[actuator] + delta
    dn = dict(centre); dn[actuator] = centre[actuator] - delta
    return (expected_outcome(regime, up, objective, n) -
            expected_outcome(regime, dn, objective, n)) / (2 * delta)


def true_step_effect(regime: RegimeParams, current: dict, proposed: dict, objective: str,
                     n: int = 2000) -> float:
    """E[obj | do(proposed)] - E[obj | do(current)]: did the move help?"""
    return (expected_outcome(regime, proposed, objective, n) -
            expected_outcome(regime, current, objective, n))


# ---------------------------------------------------------------------------
# reference-graph sign rule (the verifier of verify.py, claim-free form)
# ---------------------------------------------------------------------------

def majority_path_sign(graph: nx.DiGraph, edge_signs: dict, src: str, dst: str,
                       max_paths: int = 8, cutoff: int = 6):
    from verifier.verify import implied_sign
    return implied_sign(graph, edge_signs, src, dst, max_paths=max_paths, cutoff=cutoff)


def reward_sign_from_graph(graph, edge_signs, actuator):
    """Reward is a signed sum of QoS terms; the graph rule combines their
    majority-path signs weighted by the reward weights."""
    from rl_controller.env import REWARD_WEIGHTS
    tot = 0.0
    for q, w, s in [("latency", REWARD_WEIGHTS["latency"], -1),
                    ("energy", REWARD_WEIGHTS["energy"], -1),
                    ("packet_loss", REWARD_WEIGHTS["packet_loss"], -1),
                    ("throughput", REWARD_WEIGHTS["throughput"], 1)]:
        ps = majority_path_sign(graph, edge_signs, actuator, q)
        if ps is not None:
            tot += w * s * ps
    return None if tot == 0 else (1 if tot > 0 else -1)


def cost_sign_from_graph(graph, edge_signs, actuator):
    """Graph rule for the operator cost: weighted vote of the majority-path
    signs of each QoS term, plus the known direct emission term on tx_power
    (which a correct rule cannot get wrong)."""
    tot = 0.0
    for q, w in [("latency", COST_W["latency"]), ("energy", COST_W["energy"]),
                 ("packet_loss", COST_W["packet_loss"]), ("throughput", -COST_W["throughput"])]:
        ps = majority_path_sign(graph, edge_signs, actuator, q)
        if ps is not None:
            tot += w * ps
    if actuator == "tx_power":
        tot += COST_W["tx2"]
    return None if tot == 0 else (1 if tot > 0 else -1)


def perturb_reference(edge_signs: dict, graph: nx.DiGraph, flip: float = 0.0,
                      drop: float = 0.0, spurious: float = 0.0, seed: int = 0):
    """Misspecified reference G*: flip the sign of a fraction of edges, drop a
    fraction, and add spurious edges (as a fraction of |E|) with random sign,
    keeping the graph acyclic."""
    rng = np.random.default_rng(seed)
    g = graph.copy()
    signs = dict(edge_signs)
    edges = list(g.edges())
    for e in edges:
        r = rng.random()
        if r < drop:
            g.remove_edge(*e); signs.pop(e, None)
        elif r < drop + flip:
            signs[e] = -signs[e]
    order = list(nx.topological_sort(graph))
    n_add = int(round(spurious * len(edges)))
    tries = 0
    while n_add > 0 and tries < 1000:
        tries += 1
        i, j = sorted(rng.choice(len(order), 2, replace=False))
        u, v = order[i], order[j]
        if g.has_edge(u, v) or v in ("interference", "link_quality", "arrival_rate",
                                     "duty_cycle", "tx_power"):
            continue
        g.add_edge(u, v); signs[(u, v)] = int(rng.choice([-1, 1])); n_add -= 1
    return g, signs
