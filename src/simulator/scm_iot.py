# Author: abdkhan
"""
Synthetic Structural Causal Model (SCM) for an autonomous IoT edge cluster.

Ground-truth causal DAG over IoT QoS telemetry, used because no public
real-world IoT dataset exposes a verified causal graph (needed for
SHD/SID evaluation of causal discovery, and for exact do-calculus
ground truth when evaluating counterfactual predictions).

Node semantics (an IoT edge node, e.g. a LoRa/Wi-Fi/5G-connected sensor
gateway under a duty-cycle / tx-power control policy):

    Exogenous / slow drift:
        interference      U_I(t)  ambient RF interference level
        link_quality      U_L(t)  slowly varying channel quality
        arrival_rate      U_A(t)  application-layer packet arrival rate

    Controllable (actions; the RL controller / do-operator acts here):
        duty_cycle        fraction of time radio is active   [0,1]
        tx_power          transmit power level                [0,1]

    Endogenous (caused by the above):
        collision_rate    = f(interference, duty_cycle, tx_power, link_quality)
        retransmissions   = f(collision_rate, tx_power)
        queue_length      = f(retransmissions, duty_cycle, arrival_rate)
        cpu_load          = f(queue_length, duty_cycle)
        latency           = f(retransmissions, queue_length, cpu_load)
        throughput        = f(duty_cycle, collision_rate, retransmissions)
        packet_loss       = f(collision_rate, retransmissions, queue_length)
        energy            = f(tx_power, duty_cycle, cpu_load, retransmissions)
        battery           = state variable, integrates -energy over time

Non-stationarity: at random "regime shift" points the noise/interference
distribution changes, mimicking drift in an autonomous deployment.

All structural equations are known analytically, so do(...) interventions
can be simulated exactly -- this is the ground truth used to check
(a) causal discovery structural accuracy (SHD/SID against GROUND_TRUTH_DAG)
(b) whether a downstream predictor (SCM-fit or tiny-LLM) gets the
    interventional/counterfactual outcome right.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import networkx as nx

NODES = [
    "interference", "link_quality", "arrival_rate",   # exogenous (0,1,2)
    "duty_cycle", "tx_power",                          # actions   (3,4)
    "collision_rate", "retransmissions", "queue_length", "cpu_load",
    "latency", "throughput", "packet_loss", "energy", "battery",
]

EXOGENOUS = ["interference", "link_quality", "arrival_rate"]
ACTIONS = ["duty_cycle", "tx_power"]
ENDOGENOUS = [n for n in NODES if n not in EXOGENOUS and n not in ACTIONS]

GROUND_TRUTH_EDGES = [
    ("interference", "collision_rate"),
    ("duty_cycle", "collision_rate"),
    ("tx_power", "collision_rate"),
    ("link_quality", "collision_rate"),
    ("collision_rate", "retransmissions"),
    ("tx_power", "retransmissions"),
    ("retransmissions", "queue_length"),
    ("duty_cycle", "queue_length"),
    ("arrival_rate", "queue_length"),
    ("queue_length", "cpu_load"),
    ("duty_cycle", "cpu_load"),
    ("retransmissions", "latency"),
    ("queue_length", "latency"),
    ("cpu_load", "latency"),
    ("duty_cycle", "throughput"),
    ("collision_rate", "throughput"),
    ("retransmissions", "throughput"),
    ("collision_rate", "packet_loss"),
    ("retransmissions", "packet_loss"),
    ("queue_length", "packet_loss"),
    ("tx_power", "energy"),
    ("duty_cycle", "energy"),
    ("cpu_load", "energy"),
    ("retransmissions", "energy"),
    ("energy", "battery"),
]


def ground_truth_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(GROUND_TRUTH_EDGES)
    assert nx.is_directed_acyclic_graph(g)
    return g


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclasses.dataclass
class RegimeParams:
    """Coefficients for the structural equations; can be perturbed to
    simulate non-stationary drift (a new regime = new RegimeParams)."""
    interference_mean: float = 0.3
    interference_std: float = 0.15
    link_quality_mean: float = 0.7
    link_quality_std: float = 0.1
    arrival_rate_mean: float = 0.4
    arrival_rate_std: float = 0.1
    noise_scale: float = 0.03


class IoTClusterSCM:
    """Simulates one autonomous IoT edge cluster's telemetry stream.

    state persists battery level across steps; everything else is a
    (near-)Markovian function of the current exogenous draws + actions.
    """

    def __init__(self, seed: int = 0, regime: RegimeParams | None = None,
                 regime_shift_prob: float = 0.002):
        self.rng = np.random.default_rng(seed)
        self.regime = regime or RegimeParams()
        self.regime_shift_prob = regime_shift_prob
        self.battery = 1.0  # normalized, 1.0 = full
        self.t = 0

    def maybe_shift_regime(self):
        if self.rng.random() < self.regime_shift_prob:
            r = self.regime
            self.regime = RegimeParams(
                interference_mean=float(np.clip(r.interference_mean + self.rng.normal(0, 0.15), 0.05, 0.9)),
                interference_std=r.interference_std,
                link_quality_mean=float(np.clip(r.link_quality_mean + self.rng.normal(0, 0.1), 0.2, 0.95)),
                link_quality_std=r.link_quality_std,
                arrival_rate_mean=float(np.clip(r.arrival_rate_mean + self.rng.normal(0, 0.1), 0.1, 0.9)),
                arrival_rate_std=r.arrival_rate_std,
                noise_scale=r.noise_scale,
            )
            return True
        return False

    def sample_exogenous(self):
        r = self.regime
        interference = float(np.clip(self.rng.normal(r.interference_mean, r.interference_std), 0, 1))
        link_quality = float(np.clip(self.rng.normal(r.link_quality_mean, r.link_quality_std), 0, 1))
        arrival_rate = float(np.clip(self.rng.normal(r.arrival_rate_mean, r.arrival_rate_std), 0, 1))
        return dict(interference=interference, link_quality=link_quality, arrival_rate=arrival_rate)

    def structural_equations(self, exo: dict, actions: dict, noise_scale: float,
                              do: dict | None = None) -> dict:
        """Compute all endogenous variables given exogenous draws + actions.

        `do`: optional dict of {var: value} hard interventions (do-operator),
        overriding the structural equation for that variable entirely.
        """
        do = do or {}
        n = lambda: self.rng.normal(0, noise_scale)

        v = dict(exo)
        v.update(actions)
        # do() can override exogenous/action variables directly, before any
        # endogenous equation is evaluated downstream of them.
        for name in EXOGENOUS + ACTIONS:
            if name in do:
                v[name] = do[name]

        def set_var(name, value):
            v[name] = do[name] if name in do else float(np.clip(value, 0, 1))

        set_var("collision_rate", _sig(
            3.0 * v["interference"] + 2.5 * v["duty_cycle"] - 1.5 * v["tx_power"]
            - 2.0 * v["link_quality"] - 1.0 + n()))

        set_var("retransmissions", _sig(
            3.5 * v["collision_rate"] - 1.2 * v["tx_power"] - 0.5 + n()))

        set_var("queue_length", _sig(
            2.5 * v["retransmissions"] + 1.5 * v["arrival_rate"]
            - 1.0 * v["duty_cycle"] - 0.8 + n()))

        set_var("cpu_load", _sig(
            2.0 * v["queue_length"] + 1.0 * v["duty_cycle"] - 1.2 + n()))

        set_var("latency", _sig(
            2.2 * v["retransmissions"] + 2.0 * v["queue_length"]
            + 1.0 * v["cpu_load"] - 1.8 + n()))

        set_var("throughput", _sig(
            2.0 * v["duty_cycle"] - 1.8 * v["collision_rate"]
            - 1.5 * v["retransmissions"] + 0.3 + n()))

        set_var("packet_loss", _sig(
            2.5 * v["collision_rate"] + 1.8 * v["retransmissions"]
            + 1.0 * v["queue_length"] - 2.5 + n()))

        set_var("energy", _sig(
            2.0 * v["tx_power"] + 1.8 * v["duty_cycle"]
            + 1.0 * v["cpu_load"] + 0.8 * v["retransmissions"] - 1.5 + n()))

        return v

    def step(self, actions: dict, do: dict | None = None) -> dict:
        self.maybe_shift_regime()
        exo = self.sample_exogenous()
        v = self.structural_equations(exo, actions, self.regime.noise_scale, do=do)
        self.battery = float(np.clip(self.battery - 0.01 * v["energy"] + 0.0005, 0, 1))
        v["battery"] = do.get("battery", self.battery) if do else self.battery
        v["t"] = self.t
        self.t += 1
        return v

    def rollout(self, n_steps: int, policy=None, seed_actions: dict | None = None):
        """policy: callable(state)->actions dict, or None to use random/default actions."""
        rows = []
        for _ in range(n_steps):
            if policy is not None:
                actions = policy(self)
            elif seed_actions is not None:
                actions = seed_actions
            else:
                actions = dict(
                    duty_cycle=float(self.rng.uniform(0.1, 0.9)),
                    tx_power=float(self.rng.uniform(0.1, 0.9)),
                )
            rows.append(self.step(actions))
        return rows

    def counterfactual(self, actions: dict, do: dict, n_samples: int = 200):
        """Monte-Carlo estimate of E[outcome | do(...)] under the *current*
        regime, by resampling exogenous noise many times (ground truth for
        evaluating a distilled model's do-operator answers)."""
        outs = []
        saved_state = self.rng.bit_generator.state
        for _ in range(n_samples):
            exo = self.sample_exogenous()
            v = self.structural_equations(exo, actions, self.regime.noise_scale, do=do)
            outs.append(v)
        self.rng.bit_generator.state = saved_state
        keys = outs[0].keys()
        return {k: float(np.mean([o[k] for o in outs])) for k in keys}
