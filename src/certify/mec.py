# Author: abdkhan
"""A second, structurally different plant: a mobile-edge-computing (MEC)
offloading node.

An IoT device receives computation tasks and decides, every control tick
(1 s), what fraction of them to offload to an edge server and at which
frequency to clock its own CPU. Nothing here is a sigmoid SCM: the plant is
built from standard analytic models,

  * local execution: an M/M/1/K queue whose service rate scales with the
    DVFS frequency, with dynamic CPU power kappa * f^3 (cubic DVFS law);
  * uplink: Shannon rate W log2(1 + SNR g) (W = 5 MHz, 2 Mbit tasks) with
    block Rayleigh fading g
    (Gamma(16, 1/16): the per-tick average of 16 coherence blocks), feeding
    an M/M/1/K transmit queue at a fixed radio power;
  * edge server: an M/M/1/K queue (20 tasks/s) whose capacity shrinks with the
    server's background load;
  * blocking of an M/M/1/K queue gives the dropped-task share.

Actuators (normalized to [0.02, 0.98]):
  offload   fraction of arriving tasks sent to the edge
  cpu_freq  local CPU frequency, f_GHz = 0.8 + 1.7 * cpu_freq
Exogenous drivers (measured, normalized): arrival (task rate), channel
(mean SNR indicator), server_load (edge background utilization).
Measured channels: latency (normalized mean task latency), energy
(normalized device power), drop (share of tasks blocked).

The operator's cost is J = latency + 0.8 energy + 2 drop. Its optimum is
interior in both actuators and moves with the regime: more offloading on a
good channel and an idle server, a faster clock under heavy arrivals.
"""
from __future__ import annotations

import dataclasses

import numpy as np

ACTUATORS = ["offload", "cpu_freq"]
EXO = ["arrival", "channel", "server_load"]
OBJECTIVES = ["latency", "energy", "drop"]
COST_W = dict(latency=1.0, energy=0.8, drop=2.0)
K_BUF = 3           # queue capacity (tasks) of each M/M/1/K stage
# physical constants (see simulate_vec)
PHYS = dict(lam0=2.0, lam1=8.0, f0=0.8, f1=1.7, gcyc=0.25, bw=5.0, mbit=2.0, mu_edge=20.0,
            lat_norm=2.5, p_radio=1.2, kappa=0.25)


def operator_cost(v: dict):
    w = COST_W
    return w["latency"] * v["latency"] + w["energy"] * v["energy"] + w["drop"] * v["drop"]


def mm1k(lam, mu, K=None):
    """M/M/1/K: mean sojourn time of admitted tasks, blocking probability and
    busy probability, vectorised and smooth through rho = 1."""
    K = K_BUF if K is None else K
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    rho = np.maximum(lam / mu, 1e-9)
    near = np.abs(rho - 1.0) < 1e-6
    r = np.where(near, 1.0 + 1e-6, rho)
    rK1 = r ** (K + 1)
    p0 = (1.0 - r) / (1.0 - rK1)
    n_mean = r / (1.0 - r) - (K + 1) * rK1 / (1.0 - rK1)
    pK = p0 * r ** K
    lam_eff = np.maximum(lam * (1.0 - pK), 1e-12)
    T = np.where(lam > 1e-9, n_mean / lam_eff, 1.0 / mu)
    return T, pK, 1.0 - p0


@dataclasses.dataclass
class MECRegime:
    arrival_mean: float = 0.5
    arrival_std: float = 0.12
    channel_mean: float = 0.55
    channel_std: float = 0.12
    load_mean: float = 0.4
    load_std: float = 0.1
    noise_scale: float = 0.015


def simulate_vec(regime: MECRegime, actions: dict, n: int, rng: np.random.Generator,
                 exo: dict | None = None) -> dict:
    """n i.i.d. ticks of the plant at `actions` (scalars or length-n arrays)."""
    r = regime
    # The device measures its task arrivals, its realized uplink SNR (block
    # Rayleigh fading averaged over 16 coherence blocks per tick, folded into
    # the channel indicator) and the server's advertised load; none of them
    # is caused by its own actions, so all three are exogenous drivers.
    h = np.clip(rng.normal(r.channel_mean, r.channel_std, n), 0, 1)
    fade = rng.gamma(16.0, 1.0 / 16.0, n)
    drawn = dict(
        arrival=np.clip(rng.normal(r.arrival_mean, r.arrival_std, n), 0, 1),
        channel=np.clip(h + 10.0 * np.log10(fade) / 22.0, 0, 1),
        server_load=np.clip(rng.normal(r.load_mean, r.load_std, n), 0, 1))
    exo = drawn if exo is None else exo
    ns = r.noise_scale
    burst = np.exp(rng.normal(0.0, 0.03, n))           # unmeasured arrival burstiness
    size = np.exp(rng.normal(0.0, 0.05, n))            # unmeasured task-size variation
    meas = rng.normal(0.0, ns, (3, n))                 # measurement noise
    c = PHYS
    x = np.asarray(actions["offload"], dtype=float)
    f = c["f0"] + c["f1"] * np.asarray(actions["cpu_freq"], dtype=float)
    lam = (c["lam0"] + c["lam1"] * exo["arrival"]) * burst   # tasks / s
    # local execution: gcyc Gcycles per task
    mu_l = f / (c["gcyc"] * size)
    T_l, P_l, busy_l = mm1k(lam * (1.0 - x), mu_l)
    # uplink: 5 MHz, SNR from -2 to 20 dB, 2 Mbit per task
    snr = 10.0 ** ((-2.0 + 22.0 * exo["channel"]) / 10.0)
    rate = c["bw"] * np.log2(1.0 + snr)                # Mbit / s
    mu_u = rate / (c["mbit"] * size)
    T_u, P_u, busy_u = mm1k(lam * x, mu_u)
    # edge server: 20 tasks / s, minus background load
    mu_e = c["mu_edge"] * (1.0 - 0.8 * exo["server_load"])
    T_e, P_e, _ = mm1k(lam * x * (1.0 - P_u), mu_e)
    T_off = T_u + T_e + 0.01
    lat = (1.0 - x) * T_l * (1.0 - P_l) + x * T_off * (1.0 - P_u)
    drop = (1.0 - x) * P_l + x * (1.0 - (1.0 - P_u) * (1.0 - P_e))
    power = c["kappa"] * f ** 3 * busy_l + 0.08 + c["p_radio"] * busy_u
    out = dict(exo, offload=x, cpu_freq=np.asarray(actions["cpu_freq"], dtype=float))
    out["latency"] = np.clip(lat / c["lat_norm"] + meas[0], 0.0, 1.0)
    out["energy"] = np.clip(power / 4.5 + meas[1], 0.0, 1.0)
    out["drop"] = np.clip(drop + 0.3 * meas[2], 0.0, 1.0)
    return out


class MECPlant:
    """Stochastic MEC node; regimes (arrival, channel, server load) shift at
    random times, as in the SCM."""

    def __init__(self, seed: int = 0, regime: MECRegime | None = None,
                 regime_shift_prob: float = 0.002, noise_mult: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.regime = regime or MECRegime()
        self.regime.noise_scale *= noise_mult
        self.regime_shift_prob = regime_shift_prob
        self.t = 0

    def maybe_shift_regime(self):
        if self.rng.random() < self.regime_shift_prob:
            r = self.regime
            self.regime = MECRegime(
                arrival_mean=float(np.clip(r.arrival_mean + self.rng.normal(0, 0.15), 0.1, 0.9)),
                arrival_std=r.arrival_std,
                channel_mean=float(np.clip(r.channel_mean + self.rng.normal(0, 0.12), 0.15, 0.9)),
                channel_std=r.channel_std,
                load_mean=float(np.clip(r.load_mean + self.rng.normal(0, 0.12), 0.05, 0.85)),
                load_std=r.load_std, noise_scale=r.noise_scale)
            return True
        return False

    def step(self, actions: dict, do=None) -> dict:
        self.maybe_shift_regime()
        v = simulate_vec(self.regime, {a: float(actions[a]) for a in ACTUATORS}, 1, self.rng)
        out = {k: float(np.asarray(val).reshape(-1)[0]) for k, val in v.items()}
        out["battery"] = 1.0
        out["t"] = self.t
        self.t += 1
        return out
