# Author: abdkhan
"""Plant 3: smart-inverter Volt/VAR control on the IEEE 33-bus distribution
feeder (Baran & Wu, 1989), taken from pandapower (pandapower.networks.case33bw).

Two PV inverters, at the ends of the main feeder and of the longest lateral
(buses 18 and 33 in the usual 1-based numbering; 17 and 32 here), are the
smart-grid IoT devices whose reactive-power setpoints a distribution
management system (DMS) proposes. The DMS sees the feeder only through
metering telemetry that an attacker can forge.

Actuators (normalized to [0.02, 0.98]):
  q18, q33   inverter reactive power Q = QMIN + (QMAX - QMIN) u, mostly injection
             (the feeder is voltage-limited), from -0.2 to 1.2 MVAr
Exogenous drivers (measured, normalized to [0, 1]):
  demand     system load level (feeder-head forecast / AMI aggregate)
  irradiance plane-of-array irradiance at the PV sites
  qshare     reactive share of the load (power factor of the feeder)
Measured channels (normalized, forgeable):
  loss       feeder losses computed by the DMS from the substation meter and
             the AMI and PV meters (substation import - metered consumption
             + metered PV), with metering error
  v18, v33   inverter terminal voltages reported by the inverters
Operator cost (affine in the measured channels, eq. (1) of the paper):
  J = W_LOSS loss + W_V ((1 - v18) + (1 - v33)) + W_Q (Q18^2 + Q33^2)

Power flow: a vectorised backward/forward sweep (BFS) on the radial feeder,
exact for the AC power-flow equations with constant-power loads; it agrees
with pandapower's Newton-Raphson solver to 1e-8 p.u. (tests/test_grid.py).
Every draw has its own loads, PV output and metering noise, so the harness's
Monte-Carlo ground truth runs thousands of power flows per call.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache

import numpy as np

ACTUATORS = ["q18", "q33"]
EXO = ["demand", "irradiance", "qshare"]
OBJECTIVES = ["loss", "v18", "v33"]
INV_BUS = (17, 32)            # 0-based bus indices of the two inverters
QMIN, QMAX = -0.2, 1.2        # MVAr per inverter
PV_RATED = 0.8                # MW per PV plant
# normalization of the measured channels to [0, 1]
LOSS_LO, LOSS_HI = 0.0, 1.5   # MW (wide enough that no reachable state clips)
import os as _os
if _os.environ.get("KIC_GRID_DOMAIN") == "1":
    # certified-domain configuration: the loss telemetry range starts below 0 so
    # that the channel provably never clips (grid_certify_run.py)
    LOSS_LO = -0.05
V_LO, V_HI = 0.70, 1.15       # p.u.
COST_W = dict(loss=6.0, v=4.0, q=0.35)
NOISE = dict(load=0.03, pv=0.03, meter_loss=0.0015, meter_v=0.0005)   # relative / MW / p.u.


@lru_cache(maxsize=1)
def feeder():
    """Radial topology and per-unit data of case33bw (in-service lines)."""
    import pandapower.networks as pn
    net = pn.case33bw()
    vn = float(net.bus.vn_kv.iloc[0])
    sbase = 1.0                                          # MVA
    zbase = vn ** 2 / sbase
    lines = net.line[net.line.in_service]
    nb = len(net.bus)
    parent = -np.ones(nb, dtype=int)
    z = np.zeros(nb, dtype=complex)
    adj = {i: [] for i in range(nb)}
    for _, l in lines.iterrows():
        f, t = int(l.from_bus), int(l.to_bus)
        zz = (l.r_ohm_per_km + 1j * l.x_ohm_per_km) * l.length_km / zbase
        adj[f].append((t, zz))
        adj[t].append((f, zz))
    root = int(net.ext_grid.bus.iloc[0])
    order, seen = [root], {root}
    k = 0
    while k < len(order):
        u = order[k]
        for v, zz in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v] = u
                z[v] = zz
                order.append(v)
        k += 1
    p = np.zeros(nb)
    q = np.zeros(nb)
    for _, ld in net.load.iterrows():
        p[int(ld.bus)] += ld.p_mw
        q[int(ld.bus)] += ld.q_mvar
    return dict(nb=nb, root=root, parent=parent, z=z, order=np.array(order), p=p / sbase, q=q / sbase,
                vm_ext=float(net.ext_grid.vm_pu.iloc[0]), net=net)


@lru_cache(maxsize=1)
def _paths():
    """Path matrices of the radial feeder: DOWN[b, k] = 1 if bus k is in the
    subtree of bus b (so the current of branch b is I_inj @ DOWN.T), and
    UP[k, b] = 1 if branch b (indexed by its child bus) lies on the path from
    the root to bus k (so V = V0 - (z * I_branch) @ UP.T)."""
    F = feeder()
    nb, parent = F["nb"], F["parent"]
    UP = np.zeros((nb, nb))
    for k in range(nb):
        b = k
        while parent[b] >= 0:
            UP[k, b] = 1.0
            b = parent[b]
    DOWN = UP.T.copy()
    for b in range(nb):
        DOWN[b, b] = 1.0
    DOWN[F["root"], :] = 0.0          # the root has no branch
    return DOWN, UP


def bfs(s_load, vm0=1.0, iters=40, tol=1e-10):
    """Backward/forward sweep power flow in matrix form. s_load: (n, nb)
    complex net load (p.u., positive = consumption). Returns complex voltages
    (n, nb) and branch currents (n, nb) (branch indexed by its child bus)."""
    F = feeder()
    DOWN, UP = _paths()
    z = F["z"]
    n, nb = s_load.shape
    V = np.full((n, nb), vm0 + 0j)
    Dt, Ut = np.ascontiguousarray(DOWN.T), np.ascontiguousarray(UP.T)
    cmm = lambda A, M: (np.ascontiguousarray(A.real) @ M) + 1j * (np.ascontiguousarray(A.imag) @ M)   # real BLAS
    if n > 8:
        # warm start from the solution at the mean load: the sweep converges to the
        # same fixed point (to tol) in about half the iterations
        V0, _ = bfs(np.mean(s_load, 0, keepdims=True), vm0, iters, tol)
        V = np.repeat(V0, n, 0)
    for _ in range(iters):
        Ib = cmm(np.conj(s_load / V), Dt)
        Vn = vm0 - cmm(Ib * z[None, :], Ut)
        done = np.max(np.abs(Vn - V)) < tol
        V = Vn
        if done:
            break
    Ib = cmm(np.conj(s_load / V), Dt)
    return V, Ib


@dataclasses.dataclass
class GridRegime:
    demand_mean: float = 0.55
    demand_std: float = 0.08
    irr_mean: float = 0.5
    irr_std: float = 0.12
    qshare_mean: float = 0.5
    qshare_std: float = 0.08
    noise_scale: float = 1.0
    # certified operating domain D (grid_certify_run.py): per-tick demand and
    # reactive share confined to [dmin, dmax] and [0, qmax]; the defaults leave
    # the plant unchanged
    dmin: float = 0.0
    dmax: float = 1.0
    qmax: float = 1.0


def _q_of(u):
    return QMIN + (QMAX - QMIN) * np.asarray(u, dtype=float)


def simulate_vec(regime: GridRegime, actions: dict, n: int, rng: np.random.Generator,
                 exo: dict | None = None) -> dict:
    """n i.i.d. ticks at `actions` (scalars or length-n arrays)."""
    r = regime
    F = feeder()
    drawn = dict(
        demand=np.clip(rng.normal(r.demand_mean, r.demand_std, n), r.dmin, r.dmax),
        irradiance=np.clip(rng.normal(r.irr_mean, r.irr_std, n), 0, 1),
        qshare=np.clip(rng.normal(r.qshare_mean, r.qshare_std, n), 0, r.qmax))
    exo = drawn if exo is None else exo
    ns = NOISE
    k = r.noise_scale
    nb = F["nb"]
    lvl = 0.5 + 0.8 * np.asarray(exo["demand"])                  # load multiplier 0.5 .. 1.3
    qm = 0.6 + 0.8 * np.asarray(exo["qshare"])                   # reactive multiplier 0.6 .. 1.4
    e_p = 1.0 + k * ns["load"] * rng.standard_normal((n, nb))
    e_q = 1.0 + k * ns["load"] * rng.standard_normal((n, nb))
    P = F["p"][None, :] * lvl[:, None] * e_p
    Q = F["q"][None, :] * (lvl * qm)[:, None] * e_q
    pv = PV_RATED * np.asarray(exo["irradiance"])[:, None] * (1.0 + k * ns["pv"] * rng.standard_normal((n, 2)))
    q_inv = np.stack([np.broadcast_to(_q_of(actions[a]), (n,)) for a in ACTUATORS], 1)
    for j, b in enumerate(INV_BUS):
        P[:, b] -= pv[:, j]
        Q[:, b] -= q_inv[:, j]
    V, Ib = bfs(P + 1j * Q, F["vm_ext"])
    zr = np.real(F["z"])
    loss = np.sum(zr[None, :] * np.abs(Ib) ** 2, axis=1)       # p.u. = MW (sbase 1 MVA)
    vm = np.abs(V)
    meas_loss = loss + k * ns["meter_loss"] * rng.standard_normal(n)
    out = dict(exo)
    for a in ACTUATORS:
        out[a] = np.broadcast_to(np.asarray(actions[a], dtype=float), (n,))
    out["loss"] = np.clip((meas_loss - LOSS_LO) / (LOSS_HI - LOSS_LO), 0.0, 1.0)
    for name, b in (("v18", INV_BUS[0]), ("v33", INV_BUS[1])):
        out[name] = np.clip((vm[:, b] + k * ns["meter_v"] * rng.standard_normal(n) - V_LO) / (V_HI - V_LO), 0.0, 1.0)
    out["_vm"] = vm                                             # full voltage profile (side-channel studies)
    out["_qinv"] = q_inv
    return out


def operator_cost(v: dict):
    w = COST_W
    q18, q33 = _q_of(v["q18"]), _q_of(v["q33"])
    vv = lambda c: V_LO + (V_HI - V_LO) * np.asarray(v[c])
    return (w["loss"] * (LOSS_LO + (LOSS_HI - LOSS_LO) * np.asarray(v["loss"]))
            + w["v"] * ((1.0 - vv("v18")) + (1.0 - vv("v33")))
            + w["q"] * (q18 ** 2 + q33 ** 2))


class GridPlant:
    """Stochastic feeder; the regime (demand, irradiance, reactive share)
    shifts at random times, as in the other plants."""

    def __init__(self, seed: int = 0, regime: GridRegime | None = None, regime_shift_prob: float = 0.002,
                 noise_mult: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.regime = regime or GridRegime()
        self.regime.noise_scale *= noise_mult
        self.regime_shift_prob = regime_shift_prob
        self.t = 0

    def maybe_shift_regime(self):
        if self.rng.random() < self.regime_shift_prob:
            r = self.regime
            self.regime = GridRegime(
                demand_mean=float(np.clip(r.demand_mean + self.rng.normal(0, 0.15), 0.15, 0.9)),
                irr_mean=float(np.clip(r.irr_mean + self.rng.normal(0, 0.2), 0.05, 0.95)),
                qshare_mean=float(np.clip(r.qshare_mean + self.rng.normal(0, 0.12), 0.15, 0.85)),
                demand_std=r.demand_std, irr_std=r.irr_std, qshare_std=r.qshare_std, noise_scale=r.noise_scale)
            return True
        return False

    def step(self, actions: dict, do=None) -> dict:
        self.maybe_shift_regime()
        v = simulate_vec(self.regime, {a: float(actions[a]) for a in ACTUATORS}, 1, self.rng)
        out = {k: float(np.asarray(val).reshape(-1)[0]) for k, val in v.items() if not k.startswith("_")}
        # true bus voltages (p.u.): what a meter at bus i would read before its own noise;
        # used only by the side-channel attacker, never by a gate
        for i, vm in enumerate(np.asarray(v["_vm"])[0]):
            out[f"vb{i}"] = float(vm)
        out["battery"] = 1.0
        out["t"] = self.t
        self.t += 1
        return out
