# Author: abdkhan
"""Certified envelopes for the two simulated plants (Assumption 2).

The paper's envelopes were first *characterized*: maxima over 300 sampled
regimes and actions. A finite maximum is not a bound, so this module
*proves* upper bounds by interval branch-and-bound over the whole bounded
input space, using the simulators' equations:

  G_j  >= |D| / delta for every realization: by the mean-value theorem it
          suffices to bound |d J / d u_j| over every action in the range and
          every exogenous and noise value;
  L_j  >= the Lipschitz constant of d_j m (m = E J): the Euclidean norm of
          row j of the Hessian of m. We bound the Hessian of J conditional on
          everything except the measurement noise, whose effect (clipping to
          [0, 1]) is integrated exactly; the expectation of the rest is then
          bounded by its supremum.

Unbounded (Gaussian or log-normal) noise terms are truncated at k standard
deviations. The bounds hold on the truncation event; its complement has
per-tick probability eps_k, and there the global bound Gbar (free noise)
applies, which the theorem absorbs as an extra width Gbar * eps_k (see the
supplement). V = 0 holds exactly for a plant whose regime does not change.

    python -m certify.certify_env            # writes experiments/results/kic_certified_env.json
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from certify.ival import I, Jet, f_sigmoid, f_exp, f_log, f_smoothclip

LO, HI = 0.02, 0.98


# ---------------------------------------------------------------- boxes

def grid_boxes(bounds, n):
    """Uniform partition of a box into prod(n) sub-boxes: list of (lo, hi) arrays."""
    axes = []
    for (a, b), k in zip(bounds, n):
        e = np.linspace(a, b, k + 1)
        axes.append((e[:-1], e[1:]))
    los = np.stack(np.meshgrid(*[x[0] for x in axes], indexing="ij"), -1).reshape(-1, len(bounds))
    his = np.stack(np.meshgrid(*[x[1] for x in axes], indexing="ij"), -1).reshape(-1, len(bounds))
    return los, his


def bisect(los, his):
    """Split each box along its widest (relative) dimension."""
    w = his - los
    k = np.argmax(w, axis=1)
    idx = np.arange(len(los))
    mid = 0.5 * (los[idx, k] + his[idx, k])
    lo2, hi1 = los.copy(), his.copy()
    hi1[idx, k] = mid
    lo2[idx, k] = mid
    return np.concatenate([los, lo2]), np.concatenate([hi1, his])


def branch_and_bound(bound_fn, los, his, target, max_rounds=40, min_width=1e-4, chunk=200_000,
                     log=None):
    """Upper-bound sup over the union of boxes of a quantity whose enclosure
    on a box is bound_fn(lo, hi) -> array. Boxes whose bound is <= target are
    closed; the others are bisected. Returns (certified_upper_bound,
    closed_all, n_boxes_evaluated)."""
    best_closed = -np.inf
    n_eval = 0
    for r in range(max_rounds):
        ub = np.concatenate([bound_fn(los[i:i + chunk], his[i:i + chunk])
                             for i in range(0, len(los), chunk)])
        n_eval += len(los)
        ok = ub <= target
        if ok.any():
            best_closed = max(best_closed, float(ub[ok].max()))
        los, his, ub = los[~ok], his[~ok], ub[~ok]
        if log:
            log(f"  round {r}: open {len(los)}, max open bound {ub.max() if len(ub) else float('nan'):.4f}")
        if len(los) == 0:
            return max(best_closed, -np.inf), True, n_eval
        if np.max(his - los) < min_width or len(los) > 4_000_000:
            return float(max(best_closed, ub.max())), False, n_eval
        los, his = bisect(los, his)
    return float(max(best_closed, ub.max())), False, n_eval


# ---------------------------------------------------------------- plant 1: SCM

SCM_W = dict(latency=1.0, energy=0.6, packet_loss=1.0, throughput=2.5, tx2=0.7)
SCM_SIGMA = 0.03       # noise_scale of RegimeParams (every regime)
# inputs: duty d, tx p, interference Iq, link Lq, arrival A (exogenous clipped to [0, 1])
SCM_BOUNDS = [(LO, HI), (LO, HI), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def _scm_nodes(lo, hi, k_sigma, want_jets=True):
    """Forward pass of the SCM structural equations over boxes. Returns the
    node jets (value/gradient/Hessian enclosures wrt (d, p)) and, per node,
    the enclosure of its sigmoid slope."""
    nz = I(-k_sigma * SCM_SIGMA * np.ones(len(lo)), k_sigma * SCM_SIGMA * np.ones(len(lo)))
    d = Jet.var(I(lo[:, 0], hi[:, 0]), 0)
    p = Jet.var(I(lo[:, 1], hi[:, 1]), 1)
    Iq, Lq, A = (Jet.const(I(lo[:, i], hi[:, i])) for i in (2, 3, 4))
    slopes = {}

    def node(name, z):
        z = Jet(z.v + nz, z.g, z.h)
        slopes[name] = f_sigmoid(z.v)[1]
        return z.apply(f_sigmoid)

    col = node("col", 3.0 * Iq + 2.5 * d - 1.5 * p - 2.0 * Lq - 1.0)
    ret = node("ret", 3.5 * col - 1.2 * p - 0.5)
    q = node("q", 2.5 * ret + 1.5 * A - 1.0 * d - 0.8)
    cpu = node("cpu", 2.0 * q + 1.0 * d - 1.2)
    lat = node("lat", 2.2 * ret + 2.0 * q + 1.0 * cpu - 1.8)
    thr = node("thr", 2.0 * d - 1.8 * col - 1.5 * ret + 0.3)
    pl = node("pl", 2.5 * col + 1.8 * ret + 1.0 * q - 2.5)
    en = node("en", 2.0 * p + 1.8 * d + 1.0 * cpu + 0.8 * ret - 1.5)
    w = SCM_W
    J = (w["latency"] * lat + w["energy"] * en + w["packet_loss"] * pl - w["throughput"] * thr
         + w["tx2"] * (p * p))
    return J, slopes


_VERT = np.array(list(itertools.product([0, 1], repeat=8)), dtype=float)   # 256 x 8


def _scm_grad_vertices(s, p):
    """dJ/d(d, p) for slope vectors s (..., 8) and tx value p (...); the
    expression is multilinear in the node slopes, so over a box of slopes its
    extremes are at the vertices."""
    col = s[..., 0:1] * np.array([2.5, -1.5])
    ret = s[..., 1:2] * (3.5 * col + np.array([0.0, -1.2]))
    q = s[..., 2:3] * (2.5 * ret + np.array([-1.0, 0.0]))
    cpu = s[..., 3:4] * (2.0 * q + np.array([1.0, 0.0]))
    lat = s[..., 4:5] * (2.2 * ret + 2.0 * q + 1.0 * cpu)
    thr = s[..., 5:6] * (np.array([2.0, 0.0]) - 1.8 * col - 1.5 * ret)
    pl = s[..., 6:7] * (2.5 * col + 1.8 * ret + 1.0 * q)
    en = s[..., 7:8] * (np.array([1.8, 2.0]) + 1.0 * cpu + 0.8 * ret)
    w = SCM_W
    g = w["latency"] * lat + w["energy"] * en + w["packet_loss"] * pl - w["throughput"] * thr
    g = g + np.stack([np.zeros_like(p), 2 * w["tx2"] * p], -1)
    return g


def scm_G_bound(j, k_sigma):
    names = ["col", "ret", "q", "cpu", "lat", "thr", "pl", "en"]

    def fn(lo, hi):
        _, sl = _scm_nodes(lo, hi, k_sigma)
        slo = np.stack([sl[n].lo for n in names], -1)          # (N, 8)
        shi = np.stack([sl[n].hi for n in names], -1)
        s = slo[:, None, :] + _VERT[None] * (shi - slo)[:, None, :]   # (N, 256, 8)
        best = np.zeros(len(lo))
        for pv in (lo[:, 1], hi[:, 1]):
            g = _scm_grad_vertices(s, np.broadcast_to(pv[:, None], s.shape[:2]))
            best = np.maximum(best, np.abs(g[..., j]).max(1))
        return best * (1 + 1e-9) + 1e-12
    return fn


def scm_L_bound(j, k_sigma):
    def fn(lo, hi):
        J, _ = _scm_nodes(lo, hi, k_sigma)
        return J.hess_row_norm(j)
    return fn


def scm_free_G():
    """Global bound with unbounded noise: every node slope ranges over [0, 1/4]."""
    s = 0.25 * _VERT
    out = []
    for j in range(2):
        m = 0.0
        for pv in (LO, HI):
            g = _scm_grad_vertices(s, np.full(len(s), pv))
            m = max(m, float(np.abs(g[:, j]).max()))
        out.append(m)
    return out


def scm_free_L():
    lo, hi = grid_boxes(SCM_BOUNDS, [1] * 5)
    J, _ = _scm_nodes(lo, hi, k_sigma=1e4 / SCM_SIGMA)     # noise effectively unbounded
    return [float(J.hess_row_norm(j)[0]) for j in range(2)]


def scm_tail(k):
    return 8 * 2 * norm.sf(k)          # eight independent Gaussian noise terms


# ---------------------------------------------------------------- plant 2: MEC

from certify import mec as _M

MEC_P = _M.PHYS
MEC_SIGMA = 0.015                      # noise_scale of MECRegime (every regime)
# inputs: offload x, cpu c, arrival a, channel h, load l, log-burst b, log-size z
MEC_BURST_S, MEC_SIZE_S = 0.03, 0.05


def mec_bounds(k):
    return [(LO, HI), (LO, HI), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (-k * MEC_BURST_S, k * MEC_BURST_S), (-k * MEC_SIZE_S, k * MEC_SIZE_S)]


def _mm1k3(lam: Jet, mu: Jet):
    """M/M/1/3 in closed polynomial form, smooth through rho = 1:
    sojourn T = (1 + 2r + 3r^2) / (mu (1 + r + r^2)), blocking
    P3 = r^3 / (1 + r + r^2 + r^3), busy = (r + r^2 + r^3) / (1 + r + r^2 + r^3)."""
    r = (lam / mu).clamp(0.0)
    r2 = r * r
    r3 = r2 * r
    s3 = 1.0 + r + r2
    s4 = s3 + r3
    inv_s4 = s4.apply(_recip)
    T = ((1.0 + 2.0 * r + 3.0 * r2) * (mu * s3).apply(_recip)).clamp(0.0)
    P = (r3 * inv_s4).clamp(0.0, 1.0)
    busy = ((r + r2 + r3) * inv_s4).clamp(0.0, 1.0)
    return T, P, busy


def _recip(x: I):
    from certify.ival import f_recip
    return f_recip(x)


def _mec_channels(lo, hi):
    """Pre-noise channel values a_k (latency/2.5, power/4.5, drop) as jets
    in (x, c)."""
    c = MEC_P
    x = Jet.var(I(lo[:, 0], hi[:, 0]), 0)
    u1 = Jet.var(I(lo[:, 1], hi[:, 1]), 1)
    arr, ch, ld = (Jet.const(I(lo[:, i], hi[:, i])) for i in (2, 3, 4))
    burst = Jet.const(I(np.exp(lo[:, 5]), np.exp(hi[:, 5])))
    size = Jet.const(I(np.exp(lo[:, 6]), np.exp(hi[:, 6])))
    f = c["f0"] + c["f1"] * u1
    lam = (c["lam0"] + c["lam1"] * arr) * burst
    mu_l = f * (c["gcyc"] * size).apply(_recip)
    T_l, P_l, busy_l = _mm1k3(lam * (1.0 - x), mu_l)
    # snr and rate depend only on the channel indicator (exogenous)
    snr_lo = 10.0 ** ((-2.0 + 22.0 * lo[:, 3]) / 10.0)
    snr_hi = 10.0 ** ((-2.0 + 22.0 * hi[:, 3]) / 10.0)
    rate = Jet.const(I(c["bw"] * np.log2(1 + snr_lo), c["bw"] * np.log2(1 + snr_hi)))
    mu_u = rate * (c["mbit"] * size).apply(_recip)
    T_u, P_u, busy_u = _mm1k3(lam * x, mu_u)
    mu_e = c["mu_edge"] * (1.0 - 0.8 * ld)
    T_e, P_e, _ = _mm1k3(lam * x * (1.0 - P_u), mu_e)
    T_off = T_u + T_e + 0.01
    lat = (1.0 - x) * T_l * (1.0 - P_l) + x * T_off * (1.0 - P_u)
    drop = (1.0 - x) * P_l + x * (1.0 - (1.0 - P_u) * (1.0 - P_e))
    power = c["kappa"] * (f * f * f) * busy_l + 0.08 + c["p_radio"] * busy_u
    return [lat * (1.0 / c["lat_norm"]), power * (1.0 / 4.5), drop], [MEC_SIGMA, MEC_SIGMA, 0.3 * MEC_SIGMA]


MEC_W = [_M.COST_W["latency"], _M.COST_W["energy"], _M.COST_W["drop"]]


def mec_G_bound(j):
    """sum_k w_k |d a_k / d u_j|: clipping each channel to [0, 1] after noise
    can only shrink a contrast, channel by channel."""
    def fn(lo, hi):
        ch, _ = _mec_channels(lo, hi)
        return sum(w * a.g[j].absmax() for w, a in zip(MEC_W, ch))
    return fn


def mec_L_bound(j):
    """Row j of the Hessian of sum_k w_k E_e[clip(a_k + e_k, 0, 1)], the
    measurement noise integrated exactly (smooth clip)."""
    encs = None

    def fn(lo, hi):
        nonlocal encs
        ch, sig = _mec_channels(lo, hi)
        if encs is None:
            encs = [f_smoothclip(s) for s in sig]
        tot = None
        for w, a, e in zip(MEC_W, ch, encs):
            t = a.apply(e) * w
            tot = t if tot is None else tot + t
        return tot.hess_row_norm(j)
    return fn


def mec_tail(k):
    return 2 * 2 * norm.sf(k)          # log-burst and log-size


def mec_free_G():
    """Without truncation every channel is clipped to [0, 1], so a
    per-realization contrast is at most sum_k w_k / (2 delta) ... in slope
    units sum_k w_k / (2 delta) is too crude; we use the range bound
    sum_k w_k * 1 / (2 * delta) with delta = 0.05."""
    return sum(MEC_W) / (2 * 0.05)


# ---------------------------------------------------------------- MEC, with exact queue primitives

from certify.ival import MM1K3_P, MM1K3_B, MM1K3_K, MM1K3_H


def _mec_channels_q(lo, hi):
    """Pre-noise channels (latency/2.5, power/4.5, drop) as jets in (x, c),
    with every M/M/1/3 quantity written as an exact univariate primitive of
    the utilization r:  T(1-P) = K(r)/mu,  T = H(r)/mu,  P(r),  busy B(r)."""
    c = MEC_P
    x = Jet.var(I(lo[:, 0], hi[:, 0]), 0)
    u1 = Jet.var(I(lo[:, 1], hi[:, 1]), 1)
    lam = I((c["lam0"] + c["lam1"] * lo[:, 2]) * np.exp(lo[:, 5]),
            (c["lam0"] + c["lam1"] * hi[:, 2]) * np.exp(hi[:, 5]))
    z = I(np.exp(lo[:, 6]), np.exp(hi[:, 6]))
    snr_lo = 10.0 ** ((-2.0 + 22.0 * lo[:, 3]) / 10.0)
    snr_hi = 10.0 ** ((-2.0 + 22.0 * hi[:, 3]) / 10.0)
    rate = I(c["bw"] * np.log2(1 + snr_lo), c["bw"] * np.log2(1 + snr_hi))
    mu_e = I(c["mu_edge"] * (1.0 - 0.8 * hi[:, 4]), c["mu_edge"] * (1.0 - 0.8 * lo[:, 4]))
    f = c["f0"] + c["f1"] * u1
    inv_f = f.apply(f_recip_)
    one_m_x = 1.0 - x
    # local queue: r_l = 0.25 z lam (1 - x) / f
    r_l = (one_m_x * inv_f * (0.25 * z * lam)).clamp(0.0)
    K_l, P_l, B_l = r_l.apply(MM1K3_K.enc), r_l.apply(MM1K3_P.enc), r_l.apply(MM1K3_B.enc)
    # uplink: r_u = 2 z lam x / rate
    ru_c = 2.0 * z * lam / rate
    r_u = (x * ru_c).clamp(0.0)
    K_u, P_u, B_u = r_u.apply(MM1K3_K.enc), r_u.apply(MM1K3_P.enc), r_u.apply(MM1K3_B.enc)
    # edge: r_e = lam x (1 - P_u) / mu_e
    r_e = (x * (1.0 - P_u) * (lam / mu_e)).clamp(0.0)
    H_e, P_e = r_e.apply(MM1K3_H.enc), r_e.apply(MM1K3_P.enc)
    lat = (one_m_x * K_l * inv_f * (0.25 * z)
           + x * (K_u * (2.0 * z / rate) + (1.0 - P_u) * (H_e * (1.0 / mu_e) + 0.01)))
    drop = one_m_x * P_l + x * (1.0 - (1.0 - P_u) * (1.0 - P_e))
    power = c["kappa"] * (f * f * f) * B_l + 0.08 + c["p_radio"] * B_u
    return [lat * (1.0 / c["lat_norm"]), power * (1.0 / 4.5), drop], [MEC_SIGMA, MEC_SIGMA, 0.3 * MEC_SIGMA]


def f_recip_(x):
    from certify.ival import f_recip
    return f_recip(x)


def mec_G_bound_q(j, k=6.0):
    """|d/du_j sum_k w_k clip(a_k + e_k)| <= |sum_k w_k 1_k d a_k/du_j| with the
    indicator 1_k = 1{0 < a_k + e_k < 1} enclosed in [0, 1] unless the channel
    is surely clipped (0) or surely unclipped (1) for |e_k| <= k sigma_k."""
    def fn(lo, hi):
        ch, sig = _mec_channels_q(lo, hi)
        tot = None
        for w, a, sg in zip(MEC_W, ch, sig):
            m = k * sg
            never = (a.v.hi < -m) | (a.v.lo > 1 + m)
            always = (a.v.lo > m) & (a.v.hi < 1 - m)
            ind = I(np.where(never | ~always, 0.0, 1.0), np.where(never, 0.0, 1.0))
            t = (a.g[j] * ind) * w
            tot = t if tot is None else tot + t
        return tot.absmax()
    return fn


def mec_L_bound_q(j):
    encs = {}

    def fn(lo, hi):
        ch, sig = _mec_channels_q(lo, hi)
        tot = None
        for w, a, sg in zip(MEC_W, ch, sig):
            if sg not in encs:
                encs[sg] = f_smoothclip(sg)
            t = a.apply(encs[sg]) * w
            tot = t if tot is None else tot + t
        return tot.hess_row_norm(j)
    return fn


# ---------------------------------------------------------------- attained values (lower bounds)

def scm_attained(k=6.0, restarts=300, seed=0):
    """Local search (L-BFGS-B from random starts) over actions, exogenous
    values and noise within k sigma: values the plant actually reaches, i.e.
    lower bounds on the suprema that the branch-and-bound bounds from above."""
    from scipy.optimize import minimize
    sig = lambda z: 1 / (1 + np.exp(-z))
    w = SCM_W

    def J(x):
        d, p, Iq, Lq, A = x[:5]
        n = x[5:]
        col = sig(3 * Iq + 2.5 * d - 1.5 * p - 2 * Lq - 1 + n[0]); ret = sig(3.5 * col - 1.2 * p - 0.5 + n[1])
        q = sig(2.5 * ret + 1.5 * A - d - 0.8 + n[2]); cpu = sig(2 * q + d - 1.2 + n[3])
        lat = sig(2.2 * ret + 2 * q + cpu - 1.8 + n[4]); thr = sig(2 * d - 1.8 * col - 1.5 * ret + 0.3 + n[5])
        pl = sig(2.5 * col + 1.8 * ret + q - 2.5 + n[6]); en = sig(2 * p + 1.8 * d + cpu + 0.8 * ret - 1.5 + n[7])
        return (w["latency"] * lat + w["energy"] * en + w["packet_loss"] * pl - w["throughput"] * thr
                + w["tx2"] * p * p)

    def grad(x, j, h=1e-5):
        e = np.zeros_like(x); e[j] = h
        return (J(x + e) - J(x - e)) / (2 * h)

    kk = k * SCM_SIGMA
    bnds = [(LO, HI)] * 2 + [(0, 1)] * 3 + [(-kk, kk)] * 8
    rng = np.random.default_rng(seed)
    out = {}
    for j, a in enumerate(["duty_cycle", "tx_power"]):
        best = 0.0
        for _ in range(restarts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
            r = minimize(lambda x: -abs(grad(x, j)), x0, bounds=bnds, method="L-BFGS-B")
            best = max(best, -float(r.fun))
        out[a] = best
    return out


# ---------------------------------------------------------------- driver

SCM_TARGETS = {"G": {"duty_cycle": 1.6, "tx_power": 2.0}, "L": {"duty_cycle": 6.0, "tx_power": 6.3}}


def certify_scm(k=6.0, log=print):
    """Certify the SCM envelopes declared for the theorem-covered runs
    (G = 2.0, L = 6.3) actuator by actuator; returns the certified upper
    bounds, the free-noise global bounds, the tail probability and the
    attained (lower-bound) sensitivities."""
    t0 = time.time()
    out = dict(k_sigma=k, tail=scm_tail(k), free_G=scm_free_G(), free_L=scm_free_L(), G={}, L={})
    for j, a in enumerate(["duty_cycle", "tx_power"]):
        lo, hi = grid_boxes(SCM_BOUNDS, [8, 8, 6, 6, 6])
        for name, fn in (("G", scm_G_bound(j, k)), ("L", scm_L_bound(j, k))):
            tgt = SCM_TARGETS[name][a]
            log(f"SCM {name} {a} (target {tgt})")
            ub, closed, n = branch_and_bound(fn, lo, hi, tgt, max_rounds=60, log=log)
            out[name][a] = dict(bound=ub, target=tgt, certified=bool(closed), boxes=n)
            log(f"  -> {ub:.4f} certified: {closed} ({n} boxes, {time.time() - t0:.0f}s)")
    # tail: beyond k sigma the global (free-noise) bounds apply with probability <= tail
    out["tail_width_G"] = max(out["free_G"]) * out["tail"]
    out["L_with_tail"] = {a: out["L"][a]["bound"] + max(out["free_L"]) * out["tail"] for a in out["L"]}
    out["attained_G"] = scm_attained(k)
    log(f"attained G: {out['attained_G']}")
    return out


MEC_TARGETS = {"G": {"offload": 2.5, "cpu_freq": 2.0}, "L": {"offload": 12.0, "cpu_freq": 8.0}}


def _mec_job(args):
    name, j, a, k = args
    fn = mec_G_bound_q(j, k) if name == "G" else mec_L_bound_q(j)
    lo, hi = grid_boxes(mec_bounds(k), [8, 8, 4, 4, 4, 2, 2])
    ub, closed, n = branch_and_bound(fn, lo, hi, MEC_TARGETS[name][a], max_rounds=80, min_width=1e-6)
    return name, a, dict(bound=ub, target=MEC_TARGETS[name][a], certified=bool(closed), boxes=n)


def mec_attained(k=6.0, n_rand=20000, n_local=30, seed=1):
    """Local search over the clipped per-realization contrast |D|/delta of the
    MEC node (measurement noise within k sigma chosen adversarially)."""
    from scipy.optimize import minimize
    c = MEC_P

    def chans(x, u1, a, h, l, b, z):
        f = c["f0"] + c["f1"] * u1
        lam = (c["lam0"] + c["lam1"] * a) * np.exp(b)
        size = np.exp(z)
        T_l, P_l, busy_l = _M.mm1k(lam * (1 - x), f / (c["gcyc"] * size))
        rate = c["bw"] * np.log2(1 + 10 ** ((-2 + 22 * h) / 10))
        T_u, P_u, busy_u = _M.mm1k(lam * x, rate / (c["mbit"] * size))
        T_e, P_e, _ = _M.mm1k(lam * x * (1 - P_u), c["mu_edge"] * (1 - 0.8 * l))
        lat = (1 - x) * T_l * (1 - P_l) + x * (T_u + T_e + 0.01) * (1 - P_u)
        drop = (1 - x) * P_l + x * (1 - (1 - P_u) * (1 - P_e))
        power = c["kappa"] * f ** 3 * busy_l + 0.08 + c["p_radio"] * busy_u
        return np.array([float(lat / c["lat_norm"]), float(power / 4.5), float(drop)])

    W = np.array(MEC_W)
    S = np.array([MEC_SIGMA, MEC_SIGMA, 0.3 * MEC_SIGMA]) * k
    d = 0.05

    def contrast(v, j):
        st, e = v[:7], v[7:]
        up, dn = st.copy(), st.copy()
        up[j] += d
        dn[j] -= d
        return abs(W @ (np.clip(chans(*up) + e, 0, 1) - np.clip(chans(*dn) + e, 0, 1))) / (2 * d)

    bnds = [(LO + d, HI - d)] * 2 + [(0, 1)] * 3 + [(-k * MEC_BURST_S, k * MEC_BURST_S),
                                                  (-k * MEC_SIZE_S, k * MEC_SIZE_S)] + [(-x, x) for x in S]
    rng = np.random.default_rng(seed)
    out = {}
    for j, a in enumerate(["offload", "cpu_freq"]):
        X = np.array([[rng.uniform(lo, hi) for lo, hi in bnds] for _ in range(n_rand)])
        vals = np.array([contrast(x, j) for x in X])
        best = float(vals.max())
        for x0 in X[np.argsort(-vals)[:n_local]]:
            r = minimize(lambda v: -contrast(v, j), x0, bounds=bnds, method="Powell")
            best = max(best, -float(r.fun))
        out[a] = best
    return out


def certify_mec(k=6.0, procs=4):
    """Interval branch-and-bound with exact M/M/1/3 primitives, stopped at
    4 million open boxes per round. The bounds are valid upper bounds but far
    above the attained values: queueing near overload makes worst-case
    curvature large, so the MEC envelopes are not certified at usable values."""
    from multiprocessing import Pool
    jobs = [(nm, j, a, k) for nm in ("G", "L") for j, a in enumerate(["offload", "cpu_freq"])]
    with Pool(procs) as pool:
        res = pool.map(_mec_job, jobs)
    out = dict(k_sigma=k, tail=mec_tail(k), G={}, L={})
    for nm, a, v in res:
        out[nm][a] = v
    out["attained_G"] = mec_attained(k)
    out["certified"] = False
    return out


if __name__ == "__main__":
    import sys
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    p = Path(__file__).resolve().parents[2] / "experiments" / "results" / "kic_certified_env.json"
    res = json.load(open(p)) if p.exists() else {}
    if which in ("all", "scm"):
        res["scm"] = certify_scm(k)
    if which in ("all", "mec"):
        res["mec"] = certify_mec(k)
    json.dump(res, open(p, "w"), indent=1)
    print(json.dumps(res, indent=1))
