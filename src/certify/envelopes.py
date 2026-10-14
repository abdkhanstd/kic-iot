# Author: abdkhan
"""Derivation of the declared envelopes (G, L) from a plant's simulator.

G: largest per-realization sensitivity |J(c + d e_j, xi) - J(c - d e_j, xi)| / (2d)
   with every exogenous input and noise held fixed (common random numbers).
L: largest Lipschitz constant of the slope d_j m in the action, estimated as
   the Euclidean norm of the j-th row of the Hessian of m = E J (central
   differences), over sampled regimes and actions.
In a deployment the same quantities come from bench characterization with
paired inputs, or from physical sensitivity bounds (Sec. IV-F of the paper).
"""
from __future__ import annotations

import numpy as np


def derive(P, regime_sampler, n_states=150, d=0.05, n=2000, seed=0, per_actuator=False):
    """Paired-input envelopes. per_actuator=True also returns each actuator's own maxima
    (same draws; the plant-wide G, L are their maxima)."""
    rng = np.random.default_rng(seed)
    G = L = 0.0
    acts = P.actuators
    Gj, Lj = {a: 0.0 for a in acts}, {a: 0.0 for a in acts}
    for k in range(n_states):
        reg = regime_sampler(rng)
        c = {a: float(rng.uniform(0.1, 0.9)) for a in acts}
        def J(u, s=k):
            return P.cost(P.simulate_vec(reg, u, n, np.random.default_rng(1000 + s)))
        for j, a in enumerate(acts):
            up = dict(c); up[a] += d
            dn = dict(c); dn[a] -= d
            ju, jd = J(up), J(dn)
            Gj[a] = max(Gj[a], float(np.max(np.abs(ju - jd)) / (2 * d)))
            G = max(G, Gj[a])
            row = []
            for b in acts:
                if b == a:
                    row.append((ju.mean() - 2 * J(c).mean() + jd.mean()) / d ** 2)
                else:
                    pp = dict(c); pp[a] += d; pp[b] += d
                    pm = dict(c); pm[a] += d; pm[b] -= d
                    mp = dict(c); mp[a] -= d; mp[b] += d
                    mm = dict(c); mm[a] -= d; mm[b] -= d
                    row.append((J(pp).mean() - J(pm).mean() - J(mp).mean() + J(mm).mean()) / (4 * d * d))
            Lj[a] = max(Lj[a], float(np.linalg.norm(row)))
            L = max(L, Lj[a])
    return (G, L, Gj, Lj) if per_actuator else (G, L)


def scm_regimes(rng):
    from simulator.scm_iot import RegimeParams
    return RegimeParams(interference_mean=rng.uniform(0.1, 0.6), link_quality_mean=rng.uniform(0.4, 0.9),
                        arrival_rate_mean=rng.uniform(0.2, 0.7))


def mec_regimes(rng):
    from certify.mec import MECRegime
    return MECRegime(arrival_mean=rng.uniform(0.1, 0.9), channel_mean=rng.uniform(0.15, 0.9),
                     load_mean=rng.uniform(0.05, 0.85))


def grid_regimes(rng):
    from certify.grid import GridRegime
    return GridRegime(demand_mean=rng.uniform(0.15, 0.9), irr_mean=rng.uniform(0.05, 0.95),
                      qshare_mean=rng.uniform(0.15, 0.85))


def grid_regimes_domain(rng):
    """Stationary feeder regimes confined to the certified operating domain D
    (demand in [0.1, 0.7], reactive share in [0, 0.8]; grid_certify_run.py)."""
    from certify.grid import GridRegime
    return GridRegime(demand_mean=rng.uniform(0.2, 0.6), irr_mean=rng.uniform(0.05, 0.95),
                      qshare_mean=rng.uniform(0.15, 0.7), dmin=0.1, dmax=0.7, qmax=0.8)


def regimes_for(plant: str):
    import os
    if plant == "grid" and os.environ.get("KIC_GRID_DOMAIN") == "1":
        return grid_regimes_domain
    return {"scm": scm_regimes, "mec": mec_regimes, "grid": grid_regimes}[plant]
