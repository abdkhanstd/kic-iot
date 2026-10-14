# Author: abdkhan
"""Certified envelopes and leakage gains for the IEEE 33-bus feeder (plant 3).

The feeder's operator cost (grid.py)
    J = 6 loss + 4 [(1 - |V_18|) + (1 - |V_33|)] + 0.35 (Q_18^2 + Q_33^2)   (+ meter noise)
depends on the AC power-flow solution V of the radial network,
    V = V0 - Z conj(S / V)                                              (BFS fixed point)
with Z the path-impedance matrix (Z_kl = sum of branch impedances on the common
part of the root paths of k and l) and S the net complex load. This module
PROVES, over boxes of the operating domain, upper bounds on
  G_j  = sup |dJ/du_j|                    (per realization: every load, PV and noise value)
  L_j  = sup ||row j of the Hessian of J|| (bounds the Hessian of m = E J, noise is additive)
  k_b  = sup |d|V_b|/du_18|               (voltage gain seen by a meter at bus b: leakage)
by complex interval arithmetic in rectangular form:

1. Solution enclosure. Phi(V) = V0 - Z conj(S/V). If Phi(X) is contained in the
   box X, a fixed point lies in X (Brouwer); if the derivative bound
   rho = max_k sum_l |Z_kl| sup|S_l|/inf|V_l|^2 < 1 on X, Phi is a contraction
   there, the fixed point is unique and it is the limit of the simulator's
   backward/forward sweep from any start in X (flat start included). We then
   iterate X <- Phi(X) cap X, which keeps the fixed point.
2. Sensitivities. dV solves dV = b + T(dV), T(D) = Z conj(S D / V^2), with
   ||T|| <= rho; second derivatives solve the same linear system with a
   right-hand side built from first derivatives. Starting from the ball
   ||b||/(1 - rho) and iterating D <- (b + T(D)) cap D keeps the true value.
3. Cost derivatives follow from loss = V0 Re(sum S/V) - sum Re S and
   d|V| = Re(conj(V) dV)/|V|.

Every real operation is padded outward by a relative 1e-12 (rounding). Load
and PV noise are enclosed at +-6 sigma (their tail is priced as in the SCM).

    python -m certify.grid_cert        # writes experiments/results/kic_certified_grid_env.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from certify import grid as GR

EPS = 1e-12
K_SIGMA = 6.0


def _pad(lo, hi):
    # one correctly rounded IEEE-754 operation: error <= 2^-53 |result|, far inside EPS
    w = EPS * (np.abs(lo) + np.abs(hi)) + 1e-300
    return lo - w, hi + w


def _psum(tlo, thi):
    """Rigorous sum over axis 1 of interval terms [tlo, thi]. A floating-point sum of
    n terms errs by at most gamma_n * sum|terms| (gamma_n = n u / (1 - n u), u = 2^-53),
    which can exceed a padding relative to the (possibly cancelling) result; pad by
    1e-13 * sum|terms| >> gamma_33 instead, like matvec."""
    err = 1e-13 * (np.abs(tlo).sum(1) + np.abs(thi).sum(1)) + 1e-300
    lo, hi = _pad(tlo.sum(1), thi.sum(1))
    return lo - err, hi + err


# ------------------------------------------------------------ real intervals (arrays)

def rmul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return _pad(p.min(0), p.max(0))


def rsq(lo, hi):
    a, b = lo * lo, hi * hi
    l = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(a, b))
    return _pad(l, np.maximum(a, b))


def rabsmax(lo, hi):
    return np.maximum(np.abs(lo), np.abs(hi))


class CI:
    """Complex interval vector (rectangle per component), arrays of shape (N, n)."""
    __slots__ = ("rl", "rh", "il", "ih")

    def __init__(self, rl, rh, il, ih):
        self.rl, self.rh, self.il, self.ih = rl, rh, il, ih

    @staticmethod
    def point(z):
        z = np.asarray(z, dtype=complex)
        return CI(z.real.copy(), z.real.copy(), z.imag.copy(), z.imag.copy())

    def __add__(self, o):
        if isinstance(o, CI):
            return CI(*_pad(self.rl + o.rl, self.rh + o.rh), *_pad(self.il + o.il, self.ih + o.ih))
        o = np.asarray(o, dtype=complex)
        return CI(*_pad(self.rl + o.real, self.rh + o.real), *_pad(self.il + o.imag, self.ih + o.imag))

    def __neg__(self):
        return CI(-self.rh, -self.rl, -self.ih, -self.il)

    def __sub__(self, o):
        return self + (-o)

    def conj(self):
        return CI(self.rl, self.rh, -self.ih, -self.il)

    def mul(self, o):
        """rectangle x rectangle"""
        ac = rmul(self.rl, self.rh, o.rl, o.rh)
        bd = rmul(self.il, self.ih, o.il, o.ih)
        ad = rmul(self.rl, self.rh, o.il, o.ih)
        bc = rmul(self.il, self.ih, o.rl, o.rh)
        return CI(*_pad(ac[0] - bd[1], ac[1] - bd[0]), *_pad(ad[0] + bc[0], ad[1] + bc[1]))

    def rscale(self, lo, hi):
        """times a real interval [lo, hi]"""
        return CI(*rmul(self.rl, self.rh, lo, hi), *rmul(self.il, self.ih, lo, hi))

    def cscale(self, z):
        """times an exact complex array z (broadcast)"""
        z = np.asarray(z, dtype=complex)
        zr, zi = z.real, z.imag
        a = rmul(self.rl, self.rh, zr, zr)
        b = rmul(self.il, self.ih, zi, zi)
        c = rmul(self.rl, self.rh, zi, zi)
        d = rmul(self.il, self.ih, zr, zr)
        return CI(*_pad(a[0] - b[1], a[1] - b[0]), *_pad(c[0] + d[0], c[1] + d[1]))

    def abs2(self):
        a, b = rsq(self.rl, self.rh), rsq(self.il, self.ih)
        return _pad(a[0] + b[0], a[1] + b[1])

    def absmax(self):
        return np.sqrt(np.maximum(self.rl ** 2, self.rh ** 2) + np.maximum(self.il ** 2, self.ih ** 2)) * (1 + 1e-12)

    def recip(self):
        lo, hi = self.abs2()
        assert np.all(lo > 0), "enclosure contains 0"
        return self.conj().rscale(*_pad(1.0 / hi, 1.0 / lo))

    def matvec(self, Z):
        """Z @ x for an exact complex matrix Z (n x n) applied to each row of x."""
        mr, mi = 0.5 * (self.rl + self.rh), 0.5 * (self.il + self.ih)
        rr, ri = 0.5 * (self.rh - self.rl), 0.5 * (self.ih - self.il)
        Zr, Zi = Z.real, Z.imag
        outr = mr @ Zr.T - mi @ Zi.T
        outi = mr @ Zi.T + mi @ Zr.T
        radr = rr @ np.abs(Zr).T + ri @ np.abs(Zi).T
        radi = rr @ np.abs(Zi).T + ri @ np.abs(Zr).T
        # rounding of the two products and the sums
        er = 1e-13 * (np.abs(mr) @ np.abs(Zr).T + np.abs(mi) @ np.abs(Zi).T + radr) + 1e-300
        ei = 1e-13 * (np.abs(mr) @ np.abs(Zi).T + np.abs(mi) @ np.abs(Zr).T + radi) + 1e-300
        return CI(outr - radr - er, outr + radr + er, outi - radi - ei, outi + radi + ei)

    def intersect(self, o):
        rl, rh = np.maximum(self.rl, o.rl), np.minimum(self.rh, o.rh)
        il, ih = np.maximum(self.il, o.il), np.minimum(self.ih, o.ih)
        return CI(rl, rh, il, ih)

    def empty(self):
        return (self.rl > self.rh) | (self.il > self.ih)

    def subset_of(self, o):
        return (self.rl >= o.rl) & (self.rh <= o.rh) & (self.il >= o.il) & (self.ih <= o.ih)

    def re(self):
        return self.rl, self.rh

    def sum(self):
        return CI(*_psum(self.rl, self.rh), *_psum(self.il, self.ih))

    def col(self, k):
        return CI(self.rl[:, k], self.rh[:, k], self.il[:, k], self.ih[:, k])


def ball(N, n, R):
    R = np.broadcast_to(np.asarray(R, dtype=float).reshape(-1, 1), (N, n))
    return CI(-R.copy(), R.copy(), -R.copy(), R.copy())


# ------------------------------------------------------------ the feeder

def feeder_matrices():
    F = GR.feeder()
    nb, root, parent, z = F["nb"], F["root"], F["parent"], F["z"]
    idx = [k for k in range(nb) if k != root]              # non-root buses
    pos = {k: i for i, k in enumerate(idx)}
    paths = {}
    for k in idx:
        s, b = set(), k
        while parent[b] >= 0:
            s.add(b)
            b = parent[b]
        paths[k] = s
    n = len(idx)
    Z = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for c in range(n):
            common = paths[idx[a]] & paths[idx[c]]
            Z[a, c] = sum(z[b] for b in common)
    D = np.zeros((n, n))                                   # D[b, l] = 1: bus l in the subtree of branch b
    for a in range(n):
        for c in range(n):
            if idx[a] in paths[idx[c]]:
                D[a, c] = 1.0
    return dict(Z=Z, D=D, r=np.real(z[idx]), idx=idx, pos=pos, p=F["p"][idx], q=F["q"][idx], V0=F["vm_ext"],
                inv=[pos[b] for b in GR.INV_BUS])


M = None


def _M():
    global M
    if M is None:
        M = feeder_matrices()
    return M


def load_box(lo, hi, k_sigma=None):
    """Net load S (N x n complex interval) for boxes over (lvl, qm, irr, u18, u33)."""
    m = _M()
    k_sigma = K_SIGMA if k_sigma is None else k_sigma
    nl = k_sigma * GR.NOISE["load"]
    npv = k_sigma * GR.NOISE["pv"]
    lvl = (0.5 + 0.8 * lo[:, 0], 0.5 + 0.8 * hi[:, 0])
    qm = (0.6 + 0.8 * lo[:, 1], 0.6 + 0.8 * hi[:, 1])
    irr = (lo[:, 2], hi[:, 2])
    N = len(lo)
    ep = (np.full(N, 1 - nl), np.full(N, 1 + nl))
    Pl = rmul(lvl[0][:, None] * m["p"][None, :], lvl[1][:, None] * m["p"][None, :], ep[0][:, None], ep[1][:, None])
    lq = rmul(lvl[0], lvl[1], qm[0], qm[1])
    Ql = rmul(lq[0][:, None] * m["q"][None, :], lq[1][:, None] * m["q"][None, :], ep[0][:, None], ep[1][:, None])
    Pl = [Pl[0].copy(), Pl[1].copy()]
    Ql = [Ql[0].copy(), Ql[1].copy()]
    pv = rmul(GR.PV_RATED * irr[0], GR.PV_RATED * irr[1], np.full(N, 1 - npv), np.full(N, 1 + npv))
    for j, b in enumerate(m["inv"]):
        qlo = GR.QMIN + (GR.QMAX - GR.QMIN) * lo[:, 3 + j]
        qhi = GR.QMIN + (GR.QMAX - GR.QMIN) * hi[:, 3 + j]
        Pl[0][:, b] -= pv[1]; Pl[1][:, b] -= pv[0]
        Ql[0][:, b] -= qhi;   Ql[1][:, b] -= qlo
    return CI(*_pad(Pl[0], Pl[1]), *_pad(Ql[0], Ql[1]))


def _point_solution(S_mid):
    """Numerical BFS solution for the box midpoints (starting point only)."""
    m = _M()
    Z, V0 = m["Z"], m["V0"]
    V = np.full(S_mid.shape, V0 + 0j)
    for _ in range(200):
        Vn = V0 - np.conj(S_mid / V) @ Z.T
        if np.max(np.abs(Vn - V)) < 1e-13:
            V = Vn
            break
        V = Vn
    return V


def enclose(lo, hi, iters=60):
    """Enclosure of the power-flow solution and certification flags per box.
    epsilon-inflation around the midpoint solution: find X with Phi(X) in X."""
    m = _M()
    Z, V0 = m["Z"], m["V0"]
    S = load_box(lo, hi)
    N, n = S.rl.shape
    Sm = 0.5 * (S.rl + S.rh) + 0.5j * (S.il + S.ih)
    Vm = _point_solution(Sm)
    r = np.full((N, 1), 0.005)
    ok = np.zeros(N, bool)
    X = None
    Vc = CI.point(Vm)
    base = (-(S.mul(Vc.recip()).conj().matvec(Z))) + V0          # Phi(V_c) with the load box

    def phi(Xt):
        # centred form: Phi(V) = Phi(V_c) + Z conj(S (V - V_c) / (V V_c))
        return base + S.mul(Xt - Vc).mul(Xt.mul(Vc).recip()).conj().matvec(Z)

    for _ in range(40):
        Xt = CI(Vm.real - r, Vm.real + r, Vm.imag - r, Vm.imag + r)
        Y = phi(Xt)
        inc = np.all(Y.subset_of(Xt), axis=1)
        if X is None:
            X = Xt
        newly = inc & ~ok
        for a in ("rl", "rh", "il", "ih"):
            getattr(X, a)[newly] = getattr(Xt, a)[newly]
        ok |= inc
        if ok.all():
            break
        r = np.where(ok[:, None], r, 1.25 * r)
        # boxes whose inflated radius exceeds 0.4 are hopeless at this width
        if np.all(ok | (r[:, 0] > 0.4)):
            break
    for _ in range(iters):
        Y = phi(X)
        Xn = Y.intersect(X)
        ok &= ~np.any(Xn.empty(), axis=1)
        conv = np.max(np.abs(Xn.rh - X.rh) + np.abs(Xn.rl - X.rl) + np.abs(Xn.ih - X.ih) + np.abs(Xn.il - X.il))
        X = Xn
        if conv < 1e-12:
            break
    v2lo, _ = X.abs2()
    Smax = S.absmax()
    rho = np.max((np.abs(Z)[None, :, :] * (Smax / np.maximum(v2lo, 1e-9))[:, None, :]).sum(2), axis=1) * (1 + 1e-12)
    ok &= rho < 1
    return dict(S=S, V=X, rho=rho, ok=ok)


def _solve_linear(b, Yc, Z, rho, iters=40):
    """Enclose D = b + Z conj(Yc * D), ||T|| <= rho < 1."""
    N, n = b.rl.shape
    R = b.absmax().max(1) / (1 - rho)
    D = ball(N, n, R)
    for _ in range(iters):
        Dn = (b + Yc.mul(D).conj().matvec(Z)).intersect(D)
        if np.max(np.abs(Dn.rh - D.rh) + np.abs(Dn.il - D.il)) < 1e-13:
            D = Dn
            break
        D = Dn
    return D


def sensitivities(lo, hi, second=True):
    """Enclosures of J's gradient/Hessian in the normalized actions and of
    d|V_b|/du_18 for every bus b, over boxes (lvl, qm, irr, u18, u33)."""
    m = _M()
    Z, V0, inv = m["Z"], m["V0"], m["inv"]
    E = enclose(lo, hi)
    S, V, rho, ok = E["S"], E["V"], E["rho"], E["ok"]
    N, n = S.rl.shape
    rV = V.recip()
    rV2 = rV.mul(rV)
    Yc = S.mul(rV2)                                        # S / V^2
    s = GR.QMAX - GR.QMIN
    dS = []
    for b in inv:
        z = np.zeros((N, n), dtype=complex)
        z[:, b] = -1j * s                                  # d S / d u_j (per normalized action)
        dS.append(CI.point(z))
    dV, dW = [], []
    for j in range(2):
        bj = -(dS[j].mul(rV).conj().matvec(Z))
        Dj = _solve_linear(bj, Yc, Z, rho)
        dV.append(Dj)
        dW.append(dS[j].mul(rV) - Yc.mul(Dj))
    absV = np.sqrt(np.maximum(V.abs2()[0], 1e-300)), np.sqrt(V.abs2()[1])
    inv_abs = _pad(1.0 / absV[1], 1.0 / absV[0])

    def dabs(Dj):
        # Re(conj(V) dV) / |V|
        re = V.conj().mul(Dj)
        return rmul(re.rl, re.rh, inv_abs[0], inv_abs[1])

    dr = [dabs(dV[j]) for j in range(2)]
    Qlo = [GR.QMIN + s * lo[:, 3 + j] for j in range(2)]
    Qhi = [GR.QMIN + s * hi[:, 3 + j] for j in range(2)]
    Dm, rb = m["D"].astype(complex), m["r"]
    Ib = S.mul(rV).conj().matvec(Dm)                       # branch currents
    dI = [dW[j].conj().matvec(Dm) for j in range(2)]

    def rsum_r(ci_re_lo, ci_re_hi):
        # sum_b r_b x_b for a real interval vector x
        # r_b > 0 is exact, so x_b r_b keeps the order of the endpoints; the products'
        # rounding is inside the 1e-13 * sum|terms| of _psum
        return _psum(ci_re_lo * rb[None, :], ci_re_hi * rb[None, :])

    grad = []
    for j in range(2):
        pr = Ib.conj().mul(dI[j])                          # conj(I_b) dI_b
        dl = rsum_r(2 * pr.rl, 2 * pr.rh)                  # d loss = 2 sum r_b Re(conj(I_b) dI_b)
        g_lo = 6 * dl[0] - 4 * (dr[j][1][:, inv[0]] + dr[j][1][:, inv[1]]) + 0.7 * s * Qlo[j]
        g_hi = 6 * dl[1] - 4 * (dr[j][0][:, inv[0]] + dr[j][0][:, inv[1]]) + 0.7 * s * Qhi[j]
        grad.append(_pad(g_lo, g_hi))
    out = dict(ok=ok, rho=rho, grad=grad, dr18=dr[0], dr33=dr[1], V=V, S=S)
    # loss and |V| enclosures (channel clipping check)
    a2 = Ib.abs2()
    out["loss"] = _psum(a2[0] * rb[None, :], a2[1] * rb[None, :])
    out["absV"] = absV
    if not second:
        return out
    H = [[None, None], [None, None]]
    rV3 = rV2.mul(rV)
    for i in range(2):
        for j in range(i, 2):
            t = dS[i].mul(dV[j]).mul(rV2) + dS[j].mul(dV[i]).mul(rV2) - S.mul(dV[i]).mul(dV[j]).mul(rV3).rscale(2.0, 2.0)
            b2 = t.conj().matvec(Z)
            D2 = _solve_linear(b2, Yc, Z, rho)
            d2W = (-(dS[i].mul(dV[j]).mul(rV2))) - dS[j].mul(dV[i]).mul(rV2) - Yc.mul(D2) \
                + S.mul(dV[i]).mul(dV[j]).mul(rV3).rscale(2.0, 2.0)
            d2I = d2W.conj().matvec(Dm)
            p1 = dI[i].conj().mul(dI[j])
            p2 = Ib.conj().mul(d2I)
            d2l = rsum_r(2 * (p1.rl + p2.rl), 2 * (p1.rh + p2.rh))
            # d2|V| = [Re(conj(dV_j) dV_i) + Re(conj(V) d2V)] / |V| - d|V|_i d|V|_j / |V|
            a = dV[j].conj().mul(dV[i])
            c = V.conj().mul(D2)
            num = _pad(a.rl + c.rl, a.rh + c.rh)
            t1 = rmul(num[0], num[1], inv_abs[0], inv_abs[1])
            pr = rmul(dr[i][0], dr[i][1], dr[j][0], dr[j][1])
            t2 = rmul(pr[0], pr[1], inv_abs[0], inv_abs[1])
            d2r = _pad(t1[0] - t2[1], t1[1] - t2[0])
            h_lo = 6 * d2l[0] - 4 * (d2r[1][:, inv[0]] + d2r[1][:, inv[1]]) + (0.7 * s * s if i == j else 0.0)
            h_hi = 6 * d2l[1] - 4 * (d2r[0][:, inv[0]] + d2r[0][:, inv[1]]) + (0.7 * s * s if i == j else 0.0)
            H[i][j] = H[j][i] = _pad(h_lo, h_hi)
    out["H"] = H
    return out


def box_bounds(lo, hi):
    """Per box: G_j, L_j (row norms) upper bounds and validity flag."""
    o = sensitivities(lo, hi)
    G = [rabsmax(*o["grad"][j]) for j in range(2)]
    L = [np.sqrt(rabsmax(*o["H"][j][0]) ** 2 + rabsmax(*o["H"][j][1]) ** 2) for j in range(2)]
    return o, G, L


# ------------------------------------------------------------ driver

def grid_boxes(bounds, n):
    axes = []
    for (a, b), k in zip(bounds, n):
        e = np.linspace(a, b, k + 1)
        axes.append((e[:-1], e[1:]))
    los = np.stack(np.meshgrid(*[x[0] for x in axes], indexing="ij"), -1).reshape(-1, len(bounds))
    his = np.stack(np.meshgrid(*[x[1] for x in axes], indexing="ij"), -1).reshape(-1, len(bounds))
    return los, his


def bisect(los, his, scale):
    w = (his - los) / scale
    k = np.argmax(w, axis=1)
    idx = np.arange(len(los))
    mid = 0.5 * (los[idx, k] + his[idx, k])
    lo2, hi1 = los.copy(), his.copy()
    hi1[idx, k] = mid
    lo2[idx, k] = mid
    return np.concatenate([los, lo2]), np.concatenate([hi1, his])


def certify(domain, targets, grid=(4, 4, 3, 8, 8), max_rounds=12, chunk=4000, log=print):
    """Branch and bound over the domain box: close a box when its G, L bounds are
    at most the targets and the enclosure is valid; bisect the rest."""
    los, his = grid_boxes(domain, grid)
    scale = np.array([b - a for a, b in domain])
    best = dict(G=[0.0, 0.0], L=[0.0, 0.0], kappa=None, loss=[np.inf, -np.inf], absV=[np.inf, -np.inf], rho=0.0)
    n_eval = 0
    for r in range(max_rounds):
        closed = np.zeros(len(los), bool)
        Gs = np.zeros((len(los), 2)); Ls = np.zeros((len(los), 2)); oks = np.zeros(len(los), bool)
        for i in range(0, len(los), chunk):
            sl = slice(i, i + chunk)
            o, G, L = box_bounds(los[sl], his[sl])
            Gs[sl] = np.stack(G, 1); Ls[sl] = np.stack(L, 1); oks[sl] = o["ok"]
            kap = rabsmax(*o["dr18"])
            best["kappa"] = kap[o["ok"]].max(0) if best["kappa"] is None else np.maximum(best["kappa"], kap[o["ok"]].max(0))
            best["loss"][0] = min(best["loss"][0], float(o["loss"][0][o["ok"]].min()) if o["ok"].any() else np.inf)
            best["loss"][1] = max(best["loss"][1], float(o["loss"][1][o["ok"]].max()) if o["ok"].any() else -np.inf)
            best["absV"][0] = min(best["absV"][0], float(o["absV"][0][o["ok"]].min()) if o["ok"].any() else np.inf)
            best["absV"][1] = max(best["absV"][1], float(o["absV"][1][o["ok"]].max()) if o["ok"].any() else -np.inf)
            best["rho"] = max(best["rho"], float(o["rho"][o["ok"]].max()) if o["ok"].any() else 0.0)
        n_eval += len(los)
        closed = oks & np.all(Gs <= np.array(targets["G"]), 1) & np.all(Ls <= np.array(targets["L"]), 1)
        for j in range(2):
            if closed.any():
                best["G"][j] = max(best["G"][j], float(Gs[closed, j].max()))
                best["L"][j] = max(best["L"][j], float(Ls[closed, j].max()))
        op = ~closed
        log(f"round {r}: boxes {len(los)}, open {op.sum()}, invalid {(~oks).sum()}, "
            f"max open G {Gs[op].max(0) if op.any() else '-'}, L {Ls[op].max(0) if op.any() else '-'}")
        if not op.any():
            return dict(certified=True, n_eval=n_eval, **best)
        los, his = bisect(los[op], his[op], scale)
        if len(los) > 3_000_000:
            break
    return dict(certified=False, n_eval=n_eval, open_boxes=int(len(los)), **best)


if __name__ == "__main__":
    import sys
    out = Path(__file__).resolve().parents[2] / "experiments" / "results" / "kic_certified_grid_env.json"
    dom = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.02, 0.98), (0.02, 0.98)]
    t0 = time.time()
    res = certify(dom, dict(G=[float(sys.argv[1]) if len(sys.argv) > 1 else 3.0] * 2,
                            L=[float(sys.argv[2]) if len(sys.argv) > 2 else 8.0] * 2))
    res["seconds"] = time.time() - t0
    res["domain"] = dom
    res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(out, "w"), indent=1)
