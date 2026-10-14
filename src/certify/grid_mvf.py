# Author: abdkhan
"""Mean-value-form certification of the feeder's envelopes and leakage gains.

The plain enclosure in grid_cert.py wraps the +-6 sigma load noise of all 32
buses and is about twice the attained values. The mean-value form evaluates
each quantity q(x) at the centre x_c of the box in load space (64 coordinates:
P and Q per bus; actuators and PV enter through them) and adds
    sum_k sup_X |dq/dx_k| r_k,
with the partial derivatives enclosed over the whole box X (mean-value
theorem; X is convex). For q = dJ/du_j this needs second, for the Hessian
third, directional derivatives of the power-flow solution. All of them solve
the same linear system d^n V = b_n + T(d^n V), T(D) = Z conj(S D / V^2),
||T|| <= rho < 1, with right-hand sides built from lower orders:
    W = S R, R = 1/V,  dR = -R^2 dV,
    d2R_xy = 2 R^3 dV_x dV_y - R^2 d2V_xy,
    d3R_xyz = -6 R^4 dV_x dV_y dV_z + 2 R^3 (d2V_xy dV_z + d2V_xz dV_y + d2V_yz dV_x) - R^2 d3V_xyz,
    d^n W from Leibniz (S is affine in every coordinate), d^n V = -Z conj(d^n W),
    loss = sum_b r_b |I_b|^2, I = D conj(W);  |V| = sqrt(f), f = V conj(V).
"""
from __future__ import annotations

import numpy as np

from certify import grid as GR
from certify.grid_cert import (CI, _M, _pad, rmul, rabsmax, enclose, _point_solution, _solve_linear)


def _solve3(b, Yc, Z, rho, iters=40):
    """Enclose D = b + Z conj(Yc * D) for stacked right-hand sides b (N, K, n)."""
    R = b.absmax().max(-1) / (1 - rho)[:, None]
    shp = b.rl.shape
    Rb = np.broadcast_to(R[..., None], shp)
    D = CI(-Rb.copy(), Rb.copy(), -Rb.copy(), Rb.copy())
    for _ in range(iters):
        Dn = (b + Yc.mul(D).conj().matvec(Z)).intersect(D)
        conv = np.max(np.abs(Dn.rh - D.rh) + np.abs(Dn.il - D.il) + np.abs(Dn.rl - D.rl) + np.abs(Dn.ih - D.ih))
        D = Dn
        if conv < 1e-13:
            break
    return D


def _ex(ci):
    return CI(ci.rl[:, None], ci.rh[:, None], ci.il[:, None], ci.ih[:, None])


def _sel(ci, q):
    return CI(ci.rl[:, q:q + 1], ci.rh[:, q:q + 1], ci.il[:, q:q + 1], ci.ih[:, q:q + 1])


def _rc(A, B):
    """Re(conj(A) B) as a real interval"""
    p = A.conj().mul(B)
    return p.rl, p.rh


def _rsum(lo, hi, w):
    a, b = lo * w, hi * w
    return _pad(np.minimum(a, b).sum(-1), np.maximum(a, b).sum(-1))


def _add(*xs):
    return _pad(sum(x[0] for x in xs), sum(x[1] for x in xs))


def derivative_bounds(S, V, rho, order3=True):
    m = _M()
    Z, inv, rb = m["Z"], m["inv"], m["r"]
    Dm = m["D"].astype(complex)
    N, n = S.rl.shape
    K = 2 * n
    s = GR.QMAX - GR.QMIN
    R = V.recip(); R2 = R.mul(R); R3 = R2.mul(R); R4 = R2.mul(R2)
    S3, R_, R2_, R3_, R4_, V3 = _ex(S), _ex(R), _ex(R2), _ex(R3), _ex(R4), _ex(V)
    Yc3 = _ex(S.mul(R2))
    za = np.zeros((N, 2, n), complex)
    for j, b in enumerate(inv):
        za[:, j, b] = -1j * s
    dSa = CI.point(za)
    zk = np.zeros((N, K, n), complex)
    for l in range(n):
        zk[:, l, l] = 1.0
        zk[:, n + l, l] = 1j
    dSk = CI.point(zk)
    dVa = _solve3(-(dSa.mul(R_).conj().matvec(Z)), Yc3, Z, rho)
    dVk = _solve3(-(dSk.mul(R_).conj().matvec(Z)), Yc3, Z, rho)

    def dR(dV):
        return -(R2_.mul(dV))

    def d2R(dVx, dVy, d2V):
        return R3_.mul(dVx).mul(dVy).rscale(2.0, 2.0) - R2_.mul(d2V)

    def d2W(dSx, dVx, dSy, dVy, d2V):
        return dSx.mul(dR(dVy)) + dSy.mul(dR(dVx)) + S3.mul(d2R(dVx, dVy, d2V))

    def second(dSx, dVx, dSy, dVy):
        t = dSx.mul(R2_).mul(dVy) + dSy.mul(R2_).mul(dVx) - S3.mul(R3_).mul(dVx).mul(dVy).rscale(2.0, 2.0)
        return _solve3(t.conj().matvec(Z), Yc3, Z, rho)

    I3 = _ex(S.mul(R).conj().matvec(Dm))
    dIa = (dSa.mul(R_) + S3.mul(dR(dVa))).conj().matvec(Dm)
    dIk = (dSk.mul(R_) + S3.mul(dR(dVk))).conj().matvec(Dm)
    f0 = V.abs2()
    absV = (np.sqrt(np.maximum(f0[0], 1e-300)), np.sqrt(f0[1]))
    ia = _pad(1.0 / absV[1], 1.0 / absV[0])
    i1 = (0.5 * ia[0][:, None], 0.5 * ia[1][:, None])
    i3 = (0.25 * ia[0][:, None] ** 3, 0.25 * ia[1][:, None] ** 3)
    i5 = (0.375 * ia[0][:, None] ** 5, 0.375 * ia[1][:, None] ** 5)

    def df(dV):
        lo, hi = _rc(V3, dV)
        return 2 * lo, 2 * hi

    dfa, dfk = df(dVa), df(dVk)

    def jnet(dl, drho):
        lo = 6 * dl[0] - 4 * (drho[1][..., inv[0]] + drho[1][..., inv[1]])
        hi = 6 * dl[1] - 4 * (drho[0][..., inv[0]] + drho[0][..., inv[1]])
        return _pad(lo, hi)

    out = dict(g2=[], h3={}, kap2=None)
    V_ak, I_ak, f_ak = [], [], []
    for j in range(2):
        dSj, dVj, dIj = _sel(dSa, j), _sel(dVa, j), _sel(dIa, j)
        dfj = (dfa[0][:, j:j + 1], dfa[1][:, j:j + 1])
        V2 = second(dSj, dVj, dSk, dVk)
        I2 = d2W(dSj, dVj, dSk, dVk, V2).conj().matvec(Dm)
        l2 = _rsum(*_add(_rc(dIk, dIj), _rc(I3, I2)), 2 * rb)
        f2 = _add(_rc(dVk, dVj), _rc(V3, V2)); f2 = (2 * f2[0], 2 * f2[1])
        t1 = rmul(*f2, *i1)
        t2 = rmul(*rmul(*dfj, *dfk), *i3)
        rho2 = _pad(t1[0] - t2[1], t1[1] - t2[0])
        out["g2"].append(jnet(l2, rho2))
        if j == 0:
            out["kap2"] = rho2
        V_ak.append(V2); I_ak.append(I2); f_ak.append(f2)
    if not order3:
        return out
    V_aa, I_aa, f_aa = {}, {}, {}
    for (i, j) in ((0, 0), (0, 1), (1, 1)):
        V2 = second(_sel(dSa, i), _sel(dVa, i), _sel(dSa, j), _sel(dVa, j))
        V_aa[(i, j)] = V2
        I_aa[(i, j)] = d2W(_sel(dSa, i), _sel(dVa, i), _sel(dSa, j), _sel(dVa, j), V2).conj().matvec(Dm)
        f2 = _add(_rc(_sel(dVa, j), _sel(dVa, i)), _rc(V3, V2))
        f_aa[(i, j)] = (2 * f2[0], 2 * f2[1])
    for (i, j) in ((0, 0), (0, 1), (1, 1)):
        dSi, dSj = _sel(dSa, i), _sel(dSa, j)
        dVi, dVj = _sel(dVa, i), _sel(dVa, j)
        dIi, dIj = _sel(dIa, i), _sel(dIa, j)
        Vij, Vik, Vjk = V_aa[(i, j)], V_ak[i], V_ak[j]
        Rij, Rik, Rjk = d2R(dVi, dVj, Vij), d2R(dVi, dVk, Vik), d2R(dVj, dVk, Vjk)
        sumV = Vij.mul(dVk) + Vik.mul(dVj) + Vjk.mul(dVi)
        triple = dVi.mul(dVj).mul(dVk)
        b3 = dSi.mul(Rjk) + dSj.mul(Rik) + dSk.mul(Rij) \
            - S3.mul(R4_).mul(triple).rscale(6.0, 6.0) + S3.mul(R3_).mul(sumV).rscale(2.0, 2.0)
        V3d = _solve3(-(b3.conj().matvec(Z)), Yc3, Z, rho)
        R3d = R3_.mul(sumV).rscale(2.0, 2.0) - R4_.mul(triple).rscale(6.0, 6.0) - R2_.mul(V3d)
        I3d = (dSi.mul(Rjk) + dSj.mul(Rik) + dSk.mul(Rij) + S3.mul(R3d)).conj().matvec(Dm)
        Iij, Iik, Ijk = I_aa[(i, j)], I_ak[i], I_ak[j]
        l3 = _rsum(*_add(_rc(Ijk, dIi), _rc(dIj, Iik), _rc(dIk, Iij), _rc(I3, I3d)), 2 * rb)
        f3 = _add(_rc(Vjk, dVi), _rc(dVj, Vik), _rc(dVk, Vij), _rc(V3, V3d)); f3 = (2 * f3[0], 2 * f3[1])
        dfi = (dfa[0][:, i:i + 1], dfa[1][:, i:i + 1]); dfj = (dfa[0][:, j:j + 1], dfa[1][:, j:j + 1])
        fij, fik, fjk = f_aa[(i, j)], f_ak[i], f_ak[j]
        a1 = rmul(*f3, *i1)
        ss = _add(rmul(*fij, *dfk), rmul(*fik, *dfj), rmul(*fjk, *dfi))
        a2 = rmul(*ss, *i3)
        a3 = rmul(*rmul(*rmul(*dfi, *dfj), *dfk), *i5)
        rho3 = _pad(a1[0] - a2[1] + a3[0], a1[1] - a2[0] + a3[1])
        out["h3"][(i, j)] = jnet(l3, rho3)
    return out


def _direct_at(S, V, rho):
    """Network gradient/Hessian (without the explicit Q^2 term) and d|V|/du18 at
    point loads S with a tight solution enclosure V."""
    m = _M()
    Z, inv, rb = m["Z"], m["inv"], m["r"]
    Dm = m["D"].astype(complex)
    N, n = S.rl.shape
    s = GR.QMAX - GR.QMIN
    rV = V.recip(); rV2 = rV.mul(rV); rV3 = rV2.mul(rV)
    Yc = S.mul(rV2)
    dS = []
    for b in inv:
        z = np.zeros((N, n), dtype=complex); z[:, b] = -1j * s
        dS.append(CI.point(z))
    dV = [_solve_linear(-(dS[j].mul(rV).conj().matvec(Z)), Yc, Z, rho) for j in range(2)]
    dW = [dS[j].mul(rV) - Yc.mul(dV[j]) for j in range(2)]
    absV = np.sqrt(np.maximum(V.abs2()[0], 1e-300)), np.sqrt(V.abs2()[1])
    ia = _pad(1.0 / absV[1], 1.0 / absV[0])
    dr = []
    for j in range(2):
        re = V.conj().mul(dV[j]); dr.append(rmul(re.rl, re.rh, ia[0], ia[1]))
    Ib = S.mul(rV).conj().matvec(Dm)
    dI = [dW[j].conj().matvec(Dm) for j in range(2)]

    def jnet(dl, d):
        return _pad(6 * dl[0] - 4 * (d[1][:, inv[0]] + d[1][:, inv[1]]),
                    6 * dl[1] - 4 * (d[0][:, inv[0]] + d[0][:, inv[1]]))

    gradN = []
    for j in range(2):
        pr = Ib.conj().mul(dI[j])
        gradN.append(jnet(_rsum(pr.rl, pr.rh, 2 * rb), dr[j]))
    HN = {}
    for (i, j) in ((0, 0), (0, 1), (1, 1)):
        t = dS[i].mul(dV[j]).mul(rV2) + dS[j].mul(dV[i]).mul(rV2) - S.mul(dV[i]).mul(dV[j]).mul(rV3).rscale(2.0, 2.0)
        D2 = _solve_linear(t.conj().matvec(Z), Yc, Z, rho)
        d2W = (-(dS[i].mul(dV[j]).mul(rV2))) - dS[j].mul(dV[i]).mul(rV2) - Yc.mul(D2) \
            + S.mul(dV[i]).mul(dV[j]).mul(rV3).rscale(2.0, 2.0)
        d2I = d2W.conj().matvec(Dm)
        p1 = dI[i].conj().mul(dI[j]); p2 = Ib.conj().mul(d2I)
        d2l = _rsum(p1.rl + p2.rl, p1.rh + p2.rh, 2 * rb)
        a = dV[j].conj().mul(dV[i]); c = V.conj().mul(D2)
        t1 = rmul(a.rl + c.rl, a.rh + c.rh, ia[0], ia[1])
        t2 = rmul(*rmul(dr[i][0], dr[i][1], dr[j][0], dr[j][1]), ia[0], ia[1])
        HN[(i, j)] = jnet(d2l, _pad(t1[0] - t2[1], t1[1] - t2[0]))
    return dict(gradN=gradN, HN=HN, dr18=dr[0], absV=absV)


def mvf_bounds(lo, hi, order3=True):
    """Mean-value-form upper bounds per parameter box (lvl, qm, irr, u18, u33):
    G_j, L_j (Hessian row norms), kappa_b = sup |d|V_b|/du_18|, validity."""
    m = _M()
    s = GR.QMAX - GR.QMIN
    E = enclose(lo, hi)
    S, V, rho, ok = E["S"], E["V"], E["rho"], E["ok"]
    rP, rQ = 0.5 * (S.rh - S.rl), 0.5 * (S.ih - S.il)
    r = np.concatenate([rP, rQ], 1)
    Smid = 0.5 * (S.rl + S.rh) + 0.5j * (S.il + S.ih)
    Sc = CI.point(Smid)
    Vm = _point_solution(Smid)
    e = 1e-8
    Vc = CI(Vm.real - e, Vm.real + e, Vm.imag - e, Vm.imag + e)
    Y = (-(Sc.mul(Vc.recip()).conj().matvec(m["Z"]))) + m["V0"]
    okc = np.all(Y.subset_of(Vc), axis=1)
    Vc = Y.intersect(Vc)
    v2 = Vc.abs2()[0]
    rhoc = np.max((np.abs(m["Z"])[None] * (Sc.absmax() / v2)[:, None, :]).sum(2), 1) * (1 + 1e-12)
    oc = _direct_at(Sc, Vc, rhoc)
    db = derivative_bounds(S, V, rho, order3=order3)
    G = []
    for j in range(2):
        corr = (rabsmax(*db["g2"][j]) * r).sum(1)
        Qlo, Qhi = GR.QMIN + s * lo[:, 3 + j], GR.QMIN + s * hi[:, 3 + j]
        g = oc["gradN"][j]
        a, b = g[0] - corr + 0.7 * s * Qlo, g[1] + corr + 0.7 * s * Qhi
        G.append(np.maximum(np.abs(a), np.abs(b)) * (1 + 1e-12))
    kap = rabsmax(*oc["dr18"]) + (rabsmax(*db["kap2"]) * r[:, :, None]).sum(1)
    out = dict(ok=ok & okc & (rho < 1) & (rhoc < 1), rho=rho, G=G, kappa=kap)
    if order3:
        Hb = {}
        for (i, j) in ((0, 0), (0, 1), (1, 1)):
            corr = (rabsmax(*db["h3"][(i, j)]) * r).sum(1)
            h = oc["HN"][(i, j)]
            c = 0.7 * s * s if i == j else 0.0
            Hb[(i, j)] = np.maximum(np.abs(h[0] - corr + c), np.abs(h[1] + corr + c))
        out["L"] = [np.sqrt(Hb[(0, 0)] ** 2 + Hb[(0, 1)] ** 2), np.sqrt(Hb[(1, 1)] ** 2 + Hb[(0, 1)] ** 2)]
    return out
