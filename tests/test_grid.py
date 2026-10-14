# Author: abdkhan
"""The feeder simulator (src/certify/grid.py) against pandapower's
Newton-Raphson power flow on the same IEEE 33-bus network."""
import copy
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


def test_bfs_matches_pandapower():
    import pandapower as pp
    from certify import grid as G
    F = G.feeder()
    rng = np.random.default_rng(0)
    worst_v = worst_loss = 0.0
    for _ in range(10):
        lvl, qm, irr = rng.uniform(0.5, 1.3), rng.uniform(0.6, 1.4), rng.uniform(0, 1)
        qinv = rng.uniform(G.QMIN, G.QMAX, 2)
        net = copy.deepcopy(F["net"])
        net.load.p_mw *= lvl
        net.load.q_mvar *= lvl * qm
        for j, b in enumerate(G.INV_BUS):
            pp.create_sgen(net, b, p_mw=G.PV_RATED * irr, q_mvar=qinv[j])
        pp.runpp(net, tolerance_mva=1e-10, numba=False)
        P, Q = F["p"] * lvl, F["q"] * lvl * qm
        P, Q = P.copy(), Q.copy()
        for j, b in enumerate(G.INV_BUS):
            P[b] -= G.PV_RATED * irr
            Q[b] -= qinv[j]
        V, Ib = G.bfs((P + 1j * Q)[None, :], F["vm_ext"])
        loss = np.sum(np.real(F["z"]) * np.abs(Ib[0]) ** 2)
        worst_v = max(worst_v, np.max(np.abs(np.abs(V[0]) - net.res_bus.vm_pu.values)))
        worst_loss = max(worst_loss, abs(loss - net.res_line.pl_mw.sum()))
    assert worst_v < 1e-8 and worst_loss < 1e-8, (worst_v, worst_loss)


def test_grid_cost_is_affine_in_channels():
    """Eq. (1): J is affine in the measured channels (loss, v18, v33)."""
    from certify import grid as G
    base = {"q18": 0.4, "q33": 0.6, "loss": 0.2, "v18": 0.5, "v33": 0.5}
    j0 = G.operator_cost(base)
    for ch in ("loss", "v18", "v33"):
        a, b = dict(base), dict(base)
        a[ch] += 0.01
        b[ch] += 0.02
        assert abs((G.operator_cost(b) - j0) - 2 * (G.operator_cost(a) - j0)) < 1e-12




def test_certified_enclosures_contain_simulator():
    """grid_cert: for random boxes of the certified domain and random loads within
    the +-5 sigma noise box, the simulator's voltages and cost gradient lie inside
    the interval enclosures."""
    import numpy as np
    from certify import grid as GR, grid_cert as GC
    GC.K_SIGMA = 5.0
    F = GR.feeder()
    m = GC._M()
    rng = np.random.default_rng(4)
    dom = np.array([[0.1, 0.7], [0.0, 0.8], [0.0, 1.0], [0.02, 0.98], [0.02, 0.98]])
    lo = rng.uniform(dom[:, 0], dom[:, 1] - 0.04, (20, 5))
    hi = lo + 0.04
    o = GC.sensitivities(lo, hi, second=False)
    assert o["ok"].all()
    for i in range(20):
        x = rng.uniform(lo[i], hi[i])
        lvl, qm, irr = 0.5 + 0.8 * x[0], 0.6 + 0.8 * x[1], x[2]
        e = lambda k: 1 + 0.15 * rng.uniform(-1, 1, k)
        P = F["p"] * lvl * e(F["nb"]); Q = F["q"] * lvl * qm * e(F["nb"]); pv = GR.PV_RATED * irr * e(2)
        def cost(u18, u33):
            P2, Q2 = P.copy(), Q.copy()
            for j, (b, u) in enumerate(zip(GR.INV_BUS, (u18, u33))):
                P2[b] -= pv[j]; Q2[b] -= GR.QMIN + (GR.QMAX - GR.QMIN) * u
            V, Ib = GR.bfs((P2 + 1j * Q2)[None], F["vm_ext"], iters=200, tol=1e-14)
            loss = np.sum(np.real(F["z"]) * np.abs(Ib[0]) ** 2); vm = np.abs(V[0])
            q = lambda u: GR.QMIN + (GR.QMAX - GR.QMIN) * u
            return 6 * loss + 4 * ((1 - vm[17]) + (1 - vm[32])) + 0.35 * (q(u18) ** 2 + q(u33) ** 2), V[0]
        h = 1e-6
        _, V = cost(x[3], x[4])
        Vi = V[m["idx"]]
        assert np.all(Vi.real >= o["V"].rl[i] - 1e-9) and np.all(Vi.real <= o["V"].rh[i] + 1e-9)
        assert np.all(Vi.imag >= o["V"].il[i] - 1e-9) and np.all(Vi.imag <= o["V"].ih[i] + 1e-9)
        g = (cost(x[3] + h, x[4])[0] - cost(x[3] - h, x[4])[0]) / (2 * h)
        assert o["grad"][0][0][i] - 1e-5 <= g <= o["grad"][0][1][i] + 1e-5


if __name__ == "__main__":
    for f in [test_bfs_matches_pandapower, test_grid_cost_is_affine_in_channels, test_certified_enclosures_contain_simulator]:
        f(); print("ok", f.__name__)
