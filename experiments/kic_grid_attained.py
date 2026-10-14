# Author: abdkhan
"""Attained (sampled) sensitivity and curvature of the feeder's cost on the
certified operating domain D_f, for comparison with the certified bounds.
Loads and PV at their +-5 sigma corners half the time, uniform otherwise.

    python kic_grid_attained.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify import grid as GR

DOM = [(0.1, 0.7), (0.0, 0.8), (0.0, 1.0), (0.02, 0.98), (0.02, 0.98)]


def main(reps=8, n=10000, k=0.15):
    F = GR.feeder(); nb = F["nb"]
    rng = np.random.default_rng(11)
    G = np.zeros(2); Lm = np.zeros(2)

    def J(x, eP, eQ, epv, u18, u33):
        lvl = 0.5 + 0.8 * x[:, 0]; qm = 0.6 + 0.8 * x[:, 1]; irr = x[:, 2]
        P = F["p"][None] * lvl[:, None] * eP; Q = F["q"][None] * (lvl * qm)[:, None] * eQ
        for j, (b, u) in enumerate(zip(GR.INV_BUS, (u18, u33))):
            P[:, b] -= GR.PV_RATED * irr * epv[:, j]; Q[:, b] -= GR.QMIN + (GR.QMAX - GR.QMIN) * u
        V, Ib = GR.bfs(P + 1j * Q, F["vm_ext"], iters=100, tol=1e-13)
        loss = np.sum(np.real(F["z"])[None] * np.abs(Ib) ** 2, 1); vm = np.abs(V)
        q = lambda u: GR.QMIN + (GR.QMAX - GR.QMIN) * u
        return 6 * loss + 4 * ((1 - vm[:, 17]) + (1 - vm[:, 32])) + 0.35 * (q(u18) ** 2 + q(u33) ** 2)

    for _ in range(reps):
        x = rng.uniform([a for a, _ in DOM[:3]], [b for _, b in DOM[:3]], (n, 3))
        u = rng.uniform(0.02 + 1e-3, 0.98 - 1e-3, (n, 2))
        noise = lambda shape: 1 + k * np.where(rng.random(shape) < 0.5, rng.choice([-1, 1], shape), rng.uniform(-1, 1, shape))
        eP, eQ, epv = noise((n, nb)), noise((n, nb)), noise((n, 2))
        h = 1e-4
        f = lambda a, b: J(x, eP, eQ, epv, a, b)
        j0 = f(u[:, 0], u[:, 1])
        jp, jm = f(u[:, 0] + h, u[:, 1]), f(u[:, 0] - h, u[:, 1])
        kp, km = f(u[:, 0], u[:, 1] + h), f(u[:, 0], u[:, 1] - h)
        H00 = (jp - 2 * j0 + jm) / h ** 2; H11 = (kp - 2 * j0 + km) / h ** 2
        H01 = (f(u[:, 0] + h, u[:, 1] + h) - f(u[:, 0] + h, u[:, 1] - h) - f(u[:, 0] - h, u[:, 1] + h)
               + f(u[:, 0] - h, u[:, 1] - h)) / (4 * h * h)
        G = np.maximum(G, [np.abs((jp - jm) / (2 * h)).max(), np.abs((kp - km) / (2 * h)).max()])
        Lm = np.maximum(Lm, [np.sqrt(H00 ** 2 + H01 ** 2).max(), np.sqrt(H11 ** 2 + H01 ** 2).max()])
    out = dict(G=G.tolist(), L=Lm.tolist(), samples=reps * n, domain=DOM, noise_halfwidth=k)
    print(out)
    json.dump(out, open(ROOT / "experiments" / "results" / "kic_grid_attained.json", "w"), indent=1)


if __name__ == "__main__":
    main()
