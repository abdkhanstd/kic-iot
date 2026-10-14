# Author: abdkhan
"""Branch-and-bound certification of the feeder's envelopes over a claimed
operating domain D, in parallel.

Per box of D (demand, reactive share, irradiance, u18, u33), with every bus's
load and the PV output anywhere within +-K sigma of their means:
  1. the plain interval method (grid_cert.box_bounds) encloses the power-flow
     solution and bounds G_j, L_j and the leakage gains kappa_b;
  2. boxes whose plain bounds exceed the targets are bisected; after
     `switch_width` they are evaluated by the mean-value form (grid_mvf),
     which is tighter but slower;
  3. a box is closed when its enclosure is valid (Phi(X) in X, rho < 1) and
     its G, L bounds are at most the targets.
The certified values are the maxima over closed boxes; the run fails if any
box stays open. Also recorded: enclosures of the loss and of |V| at the two
inverters (the channels are never clipped on D), and sup rho.

    python -m certify.grid_certify_run --procs 60 --G 2.9 --L 6.5
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from certify import grid_cert as GC

RES = Path(__file__).resolve().parents[2] / "experiments" / "results"


def _eval(args):
    lo, hi, use_mvf, ksig = args
    GC.K_SIGMA = ksig
    if use_mvf:
        from certify import grid_mvf as MV
        o = MV.mvf_bounds(lo, hi)
        G, L, kap = o["G"], o["L"], o["kappa"]
        E = GC.enclose(lo, hi)
        loss = (np.full(len(lo), np.nan), np.full(len(lo), np.nan))
        absV = None
        ok = o["ok"]
    else:
        o, G, L = GC.box_bounds(lo, hi)
        kap = GC.rabsmax(*o["dr18"])
        loss = o["loss"]
        absV = o["absV"]
        ok = o["ok"]
    inv = GC._M()["inv"]
    av = (np.full(len(lo), np.nan), np.full(len(lo), np.nan)) if absV is None else \
        (absV[0][:, inv].min(1), absV[1][:, inv].max(1))
    return (np.stack(G, 1), np.stack(L, 1), kap, ok, o["rho"], loss[0], loss[1], av[0], av[1])


def run(domain, targets, grid, procs, ksig, switch_width=0.02, max_rounds=30, chunk=512, log=print, loss_min=0.0):
    los, his = GC.grid_boxes(domain, grid)
    scale = np.array([b - a for a, b in domain])
    best = dict(G=[0.0, 0.0], L=[0.0, 0.0], kappa=None, rho=0.0, loss=[np.inf, -np.inf], absV_inv=[np.inf, -np.inf])
    n_eval, n_mvf = 0, 0
    tG, tL = np.array(targets["G"]), np.array(targets["L"])
    with Pool(procs) as pool:
        for r in range(max_rounds):
            width = ((his - los) / scale).max(1)
            use_mvf = width <= switch_width + 1e-12
            jobs = []
            for flag in (False, True):
                idx = np.where(use_mvf == flag)[0]
                for i in range(0, len(idx), chunk if not flag else 64):
                    sel = idx[i:i + (chunk if not flag else 64)]
                    jobs.append((sel, (los[sel], his[sel], flag, ksig)))
            res = pool.map(_eval, [j[1] for j in jobs], chunksize=1)
            Gs = np.zeros((len(los), 2)); Ls = np.zeros((len(los), 2)); oks = np.zeros(len(los), bool)
            L0 = np.zeros(len(los))
            kaps = None
            for (sel, _), (G, L, kap, ok, rho, l0, l1, a0, a1) in zip(jobs, res):
                Gs[sel], Ls[sel], oks[sel] = G, L, ok
                L0[sel] = l0
                if kaps is None:
                    kaps = np.zeros((len(los), kap.shape[1]))
                kaps[sel] = kap
                if ok.any():
                    best["rho"] = max(best["rho"], float(rho[ok].max()))
                    if np.isfinite(l0[ok]).any():
                        best["loss"] = [min(best["loss"][0], float(np.nanmin(l0[ok]))), max(best["loss"][1], float(np.nanmax(l1[ok])))]
                    if np.isfinite(a0[ok]).any():
                        best["absV_inv"] = [min(best["absV_inv"][0], float(np.nanmin(a0[ok]))),
                                            max(best["absV_inv"][1], float(np.nanmax(a1[ok])))]
            n_eval += len(los); n_mvf += int(use_mvf.sum())
            # the loss channel must stay clear of its clipping bound (0) by the meter's k-sigma noise
            closed = oks & np.all(Gs <= tG, 1) & np.all(Ls <= tL, 1) & (L0 >= loss_min)
            if closed.any():
                for j in range(2):
                    best["G"][j] = max(best["G"][j], float(Gs[closed, j].max()))
                    best["L"][j] = max(best["L"][j], float(Ls[closed, j].max()))
                k = kaps[closed].max(0)
                best["kappa"] = k if best["kappa"] is None else np.maximum(best["kappa"], k)
            op = ~closed
            log(f"round {r}: boxes {len(los)} (mvf {int(use_mvf.sum())}), closed {int(closed.sum())}, "
                f"invalid {int((~oks).sum())}, max open G {np.round(Gs[op].max(0), 3) if op.any() else '-'} "
                f"L {np.round(Ls[op].max(0), 3) if op.any() else '-'}, min width {width.min():.4f}", flush=True)
            if not op.any():
                return dict(certified=True, n_eval=n_eval, n_mvf=n_mvf, **best)
            los, his = GC.bisect(los[op], his[op], scale)
            if len(los) > 2_000_000:
                break
    return dict(certified=False, n_eval=n_eval, n_mvf=n_mvf, open_boxes=int(len(los)), **best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=60)
    ap.add_argument("--G", type=float, default=2.9)
    ap.add_argument("--L", type=float, default=6.5)
    ap.add_argument("--ksig", type=float, default=5.0)
    ap.add_argument("--demand", type=float, nargs=2, default=[0.0, 0.8])
    ap.add_argument("--qshare", type=float, nargs=2, default=[0.0, 1.0])
    ap.add_argument("--irr", type=float, nargs=2, default=[0.0, 1.0])
    ap.add_argument("--switch", type=float, default=0.03)
    ap.add_argument("--tag", default="")
    ap.add_argument("--loss_lo", type=float, default=0.0, help="lower end of the loss telemetry range (MW)")
    a = ap.parse_args()
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    dom = [tuple(a.demand), tuple(a.qshare), tuple(a.irr), (0.02, 0.98), (0.02, 0.98)]
    t0 = time.time()
    from certify import grid as GR
    res = run(dom, dict(G=[a.G, a.G], L=[a.L, a.L]), grid=(4, 4, 4, 8, 8), procs=a.procs, ksig=a.ksig,
              switch_width=a.switch, loss_min=a.loss_lo + a.ksig * GR.NOISE["meter_loss"] + 1e-4)
    res.update(seconds=time.time() - t0, domain=dom, targets=dict(G=a.G, L=a.L), k_sigma=a.ksig, loss_lo=a.loss_lo,
               loss_channel_ok=bool(res["loss"][0] > a.loss_lo + a.ksig * GR.NOISE["meter_loss"]
                                    and res["loss"][1] < GR.LOSS_HI - a.ksig * GR.NOISE["meter_loss"]),
               v_channel_ok=bool(res["absV_inv"][0] > GR.V_LO + a.ksig * GR.NOISE["meter_v"]
                                 and res["absV_inv"][1] < GR.V_HI - a.ksig * GR.NOISE["meter_v"]))
    res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    print(json.dumps({k: v for k, v in res.items() if k != "kappa"}, indent=1))
    json.dump(res, open(RES / f"kic_certified_grid_env{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
