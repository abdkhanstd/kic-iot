# Author: abdkhan
"""How often the simulated C1 forger's range clipping binds.

The C1 forger writes clip(Y_true - kappa (x - 0.5) + offset, 0, 1) on each forged channel (x the
published nominal; offset 0 on the SCM, 0.3 on the MEC node and the feeder). C1 is additive; where
the clip binds, the written value depends on the genuine reading, and that tick's forgery is
outside C1. This script replays the shadow-mode trajectories of the in-model forging scenarios
(plant and forger only: the trajectory does not depend on any gate) and counts binding readings.

    python experiments/kic_c1_clip_rate.py      -> results/kic_c1_clip_rate.json
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np


def replay(plant_name, seeds, stationary, dwell):
    import kic_shadow as KS
    from certify import plantspec
    KS.DWELL, KS.N_DWELL, KS.STATIONARY = dwell, 1, stationary
    P = plantspec.get(plant_name)
    n = bound = 0
    for seed in range(seeds):
        class C:
            pass
        cfg = C(); cfg.seed = seed; cfg.regime_shift_prob = 0.0 if stationary else 0.002; cfg.plant_memory = 0.0
        cfg.noise_mult = 1.0; cfg.drift_amp = 0.0; cfg.drift_period = 1600.0
        plant = P.make(cfg)
        if stationary:
            from certify import envelopes
            plant.regime = envelopes.regimes_for(plant_name)(np.random.default_rng(seed + 7))
        sp = KS.spoofer_for(plant_name, "spoof_blind")
        exc = KS.KeyedRademacher(f"key-{seed}".encode())
        traj = KS.trajectory(seed, P.actuators)
        for t in range(1, len(traj) + 1):
            centre, signs, applied = KS.dither(traj[t - 1], t, exc, 0.05, P.actuators)
            ts = plant.step(applied)
            x = centre[sp.actuator]
            for k in sp.targets:
                raw = ts[k] - sp.kappa * (x - 0.5) + sp.offset
                n += 1; bound += int(raw < 0.0 or raw > 1.0)
    return dict(readings=n, bound=bound, rate=bound / n)


def main():
    out = {"scm (level S)": replay("scm", 30, False, 800), "scm (level T)": replay("scm", 30, True, 800),
           "mec (level S)": replay("mec", 30, False, 800), "mec (level C)": replay("mec", 30, True, 800)}
    os.environ["KIC_GRID_DOMAIN"] = "1"           # the feeder on D_f (level T); read at import below
    import importlib
    from certify import grid as GR
    importlib.reload(GR)
    out["feeder on D_f (level T)"] = replay("grid", 15, True, 800)
    for k, v in out.items():
        print(k, v, flush=True)
    json.dump(out, open(RES / "kic_c1_clip_rate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
