# Author: abdkhan
"""Characterized envelopes (G, L): the paired-input procedure of
certify/envelopes.py (maximum over 150 sampled regime--action pairs, both
actuators) on every plant and domain the paper reports.

    python kic_characterize_env.py

grid    : the feeder's full domain (level-C runs; declared G/L 2.2/4.5)
grid_Df : the feeder on its certified operating domain D_f (level-T runs)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify import plantspec
from certify.envelopes import derive, scm_regimes, mec_regimes, grid_regimes, grid_regimes_domain

RES = ROOT / "experiments" / "results"
PLANTS = {"scm": ("scm", scm_regimes), "mec": ("mec", mec_regimes),
          "grid": ("grid", grid_regimes), "grid_Df": ("grid", grid_regimes_domain)}


def main():
    out = {}
    for name, (plant, regimes) in PLANTS.items():
        t0 = time.time()
        G, L, Gj, Lj = derive(plantspec.get(plant), regimes, per_actuator=True)
        out[name] = dict(G=G, L=L, G_by_actuator=Gj, L_by_actuator=Lj)
        print(f"{name}: G={G:.4f} L={L:.4f} ({time.time() - t0:.0f}s)", flush=True)
    out["config"] = dict(n_states=150, d=0.05, n=2000, seed=0, numpy=np.__version__)
    json.dump(out, open(RES / "kic_characterized_env.json", "w"), indent=1)


if __name__ == "__main__":
    main()
