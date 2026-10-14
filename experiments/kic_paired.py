# Author: abdkhan
"""Paired seed-level comparison for the attack matrix. Gates are paired on
common random numbers: for a seed, every gate sees the same exogenous draws,
initial action, backdoor trigger times and key. Each gate runs its own closed
loop, so proposals diverge with the trajectory; the pairing is at the level
of the seed. The relevant uncertainty is that of the per-seed DIFFERENCE
(baseline - KIC), with KIC the main gate (kic_sound)."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
rows = json.load(open(R / "kic_attacks.json"))["rows"]
for f in ["kic_attacks_kic.json"]:   # KIC gates re-run with the window rule
    if (R / f).exists():
        extra = json.load(open(R / f))["rows"]
        redo = {r["gate"] for r in extra}
        rows = [r for r in rows if r["gate"] not in redo] + extra


def get(s, g):
    return {r["seed"]: r["ecost"] for r in rows if r["scenario"] == s and r["gate"] == g}


out, rng = {}, np.random.default_rng(0)
for s in sorted({r["scenario"] for r in rows}):
    k = get(s, "kic_sound")
    for g in ["none", "gstar_claims", "obsci", "obsci_lite", "obsrls", "wm", "wm_switch",
              "kic", "kic_hold", "kic_v1", "kic_sign", "kic_trial_sound", "kic_holdfw"]:
        b = get(s, g)
        diff = np.array([b[i] - k[i] for i in sorted(k)])
        bs = rng.choice(diff, (5000, len(diff))).mean(1)
        out[f"{s}|{g}"] = dict(mean=float(diff.mean()), lo=float(np.quantile(bs, .025)),
                               hi=float(np.quantile(bs, .975)), kic_better=int((diff > 0).sum()),
                               n=len(diff))
json.dump(out, open(R / "kic_attacks_paired.json", "w"), indent=1)
for key, v in out.items():
    print(f"{key:30s} {v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]  KIC better {v['kic_better']}/{v['n']}")
