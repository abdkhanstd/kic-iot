# Author: abdkhan
"""A1 as defined in the threat model: the attacker alters logged TELEMETRY
before structure learning, and everything learned from the log (the
discovered graph, a surrogate SCM, any observational effect estimate) is
refitted from the altered data. The earlier A1 stress test inverted graph
edges directly, which is graph tampering, not telemetry poisoning.

effect_inversion: on a fraction `frac` of logged rows, subtract
kappa*(duty_cycle - 0.5) from the QoS outcomes. Values stay in [0,1] and the
format is unchanged; in the poisoned log a higher duty cycle appears to
LOWER latency, loss and energy. Nothing about the plant changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QOS = ["latency", "packet_loss", "energy", "collision_rate", "retransmissions", "queue_length"]


def effect_inversion(df: pd.DataFrame, frac: float, kappa: float = 0.8, actuator: str = "duty_cycle",
                     targets=QOS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    rows = rng.random(len(df)) < frac
    shift = kappa * (out.loc[rows, actuator] - 0.5)
    for k in targets:
        out.loc[rows, k] = np.clip(out.loc[rows, k] - shift, 0.0, 1.0)
    return out
