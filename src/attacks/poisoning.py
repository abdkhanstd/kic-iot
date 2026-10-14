# Author: abdkhan
"""Causal-graph poisoning attacks on the telemetry feeding causal discovery
(component 4's adversarial-robustness study).

Three attack families, each parameterized by an intensity in [0,1]:

  1. Adversarial missingness: rows are dropped/masked in a way correlated
     with a target variable (not-missing-at-random), which is known to bias
     constraint-based causal discovery toward spurious independencies
     (in the spirit of the ACM TIST adversarial-missingness paper).
  2. Value perturbation ("edge injection"): additive adversarial noise
     correlated between two variables that are NOT truly causally linked,
     to try to trick discovery into adding a spurious edge.
  3. Value flipping ("edge deletion"): noise anti-correlated with a true
     parent-child pair, to try to mask a real edge (push its estimated
     dependency toward zero).

Robustness metric pipeline (see experiments/run_poisoning_ablation.py):
  clean SHD/SID -> poisoned SHD/SID -> downstream QoS-control degradation
  -> same pipeline with the verifier active, to measure how much the
  verifier mitigates the damage from a corrupted graph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adversarial_missingness(df: pd.DataFrame, target: str, intensity: float,
                             seed: int = 0) -> pd.DataFrame:
    """Drop rows where `target` is high, with probability scaling with
    intensity -- creates non-random missingness that biases the observed
    correlation structure involving `target`."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    thresh = df[target].quantile(0.5)
    high_mask = df[target] > thresh
    drop_prob = np.where(high_mask, intensity, intensity * 0.1)
    keep = rng.random(len(df)) > drop_prob
    return df[keep].reset_index(drop=True)


def edge_injection(df: pd.DataFrame, var_a: str, var_b: str, intensity: float,
                    seed: int = 0) -> pd.DataFrame:
    """Inject correlated noise between two variables with no true causal
    link, to try to fool discovery into adding a spurious edge a<->b."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    shared_noise = rng.normal(0, intensity, size=len(df))
    df[var_a] = np.clip(df[var_a] + shared_noise, 0, 1)
    df[var_b] = np.clip(df[var_b] + shared_noise, 0, 1)
    return df


def edge_masking(df: pd.DataFrame, parent: str, child: str, intensity: float,
                  seed: int = 0) -> pd.DataFrame:
    """Add anti-correlated noise to mask a true parent->child dependency,
    pushing the estimated correlation of a real edge toward zero."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    resid = df[child] - df[child].mean()
    counter_noise = -intensity * resid.to_numpy() + rng.normal(0, 0.02, size=len(df))
    df[child] = np.clip(df[child] + counter_noise, 0, 1)
    return df


ATTACKS = {
    "missingness": adversarial_missingness,
    "injection": edge_injection,
    "masking": edge_masking,
}
