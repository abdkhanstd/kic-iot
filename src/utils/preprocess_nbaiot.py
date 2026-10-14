# Author: abdkhan
"""Preprocess the real-world N-BaIoT dataset (UCI ML repo, IoT botnet
telemetry) into a labeled, subsampled combined table for:
  (a) a real-data causal-discovery demonstration (no known ground-truth
      graph exists for real deployments -- expected, discovery quality here
      is reported qualitatively/via stability, not SHD/SID),
  (b) root-cause counterfactual QA generation ("if this device had not
      been compromised, what would feature X look like?" using the benign
      distribution as the do(compromised=0) counterfactual baseline),
  (c) the second, heterogeneous domain for the federated-fusion demo.

N-BaIoT ships 115 statistical network-flow features per device, with
benign_traffic.csv plus per-attack-family/-subtype CSVs (mirai: ack, scan,
syn, udp, udpplain; gafgyt: combo, junk, scan, tcp, udp) under each of 9
IoT device folders.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nbaiot" / "extracted"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

MIRAI_TYPES = ["ack", "scan", "syn", "udp", "udpplain"]
GAFGYT_TYPES = ["combo", "junk", "scan", "tcp", "udp"]


def _load_sample(path: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df


def build(n_per_file: int = 2000, seed: int = 0) -> pd.DataFrame:
    device_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()])
    frames = []
    for dev_dir in device_dirs:
        device = dev_dir.name
        benign_path = dev_dir / "benign_traffic.csv"
        if benign_path.exists():
            df = _load_sample(benign_path, n_per_file, seed)
            df["device"] = device
            df["attack_family"] = "benign"
            df["attack_type"] = "benign"
            df["compromised"] = 0
            frames.append(df)

        mirai_dir = dev_dir / "mirai_attacks"
        if mirai_dir.exists():
            for t in MIRAI_TYPES:
                p = mirai_dir / f"{t}.csv"
                if p.exists():
                    df = _load_sample(p, n_per_file, seed)
                    df["device"] = device
                    df["attack_family"] = "mirai"
                    df["attack_type"] = t
                    df["compromised"] = 1
                    frames.append(df)

        gafgyt_dir = dev_dir / "gafgyt_attacks"
        if gafgyt_dir.exists():
            for t in GAFGYT_TYPES:
                p = gafgyt_dir / f"{t}.csv"
                if p.exists():
                    df = _load_sample(p, n_per_file, seed)
                    df["device"] = device
                    df["attack_family"] = "gafgyt"
                    df["attack_type"] = t
                    df["compromised"] = 1
                    frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def normalize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = ((df[c].clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_file", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading + subsampling N-BaIoT files...")
    combined = build(n_per_file=args.n_per_file, seed=args.seed)
    meta_cols = ["device", "attack_family", "attack_type", "compromised"]
    feature_cols = [c for c in combined.columns if c not in meta_cols]

    combined = normalize_features(combined, feature_cols)
    combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_path = OUT_DIR / "nbaiot_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows x {len(feature_cols)} features to {out_path}")
    print(combined[meta_cols].value_counts().head(20))


if __name__ == "__main__":
    main()
