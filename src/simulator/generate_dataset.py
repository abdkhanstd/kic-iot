# Author: abdkhan
"""Generate synthetic IoT telemetry datasets from the ground-truth SCM.

Produces:
  data/synthetic/clean_train.csv / clean_test.csv   -- observational data
  data/synthetic/ground_truth_graph.json            -- edge list for SHD/SID eval
  data/synthetic/interventional_probe.csv           -- do() probes for
      evaluating counterfactual-prediction accuracy (tiny-LLM + verifier)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.scm_iot import IoTClusterSCM, GROUND_TRUTH_EDGES, NODES


def gen_observational(n_steps: int, seed: int) -> pd.DataFrame:
    scm = IoTClusterSCM(seed=seed)
    rows = scm.rollout(n_steps)
    return pd.DataFrame(rows)


def gen_interventional_probes(n_probes: int, seed: int) -> pd.DataFrame:
    """For a grid of do(duty_cycle=x) / do(tx_power=x) interventions, record
    the exact SCM counterfactual mean outcome -- this is ground truth used
    later to score whether the distilled tiny LLM's answer to
    "what happens if duty_cycle -> x?" matches do-calculus."""
    scm = IoTClusterSCM(seed=seed + 10_000)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_probes):
        base_actions = dict(
            duty_cycle=float(rng.uniform(0.1, 0.9)),
            tx_power=float(rng.uniform(0.1, 0.9)),
        )
        var = rng.choice(["duty_cycle", "tx_power"])
        delta = rng.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
        new_val = float(np.clip(base_actions[var] + delta, 0.02, 0.98))
        do = {var: new_val}

        before = scm.counterfactual(base_actions, do={})
        after = scm.counterfactual(base_actions, do=do)

        rows.append(dict(
            intervened_var=var, from_value=round(base_actions[var], 3),
            to_value=round(new_val, 3),
            base_duty_cycle=round(base_actions["duty_cycle"], 3),
            base_tx_power=round(base_actions["tx_power"], 3),
            **{f"before_{k}": round(v, 4) for k, v in before.items() if k in
               ("collision_rate", "retransmissions", "queue_length", "cpu_load",
                "latency", "throughput", "packet_loss", "energy")},
            **{f"after_{k}": round(v, 4) for k, v in after.items() if k in
               ("collision_rate", "retransmissions", "queue_length", "cpu_load",
                "latency", "throughput", "packet_loss", "energy")},
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--n_probes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating observational train/test splits...")
    train_df = gen_observational(args.n_train, seed=args.seed)
    test_df = gen_observational(args.n_test, seed=args.seed + 1)
    train_df.to_csv(out / "clean_train.csv", index=False)
    test_df.to_csv(out / "clean_test.csv", index=False)

    print("Generating interventional probes (do-calculus ground truth)...")
    probes_df = gen_interventional_probes(args.n_probes, seed=args.seed + 2)
    probes_df.to_csv(out / "interventional_probes.csv", index=False)

    with open(out / "ground_truth_graph.json", "w") as f:
        json.dump({"nodes": NODES, "edges": GROUND_TRUTH_EDGES}, f, indent=2)

    print(f"Wrote {len(train_df)} train rows, {len(test_df)} test rows, "
          f"{len(probes_df)} interventional probes to {out}/")


if __name__ == "__main__":
    main()
