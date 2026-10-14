# Author: abdkhan
"""Run the closed control loop for ONE adapter and save a partial result.

Exists so that several adapters can be evaluated concurrently, one per GPU
(set CUDA_VISIBLE_DEVICES), instead of serially inside a single process --
the closed-loop eval is generation-bound, so this roughly halves wall time
on the two-GPU box. `--merge` combines the partials into one comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import ground_truth_graph
from verifier.verify import CausalVerifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_integrated_control import (load_llm, run_llm_loop, run_llm_loops_batched,
                                     episode_summary, aggregate)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--adapter")
    ap.add_argument("--tag")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--verifier", choices=["on", "off", "both"], default="both")
    ap.add_argument("--batched", action="store_true",
                     help="advance all seeds in lockstep and batch their per-step LLM "
                          "queries (exact, not an approximation; fills the otherwise "
                          "~85%%-idle GPU)")
    ap.add_argument("--out_prefix", default="control_partial")
    ap.add_argument("--merge", nargs="*", default=None,
                     help="tags to merge into one comparison file")
    ap.add_argument("--merge_out", default="federated_distillation_control.json")
    args = ap.parse_args()

    if args.merge is not None:
        merged = {}
        for tag in args.merge:
            p = RESULTS_DIR / f"{args.out_prefix}_{tag}.json"
            merged.update(json.load(open(p)))
        with open(RESULTS_DIR / args.merge_out, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"=== merged -> {args.merge_out} ===")
        for name, r in merged.items():
            print(f"{name:34s} reward={r['reward']['mean']:7.3f}+-{r['reward']['std']:5.3f}  "
                  f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
                  f"energy={r['energy']['mean']:.3f}+-{r['energy']['std']:.3f}  "
                  f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")
        return

    seeds = list(range(1000, 1000 + args.n_seeds))
    g_true = ground_truth_graph()
    clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)

    print(f"Loading {args.adapter} ...")
    model, tokenizer = load_llm(args.base_model, args.adapter)

    modes = {"off": [False], "on": [True], "both": [False, True]}[args.verifier]
    results = {}
    for use_verifier in modes:
        name = f"{args.tag}_{'with' if use_verifier else 'no'}_verifier"
        print(f"\n=== {name} ({args.n_seeds} seeds x {args.n_steps} steps"
              f"{', batched' if args.batched else ''}) ===", flush=True)
        if args.batched:
            rows_by_seed = run_llm_loops_batched(model, tokenizer, seeds, args.n_steps,
                                                  use_verifier, verifier if use_verifier else None)
            per_seed = []
            for s in seeds:
                per_seed.append(episode_summary(rows_by_seed[s]))
                print(f"  seed={s}: reward={per_seed[-1]['reward']:.3f} "
                      f"latency={per_seed[-1]['latency']:.3f}", flush=True)
        else:
            per_seed = []
            for s in seeds:
                rows = run_llm_loop(model, tokenizer, s, args.n_steps, use_verifier,
                                     verifier if use_verifier else None)
                per_seed.append(episode_summary(rows))
                print(f"  seed={s}: reward={per_seed[-1]['reward']:.3f} "
                      f"latency={per_seed[-1]['latency']:.3f}", flush=True)
        results[name] = aggregate(per_seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.out_prefix}_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    for name, r in results.items():
        print(f"{name:34s} reward={r['reward']['mean']:7.3f}+-{r['reward']['std']:5.3f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}")


if __name__ == "__main__":
    main()
