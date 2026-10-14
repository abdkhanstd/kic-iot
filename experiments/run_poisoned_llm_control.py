# Author: abdkhan
"""The distillation-time-poisoning closed-loop experiment: a tiny LLM
fine-tuned on a corpus where 20% of the interventional training labels
were adversarially flipped (src/attacks/corpus_poisoning.py) is run through
the SAME closed control loop as run_integrated_control.py, with and
without the neuro-symbolic verifier.

The verifier's reference statistics come from clean_train.csv, which never
passes through the (here, poisoned) distillation pipeline -- this is the
scenario none of the adjacent neuro-symbolic-verification-for-LLM-actions
papers found in this project's literature check test: does verification
still protect the system when the model PRODUCING the actions was itself
trained on adversarially corrupted causal labels, not just when the
online/live causal graph is attacked (that threat surface is covered
separately by run_poisoning_ablation.py).
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
from run_integrated_control import load_llm, run_llm_loop, episode_summary, aggregate

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--poisoned_adapter", default="models/checkpoints/qwen05b-causal-poisoned")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()

    seeds = list(range(1000, 1000 + args.n_seeds))
    g_true = ground_truth_graph()
    clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)

    print(f"Loading distillation-corpus-POISONED tiny LLM ({args.poisoned_adapter})...")
    model, tokenizer = load_llm(args.base_model, args.poisoned_adapter)

    results = {}
    for name, use_verifier in [("poisoned_llm_no_verifier", False), ("poisoned_llm_with_verifier", True)]:
        print(f"\n=== {name} ({args.n_seeds} seeds x {args.n_steps} steps) ===")
        per_seed = []
        for s in seeds:
            rows = run_llm_loop(model, tokenizer, s, args.n_steps, use_verifier,
                                 verifier if use_verifier else None)
            per_seed.append(episode_summary(rows))
            print(f"  seed={s}: reward={per_seed[-1]['reward']:.2f} latency={per_seed[-1]['latency']:.3f}")
        results[name] = aggregate(per_seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "poisoned_llm_control_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL: distillation-corpus poisoning, mean +/- std over seeds ===")
    for name in ["poisoned_llm_no_verifier", "poisoned_llm_with_verifier"]:
        r = results[name]
        print(f"{name:28s} reward={r['reward']['mean']:7.2f}+-{r['reward']['std']:5.2f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
              f"energy={r['energy']['mean']:.3f}+-{r['energy']['std']:.3f}  "
              f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")


if __name__ == "__main__":
    main()
