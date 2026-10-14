# Author: abdkhan
"""Novel angle #3: the neuro-symbolic verifier as a ZERO-COST ONLINE
COMPROMISE DETECTOR, not merely an action filter.

Every adjacent LLM-action-verification paper found in this project's
literature check (CIVeX, GNNVerifier, PASE, the process-control
hallucination-prevention work) uses verification only as a *gate*: an
action is accepted or rejected, and the rejected action is discarded. But
the gate's own decision stream is itself telemetry. If the tiny LLM was
distilled from a poisoned corpus, or the live causal graph feeding it has
been attacked, the model starts emitting causally-inconsistent claims at a
measurably higher rate -- so the verifier's REJECTION RATE is a free
runtime signal that something upstream has been compromised, requiring no
extra model, no labelled attack data, and no additional inference cost
(the verifier is already running on every action).

This script quantifies that: it runs the clean-corpus LLM and the
poisoned-corpus LLM through the identical closed control loop, logs every
verifier verdict, and asks whether the per-window rejection rate separates
the two populations well enough to serve as a detector (reported as a
threshold-free AUC over sliding windows plus a simple operating point).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import IoTClusterSCM, ground_truth_graph
from distillation.build_corpus import PROMPT_TEMPLATE, LABELS
from verifier.verify import CausalVerifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from run_integrated_control import (load_llm, query_llm, build_instruction,
                                     QOS_FOR_OBJECTIVE)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


@torch.no_grad()
def run_loops_logging_verdicts_batched(model, tokenizer, verifier, seeds: list[int],
                                        n_steps: int, max_new_tokens: int = 70) -> dict[int, list[dict]]:
    """Batched (lockstep-across-seeds) version of run_loop_logging_verdicts.
    Equivalent to running each seed alone -- see run_llm_loops_batched in
    run_integrated_control.py for why this is exact rather than approximate."""
    scms = {s: IoTClusterSCM(seed=s) for s in seeds}
    actions = {s: dict(duty_cycle=0.5, tx_power=0.5) for s in seeds}
    states = {s: scms[s].step(actions[s]) for s in seeds}
    logs = {s: [] for s in seeds}

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for step in range(n_steps):
            prompts = []
            for s in seeds:
                objective = max(QOS_FOR_OBJECTIVE, key=lambda v: states[s][v])
                prompts.append(PROMPT_TEMPLATE.format(
                    instruction=build_instruction(actions[s], states[s], objective)))
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
            gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for s, text in zip(seeds, gen):
                result = verifier.verify(text, actions[s])
                actions[s] = dict(result.safe_action)
                states[s] = scms[s].step(actions[s])
                logs[s].append(dict(step=step, verdict=result.verdict,
                                    reason=result.reason.split(":")[0],
                                    rejected=int(result.verdict == "REJECT"),
                                    latency=states[s]["latency"]))
    finally:
        tokenizer.padding_side = prev_side
    return logs


def run_loop_logging_verdicts(model, tokenizer, verifier, seed: int, n_steps: int,
                               max_new_tokens: int = 70) -> list[dict]:
    """Same closed loop as run_integrated_control, but records the verifier's
    verdict and reason at every control tick.

    max_new_tokens is 70 rather than 100 here: the adapters were trained
    without an EOS token so generation never stops early, and everything the
    verifier parses (the do(...) action and the primary directional claim)
    is emitted well inside 70 tokens -- the rest is the model rambling past
    the end of its answer at ~0.07 s/token.
    """
    scm = IoTClusterSCM(seed=seed)
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    log = []
    for step in range(n_steps):
        objective = max(QOS_FOR_OBJECTIVE, key=lambda v: state[v])
        gen_text = query_llm(model, tokenizer, action, state, objective,
                              max_new_tokens=max_new_tokens)
        result = verifier.verify(gen_text, action)
        action = dict(result.safe_action)
        state = scm.step(action)
        log.append(dict(step=step, verdict=result.verdict, reason=result.reason.split(":")[0],
                        rejected=int(result.verdict == "REJECT"), latency=state["latency"]))
    return log


def sliding_window_rates(rejected: list[int], window: int) -> list[float]:
    r = np.asarray(rejected, dtype=float)
    if len(r) < window:
        return [float(r.mean())] if len(r) else []
    return [float(r[i:i + window].mean()) for i in range(len(r) - window + 1)]


def auc_from_scores(pos: list[float], neg: list[float]) -> float:
    """Threshold-free separability (Mann-Whitney U / probability that a
    randomly drawn compromised window scores higher than a clean one),
    with ties counted as 0.5 -- no sklearn dependency needed."""
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _detection_metrics(clean: dict, poisoned: dict, window: int) -> dict:
    # Two candidate detector statistics computed over the same windows:
    #   (a) overall rejection RATE  -- the obvious choice, and it turns out
    #       to be a weak signal (both models get rejected at similar rates);
    #   (b) the share of rejections attributable to OUT_OF_RANGE -- the
    #       rejection REASON, which separates the two models categorically.
    auc_rate = auc_from_scores(poisoned["window_rates"], clean["window_rates"])
    thr = 0.5 * (clean["window_rate_mean"] + poisoned["window_rate_mean"])
    out = dict(
        window=window,
        clean_rejection_rate=clean["overall_rejection_rate"],
        poisoned_rejection_rate=poisoned["overall_rejection_rate"],
        detection_auc=auc_rate, operating_threshold=thr,
        tpr_at_threshold=float(np.mean([w > thr for w in poisoned["window_rates"]])),
        fpr_at_threshold=float(np.mean([w > thr for w in clean["window_rates"]])),
        clean_reason_counts=clean["verdict_reason_counts"],
        poisoned_reason_counts=poisoned["verdict_reason_counts"],
    )
    if "window_oor_rates" in clean and "window_oor_rates" in poisoned:
        auc_reason = auc_from_scores(poisoned["window_oor_rates"], clean["window_oor_rates"])
        thr_r = 0.5 * (clean["window_oor_mean"] + poisoned["window_oor_mean"])
        out.update(
            clean_out_of_range_rate=clean["oor_rate"],
            poisoned_out_of_range_rate=poisoned["oor_rate"],
            detection_auc_by_reason=auc_reason,
            reason_operating_threshold=thr_r,
            reason_tpr=float(np.mean([w > thr_r for w in poisoned["window_oor_rates"]])),
            reason_fpr=float(np.mean([w > thr_r for w in clean["window_oor_rates"]])),
        )
    return out


def merge_partials(args):
    parts = {}
    for tag in ["clean_corpus_llm", "poisoned_corpus_llm"]:
        p = RESULTS_DIR / f"verifier_detector_partial_{tag}.json"
        if not p.exists():
            raise SystemExit(f"missing partial for {tag}: {p}")
        parts[tag] = json.load(open(p))
    window = parts["clean_corpus_llm"]["window"]
    out = _detection_metrics(parts["clean_corpus_llm"]["clean_corpus_llm"],
                              parts["poisoned_corpus_llm"]["poisoned_corpus_llm"], window)
    with open(RESULTS_DIR / "verifier_as_detector.json", "w") as f:
        json.dump(out, f, indent=2)
    print("=== Verifier-as-detector ===")
    print(f"rejection rate  clean={out['clean_rejection_rate']:.3f}  "
          f"poisoned={out['poisoned_rejection_rate']:.3f}")
    print(f"detection AUC over {window}-step windows = {out['detection_auc']:.3f}  "
          f"(TPR={out['tpr_at_threshold']:.2f} @ FPR={out['fpr_at_threshold']:.2f}, "
          f"threshold={out['operating_threshold']:.3f})")
    print(f"clean reasons:    {out['clean_reason_counts']}")
    print(f"poisoned reasons: {out['poisoned_reason_counts']}")
    if "detection_auc_by_reason" in out:
        print(f"REASON-based detector (share of OUT_OF_RANGE rejections):")
        print(f"  clean={out['clean_out_of_range_rate']:.3f}  poisoned={out['poisoned_out_of_range_rate']:.3f}"
              f"  AUC={out['detection_auc_by_reason']:.3f}  "
              f"TPR={out['reason_tpr']:.2f} @ FPR={out['reason_fpr']:.2f}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--clean_adapter", default="models/checkpoints/qwen05b-causal")
    ap.add_argument("--poisoned_adapter", default="models/checkpoints/qwen05b-causal-poisoned")
    ap.add_argument("--n_steps", type=int, default=60)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--only", choices=["clean_corpus_llm", "poisoned_corpus_llm"], default=None,
                     help="run a single condition and write a partial file (lets the two "
                          "conditions run concurrently, one per GPU, then be merged)")
    ap.add_argument("--merge", action="store_true",
                     help="merge previously written partial files and compute detection metrics")
    args = ap.parse_args()

    if args.merge:
        merge_partials(args)
        return

    g_true = ground_truth_graph()
    clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)
    seeds = list(range(3000, 3000 + args.n_seeds))

    conditions = [("clean_corpus_llm", args.clean_adapter),
                   ("poisoned_corpus_llm", args.poisoned_adapter)]
    if args.only:
        conditions = [c for c in conditions if c[0] == args.only]

    all_logs, summary = {}, {}
    for tag, adapter in conditions:
        print(f"\n=== {tag} ({adapter}) ===")
        model, tokenizer = load_llm(args.base_model, adapter)
        logs, window_rates, window_oor = [], [], []
        logs_by_seed = run_loops_logging_verdicts_batched(model, tokenizer, verifier,
                                                           seeds, args.n_steps)
        for s in seeds:
            log = logs_by_seed[s]
            logs.extend(log)
            window_rates.extend(sliding_window_rates([e["rejected"] for e in log], args.window))
            # reason-based score: share of steps rejected specifically for
            # OUT_OF_RANGE. The aggregate reason counts separate the clean and
            # poisoned models categorically (one fails only causal-sign checks,
            # the other only range checks), so this -- not the overall
            # rejection rate -- is the signal worth scoring per window.
            window_oor.extend(sliding_window_rates(
                [int(e["reason"] == "OUT_OF_RANGE") for e in log], args.window))
            rr = np.mean([e["rejected"] for e in log])
            print(f"  seed={s}: rejection_rate={rr:.3f}", flush=True)
        all_logs[tag] = logs
        reasons = pd.Series([e["reason"] for e in logs]).value_counts().to_dict()
        summary[tag] = dict(
            overall_rejection_rate=float(np.mean([e["rejected"] for e in logs])),
            window_rate_mean=float(np.mean(window_rates)), window_rate_std=float(np.std(window_rates)),
            n_windows=len(window_rates), verdict_reason_counts=reasons,
            window_rates=window_rates,
            oor_rate=float(np.mean([e["reason"] == "OUT_OF_RANGE" for e in logs])),
            window_oor_mean=float(np.mean(window_oor)), window_oor_rates=window_oor,
        )
        del model

    if args.only:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        partial = RESULTS_DIR / f"verifier_detector_partial_{args.only}.json"
        with open(partial, "w") as f:
            json.dump({args.only: summary[args.only], "window": args.window}, f, indent=2)
        print(f"\nWrote partial: {partial}")
        return

    out = _detection_metrics(summary["clean_corpus_llm"], summary["poisoned_corpus_llm"],
                              args.window)
    out.update(n_steps=args.n_steps, n_seeds=args.n_seeds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "verifier_as_detector.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Verifier-as-detector ===")
    print(f"rejection rate  clean={out['clean_rejection_rate']:.3f}  "
          f"poisoned={out['poisoned_rejection_rate']:.3f}")
    print(f"detection AUC over {args.window}-step windows = {out['detection_auc']:.3f}  "
          f"(TPR={out['tpr_at_threshold']:.2f} @ FPR={out['fpr_at_threshold']:.2f}, "
          f"threshold={out['operating_threshold']:.3f})")
    print(f"clean reasons:    {out['clean_reason_counts']}")
    print(f"poisoned reasons: {out['poisoned_reason_counts']}")
    if "detection_auc_by_reason" in out:
        print(f"REASON-based detector (share of OUT_OF_RANGE rejections):")
        print(f"  clean={out['clean_out_of_range_rate']:.3f}  poisoned={out['poisoned_out_of_range_rate']:.3f}"
              f"  AUC={out['detection_auc_by_reason']:.3f}  "
              f"TPR={out['reason_tpr']:.2f} @ FPR={out['reason_fpr']:.2f}")



if __name__ == "__main__":
    main()
