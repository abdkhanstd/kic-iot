# Author: abdkhan
"""Verifier baselines evaluated on IDENTICAL decisions to our causal gate.

The action-verification literature gates a model's proposed action before
execution, but the reference it checks against is typically the model's own
consistency or a symbolic model that is assumed correct -- not a causal
graph fitted from telemetry and held provenance-separate. Those systems
report task-specific success rates that cannot be compared with ours
directly, so we re-implement the two mechanisms they rely on and score them
on exactly the decisions our gate saw:

  llm-self-consistency  re-query the model at higher temperature n times and
                        accept only if the proposed action direction agrees
                        across samples (the self-consistency mechanism used
                        by prompt-level verification methods)
  llm-as-judge          ask the same on-device model to judge whether its own
                        causal claim is correct, and accept on "yes"
                        (the LLM-as-judge mechanism)
  reachability-only     accept iff a directed path exists from the actuator to
                        the claimed QoS variable, ignoring the sign -- the
                        graph-topology check used by graph-based verifiers
  causal verifier       ours (sign + range + rate + no-op)

Ground truth for "was the claim correct" is the true SCM's do-calculus, so
all gates are scored by the same oracle. These are re-implementations of the
described mechanisms, not the original authors' code.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from simulator.scm_iot import IoTClusterSCM, ground_truth_graph  # noqa: E402
from verifier.verify import (CausalVerifier, parse_action, parse_claims,  # noqa: E402
                              ACTIONABLE_VARS)
from distillation.build_corpus import PROMPT_TEMPLATE  # noqa: E402
from run_integrated_control import (load_llm, query_llm, build_instruction,  # noqa: E402
                                     QOS_FOR_OBJECTIVE)
from eval_verifier_precision import true_effect_sign  # noqa: E402

OUT = ROOT / "experiments" / "results" / "comparison_verifiers.json"


@torch.no_grad()
def _sample(model, tok, prompt, n, temp, max_new=70):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    outs = []
    for _ in range(n):
        o = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                            temperature=temp, top_p=0.95,
                            pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def gate_self_consistency(rec, model, tok, n=3, temp=0.8):
    """Accept only if resampled generations agree on the action direction."""
    outs = _sample(model, tok, rec["prompt"], n, temp)
    dirs = []
    for o in outs:
        p = parse_action(o)
        if p is None:
            continue
        (v, val), = p.items()
        if v in ACTIONABLE_VARS:
            dirs.append(np.sign(val - rec["cur"].get(v, 0.5)))
    return bool(dirs) and len(set(dirs)) == 1


@torch.no_grad()
def gate_llm_judge(rec, model, tok):
    """Ask the model to judge its own claim; accept on an affirmative reply."""
    q = (f"{rec['instruction']}\n\nA controller proposed: {rec['text'][:300]}\n"
         f"Is the stated direction of the causal effect correct? Answer yes or no.")
    prompt = PROMPT_TEMPLATE.format(instruction=q)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    o = model.generate(**inp, max_new_tokens=6, do_sample=False,
                        pad_token_id=tok.pad_token_id)
    ans = tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    return not re.search(r"\bno\b", ans)


def gate_reachability(rec, g_true):
    """Topology-only: is the claimed QoS variable reachable from the actuator?"""
    p, claims = rec["parsed"], rec["claims"]
    if p is None or not claims:
        return False
    (v, _), = p.items()
    if v not in ACTIONABLE_VARS:
        return False
    qos = claims[0][0]
    return v in g_true and qos in g_true and nx.has_path(g_true, v, qos)


def summarise(name, accepts, scored):
    tp = sum(1 for a, r in zip(accepts, scored) if a and r["correct"] == 1)
    fp = sum(1 for a, r in zip(accepts, scored) if a and r["correct"] == 0)
    fn = sum(1 for a, r in zip(accepts, scored) if not a and r["correct"] == 1)
    tn = sum(1 for a, r in zip(accepts, scored) if not a and r["correct"] == 0)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec_ = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = (2 * prec * rec_ / (prec + rec_)) if (prec == prec and rec_ == rec_ and prec + rec_) else float("nan")
    return dict(name=name, rejection_rate=1 - float(np.mean(accepts)),
                kept_correct=tp, kept_wrong=fp, rejected_correct=fn, rejected_wrong=tn,
                precision=prec, recall=rec_, specificity=spec, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--adapter", default="models/checkpoints/qwen05b-causal")
    ap.add_argument("--n_states", type=int, default=80)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    g_true = ground_truth_graph()
    clean_df = pd.read_csv(ROOT / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)
    model, tok = load_llm(args.base_model, str(ROOT / args.adapter))

    scm = IoTClusterSCM(seed=args.seed)
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    recs = []
    for i in range(args.n_states):
        obj = max(QOS_FOR_OBJECTIVE, key=lambda v: state[v])
        instruction = build_instruction(action, state, obj)
        text = query_llm(model, tok, action, state, obj, max_new_tokens=70)
        parsed, claims = parse_action(text), parse_claims(text)
        res = verifier.verify(text, action)

        correct = None
        if parsed is not None and claims:
            (v, val), = parsed.items()
            if v in ACTIONABLE_VARS and 0.0 <= val <= 1.0:
                t = true_effect_sign(scm, action, {v: val}, claims[0][0])
                if t is not None:
                    correct = int(t == claims[0][1])
        recs.append(dict(text=text, instruction=instruction,
                          prompt=PROMPT_TEMPLATE.format(instruction=instruction),
                          parsed=parsed, claims=claims, cur=dict(action),
                          correct=correct, causal_accept=res.verdict != "REJECT"))
        action = dict(res.safe_action)
        state = scm.step(action)
        if (i + 1) % 20 == 0:
            print(f"  collected {i+1}/{args.n_states}", flush=True)

    scored = [r for r in recs if r["correct"] is not None]
    print(f"scored {len(scored)}/{len(recs)}", flush=True)

    out = {}
    print("running self-consistency gate...", flush=True)
    out["LLM self-consistency"] = summarise(
        "LLM self-consistency", [gate_self_consistency(r, model, tok) for r in scored], scored)
    print("running LLM-as-judge gate...", flush=True)
    out["LLM-as-judge"] = summarise(
        "LLM-as-judge", [gate_llm_judge(r, model, tok) for r in scored], scored)
    out["Graph reachability only"] = summarise(
        "Graph reachability only", [gate_reachability(r, g_true) for r in scored], scored)
    out["Causal verifier (ours)"] = summarise(
        "Causal verifier (ours)", [r["causal_accept"] for r in scored], scored)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(n_scored=len(scored), n_states=args.n_states, methods=out), f, indent=2)

    print(f"\n{'gate':26s}{'rej':>8}{'prec':>8}{'rec':>8}{'spec':>8}{'F1':>8}")
    for k, v in out.items():
        g = lambda x: "   ---" if x != x else f"{x:6.3f}"
        print(f"{k:26s}{v['rejection_rate']:>8.3f}{g(v['precision'])}{g(v['recall'])}"
              f"{g(v['specificity'])}{g(v['f1'])}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
