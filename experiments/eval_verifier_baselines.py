# Author: abdkhan
"""Compare our causal verifier against simpler gates on the same decisions.

A gate that rejects half of everything will trivially block many bad actions;
the question is whether the *causal* machinery buys anything over cheaper
heuristics at the same rejection budget. We therefore replay one fixed set of
model proposals (so every gate sees identical inputs) through:

  accept-all            no gate at all -- the floor
  random@rate           rejects uniformly at random, rate matched to ours --
                        controls for "rejecting a lot is what helps"
  syntax-only           actionability + range + rate limit, i.e. our gate
                        with the causal sign/no-op reasoning removed
  magnitude-only        rejects only no-op proposals (no causal reasoning)
  causal verifier       the full gate
Ground truth for "was this claim correct" is the true SCM's do-calculus, as
in eval_verifier_precision.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import IoTClusterSCM, ground_truth_graph
from verifier.verify import (CausalVerifier, parse_action, parse_claims,
                              ACTIONABLE_VARS, MAX_STEP, NO_OP_EPS)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_integrated_control import load_llm, query_llm, QOS_FOR_OBJECTIVE
from eval_verifier_precision import true_effect_sign

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def gate_accept_all(rec, rng, rate):
    return True


def gate_random(rec, rng, rate):
    return rng.random() >= rate


def gate_syntax(rec, rng, rate):
    p = rec["parsed"]
    if p is None:
        return False
    (var, val), = p.items()
    if var not in ACTIONABLE_VARS or not (0.0 <= val <= 1.0):
        return False
    return abs(val - rec["cur"]) <= MAX_STEP


def gate_magnitude(rec, rng, rate):
    if not gate_syntax(rec, rng, rate):
        return False
    (var, val), = rec["parsed"].items()
    return abs(float(np.clip(val, 0.02, 0.98)) - rec["cur"]) >= NO_OP_EPS


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--adapter", default="models/checkpoints/qwen05b-causal")
    ap.add_argument("--n_states", type=int, default=120)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    g_true = ground_truth_graph()
    clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)
    model, tokenizer = load_llm(args.base_model, args.adapter)

    # --- collect one fixed decision set on the on-policy distribution ---
    scm = IoTClusterSCM(seed=args.seed)
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    records = []
    for i in range(args.n_states):
        objective = max(QOS_FOR_OBJECTIVE, key=lambda v: state[v])
        text = query_llm(model, tokenizer, action, state, objective, max_new_tokens=70)
        parsed, claims = parse_action(text), parse_claims(text)
        res = verifier.verify(text, action)

        correct = None
        if parsed is not None and claims:
            (var, val), = parsed.items()
            if var in ACTIONABLE_VARS and 0.0 <= val <= 1.0:
                qv, sign = claims[0]
                t = true_effect_sign(scm, action, {var: val}, qv)
                if t is not None:
                    correct = int(t == sign)
        records.append(dict(parsed=parsed, cur=action.get(
            list(parsed.keys())[0] if parsed else "duty_cycle", 0.5),
            correct=correct, causal_accept=int(res.verdict != "REJECT")))
        action = dict(res.safe_action)
        state = scm.step(action)
        if (i + 1) % 30 == 0:
            print(f"  collected {i+1}/{args.n_states}", flush=True)

    scored = [r for r in records if r["correct"] is not None]
    causal_rate = 1.0 - np.mean([r["causal_accept"] for r in records])

    gates = {"accept-all": gate_accept_all, "random@matched-rate": gate_random,
             "syntax-only": gate_syntax, "magnitude-only": gate_magnitude}
    out = {}
    for name, fn in gates.items():
        rng = np.random.default_rng(0)
        acc = [int(bool(fn(r, rng, causal_rate))) for r in scored]
        out[name] = summarise(name, acc, scored)
    out["causal verifier (ours)"] = summarise(
        "causal verifier (ours)", [r["causal_accept"] for r in scored], scored)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "verifier_baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'gate':26s}{'rej.rate':>10}{'precision':>11}{'recall':>9}{'specif.':>9}{'F1':>7}")
    for k, v in out.items():
        g = lambda x: "  --- " if x != x else f"{x:6.3f}"
        print(f"{k:26s}{v['rejection_rate']:>10.3f}{g(v['precision'])}{g(v['recall'])}"
              f"{g(v['specificity'])}{g(v['f1'])}")


def summarise(name, accepts, scored):
    tp = sum(1 for a, r in zip(accepts, scored) if a == 1 and r["correct"] == 1)
    fp = sum(1 for a, r in zip(accepts, scored) if a == 1 and r["correct"] == 0)
    fn = sum(1 for a, r in zip(accepts, scored) if a == 0 and r["correct"] == 1)
    tn = sum(1 for a, r in zip(accepts, scored) if a == 0 and r["correct"] == 0)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec) else float("nan")
    return dict(name=name, rejection_rate=1 - float(np.mean(accepts)),
                kept_correct=tp, kept_wrong=fp, rejected_correct=fn, rejected_wrong=tn,
                precision=prec, recall=rec, specificity=spec, f1=f1)


if __name__ == "__main__":
    main()
