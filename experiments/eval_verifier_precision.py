# Author: abdkhan
"""How good is the VERIFIER itself? (missing from every earlier experiment)

Everything so far measured what the verifier does to downstream control.
Nobody measured the verifier as what it actually is: a binary classifier over
"is this LLM causal claim correct?". Without that, a 50% rejection rate on a
correctly-distilled model is uninterpretable -- it could mean the model is
wrong half the time (verifier working) or that the verifier is over-strict
(verifier broken). This settles it by scoring the verifier against the TRUE
SCM.

For each sampled state we take the tiny LLM's proposal, then:
  ground truth : apply the proposed do() to the true SCM and measure whether
                 the claimed QoS variable actually moves in the claimed
                 direction (Monte-Carlo do-calculus, the same oracle used to
                 build the distillation corpus);
  verifier     : ACCEPT / REJECT from the trusted-graph sign check.
Cross-tabulating gives verifier precision, recall, and -- most importantly --
its false-rejection rate on claims that were in fact correct.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import IoTClusterSCM, ground_truth_graph
from verifier.verify import CausalVerifier, parse_action, parse_claims

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_integrated_control import load_llm, query_llm, QOS_FOR_OBJECTIVE

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
TOL = 0.005  # |delta| below this counts as "no real change"


def true_effect_sign(scm: IoTClusterSCM, action: dict, do: dict, qos_var: str,
                      n_samples: int = 300) -> int | None:
    """Ground-truth sign of the effect of `do` on `qos_var`, from the true SCM."""
    before = scm.counterfactual(action, do={}, n_samples=n_samples)
    after = scm.counterfactual(action, do=do, n_samples=n_samples)
    delta = after[qos_var] - before[qos_var]
    if abs(delta) < TOL:
        return 0
    return 1 if delta > 0 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--adapter", default="models/checkpoints/qwen05b-causal")
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--n_states", type=int, default=120)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--state_source", choices=["uniform", "onpolicy"], default="uniform",
                     help="'uniform' samples fresh mid-range actions i.i.d.; 'onpolicy' scores "
                          "the states the closed-loop controller actually drives itself into "
                          "(where the action saturates at the actuator limits). The two give "
                          "very different answers, which is the point.")
    args = ap.parse_args()

    g_true = ground_truth_graph()
    clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)
    model, tokenizer = load_llm(args.base_model, args.adapter)

    rng = np.random.default_rng(args.seed)
    scm = IoTClusterSCM(seed=args.seed)
    rows = []
    # on-policy: the action persists and is updated by the verifier's verdict,
    # exactly as in the deployed control loop, so the trajectory drifts toward
    # the actuator limits the way it does in run_integrated_control.
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    for i in range(args.n_states):
        if args.state_source == "uniform":
            action = dict(duty_cycle=float(rng.uniform(0.15, 0.85)),
                          tx_power=float(rng.uniform(0.15, 0.85)))
            state = scm.step(action)
        objective = max(QOS_FOR_OBJECTIVE, key=lambda v: state[v])
        text = query_llm(model, tokenizer, action, state, objective, max_new_tokens=70)

        parsed = parse_action(text)
        claims = parse_claims(text)
        result = verifier.verify(text, action)
        accepted = int(result.verdict != "REJECT")

        claim_correct = None
        if parsed is not None and claims:
            (var, value), = parsed.items()
            if var in ("duty_cycle", "tx_power") and 0.0 <= value <= 1.0:
                qos_var, claimed_sign = claims[0]
                truth = true_effect_sign(scm, action, {var: value}, qos_var)
                if truth is not None:
                    # a claim is "correct" if the true effect moves the way it says
                    claim_correct = int(truth == claimed_sign)
        rows.append(dict(i=i, accepted=accepted, verdict=result.verdict,
                          reason=result.reason.split(":")[0], claim_correct=claim_correct,
                          duty_cycle=action["duty_cycle"], tx_power=action["tx_power"]))
        if args.state_source == "onpolicy":
            action = dict(result.safe_action)
            state = scm.step(action)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.n_states}", flush=True)

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["claim_correct"]).copy()
    scored["claim_correct"] = scored["claim_correct"].astype(int)

    # confusion of the verifier as a classifier of "claim is correct"
    tp = int(((scored.accepted == 1) & (scored.claim_correct == 1)).sum())  # correctly kept
    fp = int(((scored.accepted == 1) & (scored.claim_correct == 0)).sum())  # wrongly kept
    fn = int(((scored.accepted == 0) & (scored.claim_correct == 1)).sum())  # wrongly rejected
    tn = int(((scored.accepted == 0) & (scored.claim_correct == 0)).sum())  # correctly rejected

    precision = tp / (tp + fp) if (tp + fp) else float("nan")   # of accepted, share truly correct
    recall = tp / (tp + fn) if (tp + fn) else float("nan")      # of correct claims, share kept
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")  # of wrong claims, share caught
    false_rejection = fn / (fn + tp) if (fn + tp) else float("nan")

    out = dict(
        tag=args.tag, adapter=args.adapter, state_source=args.state_source,
        mean_duty_cycle=float(df.duty_cycle.mean()), mean_tx_power=float(df.tx_power.mean()),
        frac_steps_at_actuator_limit=float((
            (df.duty_cycle <= 0.03) | (df.duty_cycle >= 0.97) |
            (df.tx_power <= 0.03) | (df.tx_power >= 0.97)).mean()),
        n_states=args.n_states,
        n_scored=len(scored), n_unscorable=int(df.claim_correct.isna().sum()),
        overall_accept_rate=float(df.accepted.mean()),
        base_rate_claims_correct=float(scored.claim_correct.mean()),
        confusion=dict(kept_correct=tp, kept_wrong=fp, rejected_correct=fn, rejected_wrong=tn),
        precision_of_accepted=precision, recall_of_correct=recall,
        specificity_caught_wrong=specificity, false_rejection_rate=false_rejection,
        reason_counts=df.reason.value_counts().to_dict(),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"verifier_precision_{args.tag}_{args.state_source}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Verifier as a classifier of 'LLM claim is causally correct' "
          f"[{args.tag}, {args.state_source} states] ===")
    print(f"mean action: duty_cycle={out['mean_duty_cycle']:.3f} tx_power={out['mean_tx_power']:.3f}"
          f"  | steps at actuator limit: {out['frac_steps_at_actuator_limit']:.1%}")
    print(f"scored {len(scored)}/{args.n_states} proposals "
          f"({out['n_unscorable']} unscorable: no parsable claim/action)")
    print(f"base rate -- claims that were actually correct : {out['base_rate_claims_correct']:.3f}")
    print(f"                       kept(correct) kept(wrong) | rej(correct) rej(wrong)")
    print(f"                       {tp:>12} {fp:>11} | {fn:>12} {tn:>9}")
    print(f"precision of accepted claims  : {precision:.3f}")
    print(f"recall of correct claims      : {recall:.3f}")
    print(f"specificity (wrong caught)    : {specificity:.3f}")
    print(f"FALSE REJECTION rate          : {false_rejection:.3f}  <- cost of the gate")


if __name__ == "__main__":
    main()
