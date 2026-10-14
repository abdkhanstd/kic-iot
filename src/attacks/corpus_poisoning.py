# Author: abdkhan
"""Distillation-corpus poisoning: a second, distinct threat surface from the
discovery-time poisoning in attacks/poisoning.py. There, an attacker
corrupts the TELEMETRY feeding causal discovery. Here, an attacker corrupts
a fraction of the INSTRUCTION-TUNING LABELS used to distill the SCM into
the tiny LLM -- e.g. a compromised data pipeline, a poisoned crowd-sourced
labeling step, or a supply-chain attack on a shared distillation corpus.

None of the adjacent neuro-symbolic-verification-for-LLM-actions work found
in this project's literature check (CIVeX, GNNVerifier, PASE, the
process-control hallucination-prevention paper) considers this: they all
verify LLM outputs against a given, trusted symbolic/causal model, but
none stress-tests what happens when the MODEL PRODUCING those outputs was
itself trained on adversarially mislabeled data. The point of this
experiment is to show the verifier -- built from a trusted graph fit on
data that never passed through the (possibly compromised) distillation
pipeline -- still catches confidently-wrong claims even when the LLM's own
weights have been corrupted at training time, which discovery-time-only
poisoning defenses cannot address.

For each selected "interventional"-kind example, the true after-effect is
reflected around the before-value (poisoned = 2*before - after), which
flips both the direction words AND the numeric "Predicted values" in the
regenerated output text -- an internally consistent but causally WRONG
label, rather than random noise a model could easily learn to ignore.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distillation.build_corpus import QOS_VARS, LABELS, _direction


def poison_example(ex: dict) -> dict:
    meta = ex["meta"]
    before, after = meta["before"], meta["after"]
    poisoned_after = {k: float(np.clip(2 * before[k] - after[k], 0.0, 1.0)) for k in QOS_VARS}

    var_label = "duty cycle" if meta["var"] == "duty_cycle" else "transmit power"
    to_val = meta["to_value"]

    effects = [f"{LABELS[qv]} will {_direction(before[qv], poisoned_after[qv])}" for qv in QOS_VARS]
    output = (
        f"Setting {var_label} to {to_val*100:.0f}% (do({meta['var']}={to_val:.2f})): "
        + "; ".join(effects) + ". "
        f"Predicted values: latency={poisoned_after['latency']:.2f}, "
        f"throughput={poisoned_after['throughput']:.2f}, packet_loss={poisoned_after['packet_loss']:.2f}, "
        f"energy={poisoned_after['energy']:.2f} (baseline: latency={before['latency']:.2f}, "
        f"throughput={before['throughput']:.2f}, packet_loss={before['packet_loss']:.2f}, "
        f"energy={before['energy']:.2f})."
    )
    poisoned = dict(ex)
    poisoned["output"] = output
    poisoned["meta"] = dict(meta, after=poisoned_after, poisoned=True)
    return poisoned


_DIR_SWAP = {"increase": "decrease", "decrease": "increase"}


def poison_example_direction_only(ex: dict) -> dict:
    """FORMAT-PRESERVING causal attack: flip only the direction WORDS
    ("increase" <-> "decrease"), leaving every numeric value byte-identical.

    Motivation: the reflection attack in poison_example() corrupts the
    numbers as well, and empirically that destroys the model's numeric
    coherence -- the resulting adapter emits "do(duty_cycle=2)" (a
    percentage where a fraction belongs), which the verifier catches as
    OUT_OF_RANGE. That makes the resulting detector look like it detects
    causal poisoning when it actually detects malformed output.

    This attack is the control for that confound: every number the model
    sees stays valid, in-range and drawn from the clean distribution, and
    ONLY the stated causal direction is wrong. If the verifier's rejection
    reasons still separate this model from a clean one -- but now via
    CAUSAL_CHECK_FAILED rather than OUT_OF_RANGE -- then the rejection-reason
    distribution genuinely fingerprints the *kind* of compromise.
    """
    out = ex["output"]
    poisoned = re.sub(r"\b(increase|decrease)\b",
                       lambda m: _DIR_SWAP[m.group(1).lower()], out, flags=re.IGNORECASE)
    new = dict(ex)
    new["output"] = poisoned
    new["meta"] = dict(ex["meta"], poisoned=True, poison_kind="direction_words_only")
    return new


def poison_corpus_targeted(examples: list[dict], fraction: float, seed: int = 0,
                            objective: str = "latency") -> tuple[list[dict], list[int]]:
    """TARGETED attack: spend the same poison budget only on the examples that
    matter for the control objective.

    The uniform attacks above scatter corrupted labels across the corpus, so
    most of the budget lands on examples the controller rarely relies on. A
    real adversary who knows the deployment objective would instead corrupt
    only the claims about the variable being optimised. This selects examples
    whose stated effect concerns `objective` and flips just those direction
    words, leaving every numeral untouched -- so at equal budget it is a
    strictly more efficient use of the same corruption, and it is the
    realistic threat model for an attacker with knowledge of the control
    loop. Comparing this against poison_corpus_direction_only at matched
    fraction isolates the value of targeting.
    """
    rng = np.random.default_rng(seed)
    obj_words = {"latency": ["latency"], "energy": ["energy"],
                 "packet_loss": ["packet loss", "packet_loss"],
                 "throughput": ["throughput"]}[objective]
    pat = re.compile("|".join(re.escape(w) for w in obj_words), re.IGNORECASE)

    eligible = [i for i, ex in enumerate(examples)
                if pat.search(ex["output"])
                and re.search(r"\b(increase|decrease)\b", ex["output"], re.IGNORECASE)]
    n_poison = int(len(examples) * fraction)          # budget relative to WHOLE corpus,
    n_poison = min(n_poison, len(eligible))            # so budgets are comparable
    idx = set(rng.choice(eligible, size=n_poison, replace=False).tolist())

    out = []
    for i, ex in enumerate(examples):
        if i not in idx:
            out.append(ex)
            continue
        new = poison_example_direction_only(ex)
        new["meta"] = dict(new["meta"], poison_kind=f"targeted_{objective}")
        out.append(new)
    return out, sorted(idx)


def poison_corpus_direction_only(examples: list[dict], fraction: float,
                                  seed: int = 0) -> tuple[list[dict], list[int]]:
    """Apply the format-preserving attack to `fraction` of ALL examples that
    contain a direction word (both interventional and proposal styles)."""
    rng = np.random.default_rng(seed)
    eligible = [i for i, ex in enumerate(examples)
                if re.search(r"\b(increase|decrease)\b", ex["output"], re.IGNORECASE)]
    n_poison = int(len(eligible) * fraction)
    idx = set(rng.choice(eligible, size=n_poison, replace=False).tolist())
    return [poison_example_direction_only(ex) if i in idx else ex
            for i, ex in enumerate(examples)], sorted(idx)


def poison_corpus(examples: list[dict], fraction: float, seed: int = 0) -> tuple[list[dict], list[int]]:
    rng = np.random.default_rng(seed)
    interventional_idx = [i for i, ex in enumerate(examples) if ex["meta"]["kind"] == "interventional"]
    n_poison = int(len(interventional_idx) * fraction)
    poison_idx = set(rng.choice(interventional_idx, size=n_poison, replace=False).tolist())

    out = []
    for i, ex in enumerate(examples):
        out.append(poison_example(ex) if i in poison_idx else ex)
    return out, sorted(poison_idx)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="data/synthetic/distill_train.jsonl")
    ap.add_argument("--out_file", default="data/synthetic/distill_train_poisoned.jsonl")
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=["reflect", "direction_only", "targeted"], default="reflect",
                     help="'reflect' corrupts the predicted numbers too (and collaterally "
                          "the model's numeric coherence); 'direction_only' flips just the "
                          "direction words, leaving all numerics valid -- the control that "
                          "isolates causal corruption from formatting corruption")
    args = ap.parse_args()

    with open(args.in_file) as f:
        examples = [json.loads(line) for line in f]

    if args.kind == "targeted":
        poisoned, poison_idx = poison_corpus_targeted(examples, args.fraction, args.seed)
    elif args.kind == "direction_only":
        poisoned, poison_idx = poison_corpus_direction_only(examples, args.fraction, args.seed)
    else:
        poisoned, poison_idx = poison_corpus(examples, args.fraction, args.seed)
    with open(args.out_file, "w") as f:
        for ex in poisoned:
            f.write(json.dumps(ex) + "\n")
    print(f"Poisoned {len(poison_idx)}/{len(examples)} examples ({args.fraction:.0%}) -> {args.out_file}")


if __name__ == "__main__":
    main()
