# Author: abdkhan
"""Distill the ground-truth SCM into a natural-language instruction corpus
of interventional ("do") and counterfactual QoS questions, used to LoRA
fine-tune a sub-1B tiny LLM so it can answer "what would happen if..."
queries and propose interventions without running the symbolic solver at
inference time.

Two question families:
  1. Interventional: "If duty_cycle is set to X (from Y), what happens to
     latency/energy/throughput/packet_loss?" -- answer derived from
     scm.counterfactual(do={...}).
  2. Intervention-proposal (control task): "Latency is high and battery is
     draining fast -- what single action would most reduce latency without
     increasing energy above Z?" -- answer chosen by grid-searching
     do(duty_cycle=...) / do(tx_power=...) against the SCM and reporting the
     best micro-intervention plus its predicted effect (this is the
     behaviour the verifier will later check against the live SCM/graph).

Output: JSONL, one instruction-following example per line, in a simple
Alpaca-style {instruction, input, output} schema.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.scm_iot import IoTClusterSCM

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

QOS_VARS = ["latency", "throughput", "packet_loss", "energy"]
LABELS = {
    "latency": "end-to-end latency", "throughput": "throughput",
    "packet_loss": "packet loss rate", "energy": "energy consumption",
    "collision_rate": "collision rate", "retransmissions": "retransmission rate",
    "queue_length": "queue length", "cpu_load": "CPU load",
}


def _direction(before, after, tol=0.02):
    d = after - before
    if abs(d) < tol:
        return "stay roughly unchanged"
    pct = abs(d) * 100
    return f"{'increase' if d > 0 else 'decrease'} by about {pct:.1f} percentage points"


def interventional_example(scm: IoTClusterSCM, rng: np.random.Generator) -> dict:
    var = rng.choice(["duty_cycle", "tx_power"])
    var_label = "duty cycle" if var == "duty_cycle" else "transmit power"
    base_actions = dict(duty_cycle=float(rng.uniform(0.15, 0.85)),
                         tx_power=float(rng.uniform(0.15, 0.85)))
    delta_pct = int(rng.choice([10, 15, 20, 25, 30]))
    sign = rng.choice([-1, 1])
    from_val = base_actions[var]
    to_val = float(np.clip(from_val + sign * delta_pct / 100.0, 0.02, 0.98))

    before = scm.counterfactual(base_actions, do={})
    after = scm.counterfactual(base_actions, do={var: to_val})

    verb = "reduced" if sign < 0 else "increased"
    instruction = (
        f"On an autonomous IoT edge node currently running {var_label} at "
        f"{from_val*100:.0f}% and the other radio settings unchanged, what happens "
        f"to QoS if {var_label} is {verb} by {delta_pct} percentage points "
        f"(to {to_val*100:.0f}%)?"
    )
    effects = []
    for qv in QOS_VARS:
        effects.append(f"{LABELS[qv]} will {_direction(before[qv], after[qv])}")
    output = (
        f"Setting {var_label} to {to_val*100:.0f}% (do({var}={to_val:.2f})): "
        + "; ".join(effects) + ". "
        f"Predicted values: latency={after['latency']:.2f}, "
        f"throughput={after['throughput']:.2f}, packet_loss={after['packet_loss']:.2f}, "
        f"energy={after['energy']:.2f} (baseline: latency={before['latency']:.2f}, "
        f"throughput={before['throughput']:.2f}, packet_loss={before['packet_loss']:.2f}, "
        f"energy={before['energy']:.2f})."
    )
    return dict(instruction=instruction, input="", output=output,
                meta=dict(kind="interventional", var=var, from_value=round(from_val, 3),
                          to_value=round(to_val, 3),
                          before={k: round(before[k], 4) for k in QOS_VARS},
                          after={k: round(after[k], 4) for k in QOS_VARS}))


def proposal_example(scm: IoTClusterSCM, rng: np.random.Generator) -> dict:
    base_actions = dict(duty_cycle=float(rng.uniform(0.15, 0.85)),
                         tx_power=float(rng.uniform(0.15, 0.85)))
    objective = rng.choice(["latency", "energy", "packet_loss"])
    obj_label = LABELS[objective]

    candidates = []
    for var in ["duty_cycle", "tx_power"]:
        for delta in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
            to_val = float(np.clip(base_actions[var] + delta, 0.02, 0.98))
            after = scm.counterfactual(base_actions, do={var: to_val})
            candidates.append((var, to_val, after))
    before = scm.counterfactual(base_actions, do={})
    best = min(candidates, key=lambda c: c[2][objective])
    var, to_val, after = best
    var_label = "duty cycle" if var == "duty_cycle" else "transmit power"

    instruction = (
        f"An autonomous IoT edge node is running duty cycle={base_actions['duty_cycle']*100:.0f}% "
        f"and transmit power={base_actions['tx_power']*100:.0f}%, with current "
        f"latency={before['latency']:.2f}, throughput={before['throughput']:.2f}, "
        f"packet_loss={before['packet_loss']:.2f}, energy={before['energy']:.2f}. "
        f"Propose a single intervention on duty cycle or transmit power to minimize {obj_label}."
    )
    output = (
        f"Recommended action: set {var_label} to {to_val*100:.0f}% "
        f"(do({var}={to_val:.2f})). Predicted effect: {obj_label} moves from "
        f"{before[objective]:.2f} to {after[objective]:.2f} "
        f"({_direction(before[objective], after[objective])}); side effects: "
        + "; ".join(f"{LABELS[qv]} {before[qv]:.2f}->{after[qv]:.2f}"
                     for qv in QOS_VARS if qv != objective) + "."
    )
    return dict(instruction=instruction, input="", output=output,
                meta=dict(kind="proposal", objective=objective, var=var,
                          to_value=round(to_val, 3),
                          do={var: round(to_val, 3)}))


def build(n_interventional: int, n_proposal: int, seed: int) -> list[dict]:
    scm = IoTClusterSCM(seed=seed + 1)
    return build_from_predictor(scm, n_interventional, n_proposal, seed)


def build_from_predictor(predictor, n_interventional: int, n_proposal: int, seed: int) -> list[dict]:
    """Generic corpus builder: `predictor` needs only a
    `.counterfactual(actions, do={...})` method. Used with IoTClusterSCM
    (ground-truth oracle, do-calculus with real exogenous noise) in the
    default pipeline, or with distillation.surrogate_scm.LinearSurrogateSCM
    (a realistic, no-oracle-access, discovered-graph-only surrogate) for
    the federation-feeds-distillation experiment, where the interesting
    question is whether a BETTER discovered graph (e.g. federated-fused
    instead of one data-scarce site's local discovery) yields a better
    tiny LLM even though neither ever sees the true SCM directly.
    """
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n_interventional):
        examples.append(interventional_example(predictor, rng))
    for _ in range(n_proposal):
        examples.append(proposal_example(predictor, rng))
    rng.shuffle(examples)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--n_train_interventional", type=int, default=4000)
    ap.add_argument("--n_train_proposal", type=int, default=2000)
    ap.add_argument("--n_val_interventional", type=int, default=400)
    ap.add_argument("--n_val_proposal", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = build(args.n_train_interventional, args.n_train_proposal, seed=args.seed)
    val = build(args.n_val_interventional, args.n_val_proposal, seed=args.seed + 999)

    for name, data in [("distill_train.jsonl", train), ("distill_val.jsonl", val)]:
        with open(out / name, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(data)} examples to {out/name}")


if __name__ == "__main__":
    main()
