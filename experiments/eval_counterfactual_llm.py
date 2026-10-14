# Author: abdkhan
"""Experiment: counterfactual/interventional accuracy of the distilled tiny
LLM (Section 5, second bullet) -- mean absolute error between the tiny
LLM's predicted counterfactual QoS outcome and the SCM's exact do-calculus
prediction (data/synthetic/interventional_probes.csv), evaluated BEFORE and
AFTER the neuro-symbolic verifier gate (component 3), and, when multiple
adapters are supplied, as the model-size ablation (Section 5's "as much
experiments as possible" / model-size sweep).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distillation.build_corpus import PROMPT_TEMPLATE  # noqa: E402
from verifier.verify import CausalVerifier
from simulator.scm_iot import ground_truth_graph

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
QOS_VARS = ["latency", "throughput", "packet_loss", "energy"]

NUM_PATTERN = {v: re.compile(rf"{v}=([0-9]*\.?[0-9]+)") for v in QOS_VARS}


def parse_predicted_values(text: str) -> dict:
    out = {}
    for v, pat in NUM_PATTERN.items():
        matches = pat.findall(text)
        if matches:
            out[v] = float(matches[-1])  # "Predicted values:" block comes after "baseline:" mentions, take last
    return out


def load_model(base_model_path: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16,
                                                  device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, instruction: str, max_new_tokens: int = 150) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def build_query(row) -> str:
    var_label = "duty cycle" if row["intervened_var"] == "duty_cycle" else "transmit power"
    verb = "increased" if row["to_value"] > row["from_value"] else "reduced"
    delta_pct = abs(row["to_value"] - row["from_value"]) * 100
    return (f"On an autonomous IoT edge node currently running {var_label} at "
            f"{row['from_value']*100:.0f}% and the other radio settings unchanged, what happens "
            f"to QoS if {var_label} is {verb} by {delta_pct:.0f} percentage points "
            f"(to {row['to_value']*100:.0f}%)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n_probes", type=int, default=60)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--use_verifier", action="store_true")
    args = ap.parse_args()

    probes = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "interventional_probes.csv")
    probes = probes.sample(n=min(args.n_probes, len(probes)), random_state=0).reset_index(drop=True)

    model, tokenizer = load_model(args.base_model, args.adapter)

    verifier = None
    if args.use_verifier:
        g_true = ground_truth_graph()
        clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
        verifier = CausalVerifier(g_true, clean_df)

    rows = []
    n_rejected = 0
    for _, row in probes.iterrows():
        instruction = build_query(row)
        gen_text = generate(model, tokenizer, instruction)
        pred = parse_predicted_values(gen_text)

        if verifier is not None:
            current_action = {row["intervened_var"]: row["from_value"],
                               ("tx_power" if row["intervened_var"] == "duty_cycle" else "duty_cycle"):
                               row["base_tx_power"] if row["intervened_var"] == "duty_cycle" else row["base_duty_cycle"]}
            result = verifier.verify(gen_text, current_action)
            if result.verdict == "REJECT":
                n_rejected += 1
                pred = {}  # rejected actions contribute no prediction (treated as abstention)

        truth = {v: row[f"after_{v}"] for v in QOS_VARS}
        errs = {v: abs(pred[v] - truth[v]) for v in QOS_VARS if v in pred}
        rows.append(dict(parsed=len(pred) > 0, **{f"err_{v}": errs.get(v, np.nan) for v in QOS_VARS}))

    df = pd.DataFrame(rows)
    mae = {v: float(df[f"err_{v}"].mean(skipna=True)) for v in QOS_VARS}
    parse_rate = float(df["parsed"].mean())

    result = dict(tag=args.tag, base_model=args.base_model, adapter=args.adapter,
                  use_verifier=args.use_verifier, n_probes=len(probes),
                  parse_rate=parse_rate, n_rejected=n_rejected, mae=mae)
    print(json.dumps(result, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"counterfactual_eval_{args.tag}{'_verified' if args.use_verifier else ''}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
