# Author: abdkhan
"""On-device feasibility benchmark (Section 5, "energy-efficient and
scalable causal optimization"): inference latency and memory footprint of
the distilled tiny LLM under CPU-only execution (a stand-in for an
IoT-class edge CPU -- e.g. a Raspberry Pi 4/5 class ARM core -- since this
machine has no such device attached; the paper should report both this
GPU-host CPU-only figure and, if a Pi/Jetson is available before
submission, the true edge-hardware number), at fp32, bf16, and int8
(bitsandbytes) precision, across the model-size ablation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distillation.build_corpus import PROMPT_TEMPLATE  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"

SAMPLE_QUERIES = [
    "On an autonomous IoT edge node currently running duty cycle at 60% and the "
    "other radio settings unchanged, what happens to QoS if duty cycle is decreased "
    "by 20 percentage points (to 40%)?",
    "An autonomous IoT edge node is running duty cycle=70% and transmit power=45%, "
    "with current latency=0.81, throughput=0.42, packet_loss=0.55, energy=0.83. "
    "Propose a single intervention on duty cycle or transmit power to minimize latency.",
]


def bench_one(model_path: str, precision: str, device: str, n_reps: int = 10) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict()
    if precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "int8":
        kwargs["load_in_8bit"] = True
    else:
        raise ValueError(precision)

    if device == "cpu" and precision == "int8":
        return dict(error="int8 (bitsandbytes) requires CUDA; skipped on CPU")

    t_load0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, **kwargs)
    load_time = time.time() - t_load0
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    latencies = []
    for q in SAMPLE_QUERIES:
        prompt = PROMPT_TEMPLATE.format(instruction=q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)  # warmup
        for _ in range(n_reps):
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id)
            dt = time.time() - t0
            n_new = out.shape[1] - inputs["input_ids"].shape[1]
            latencies.append(dt / max(n_new, 1))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(
        n_params=n_params, param_megabytes=round(param_bytes / 1e6, 1),
        load_time_sec=round(load_time, 2),
        mean_sec_per_token=round(float(np.mean(latencies)), 4),
        p50_sec_per_token=round(float(np.median(latencies)), 4),
        tokens_per_sec=round(1.0 / float(np.mean(latencies)), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                     help="local model dirs, one per size in the ablation")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    results = {}
    for path, label in zip(args.models, args.labels):
        results[label] = {}
        for precision in (["fp32", "bf16"] if args.device == "cpu" else ["fp32", "bf16", "int8"]):
            print(f"Benchmarking {label} [{precision}] on {args.device}...")
            try:
                r = bench_one(path, precision, args.device)
            except Exception as e:
                r = dict(error=str(e))
            results[label][precision] = r
            print(f"  -> {r}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"ondevice_benchmark_{args.device}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
