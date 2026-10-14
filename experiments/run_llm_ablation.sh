#!/bin/bash
# Author: abdkhan
# Component 2 experiment pipeline: LoRA-distill two tiny-LLM sizes on the
# SCM counterfactual corpus, then evaluate counterfactual accuracy
# zero-shot vs fine-tuned vs fine-tuned+verifier (model-size ablation).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

QWEN05B=~/.cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master
QWEN15B=~/.cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master
N_PROBES=100

echo "=== [1/6] Zero-shot baseline: Qwen2.5-0.5B (no distillation) ==="
python experiments/eval_counterfactual_llm.py --base_model "$QWEN05B" \
  --n_probes $N_PROBES --tag qwen05b_zeroshot

echo "=== [2/6] LoRA fine-tune Qwen2.5-0.5B on SCM-distilled corpus ==="
python src/distillation/train_lora.py --model_name "$QWEN05B" \
  --output_dir models/checkpoints/qwen05b-causal --epochs 3

echo "=== [3/6] Eval Qwen2.5-0.5B + LoRA (no verifier) ==="
python experiments/eval_counterfactual_llm.py --base_model "$QWEN05B" \
  --adapter models/checkpoints/qwen05b-causal --n_probes $N_PROBES --tag qwen05b_lora

echo "=== [4/6] Eval Qwen2.5-0.5B + LoRA + verifier ==="
python experiments/eval_counterfactual_llm.py --base_model "$QWEN05B" \
  --adapter models/checkpoints/qwen05b-causal --n_probes $N_PROBES --tag qwen05b_lora --use_verifier

echo "=== [5/6] LoRA fine-tune Qwen2.5-1.5B (model-size ablation) ==="
python src/distillation/train_lora.py --model_name "$QWEN15B" \
  --output_dir models/checkpoints/qwen15b-causal --epochs 3 --batch_size 4 --grad_accum 4

echo "=== [6/6] Eval Qwen2.5-1.5B + LoRA (+/- verifier) ==="
python experiments/eval_counterfactual_llm.py --base_model "$QWEN15B" \
  --adapter models/checkpoints/qwen15b-causal --n_probes $N_PROBES --tag qwen15b_lora
python experiments/eval_counterfactual_llm.py --base_model "$QWEN15B" \
  --adapter models/checkpoints/qwen15b-causal --n_probes $N_PROBES --tag qwen15b_lora --use_verifier

echo "=== On-device benchmark (CPU) for both sizes, base + LoRA merged not needed for latency shape ==="
python experiments/benchmark_ondevice.py --device cpu \
  --models "$QWEN05B" "$QWEN15B" --labels qwen0.5b qwen1.5b

echo "ALL DONE"
