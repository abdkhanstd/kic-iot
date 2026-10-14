# Author: abdkhan
"""LoRA fine-tune a sub-1B tiny LLM on the SCM-distilled counterfactual
instruction corpus (component 2). Candidates evaluated in the model-size
ablation: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct.

huggingface.co is unreachable from this network (TLS handshake blocked),
and hf-mirror.com's /resolve/ endpoint 308-redirects straight back to the
blocked origin rather than proxying -- so base models are fetched once via
`modelscope.snapshot_download` (see src/distillation/fetch_base_model.py)
to a local directory, and --model_name here should point at that local
path rather than a hub repo id.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType

from distillation.build_corpus import PROMPT_TEMPLATE


def format_example(ex, eos: str = ""):
    prompt = PROMPT_TEMPLATE.format(instruction=ex["instruction"])
    # Append EOS so the model learns to STOP. Without it (as in the first
    # round of adapters trained here) generate() never terminates early and
    # every call burns the full max_new_tokens budget, ~35% of which is the
    # model rambling past the end of its answer -- measured at 6.7 s/call
    # for a 0.5B model emitting 100 tokens where ~65 would do.
    full = prompt + ex["output"] + eos
    return {"text": full, "prompt_len": len(prompt)}


def build_tokenize_fn(tokenizer, max_len):
    def _tok(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_len,
                         padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out
    return _tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--train_file", default="data/synthetic/distill_train.jsonl")
    ap.add_argument("--val_file", default="data/synthetic/distill_val.jsonl")
    ap.add_argument("--output_dir", default="models/checkpoints/smollm2-360m-causal")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})
    ds = ds.map(lambda ex: format_example(ex, eos=tokenizer.eos_token or ""))
    tok_fn = build_tokenize_fn(tokenizer, args.max_len)
    ds = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
