# Author: abdkhan
"""The end-to-end integration experiment that was previously missing: the
paper's main claim is a closed loop of

    tiny LLM proposes an intervention (NL) -> neuro-symbolic verifier gates
    it against a TRUSTED causal graph -> the (possibly corrected/rejected)
    action is applied to the real system -> QoS is measured

but until now the RL-controller experiment and the LLM-counterfactual-
accuracy experiment were two separate, disconnected demonstrations. This
script runs the actual closed loop and compares it against:
  - PPO (the causal-RL controller, trained separately)
  - the tiny LLM with NO verifier (raw hallucination-risk baseline)
  - ReactivePID / Static (non-learned heuristics)
across multiple seeds, reporting mean +/- std QoS so the comparison is
statistically checkable rather than a single anecdotal run.

At each control tick, the LLM is asked to minimize whichever QoS variable
(latency, energy, packet_loss) is currently worst -- a simple, defensible
real-world policy ("fix the biggest problem first") that requires no
retraining of the existing single-objective distillation corpus.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import IoTClusterSCM, ground_truth_graph
from distillation.build_corpus import PROMPT_TEMPLATE, LABELS
from verifier.verify import CausalVerifier, parse_action
from rl_controller.env import compute_reward
from rl_controller.baselines import ReactivePID, StaticPolicy

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
QOS_FOR_OBJECTIVE = ["latency", "energy", "packet_loss"]


def load_llm(base_model_path: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16,
                                                  device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def query_llm(model, tokenizer, action: dict, state: dict, objective: str,
              max_new_tokens: int = 100) -> str:
    instruction = (
        f"An autonomous IoT edge node is running duty cycle={action['duty_cycle']*100:.0f}% "
        f"and transmit power={action['tx_power']*100:.0f}%, with current "
        f"latency={state['latency']:.2f}, throughput={state['throughput']:.2f}, "
        f"packet_loss={state['packet_loss']:.2f}, energy={state['energy']:.2f}. "
        f"Propose a single intervention on duty cycle or transmit power to minimize {LABELS[objective]}."
    )
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_llm_loop(model, tokenizer, seed: int, n_steps: int, use_verifier: bool,
                  verifier: CausalVerifier | None, max_new_tokens: int = 70) -> list[dict]:
    scm = IoTClusterSCM(seed=seed)
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    rows = [state]
    for _ in range(n_steps):
        objective = max(QOS_FOR_OBJECTIVE, key=lambda v: state[v])
        gen_text = query_llm(model, tokenizer, action, state, objective,
                              max_new_tokens=max_new_tokens)
        parsed = parse_action(gen_text)

        if use_verifier:
            result = verifier.verify(gen_text, action)
            action = dict(result.safe_action)
        else:
            # raw hallucination-risk baseline: execute whatever the LLM said,
            # clipped only to the physical actuator range (no rate limit, no
            # causal sign-consistency check) -- this is deliberately the
            # "no safety net" condition the verifier is meant to improve on.
            if parsed is not None:
                (var, value), = parsed.items()
                if var in action:
                    action = {**action, var: float(np.clip(value, 0.0, 1.0))}
            # unparseable / non-actionable output -> action unchanged

        state = scm.step(action)
        rows.append(state)
    return rows


def build_instruction(action: dict, state: dict, objective: str) -> str:
    return (
        f"An autonomous IoT edge node is running duty cycle={action['duty_cycle']*100:.0f}% "
        f"and transmit power={action['tx_power']*100:.0f}%, with current "
        f"latency={state['latency']:.2f}, throughput={state['throughput']:.2f}, "
        f"packet_loss={state['packet_loss']:.2f}, energy={state['energy']:.2f}. "
        f"Propose a single intervention on duty cycle or transmit power to minimize {LABELS[objective]}."
    )


@torch.no_grad()
def run_llm_loops_batched(model, tokenizer, seeds: list[int], n_steps: int,
                           use_verifier: bool, verifier: CausalVerifier | None,
                           max_new_tokens: int = 70) -> dict[int, list[dict]]:
    """Run one closed loop per seed, advancing all of them in LOCKSTEP so the
    per-step LLM queries can be issued as a single batch.

    Each seed's loop is inherently sequential (the next prompt depends on the
    action just taken), so batching *within* a seed is impossible -- but the
    seeds are independent, so batching *across* them is exact, not an
    approximation: every seed still sees precisely the prompt it would have
    seen when run alone. This matters because batch=1 autoregressive decoding
    leaves the GPU ~85% idle (measured 12-16% utilisation); batching the
    seeds fills it and cuts wall time roughly in proportion to len(seeds).

    Decoder-only models need LEFT padding for batched generation, otherwise
    the continuation for a short prompt starts after the pad run and decodes
    misaligned.
    """
    n = len(seeds)
    scms = {s: IoTClusterSCM(seed=s) for s in seeds}
    actions = {s: dict(duty_cycle=0.5, tx_power=0.5) for s in seeds}
    states = {s: scms[s].step(actions[s]) for s in seeds}
    rows = {s: [states[s]] for s in seeds}

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for _ in range(n_steps):
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
                if use_verifier:
                    actions[s] = dict(verifier.verify(text, actions[s]).safe_action)
                else:
                    parsed = parse_action(text)
                    if parsed is not None:
                        (var, value), = parsed.items()
                        if var in actions[s]:
                            actions[s] = {**actions[s], var: float(np.clip(value, 0.0, 1.0))}
                states[s] = scms[s].step(actions[s])
                rows[s].append(states[s])
    finally:
        tokenizer.padding_side = prev_side
    assert len(rows) == n
    return rows


def episode_summary(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return dict(
        reward=float(np.mean([compute_reward(r) for r in rows])),
        latency=float(df["latency"].mean()), energy=float(df["energy"].mean()),
        packet_loss=float(df["packet_loss"].mean()), throughput=float(df["throughput"].mean()),
    )


def run_baseline_loop(policy, seed: int, n_steps: int) -> dict:
    scm = IoTClusterSCM(seed=seed)
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    rows = [state]
    obs_fields = [k for k in state.keys() if k != "t"]
    for _ in range(n_steps):
        obs = np.array([state[f] for f in obs_fields])
        act = policy.act(obs, obs_fields)
        action = dict(duty_cycle=float(np.clip(act[0], 0.02, 0.98)),
                      tx_power=float(np.clip(act[1], 0.02, 0.98)))
        state = scm.step(action)
        rows.append(state)
    return episode_summary(rows)


def run_ppo_loop(model_path: str, seed: int, n_steps: int) -> dict:
    from stable_baselines3 import PPO
    ppo = PPO.load(model_path)
    scm = IoTClusterSCM(seed=seed)
    action = dict(duty_cycle=0.5, tx_power=0.5)
    state = scm.step(action)
    rows = [state]
    obs_fields = [k for k in state.keys() if k != "t"]
    for _ in range(n_steps):
        obs = np.array([state[f] for f in obs_fields], dtype=np.float32)
        act, _ = ppo.predict(obs, deterministic=True)
        action = dict(duty_cycle=float(np.clip(act[0], 0.02, 0.98)),
                      tx_power=float(np.clip(act[1], 0.02, 0.98)))
        state = scm.step(action)
        rows.append(state)
    return episode_summary(rows)


def aggregate(per_seed: list[dict]) -> dict:
    keys = per_seed[0].keys()
    return {k: dict(mean=float(np.mean([r[k] for r in per_seed])),
                     std=float(np.std([r[k] for r in per_seed])))
            for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                     default=str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"))
    ap.add_argument("--adapter", default="models/checkpoints/qwen05b-causal")
    ap.add_argument("--ppo_model", default="models/checkpoints/ppo_causal_controller")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()

    seeds = list(range(1000, 1000 + args.n_seeds))
    g_true = ground_truth_graph()
    clean_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clean_train.csv")
    verifier = CausalVerifier(g_true, clean_df)

    results = {}

    print("=== Non-LLM baselines (cheap, more seeds) ===")
    baseline_seeds = list(range(2000, 2000 + max(args.n_seeds, 10)))
    for name, policy_cls in [("static", StaticPolicy), ("reactive_pid", ReactivePID)]:
        per_seed = [run_baseline_loop(policy_cls(), s, args.n_steps) for s in baseline_seeds]
        results[name] = aggregate(per_seed)
        print(f"{name:15s} reward={results[name]['reward']['mean']:.2f}+-{results[name]['reward']['std']:.2f}")

    print("\n=== PPO causal-RL controller ===")
    per_seed = [run_ppo_loop(args.ppo_model, s, args.n_steps) for s in baseline_seeds]
    results["ppo"] = aggregate(per_seed)
    print(f"ppo             reward={results['ppo']['reward']['mean']:.2f}+-{results['ppo']['reward']['std']:.2f}")

    print(f"\n=== Loading fine-tuned tiny LLM ({args.adapter}) ===")
    model, tokenizer = load_llm(args.base_model, args.adapter)

    for name, use_verifier in [("llm_no_verifier", False), ("llm_with_verifier", True)]:
        print(f"\n=== {name} ({args.n_seeds} seeds x {args.n_steps} steps) ===")
        per_seed = []
        for s in seeds:
            rows = run_llm_loop(model, tokenizer, s, args.n_steps, use_verifier,
                                 verifier if use_verifier else None)
            per_seed.append(episode_summary(rows))
            print(f"  seed={s}: reward={per_seed[-1]['reward']:.2f} latency={per_seed[-1]['latency']:.3f}")
        results[name] = aggregate(per_seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "integrated_control_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL: mean +/- std over seeds ===")
    for name in ["static", "reactive_pid", "ppo", "llm_no_verifier", "llm_with_verifier"]:
        r = results[name]
        print(f"{name:20s} reward={r['reward']['mean']:7.2f}+-{r['reward']['std']:5.2f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
              f"energy={r['energy']['mean']:.3f}+-{r['energy']['std']:.3f}  "
              f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")


if __name__ == "__main__":
    main()
