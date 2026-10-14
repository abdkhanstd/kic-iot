# Author: abdkhan
"""Experiment 3: causal-RL QoS controller vs. non-learned baselines
(component 3, Section 5 "sample efficiency" + "QoS/control performance").

Trains PPO (stable-baselines3) on the IoTQoSEnv and compares its converged
policy against RandomPolicy, StaticPolicy, and the hand-tuned ReactivePID
baseline -- the kind of non-causal heuristic an engineer would write without
an SCM.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_controller.env import IoTQoSEnv, OBS_FIELDS
from rl_controller.baselines import RandomPolicy, StaticPolicy, ReactivePID

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def eval_baseline(policy, n_episodes=10, episode_len=500, seed=100):
    all_rewards, all_states = [], []
    for ep in range(n_episodes):
        env = IoTQoSEnv(seed=seed + ep, episode_len=episode_len)
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(episode_len):
            action = policy.act(obs, OBS_FIELDS)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            all_states.append(info["state"])
            if term or trunc:
                break
        all_rewards.append(ep_reward)
    df = pd.DataFrame(all_states)
    return dict(mean_episode_reward=float(np.mean(all_rewards)),
                latency=float(df["latency"].mean()), energy=float(df["energy"].mean()),
                packet_loss=float(df["packet_loss"].mean()),
                throughput=float(df["throughput"].mean()))


def train_ppo(total_timesteps=100_000, seed=0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor as SB3Monitor

    env = SB3Monitor(IoTQoSEnv(seed=seed, episode_len=500))
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                n_steps=1024, batch_size=64, learning_rate=3e-4)

    # Track convergence (sample efficiency): eval every N steps.
    convergence = []
    chunk = 10_000
    steps_done = 0
    while steps_done < total_timesteps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk

        class _PPOPolicy:
            def act(self, obs, obs_fields):
                a, _ = model.predict(obs, deterministic=True)
                return a
        metrics = eval_baseline(_PPOPolicy(), n_episodes=3, episode_len=300, seed=999)
        convergence.append(dict(timesteps=steps_done, **metrics))
        print(f"  PPO @ {steps_done} steps: reward={metrics['mean_episode_reward']:.1f} "
              f"latency={metrics['latency']:.3f} energy={metrics['energy']:.3f}")

    return model, convergence


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("Evaluating RandomPolicy...")
    results["random"] = eval_baseline(RandomPolicy(seed=0))
    print("Evaluating StaticPolicy...")
    results["static"] = eval_baseline(StaticPolicy())
    print("Evaluating ReactivePID...")
    results["reactive_pid"] = eval_baseline(ReactivePID())

    print("Training PPO causal-RL controller...")
    model, convergence = train_ppo(total_timesteps=100_000, seed=0)

    class _PPOPolicy:
        def act(self, obs, obs_fields):
            a, _ = model.predict(obs, deterministic=True)
            return a
    results["ppo"] = eval_baseline(_PPOPolicy(), n_episodes=10, episode_len=500, seed=100)
    results["ppo_convergence"] = convergence

    with open(RESULTS_DIR / "rl_controller_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Final comparison (lower latency/energy/packet_loss, higher throughput = better) ===")
    for name in ["random", "static", "reactive_pid", "ppo"]:
        r = results[name]
        print(f"{name:15s} reward={r['mean_episode_reward']:8.1f}  latency={r['latency']:.3f} "
              f"energy={r['energy']:.3f} packet_loss={r['packet_loss']:.3f} throughput={r['throughput']:.3f}")

    model.save(str(RESULTS_DIR.parent.parent / "models" / "checkpoints" / "ppo_causal_controller"))


if __name__ == "__main__":
    main()
