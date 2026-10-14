# Author: abdkhan
"""Multi-seed statistical completeness for the PPO causal-RL controller
result: the original run_rl_controller.py trained ONE policy (seed=0) and
evaluated it -- a single training run cannot rule out "got lucky/unlucky
with initialization" as the explanation for beating the heuristic
baselines. This retrains PPO from scratch with 3 independent seeds and
reports mean +/- std of the converged policy's evaluation performance.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_controller.env import IoTQoSEnv, OBS_FIELDS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def eval_policy(model, n_episodes=10, episode_len=500, seed=100):
    import pandas as pd
    all_rewards, all_states = [], []
    for ep in range(n_episodes):
        env = IoTQoSEnv(seed=seed + ep, episode_len=episode_len)
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(episode_len):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            all_states.append(info["state"])
            if term or trunc:
                break
        all_rewards.append(ep_reward)
    df = pd.DataFrame(all_states)
    return dict(mean_episode_reward=float(np.mean(all_rewards)),
                latency=float(df["latency"].mean()), energy=float(df["energy"].mean()),
                packet_loss=float(df["packet_loss"].mean()), throughput=float(df["throughput"].mean()))


def train_and_eval(seed: int, total_timesteps: int = 100_000, save_dir: Path | None = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor as SB3Monitor

    env = SB3Monitor(IoTQoSEnv(seed=seed, episode_len=500))
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                n_steps=1024, batch_size=64, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    if save_dir is not None:
        # keep every seed's policy, not just the metrics -- the multi-seed
        # claim ("PPO converges to an identical policy regardless of seed")
        # is only checkable by a reader if the policies themselves are kept.
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir / f"ppo_seed{seed}"))
    return eval_policy(model, n_episodes=10, episode_len=500, seed=100)


def main():
    seeds = [0, 1, 2]
    ckpt_dir = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
    per_seed = []
    for s in seeds:
        print(f"Training PPO with seed={s}...")
        r = train_and_eval(s, save_dir=ckpt_dir)
        per_seed.append(r)
        print(f"  seed={s}: reward={r['mean_episode_reward']:.1f} latency={r['latency']:.3f} "
              f"packet_loss={r['packet_loss']:.3f}")

    keys = per_seed[0].keys()
    summary = {k: dict(mean=float(np.mean([r[k] for r in per_seed])),
                        std=float(np.std([r[k] for r in per_seed])))
               for k in keys}
    summary["n_seeds"] = len(seeds)
    summary["per_seed"] = per_seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rl_controller_multiseed.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== PPO over {len(seeds)} seeds ===")
    for k in ["mean_episode_reward", "latency", "energy", "packet_loss", "throughput"]:
        print(f"{k:20s} = {summary[k]['mean']:.3f} +- {summary[k]['std']:.3f}")


if __name__ == "__main__":
    main()
