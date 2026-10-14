# Author: abdkhan
"""Print a consolidated summary of all experiment results so far."""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def show_json(name, fn=None):
    p = RESULTS_DIR / name
    if not p.exists():
        print(f"[missing] {name}")
        return
    d = json.load(open(p))
    print(f"\n--- {name} ---")
    if fn:
        fn(d)
    else:
        print(json.dumps(d, indent=2)[:2000])


def main():
    print("=" * 70)
    print("CausalTinyIoT -- consolidated results summary")
    print("=" * 70)

    show_json("causal_discovery_clean.json", lambda d: [
        print(f"{m:10s}: SHD={d[m]['shd']:3d} SID={d[m]['sid']:3d} "
              f"P={d[m]['precision']:.2f} R={d[m]['recall']:.2f} F1={d[m]['f1']:.2f}")
        for m in ["notears", "pc"] if "error" not in d.get(m, {"error": 1})
    ])

    p = RESULTS_DIR / "poisoning_ablation.csv"
    if p.exists():
        df = pd.read_csv(p)
        print("\n--- poisoning_ablation.csv ---")
        print(df.to_string(index=False))

    show_json("sign_flip_stress_test.json")

    show_json("federated_fusion.json", lambda d: [
        print(f"fused F1 = {d['fused_f1_mean']:.3f}+-{d['fused_f1_std']:.3f} over {d['n_trials']} trials")
    ] + [
        print(f"  vs {label:11s}: baseline={c['baseline_mean']:.3f}+-{c['baseline_std']:.3f}  "
              f"delta={c['delta_mean']:+.3f}+-{c['delta_std']:.3f}  win_rate={c['frac_trials_fusion_wins']:.0%}")
        for label, key in [("best site", "vs_best_site"), ("mean site", "vs_mean_site"),
                            ("worst site", "vs_worst_site")]
        for c in [d[key]]
    ])

    show_json("rl_controller_comparison.json", lambda d: [
        print(f"{name:15s} reward={d[name]['mean_episode_reward']:8.1f} "
              f"latency={d[name]['latency']:.3f} energy={d[name]['energy']:.3f} "
              f"packet_loss={d[name]['packet_loss']:.3f} throughput={d[name]['throughput']:.3f}")
        for name in ["random", "static", "reactive_pid", "ppo"] if name in d
    ])

    show_json("rl_controller_multiseed.json", lambda d: [
        print(f"PPO ({d['n_seeds']} independently trained seeds): "
              f"reward={d['mean_episode_reward']['mean']:.1f}+-{d['mean_episode_reward']['std']:.1f}  "
              f"latency={d['latency']['mean']:.3f}+-{d['latency']['std']:.3f}  "
              f"packet_loss={d['packet_loss']['mean']:.3f}+-{d['packet_loss']['std']:.3f}")
    ])

    show_json("integrated_control_comparison.json", lambda d: [
        print(f"{name:20s} reward={r['reward']['mean']:7.2f}+-{r['reward']['std']:5.2f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
              f"energy={r['energy']['mean']:.3f}+-{r['energy']['std']:.3f}  "
              f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")
        for name, r in d.items()
    ])

    show_json("poisoned_llm_control_comparison.json", lambda d: [
        print(f"{name:28s} reward={r['reward']['mean']:7.2f}+-{r['reward']['std']:5.2f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
              f"energy={r['energy']['mean']:.3f}+-{r['energy']['std']:.3f}  "
              f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")
        for name, r in d.items()
    ])

    show_json("byzantine_fusion.json", lambda d: [
        print(f"{k:34s} F1={v['f1_mean']:.3f}+-{v['f1_std']:.3f}  "
              f"fabricated edges surviving={v['mean_fabricated_edges_survived']:.2f}"
              f"/{v.get('mean_fabricated_edges_submitted', 0):.0f}")
        for k, v in d.items()
    ])

    show_json("verifier_as_detector.json", lambda d: [
        print(f"rejection rate: clean={d['clean_rejection_rate']:.3f}  "
              f"poisoned={d['poisoned_rejection_rate']:.3f}"),
        print(f"detection AUC ({d['window']}-step windows) = {d['detection_auc']:.3f}  "
              f"TPR={d['tpr_at_threshold']:.2f} @ FPR={d['fpr_at_threshold']:.2f}"),
    ])

    show_json("verified_closedloop_control.json", lambda d: [
        print(f"{name:34s} reward={r['reward']['mean']:7.3f}+-{r['reward']['std']:5.3f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
              f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")
        for name, r in d.items()
    ])

    show_json("federated_distillation_prep.json")

    show_json("federated_distillation_control.json", lambda d: [
        print(f"{name:34s} reward={r['reward']['mean']:7.3f}+-{r['reward']['std']:5.3f}  "
              f"latency={r['latency']['mean']:.3f}+-{r['latency']['std']:.3f}  "
              f"packet_loss={r['packet_loss']['mean']:.3f}+-{r['packet_loss']['std']:.3f}")
        for name, r in d.items()
    ])

    show_json("causal_discovery_nbaiot_real.json", lambda d: [
        print(f"{cond:8s}: compromised out-edges={r['compromised_out_edges']}  "
              f"stability={r['compromised_edge_bootstrap_stability']}")
        for cond, r in d.items()
    ])

    for f in sorted(RESULTS_DIR.glob("counterfactual_eval_*.json")):
        show_json(f.name)

    for f in sorted(RESULTS_DIR.glob("ondevice_benchmark_*.json")):
        show_json(f.name)


if __name__ == "__main__":
    main()
