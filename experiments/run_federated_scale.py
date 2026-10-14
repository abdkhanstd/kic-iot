# Author: abdkhan
"""Federation at deployment scale: 10 heterogeneous sites, collusion up to 40%.

The earlier federated experiments use 3 sites and at most 2 attackers, which
is enough to expose a breakdown point but not to claim anything about a
realistic deployment. This scales both axes:

  * 10 honest sites drawn from a spread of operating regimes (interference,
    link quality, arrival rate) and sample budgets, so heterogeneity is not
    a three-point caricature;
  * malicious participants from 0 up to 40% of the federation, colluding on
    one fabricated edge set (the strictly stronger attack established in the
    3-site study).

Two quantities are reported per configuration: the fraction of fabricated
edges that survive into the shared graph (the attack's success), and fused
F1 against ground truth measured against the worst, mean and best individual
site (the federation's benefit). Both matter: an aggregator that resists the
attack by discarding everything would look robust on the first and fail the
second.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from simulator.scm_iot import IoTClusterSCM, RegimeParams, ground_truth_graph, NODES  # noqa: E402
from causal_discovery.discover import evaluate_discovery  # noqa: E402
from federated.fusion import bootstrap_edge_confidence, FederatedCoordinator  # noqa: E402
from run_byzantine_fusion import malicious_fabrication_report  # noqa: E402

OUT = ROOT / "experiments" / "results" / "federated_scale.json"
CACHE = ROOT / "experiments" / "results" / "_scale_reports_cache.pkl"
COLS = [c for c in NODES if c != "t"]


def site_configs(n_sites: int, trial: int, rng: np.random.Generator) -> dict:
    """Heterogeneous sites: regimes and sample budgets both vary, so some
    sites are genuinely better informed than others."""
    cfg = {}
    for i in range(n_sites):
        cfg[f"site_{i:02d}"] = dict(
            seed=1000 * trial + i,
            n_steps=int(rng.choice([600, 900, 1500, 3000, 6000])),
            regime=RegimeParams(
                interference_mean=float(rng.uniform(0.15, 0.65)),
                interference_std=float(rng.uniform(0.08, 0.20)),
                link_quality_mean=float(rng.uniform(0.45, 0.90)),
                arrival_rate_mean=float(rng.uniform(0.20, 0.65))),
        )
    return cfg


def honest_reports(n_sites: int, trial: int) -> dict:
    rng = np.random.default_rng(7_000 + trial)
    out = {}
    for sid, cfg in site_configs(n_sites, trial, rng).items():
        scm = IoTClusterSCM(seed=cfg["seed"], regime=cfg["regime"])
        df = pd.DataFrame(scm.rollout(cfg["n_steps"]))
        ec = bootstrap_edge_confidence(df, COLS, n_bootstrap=6, sample_frac=0.7,
                                        seed=cfg["seed"], lambda1=0.08, w_threshold=0.1)
        out[sid] = dict(edges=ec, n=len(df))
    return out


def local_f1(report, g_true) -> float:
    g = nx.DiGraph(); g.add_nodes_from(COLS)
    for e, c in report["edges"].items():
        if c >= 0.5:
            g.add_edge(*e)
    return evaluate_discovery(g_true, g)["f1"]


def one_config(reports, g_true, trial, n_mal, agg, trim_k):
    coord = FederatedCoordinator(min_sites_agreeing=1, confidence_threshold=0.2)
    for sid, r in reports.items():
        coord.submit(sid, r["edges"], r["n"])
    fabricated = set()
    for m in range(n_mal):
        rep = malicious_fabrication_report(g_true, COLS, seed=trial * 10)  # colluding
        fabricated |= set(rep.keys())
        coord.submit(f"mal_{m}", rep, n_samples=6000)
    g_fused = coord.fuse(all_nodes=COLS, aggregation=agg, trim_k=trim_k)
    m = evaluate_discovery(g_true, g_fused)
    survived = sum(1 for e in fabricated if g_fused.has_edge(*e))
    site_f1s = [local_f1(r, g_true) for r in reports.values()]
    return dict(fused_f1=m["f1"], shd=m["shd"],
                leak=(survived / len(fabricated)) if fabricated else 0.0,
                worst_site=min(site_f1s), mean_site=float(np.mean(site_f1s)),
                best_site=max(site_f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sites", type=int, default=10)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--malicious", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    g_true = ground_truth_graph()
    if CACHE.exists():
        print(f"loading cached site reports from {CACHE.name}")
        cached = pickle.load(open(CACHE, "rb"))
    else:
        cached = []
        for t in range(args.trials):
            print(f"  discovering {args.n_sites} sites, trial {t+1}/{args.trials}", flush=True)
            cached.append(honest_reports(args.n_sites, t))
        pickle.dump(cached, open(CACHE, "wb"))

    aggs = [("sample_weighted", 0), ("mean", 0), ("median", 0),
            ("trimmed_mean", 1), ("trimmed_mean_k3", 3)]
    results = {}
    for agg_label, trim_k in aggs:
        agg = "trimmed_mean" if agg_label.startswith("trimmed_mean") else agg_label
        for n_mal in args.malicious:
            runs = [one_config(cached[t], g_true, t, n_mal, agg, trim_k)
                    for t in range(args.trials)]
            key = f"{agg_label}_mal{n_mal}"
            results[key] = dict(
                aggregation=agg_label, trim_k=trim_k, n_malicious=n_mal,
                collusion_frac=n_mal / (args.n_sites + n_mal),
                **{k: dict(mean=float(np.mean([r[k] for r in runs])),
                            std=float(np.std([r[k] for r in runs])))
                   for k in ("fused_f1", "leak", "worst_site", "mean_site", "best_site")})
            r = results[key]
            print(f"{agg_label:18s} mal={n_mal} ({r['collusion_frac']:.0%}) "
                  f"leak={r['leak']['mean']:.2f} fusedF1={r['fused_f1']['mean']:.3f} "
                  f"worst={r['worst_site']['mean']:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(dict(n_sites=args.n_sites, trials=args.trials, methods=results), f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
