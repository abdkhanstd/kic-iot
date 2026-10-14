# Comparison baselines

Re-implementations of competing methods, evaluated under **identical
conditions** to ours. These are our implementations following each method's
published description: **not** the original authors' code. Absolute numbers
therefore differ from the source papers, which is the intent: the point is to
remove the confounds (different dataset, split, classifier, metric) that make
published numbers incomparable, so that the *only* thing differing between
rows is the mechanism under test.

Two comparisons are needed because the literature splits into two families
that report on different tasks.

---

## 1. `baseline_causal_ids.py`: our method on *their* task

Recent IoT causal-analysis systems report **supervised attack-classification**
accuracy (e.g. a LiNGAM-based pipeline reporting ~100% F1 on CICIoT2023),
while our pipeline is scored on **structural recovery** (SHD/SID). Those two
numbers cannot share a table. This script removes the confound by running
every causal stage on the classification task.

**Held fixed across all rows:** N-BaIoT, balanced subsample per class,
per-device standardisation, 70/30 stratified split, `RandomForest(200)`,
feature budget `k=8`, seeds `{0,1,2}`, and the metric set.
**Varied:** only the feature-selection stage.

| Row | Stage under test |
|---|---|
| `all-features` | none (control) |
| `correlation-topk` | non-causal filter, top-k \|Pearson r\| |
| `mutual-info-topk` | non-causal filter, top-k mutual information |
| `LiNGAM-causal` | DirectLiNGAM causal neighbourhood of the label (TOCA-IoT style) |
| `NOTEARS-causal` | our discovery stage, used identically |

```bash
python comparison/baseline_causal_ids.py --n_per_class 6000 --seeds 0 1 2
# -> experiments/results/comparison_ids.json
```

**Expected reading of the result.** All pipelines, *including the two
non-causal filters*, exceed 0.98 F1. This is a negative result about the
benchmark, not a positive one about any method: the task saturates, so
near-100% classification accuracy is weak evidence that a method recovered
correct causal structure. The same methods separate clearly under structural
metrics on a graph with known ground truth
(`experiments/eval_discovery_baselines.py`). This is the paper's argument for
reporting SHD/SID rather than downstream detection accuracy.

---

## 2. `baseline_llm_judge_verifier.py`: verification mechanisms

The action-verification literature gates a model's proposal before execution,
but the reference is typically the model's own consistency or a symbolic model
assumed correct. We re-implement those mechanisms and score them on
**identical proposals** against the same do-calculus oracle used everywhere
else in the paper.

| Row | Mechanism |
|---|---|
| `LLM self-consistency` | resample at temperature, accept iff action direction agrees |
| `LLM-as-judge` | ask the same model whether its own causal claim is correct |
| `Graph reachability only` | accept iff a directed path exists actuator → QoS var (topology, no sign) |
| `Causal verifier (ours)` | sign + range + rate-limit + no-op |

```bash
CUDA_VISIBLE_DEVICES=1 python comparison/baseline_llm_judge_verifier.py --n_states 80
# -> experiments/results/comparison_verifiers.json
```

**Result.** LLM-as-judge accepts **every** incorrect claim (specificity
0.000): the model cannot audit its own causal reasoning, confirming the
"causal parrots" critique on a control task. Graph topology alone is equally
uninformative (0.000), because a directed path almost always exists; the
discriminating information is the *sign*. Self-consistency reaches 0.571
specificity but still admits 43% of wrong claims.

---

## Fidelity notes and limitations

- The LiNGAM arm uses `DirectLiNGAM` from `causal-learn` with the label
  included in the variable set, then takes the label's causal neighbourhood.
  The source systems add threshold-optimisation and preprocessing steps we do
  not replicate; we hold those constant across all rows instead, so the
  comparison isolates the causal stage rather than reproducing their full
  pipeline.
- `LLM-as-judge` and `self-consistency` are prompt-level mechanisms; their
  effectiveness depends on the judge model's capability. We use the same
  on-device 0.5B model, which is the deployment-realistic choice for this
  paper but weaker than a cloud judge. A larger judge would likely score
  better, and we state this rather than claim the mechanism is useless in
  general.
- All rows share seeds, so differences are not attributable to sampling.
