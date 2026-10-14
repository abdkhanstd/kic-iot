#!/bin/bash
# Author: abdkhan
# KIC experiment sequence (CPU); the LLM study runs on the GPUs separately.
set -x
export OMP_NUM_THREADS=1
PY=${PY:-python}
cd "$(dirname "$0")"
$PY kic_attacks.py --gates kic_sound kic_holdfw kic_hold kic kic_trial_sound kic_v1 kic_sign kic_cautious kic_fb_only kic_nofb --out kic_attacks_kic.json --procs ${PROCS:-8} > results/log_attacks_kic.txt 2>&1
$PY kic_paired.py > results/log_paired.txt 2>&1
$PY kic_mec.py > results/log_mec.txt 2>&1
$PY kic_shadow.py --dwell 800 --n_dwell 1 --out kic_shadow.json --procs ${PROCS:-8} > results/log_shadow.txt 2>&1
$PY kic_shadow.py --dwell 200 --n_dwell 4 --out kic_shadow_dwell200.json --procs ${PROCS:-8} > results/log_shadow200.txt 2>&1
$PY kic_envelopes.py > results/log_env.txt 2>&1
$PY kic_frontier.py > results/log_frontier.txt 2>&1
$PY kic_weights.py > results/log_weights.txt 2>&1
$PY kic_cost_of_certification.py > results/log_cost.txt 2>&1
$PY kic_reference_sweep.py > results/log_ref.txt 2>&1
echo ALLDONE
