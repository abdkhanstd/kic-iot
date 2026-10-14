#!/bin/bash
# Author: abdkhan
# Theorem-covered feeder runs on the certified operating domain D
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KIC_GRID_DOMAIN=1
PY=python
cd "$(dirname "$0")"
$PY -W ignore kic_shadow.py --seeds 15 --dwell 800 --n_dwell 1 --stationary --certified --plants grid --scenarios none a1_telemetry spoof_blind a1_spoof --out kic_shadow_grid_cert.json --procs 60 > results/log_shadow_grid_cert.txt 2>&1
$PY -W ignore kic_certified.py --plants grid --seeds 15 --procs 60 --out kic_certified_grid_D.json > results/log_certified_grid_D.txt 2>&1
echo ALLDONE >> results/log_certified_grid_D.txt
