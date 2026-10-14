#!/bin/bash
# Author: abdkhan
# Feeder experiments, second half (lighter: the feeder simulator is slow under full load)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
cd "$(dirname "$0")"
until [ -f results/kic_certified_grid.json ]; do sleep 20; done
$PY -W ignore kic_shadow.py --seeds 15 --dwell 800 --n_dwell 1 --stationary --plants grid --scenarios none a1_telemetry spoof_blind a1_spoof --out kic_shadow_grid_stat.json --procs 60 > results/log_shadow_grid_stat.txt 2>&1
$PY -W ignore kic_shadow.py --seeds 15 --dwell 800 --n_dwell 1 --plants grid --out kic_shadow_grid.json --procs 60 > results/log_shadow_grid.txt 2>&1
$PY -W ignore kic_grid.py --seeds 15 --harm_n 2000 --procs 60 > results/log_grid.txt 2>&1
$PY -W ignore kic_grid_leakprice.py --seeds 15 --procs 60 > results/log_grid_leakprice.txt 2>&1
echo ALLDONE >> results/log_grid.txt
