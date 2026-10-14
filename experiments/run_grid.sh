#!/bin/bash
# Author: abdkhan
# Feeder experiments (third plant, IEEE 33-bus feeder)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
cd "$(dirname "$0")"
$PY -W ignore kic_grid_sidechannel.py --procs 60 > results/log_grid_side.txt 2>&1
$PY -W ignore kic_certified.py --plants grid --procs 60 --out kic_certified_grid.json > results/log_certified_grid.txt 2>&1
$PY -W ignore kic_shadow.py --dwell 800 --n_dwell 1 --stationary --plants grid --scenarios none a1_telemetry spoof_blind a1_spoof --out kic_shadow_grid_stat.json --procs 60 > results/log_shadow_grid_stat.txt 2>&1
$PY -W ignore kic_shadow.py --dwell 800 --n_dwell 1 --plants grid --out kic_shadow_grid.json --procs 60 > results/log_shadow_grid.txt 2>&1
$PY -W ignore kic_grid.py --procs 60 > results/log_grid.txt 2>&1
$PY -W ignore kic_grid_leakprice.py --procs 60 > results/log_grid_leakprice.txt 2>&1
echo ALLDONE >> results/log_grid.txt
