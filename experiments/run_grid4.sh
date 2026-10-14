#!/bin/bash
# Author: abdkhan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
cd "$(dirname "$0")"
until [ -f results/kic_shadow_grid_stat.json ]; do sleep 20; done
$PY -W ignore kic_grid.py --seeds 15 --harm_n 2000 --procs 60 > results/log_grid.txt 2>&1
$PY -W ignore kic_grid_leakprice.py --seeds 15 --procs 60 > results/log_grid_leakprice.txt 2>&1
echo ALLDONE >> results/log_grid.txt
