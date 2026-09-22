#!/bin/bash
#SBATCH -J flow-cal
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/flowcal_%j.out
#SBATCH -e /work/11006/nikhilgaruda/ls6/research/hubersed/logs/flowcal_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A AST25022

# Calibration runs for the ML4PS OOD paper:
#   selfdist -> split the mocks, train on A, score B. B's log p is the
#               same-distribution null the 0.1% threshold currently lacks.
#   reverse  -> train on DESI, score the mocks. Consistency check on the
#               "it is only prior coverage" objection.
# 5 tags x 2 modes x 3 seeds = 30 flows, spread over the node's 3 A100s.

set -u
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd "$REPO" || exit 1
unset PYTHONPATH
source .venv/bin/activate || exit 1
mkdir -p logs results/flow_calibration

# 2026-08-31b sec 1: an unset SPS_HOME kills EVERY prospector import at module
# scope via the unguarded AGN module. Nothing here imports prospector, but this
# costs nothing and removes the landmine that produced a zero-fit job before.
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/fsps

SCRIPT=bin/spender/noise/flow_null_and_reverse.py
TAGS="6latent 10latent 15latent cont10latent cont15latent"
MODES="selfdist reverse"
SEEDS=3

echo "=== node ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

# Preflight: load every latent file and split it, train nothing. A missing h5 or
# an encoder mismatch should cost seconds, not a wasted GPU allocation.
echo "=== preflight ==="
fail=0
for m in $MODES; do
  for t in $TAGS; do
    python "$SCRIPT" --mode "$m" --tag "$t" --dry-run || { echo "PREFLIGHT FAIL $m/$t"; fail=1; }
  done
done
if [ "$fail" -ne 0 ]; then echo "aborting: preflight failed"; exit 1; fi

echo "=== 30 flows, 3 concurrent (one per GPU) ==="
i=0
for m in $MODES; do
  for t in $TAGS; do
    g=$(( i % 3 ))
    (
      python "$SCRIPT" --mode "$m" --tag "$t" --seeds "$SEEDS" --device "cuda:$g" \
        > "logs/flowcal_${m}_${t}.log" 2>&1
      echo "done ${m}/${t} gpu${g} rc=$?"
    ) &
    i=$(( i + 1 ))
    if [ $(( i % 3 )) -eq 0 ]; then wait; fi
  done
done
wait

echo "=== outputs ==="
ls -la results/flow_calibration/
echo "=== per-combo tails ==="
for f in logs/flowcal_*_*.log; do echo "--- $f"; tail -4 "$f"; done
