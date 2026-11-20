#!/bin/bash
# Submit L_recoadd fits on skx-dev as slots free up (4 jobs submitted per user there). First 64, then
# 80 once its recoadd job ends COMPLETED. Fits run prod.slurm with a 2 h limit and stop at TLIM, so
# they requeue themselves. Run on the login node from $RUN:
#   nohup bash experiments/2026-11-14_poco_psb_sfh/queue_recoadd_fits.sh <recoadd jobid of 80> > <log> 2>&1 &
set -uo pipefail
REC80=$1
E=experiments/2026-11-14_poco_psb_sfh

free_slot() {
    until [ "$(squeue -u "$USER" -h -p skx-dev | wc -l)" -lt 4 ]; do sleep 60; done
}

submit() {
    TID=$1 MODEL=L_recoadd GIB=10 TLIM=6600 sbatch --parsable --export=ALL -p skx-dev -t 02:00:00 \
        -J "psbLr_$1" "$E/prod.slurm" | tail -n 1
}

free_slot
echo "$(date +%T) 64 $(submit 39633127609469239)"
while squeue -h -j "$REC80" | grep -q "$REC80"; do sleep 60; done
state=$(sacct -X -n -j "$REC80" -o State | head -1 | tr -d ' ')
if [ "$state" = COMPLETED ]; then
    free_slot
    echo "$(date +%T) 80 $(submit 39633318999755849)"
else
    echo "$(date +%T) recoadd of 80 ended $state; its fit is not submitted"
fi
echo "QUEUE_DONE $(date +%T)"
