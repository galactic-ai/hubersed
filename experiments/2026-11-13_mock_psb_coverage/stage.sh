#!/bin/bash
# Copy the v2 mock file from Corral (not mounted on compute nodes) to /scratch, if not there yet.
# Run on the stampede3 login node:
#   bash /scratch/11006/nikhilgaruda/claude_runs/2026-11-13_mock_psb_coverage/experiments/2026-11-13_mock_psb_coverage/stage.sh
set -euo pipefail
SRC=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/prospector_model/mocks_v2_500000.h5
DST=/scratch/11006/nikhilgaruda/hubersed_data/mocks_v2_500000.h5
mkdir -p "$(dirname "$DST")"
if [ ! -s "$DST" ] || [ "$(stat -c %s "$SRC")" != "$(stat -c %s "$DST")" ]; then
    cp "$SRC" "$DST.part" && mv "$DST.part" "$DST"
fi
ls -la "$DST"
echo STAGED
