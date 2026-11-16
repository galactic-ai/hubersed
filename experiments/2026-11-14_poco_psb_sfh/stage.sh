#!/bin/bash
# Copy the spender chunk files of targets.txt (TARGETID, healpix, chunk) and of the repro target
# 39633332836764152 (chunk 61, 2026-11-01_poco_repeats targets_quiescent3.txt) from Corral to /scratch,
# where load_spectrum reads them under SPEC_DATA. Run on the login node from this folder's
# parent's parent: bash experiments/2026-11-14_poco_psb_sfh/stage.sh
set -euo pipefail
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/desi_spectra
S=/scratch/11006/nikhilgaruda/hubersed_data/desi_spectra
{ cat experiments/2026-11-14_poco_psb_sfh/targets.txt; echo "39633332836764152 10157 61"; } |
while read -r tid hp chunk; do
    f=DESIchunk1024_$chunk.pkl
    [ -s "$S/$f" ] || cp -p "$C/$f" "$S/$f"
    sha256sum "$C/$f" "$S/$f" | cut -c1-16 | paste -sd' ' | sed "s/^/$tid $f /"
done
echo STAGED
