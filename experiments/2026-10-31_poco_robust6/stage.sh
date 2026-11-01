#!/bin/bash
# Copy the spender chunk files of targets.txt (TARGETID, healpix, chunk) from Corral to /scratch,
# where load_spectrum reads them under SPEC_DATA. Run on the stampede3 login node from the
# repository root: bash experiments/2026-10-31_poco_robust6/stage.sh
set -euo pipefail
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/desi_spectra
S=/scratch/11006/nikhilgaruda/hubersed_data/desi_spectra
while read -r tid hp chunk; do
    f=DESIchunk1024_$chunk.pkl
    [ -s "$S/$f" ] || cp -p "$C/$f" "$S/$f"
    sha256sum "$C/$f" "$S/$f" | cut -c1-16 | paste -sd' ' | sed "s/^/$tid $f /"
done < experiments/2026-10-31_poco_robust6/targets.txt
echo STAGED
