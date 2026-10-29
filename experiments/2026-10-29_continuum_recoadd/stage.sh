#!/bin/bash
# Copy the iron inputs of recoadd.slurm from Corral to /scratch on the stampede3 login node, since
# compute nodes do not mount Corral: the spectra files of healpix 26004 (the #14 reference) and
# 25598, and the exposure folders of the reference (87113, 88269), SV3 83526 and main 122064.
# Run on the login node: bash experiments/2026-10-29_continuum_recoadd/stage.sh
set -euo pipefail
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data
S=/scratch/11006/nikhilgaruda/hubersed_data
mkdir -p "$S/desi_iron_spectra" "$S/desi_iron_exposures"
for name in spectra-sv3-bright-26004 spectra-sv3-bright-25598 spectra-main-bright-25598; do
    rsync -a "$C/desi_iron_spectra/$name.fits.gz" "$S/desi_iron_spectra/"
done
for d in 20210503/00087113 20210513/00088269 20210405/00083526 20220210/00122064; do
    mkdir -p "$S/desi_iron_exposures/$d"
    rsync -a "$C/desi_iron_exposures/$d/" "$S/desi_iron_exposures/$d/"
done
ls -la "$S/desi_iron_spectra"
for d in 20210503/00087113 20210513/00088269 20210405/00083526 20220210/00122064; do
    echo "$d $(ls "$S/desi_iron_exposures/$d" | wc -l) files"
done
echo STAGED
