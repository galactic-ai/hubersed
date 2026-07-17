#!/bin/bash
# Encode DESI + Cue mocks through each trained spender encoder (S/N>3), then run
# IsolationForest outliers per encoder. Edit DATA / MOCKDIR / CKPTDIR + the ENCODERS
# map to match your machine, then:  bash run_matrix.sh
set -eo pipefail

DESIDIR=../../../data/desi_spectra       # dir holding DESIchunk1024_*.pkl
MOCKDIR=../../../data/prospector_model   # dir holding DESIcueprospector1024_*.pkl
OUTDIR=../../../data                      # where latent files are written (= PATHS['DATA'])
CKPTDIR=../../../data                     # dir holding the trained spender_*.pt checkpoints
SNRMIN=3

# tag <-> checkpoint filename (parallel arrays; bash 3.2 safe, no associative arrays).
# Add cue-encoder rows once that job finishes.
TAGS=(  10latent                       15latent                       cont10latent )
CKPTS=( spender_asc_run_10latent_zmax.pt spender_asc_run_15latent_zmax.pt spender_desi_cont_10latent_zmax.pt )

for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"
  CKPT="$CKPTDIR/${CKPTS[$i]}"
  DESI_OUT="$OUTDIR/spender_spec_${TAG}_snr3.h5"
  MOCK_OUT="$OUTDIR/prospector_noise_spec_cue_${TAG}_snr3.h5"
  echo "================  $TAG  ($CKPT)  ================"

  echo "-- encode DESI --"
  python get_latent_space.py "$DESIDIR" "$CKPT" "$DESI_OUT" \
      --mode spec --snr_min $SNRMIN --tag chunk1024 -b 256

  echo "-- encode Cue mocks --"
  python get_latent_space.py "$MOCKDIR" "$CKPT" "$MOCK_OUT" \
      --mode spec --snr_min $SNRMIN --tag cueprospector1024 -b 256

  echo "-- IsolationForest outliers --"
  python get_outliers.py --desi "spender_spec_${TAG}_snr3.h5" \
      --mock "prospector_noise_spec_cue_${TAG}_snr3.h5" \
      --out "desi_outliers_${TAG}_snr3.pt"
done

echo "DONE. outlier files: results/desi_outliers_{10latent,15latent,cont10latent}_snr3.pt"
