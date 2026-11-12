#!/bin/bash
# The four post.slurm steps on the Mac (stampede3 skx-dev drained 2026-11-10): merge, noise (cpu),
# encode (five encoders, mps), flows (five, cpu). Same settings as post.slurm. Run from the
# repository root: bash experiments/2026-11-09_mocks_v2_outliers/post_mac.sh [step ...]
# with steps from merge noise encode flows (default all four, in order).
set -euo pipefail
RES=results/2026-11-09_mocks_v2_outliers
NOISED=data/prospector_model/noised_cuev2
LAT=data/latents/mocks_v2
TAG=cuev2
steps=${*:-merge noise encode flows}
for a in $steps; do
    echo "== $a start $(date)"
    case $a in
        merge) uv run python experiments/2026-11-09_mocks_v2_outliers/merge.py ;;
        noise)
            mkdir -p $NOISED
            uv run python scripts/make_prospector_noisy_sed.py --input data/prospector_model/mocks_v2_500000.h5 \
                --outdir $NOISED --tag $TAG --device cpu ;;
        encode)
            mkdir -p $LAT
            for t in 6 10 15 cont10 cont15; do
                case $t in
                    cont*) ck=spender_desi_cont_${t#cont}latent_zmax.pt ;;
                    *) ck=spender_asc_run_${t}latent_zmax.pt ;;
                esac
                uv run python scripts/get_latent_space.py $NOISED data/checkpoints/$ck \
                    $LAT/prospector_noise_spec_${TAG}_${t}latent_snr3.h5 \
                    --tag $TAG --mode spec --snr_min 3
            done ;;
        flows)
            mkdir -p $RES/flows
            for t in 6 10 15 cont10 cont15; do
                uv run python scripts/get_outliers_flow.py --method nsf --tag ${t}latent \
                    --epochs 80 --num_transforms 10 --hidden 128 --num_bins 10 --seed 0 \
                    --device cpu --desi data/latents/spender_spec_${t}latent_snr3.h5 \
                    --mock $LAT/prospector_noise_spec_${TAG}_${t}latent_snr3.h5 \
                    --outdir $RES/flows
            done ;;
        *) echo "unknown step $a"; exit 1 ;;
    esac
    echo "== $a done $(date)"
done
