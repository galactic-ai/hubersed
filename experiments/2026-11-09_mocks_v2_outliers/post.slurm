#!/bin/bash
# After the mock run: ACTION=all runs the four steps in order; ACTION=merge (merge.py, then a copy to /scratch), noise
# (scripts/make_prospector_noisy_sed.py), encode (scripts/get_latent_space.py, five encoders) or
# flows (scripts/get_outliers_flow.py, five flows). Same settings as the old 500k Cue set: noise flow
# desi_noise_spender_10latent{,_flow}.pt, seed 42; encode mode spec, snr_min 3, zmax 0; flows nsf,
# 10 transforms, 128 hidden, 10 bins, 80 epochs, seed 0, cpu. Pick -p at submit (test-only first):
#   ACTION=all sbatch --export=ALL -p spr -t 06:00:00 --dependency=afterok:<mock job> experiments/2026-11-09_mocks_v2_outliers/post.slurm
#SBATCH -J mocks2post
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/mocks2post_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-09_mocks_v2_outliers
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
SCR=/scratch/11006/nikhilgaruda/hubersed_data/mocks_v2
TAG=cuev2
ACTION=${ACTION:-merge}
DEVICE=${DEVICE:-cpu}
cd $REPO
export PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID $ACTION on $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION), repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $EXP/merge.py $EXP/post.slurm scripts/make_prospector_noisy_sed.py scripts/get_latent_space.py scripts/get_outliers_flow.py
mkdir -p $SCR
set -o pipefail
step() {
case $1 in
    merge)
        $PY $EXP/merge.py && cp -v data/prospector_model/mocks_v2_500000.h5 $SCR/ ;;
    noise)
        mkdir -p $SCR/noised
        $PY scripts/make_prospector_noisy_sed.py --input $SCR/mocks_v2_500000.h5 \
            --outdir $SCR/noised --tag $TAG --device $DEVICE ;;
    encode)
        mkdir -p $SCR/latents
        for t in 6 10 15 cont10 cont15; do
            case $t in
                cont*) ck=spender_desi_cont_${t#cont}latent_zmax.pt ;;
                *) ck=spender_asc_run_${t}latent_zmax.pt ;;
            esac
            $PY scripts/get_latent_space.py $SCR/noised data/checkpoints/$ck \
                $SCR/latents/prospector_noise_spec_${TAG}_${t}latent_snr3.h5 \
                --tag $TAG --mode spec --snr_min 3 || return 1
        done ;;
    flows)
        mkdir -p results/2026-11-09_mocks_v2_outliers/flows
        for t in 6 10 15 cont10 cont15; do
            $PY scripts/get_outliers_flow.py --method nsf --tag ${t}latent --epochs 80 \
                --num_transforms 10 --hidden 128 --num_bins 10 --seed 0 --device cpu \
                --desi data/latents/spender_spec_${t}latent_snr3.h5 \
                --mock $SCR/latents/prospector_noise_spec_${TAG}_${t}latent_snr3.h5 \
                --outdir results/2026-11-09_mocks_v2_outliers/flows || return 1
        done ;;
    *) echo "unknown ACTION $1"; return 1 ;;
esac
}
status=0
if [ "$ACTION" = all ]; then
    for a in merge noise encode flows; do
        echo "== $a start $(date)"
        step $a || { status=1; echo "$a failed"; break; }
    done
else
    step $ACTION || status=1
fi
echo "exit $status, end $(date)"
exit $status
