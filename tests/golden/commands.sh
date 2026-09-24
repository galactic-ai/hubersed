#!/usr/bin/env bash
# Golden jobs for the bin/ to src/ moves. Compare runs with tests/golden/compare.py.
# Usage is bash tests/golden/commands.sh [OUTDIR]. Local only, needs FSPS and data/.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:-$ROOT/tests/golden/out}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# make_model_seds only reads priors from data/prospector_model/
uv run python scripts/get_stochastic_priors.py --no-cue -n 8 -s 0 -f
cp data/prospector_model/stochastic_priors_sample_8.npz "$OUT/priors_fsps_8.npz"
uv run python scripts/make_model_seds.py --nebular fsps -n 8 --seed 0 --workers 1 \
    -f -o "$OUT/mocks_fsps_8.h5"

uv run python -m hubersed.fitting.run_map_fits_outliers \
    -s results/emline_outlier_sample20.npz --limit 1 -n 1 -m 2000 -w 1 -o "$OUT/map"

# chunk 10 sorts before chunk 2 as text
mkdir -p "$OUT/_chunks"
ln -sf "$ROOT/data/desi_spectra/DESIchunk1024_2.pkl" "$OUT/_chunks/"
ln -sf "$ROOT/data/desi_spectra/DESIchunk1024_10.pkl" "$OUT/_chunks/"
uv run python scripts/get_latent_space.py "$OUT/_chunks" \
    data/checkpoints/spender_asc_run_10latent_zmax.pt "$OUT/latents.h5" --snr_min 3

for script in \
    scripts/make_alf_input.py "-m hubersed.alf.read_alf_sample" \
    experiments/2026-08-25_compare_alf_solar_scaled.py \
    scripts/get_stochastic_priors.py scripts/make_model_seds.py \
    "-m hubersed.fitting.run_dynesty_outliers" "-m hubersed.fitting.run_map_fits_outliers" \
    scripts/agn_star_screen.py scripts/build_cont_outlier_sample.py \
    scripts/contam_screens.py scripts/simbad_screen.py \
    scripts/get_latent_space.py \
    scripts/get_outliers.py scripts/get_outliers_flow.py \
    "-m hubersed.detect.train_DESI_noise" \
    scripts/plot_latent_umap_score.py; do
    uv run python $script --help > /dev/null || { echo "smoke failed: $script"; exit 1; }
done
echo "golden outputs written to $OUT"
