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
uv run python -m hubersed.mocks.get_stochastic_priors --no-cue -n 8 -s 0 -f
cp data/prospector_model/stochastic_priors_sample_8.npz "$OUT/priors_fsps_8.npz"
uv run python -m hubersed.mocks.make_model_seds --nebular fsps -n 8 --seed 0 --workers 1 \
    -f -o "$OUT/mocks_fsps_8.h5"

uv run python -m hubersed.fitting.run_map_fits_outliers \
    -s results/emline_outlier_sample20.npz --limit 1 -n 1 -m 2000 -w 1 -o "$OUT/map"

# chunk 10 sorts before chunk 2 as text
mkdir -p "$OUT/_chunks"
ln -sf "$ROOT/data/desi_spectra/DESIchunk1024_2.pkl" "$OUT/_chunks/"
ln -sf "$ROOT/data/desi_spectra/DESIchunk1024_10.pkl" "$OUT/_chunks/"
uv run python -m hubersed.detect.get_latent_space "$OUT/_chunks" \
    data/checkpoints/spender_asc_run_10latent_zmax.pt "$OUT/latents.h5" --snr_min 3

for script in \
    "-m hubersed.alf.make_alf_input" "-m hubersed.alf.read_alf_sample" \
    experiments/2026-08-25_compare_alf_solar_scaled.py \
    "-m hubersed.mocks.get_stochastic_priors" "-m hubersed.mocks.make_model_seds" \
    "-m hubersed.fitting.run_dynesty_outliers" "-m hubersed.fitting.run_map_fits_outliers" \
    "-m hubersed.detect.agn_star_screen" "-m hubersed.detect.build_cont_outlier_sample" \
    "-m hubersed.detect.contam_screens" scripts/simbad_screen.py \
    "-m hubersed.detect.ensemble_flow_seeds" "-m hubersed.detect.ensemble_flow_stats" \
    "-m hubersed.detect.flow_null_and_reverse" "-m hubersed.detect.get_latent_space" \
    "-m hubersed.detect.get_outliers" "-m hubersed.detect.get_outliers_flow" \
    "-m hubersed.detect.train_DESI_noise" \
    "-m hubersed.plotting.plot_latent_umap_score"; do
    uv run python $script --help > /dev/null || { echo "smoke failed: $script"; exit 1; }
done
echo "golden outputs written to $OUT"
