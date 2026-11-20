# mocks_cue_noised

The baseline Prospector mocks (MILES, Cue nebular, median DESI LSF, src/hubersed/mocks/priors.py)
with DESI-like noise, as a spender Parquet dataset. These are the mocks behind the 310 outliers.

- Mocks: 500,000. Made with `scripts/chunks_to_parquet.py` (experiments/2026-11-20_corral_parquet,
  convert.slurm) from the 489 pickle chunks `DESIcueprospector1024_*.pkl` of
  data/prospector_model/noised_cue_meanzero_wide. Values are the pickles' float64 cast to float32.
- The pickles were made by `scripts/make_prospector_noisy_sed.py` from
  prospector_stochastic_model_seds_cue_500000.h5: one S/N spectrum per mock from the noise flow and
  noise spender, conditioned on `norm`; sigma capped at the 99.5th percentile; sky-line pixels have
  zero ivar.

## Priors (src/hubersed/mocks/priors.py, Cue sample)
| Parameter | Prior |
|---|---|
| z | U(0.01, 0.6) |
| logmass | U(7, 12) |
| logzsol | U(-2.5, 0.5) |
| sigma_reg | LogU(0.1, 5) |
| tau_eq, tau_in | U(0.01, t_H) each |
| sigma_dyn | LogU(0.001, 0.5) |
| tau_dyn | N(0.01, 0.02)[0.005, 0.2] |
| dust_index | U(-2.5, 0.4) |
| dust2 | N(0.3, 1.0)[0, 4] |
| dust_ratio | N(1.0, 0.3)[0, 2] |
| duste_umin, duste_gamma, duste_qpah | N(2, 1)[0.1, 15], LogU(1e-4, 0.1), U(0.5, 7) |
| eline_sigma (km/s) | U(10, 250) |
| sigma_smooth (km/s) | U(10, 400) |
| gas_logz | U(-2.2, 0.5) |
| gas_logu | U(-4, -1) |
| gas_lognH | U(1, 4) |
| gas_logno, gas_logco | U(-1, 0.732) each |

t_H is the age of the Universe at z. Whether the model used the duste draws was not checked.

## Columns
- `spec` (7780 float32): noisy flux divided by `norm`, DESI grid without its last pixel.
- `w` (7780 float32): inverse variance of `spec`.
- `z` (float32): mock redshift. `zerr` (float32): 0.
- `norm` (float32): median noiseless flux over rest-frame 5300-5850 A, in 1e-17 erg/s/cm^2/A.
- `target_id` (int64): row index in the mock h5 (0-499,999).

Splits are spender's hash of `target_id` (70/15/15).
