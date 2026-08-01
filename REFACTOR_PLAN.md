# hubersed refactor plan

**Status:** proposal, not executed. Nikhil executes; nothing here has been applied to code.
**Date:** 2026-07-16
**Scope decisions (locked by NG):** split the knowledge log; full restructure + workflow DAG; promote `tmp/` by triage.

---

## 0. What this refactor is for

**Non-goal.** hubersed is not an open-source library. One user, no downstream consumers, no API contract.
Importing library standards wholesale — semver, PyPI/conda-forge release, coverage targets, deprecation
policy — is cargo cult. Skip them.

**Goal.** Requirements are derived from *observed* failure modes in
`UberSED/knowledge/outlier_investigation_log.md`, not from a generic checklist. Three classes:

### (a) Mechanical bugs that tests would have caught

From the log's own retractions. 10 of 12 are test-catchable, and **zero tests exist**:

| bug | log ref | catchable by |
|---|---|---|
| lexicographic `sorted(glob)` vs numeric `chunk*1024+row` → invalidated ~all pre-06-10 catalogue matching | §CRITICAL CORRECTION 2026-06-10 | TID round-trip assert |
| `f_nu` vs `f_lambda` Dn4000 (factor 0.9273 = (3900/4050)²) | §RESOLUTION 2026-07-15 | unit test on known spectrum |
| theta label shift (`len(theta)=26` vs 18 labels; `logsfr_ratios` is 9-element) | §MAP reach test | dict return via `theta_index` |
| line-free mocks (`hasspec=False` → lines never injected) → 4456 vs 352 outliers | §1 Line-free mock bug | assert intrinsic Hα/Hβ ≈ 2.86 |
| gas params never drawn in Step 1 → artificially tight EW | §Step 1 | prior-draw completeness test |
| `np.clip(ratios,-5,5)` (fit bound) compressing the σ_reg axis → false "separable" | §Step 1 | bound-provenance test |
| `MultiVariateNormal.__call__` → 9×9 NaN matrix (upstream prospector) | §UPSTREAM BUG | prior-returns-finite-scalar test |
| `fmodel.predict()` returns `(list, mfrac)`; `[0]` indexes the tuple | §2026-07-13 bugfix | smoke test |
| `sigma_kms =- ...` typo in `parameter_file.py` | §LSF note | test exercising that path |
| base set ≠ generator (different priors + template + gas treatment) → "3.22" never transferred | §RETRACTIONS 2026-07-15 | equivalence test base ↔ generator |

Not catchable (genuine science): PolyOptCal mass railing; logzsol operating-point.

### (b) Structural: `bin/` is not importable → code gets retyped instead of reused

`paste_desi_noise` was reinvented in `mualpha_infer.py` as a hand-rolled Gaussian instead of importing
`bin/spender/noise/make_prospector_noisy_sed.py`. The log calls it *"a repeat of the mock-noise incident —
reinventing instead of using the existing pipeline."* Twice is a layout defect, not a discipline defect.
`bin/` has no `__init__.py`, is not on the path, is not installed. It cannot be imported. So it isn't.

### (c) Provenance: the log points at files that do not exist

**UPDATE 2026-07-16 10:39 — partial rescue.** A prior session copied the μ_α programme out of its
sandbox into `tmp/` (41 → 53 entries, plus `tmp/figs/` with 37 figures). The list below is corrected.
The rescue does not change the conclusion; it changes the size of the loss.

**Recovered:** `mualpha_infer.py`, `diag_railing.py`, `mualpha_bin_grid.json` (the locked 37-bin /
87,851-galaxy grid), `mualpha_step0.py`, `lock_bin_grid.py`, `pozzetti_limit.py`,
`bgs_completeness.py`, `ssfr_mstar.py`, `plot_mualpha_step1.py`, `plot_zmet_plane.py`,
`mualpha_results.json`. Figures: `tmp/figs/` holds `outlier_ewmass.png`, `ssfr_mass_plane.png`,
`zmet_plane.png`, `fit5_compare.png`, `sfh_three.png`, `mock_vs_obs_dn4000.png`,
`mualpha_bin_grid.png`, `diag_railing.png`, `pozzetti_limit.png` + 28 more.

**Still gone (script lost; figure may survive):**

```
GONE  enc.py, flowmod.py          <- pipeline-verification legs for injection-recovery
                                     (the "re-encode reproduces saved latents to 2e-4" claim)
GONE  outliers_check_clean.ipynb  <- canonical TID-based characterization; ISSUE #7's headline table
                                     recovery path: tmp/outliers_analysis.py ("same as a script" per log)
GONE  tpagb_test_42580.py         <- the TP-AGB null
                                     partial: agb_pagb_profilemap_engineered.py + the pkl survive
GONE  ew_mock.py ew_vac.py ew_joint.py ssfr_joint.py ssfr_plane.py outlier_ewmass.py
                                  <- scripts for 2026-07-09 "both OOD families collapse to ONE fix"
                                     figures survive (outlier_ewmass.png, ssfr_mass_plane.png);
                                     ssfr_mstar.py may be a partial rewrite. Re-plottable, NOT re-runnable.
GONE  outputs/                    <- directory never existed in-repo; superseded by tmp/figs/
```

**The mechanism is unchanged and still live.** These ran in ephemeral cowork sandboxes. The 2026-07-09
entry admits it in-line: *"Artifacts (cowork session outputs, NOT yet in repo — TODO copy to
`tmp/`+`results/`)"*. That TODO sat undone for a week and cost six scripts. **And everything rescued is
now sitting in `tmp/`, which is gitignored** — one `rm -rf tmp` from a second, larger loss. The rescue
raises the stakes on Phase 0 step 4; it does not retire it.

Distinguish two states when triaging: **re-runnable** (script survives) vs **re-plottable only**
(figure survives, script does not). The second is not evidence under your own source-discipline rule —
a PNG cannot be re-derived or audited.

Corollary: `results/*.pkl` (94 files, 188 MB, gitignored) carry **no** record of which code or config
produced them. `alpha_tilt_map.pkl` exists; the α grid, seed count, and `maxfev` that made it are
recoverable only because `tmp/alpha_tilt_map.py` survived. For the missing dozen, nothing is recoverable.

**Phase 0 exists because of this.** Stop the bleeding before restructuring anything.

---

## 1. Target layout

```
hubersed/
├── pyproject.toml            # + ruff, pytest, deps groups, [project.scripts]
├── README.md                 # real one: what/why/how-to-run/DAG picture
├── CHANGELOG.md
├── Snakefile                 # the pipeline DAG (see §5)
├── config/
│   ├── default.yaml          # priors, cuts, model choices — currently hard-coded
│   ├── mocks_cue_500k.yaml
│   └── smoke.yaml            # tiny end-to-end for CI
├── src/hubersed/
│   ├── __init__.py           # + __version__
│   ├── paths.py              # KEEP, but drop the import-time mkdir side effect (§6)
│   ├── config.py             # NEW: load/validate YAML -> typed object
│   ├── provenance.py         # NEW: git SHA + config hash + env stamp on every artifact
│   ├── io/                   # h5/pkl readers, TID<->index, load_by_index
│   ├── data/                 # DESI/VAC access, S/N cut, z-floor, quality flags
│   ├── mocks/                # priors, SED generation, noise (from bin/model_seds/, bin/spender/noise/)
│   ├── sps/                  # model builders: Cue/FSPS, KC13 dust, LSF, WorkingMVN
│   ├── fitting/              # MAP driver, optimizer wrapper, fit_config
│   ├── detect/               # spender encode, NSF flow, IsoForest, thresholds
│   ├── features/             # dn4000(), halpha_ew() — WITH the is_flambda flag
│   └── plotting/
├── scripts/                  # thin CLIs, tracked, argparse -> package calls
├── experiments/              # tracked one-offs, dated, import the package (see §4)
├── tests/
├── docs/
│   ├── LOG.md                # mechanical/verification half of the split log (§3)
│   └── decisions/            # ADRs: 0001-alpha-tilt.md, 0002-polyoptcal.md, ...
├── notebooks/                # renamed from nb/, nbstripout in pre-commit
└── .github/workflows/tests.yml
```

**`bin/` disappears.** Its contents split: reusable → `src/hubersed/`, entry points → `scripts/`.
**`tmp/` disappears.** Triage per §4.
`results/`, `data/` stay gitignored — but see §2 (DVC) and §7 (provenance stamps).

### Precedent, from your own dependency tree

Upstream `prospector` (mounted at `~/Astronomy_Research/prospector`) already does the things worth
copying, so there is no argument that they're impractical here:

- `tests/` — 20 test files; `testpaths = ["tests"]` in `pyproject.toml`
- `.github/workflows/tests.yml` — matrix py3.10–3.14 × ubuntu/macos, **and it builds FSPS from
  `cconroy20/fsps` in CI**. So "FSPS makes CI impossible" is falsified in-house.
- `setuptools_scm` dynamic versioning
- SPEC 0 compliance comments on every pin

Standards checked from source, not memory:
[PyPA src-vs-flat](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) (you're
already correct here), [Scientific Python SPECs](https://scientific-python.org/specs/) — SPEC 0 (min
supported deps) and **SPEC 7 (seeding RNG)** are the two that bind you,
[`sp-repo-review`](https://learn.scientific-python.org/development/guides/repo-review/),
[pyOpenSci packaging guide](https://www.pyopensci.org/python-package-guide/package-structure-code/intro.html).

---

## 2. Phases

Ordered by failure-modes-closed per hour. Each phase ends green and is separately committable.

### Phase 0 — STOP THE BLEEDING (do today, before anything else)

1. `git add` the 5 untracked pipeline files: `map_chi2.py`, `flow_density.py`, `residual_anatomy.py`,
   `refit_clean_gps.py`, `recon_error.py`. These are load-bearing and one `rm -rf` from gone.
2. Commit the 3 dirty files (the shipped α fix: `get_stochastic_priors.py`, `make_cue_model_sed.py`,
   `fit_config.py`). Right now the fix the log calls *"the real fix and it stands"* exists only in your
   working tree.
3. `git rm --cached` the `.DS_Store`s; add to `.gitignore`.
4. **Un-gitignore `tmp/` and commit it as-is.** Triage later (§4). Preserving 41 files beats sorting them.
5. Write `docs/DANGLING_ARTIFACTS.md`: the GONE list above, marked *irrecoverable* vs *rerunnable*.
   For each, decide **rewrite / retract the claim / leave flagged**. Blunt version: any log claim whose
   only evidence is a GONE script is currently **unverifiable**, and by your own source-discipline rule
   should be demoted from "verified-from-run" to "claimed, evidence lost."

**Checkpoint:** `git status` clean; nothing load-bearing untracked.

### Phase 1 — Make it importable (unblocks everything)

- Add `__init__.py` to `src/hubersed/prospector/`, `src/hubersed/spender/` (currently implicit namespace
  packages — works by luck under `uv_build`).
- Move `bin/**` logic into the `src/hubersed/` tree per §1. Keep function bodies **byte-identical** in
  this phase; only relocate + import. Resist the urge to improve — a pure move is reviewable, a
  move+rewrite is not.
- Thin `scripts/*.py` wrappers preserving today's CLI flags (`--outliers`, `--cue`, `--limit`, `--tag`).
- Register `[project.scripts]` so `hubersed-map-chi2` etc. exist.

**Checkpoint:** every current `python bin/...` invocation has a working equivalent, verified by rerunning
one cheap job and diffing output bit-for-bit against the existing pkl.

### Phase 2 — Tests for the 10 bug classes (§0a)

One test per row in that table. This is the phase that pays for the refactor. Start with:

- `test_tid_roundtrip` — encode → outlier → TID → `load_by_index` → same TARGETID, for N random rows.
  The 06-10 bug, permanently.
- `test_dn4000_units` — `dn4000(f_nu)` vs `dn4000(f_lambda, is_flambda=True)` agree to <1e-6; assert the
  0.9273 factor explicitly so the bug can't return silently.
- `test_theta_is_dict` — MAP returns `{name: value}`, not a positional array. Kills the label-shift class.
- `test_mock_has_lines` — generated mock has intrinsic Hα/Hβ ≈ 2.86 (Case B). Kills the `hasspec` class.
- `test_mvn_prior_finite` — `MultiVariateNormal.__call__(9-vector)` returns a finite scalar. Currently
  **fails against upstream** → keep `WorkingMVN` in `src/hubersed/sps/` with a comment linking the
  upstream issue, and make the test assert the workaround, not the bug.
- `test_base_matches_generator` — the equivalence test that would have caught the 3.22-never-transferred
  retraction. Draw K params both ways, assert identical model spectra.

Add `tests/data/` with 2–3 tiny fixtures (one DESI spectrum, one mock row). No FSPS needed for most.

### Phase 3 — Provenance (`provenance.py`)

Every artifact write goes through one function. Stamp: git SHA, dirty-flag, config hash, resolved config
dict, `hubersed.__version__`, key dep versions (fsps/prospect/cue), UTC timestamp, RNG seed (**SPEC 7**:
pass `numpy.random.Generator`, never touch global `np.random`; your mocks are 500k draws and currently
unreseedable).

Then: `hubersed-provenance results/alpha_tilt_map.pkl` prints what made it. That question is currently
unanswerable for all 94 files.

Backfill what you can from surviving `tmp/` scripts; mark the rest `provenance: unknown (pre-refactor)`.

### Phase 4 — Config extraction

Hard-coded values that should be config, all named in the log as things you changed and had to
reason about later:

- `get_stochastic_priors.py`: `stellar_metallicities U(-1.5,0.4)`, `alphas U(-1.0,2.5)`
- `fit_config.py`: `logzsol U(-2.5,0.5)`, `dust_type=4`, `dust_index TopHat(-1.2,0.4)`
- S/N > 3 cut; `Z_FLOOR = 0.01`; 0.1% mock quantile threshold
- optimizer budget (`NSEEDS`, `MAXFEV`) — the log has a whole retraction (*"the free-SFH ceiling"*)
  caused by **comparing two arms at unequal budget**. Config makes that visible in a diff.

The mock/fit prior asymmetry the log flags (mocks `U(-1.0,+0.19)`, fits `U(-2.5,+0.5)` → *"the FITTER can
reach metal-rich galaxies the TRAINING SET has never seen"*) is exactly the bug class that a single
config file with **both** priors side by side makes obvious. Consider one `priors:` block with explicit
`mock:`/`fit:` sub-keys so a mismatch is visible on one screen.

**The table that should exist** (audited 2026-07-17 from `get_stochastic_priors.py` vs
`fit_config.get_priors`). Every ✅ is a coincidence maintained by hand across two files:

| param | mock | fit | |
|---|---|---|---|
| `logmass` | `U(7, 12)` | `Uniform(7.0, 12.0)` | ✅ |
| `logzsol` | `U(-1.5, +0.4)` | `Uniform(-2.5, +0.5)` | ❌ fit wider both ends |
| `sigma_reg` | `LogU(0.1, 5)` | `LogUniform(0.1, 5.0)` | ✅ |
| `tau_eq` / `tau_in` | `U(0.01, t_H)` | `Uniform(0.01, t_H)` | ✅ |
| `sigma_dyn` | `LogU(0.001, 0.5)` | `LogUniform(0.001, 0.5)` | ✅ |
| `tau_dyn` | `CN(0.01, 0.02, 0.005, 0.2)` | same | ✅ |
| `dust_index` | `U(-1.0, +0.4)` | `TopHat(-2.5, +0.4)` | ❌ fit 1.5 dex wider at floor |
| `dust2` | `CN(0.3, 1.0, 0, 4)` | same | ✅ |
| `dust_ratio` | `CN(1.0, 0.3, 0, 2)` | same | ✅ |
| `sigma_smooth` | `U(10, 400)` | `TopHat(10, 400)` | ✅ |
| `eline_sigma` | `U(20, 250)` | `TopHat(20, 250)` | ✅ |
| `gas_logu` | `U(-4, -1)` | `TopHat(-4, -1)` (fsps path) | ✅ |
| `gas_logz` | `U(-2.2, 0.5)` (cue) | `TopHat(-2.0, 0.5)` (fsps path) | ⚠️ fsps only |
| `dust_type` | 4 | 4 | ✅ (was 0 vs 4 until 2026-07-17) |

Both ❌ rows are the same failure: **fit prior ⊋ mock prior ⇒ the fitter reaches θ the training set has
never produced ⇒ those galaxies are OOD by construction.** `dust_index` is the more actionable one: the
log records the fits *railing* at the floor (`dust_index` railed at −1 in 9/11 worst cont-only; "railed
at −1.2 in ALL nearby strong-break"), so real galaxies demonstrably want δ < −1.0 and no mock is there.
Caveat from the same log entries: δ ≈ −2.5 is *"unphysical … laundering a smooth continuum tilt"*, so
widening the mock to match the fit is not obviously right either — it may be the **fit** prior that wants
narrowing, plus a calibration term. That is a science decision, not a config edit.

**Third gap, found the same day and worse than either range mismatch — a SHAPE mismatch.**
prospect's `ClippedNormal` (priors.py:331) sets `distribution = scipy.stats.truncnorm`: it is a
**truncated** normal despite the name. `hubersed.distributions.sample_clipped_normal` is genuinely
clipped (`np.clip(rng.normal(...))`). Different distributions. Whoever wrote the sampler implemented
prospect's class *name*; prospect's implementation contradicts it.

Fraction of each mock draw landing exactly ON a bound (computed, not estimated):

| param | mean | sigma | mini | maxi | at mini | at maxi |
|---|---|---|---|---|---|---|
| `tau_dyn` | 0.01 | 0.02 | 0.005 | 0.2 | **40.1%** | 0.0% |
| `dust2` | 0.30 | 1.00 | 0.0 | 4.0 | **38.2%** | 0.0% |
| `dust_ratio` | 1.00 | 0.30 | 0.0 | 2.0 | 0.0% | 0.0% |
| `duste_umin` | 2.00 | 1.00 | 0.1 | 15.0 | 2.9% | 0.0% |

**~38% of every mock ever generated has exactly `dust2 = 0.0`; ~40% have exactly `tau_dyn = 0.005`.**
Baked into the 14.9 GB h5. The fit prior says those atoms do not exist. The flow trains on this, so a
delta function at zero dust is training-set structure real galaxies do not have → contributes to
mock↔DESI shift → OOD flags, for purely a sampler reason. The log names the encoder domain shift as
*the* confounder on C-2ST; this is a second, unnamed one.

`sample_truncated_normal` is added to `distributions.py` and **deliberately not wired in**: switching
changes the training distribution and forces a regen. Blocked on a source question — the generator says
"from Wan+24 Stochastic prior model"; if Wan+24 specify a *clipped* normal, then prospect's prior is the
side that's wrong for the science and `fit_config` should change instead. Read arXiv:2404.14494 first.

Also found in the same audit, both minor:
- `fit_config.py:197` still lists `gas_logqion` in `build_full_cue_model`'s `vary_params`; the log
  already established it is a **dead no-op** (not a template key).
- `duste_umin` / `duste_qpah` / `duste_gamma` are drawn per-mock but Draine&Li emission peaks ~100 µm,
  while DESI at z≤0.6 covers rest 2250–9824 Å — **no dust emission in band**. Likely 3 of 21 prior arrays
  with no effect on any spectrum. UNVERIFIED: confirm by generating one mock at `duste_umin` 0.1 vs 15
  and diffing the flux before removing them.

### Phase 5 — DAG (Snakemake)

```
priors → mocks(500k, 14.9GB, ~40min) → noise → encode → flow → detect ─┐
                                                                        ├→ figures
                                          DESI spectra → encode ────────┘
                                          detect → map_chi2 → analysis
```

**Honest scoping — read this before committing to it.** `data/` is **276 GB**; mock regen is 500k FSPS
calls. You will never re-run this DAG from scratch in CI, and probably not often locally. So:

- The DAG's real value here is **declared dependencies + resumability + "which stage is stale"**, not
  one-click reproduction. Sell it to yourself on that, or don't do it.
- CI runs `config/smoke.yaml` only: ~100 mocks, whole DAG, minutes. That's a genuine end-to-end
  regression test and would have caught the `hasspec` bug on day one.
- Heavy stages get `Zenodo` deposits (`core:zenodo` skill can do this) so the DAG can start from cached
  artifacts. Otherwise the DAG is decorative.

**showyourwork:** right tool, wrong repo. It wraps Snakemake + tectonic + GHA so the *paper PDF* rebuilds
from source ([intro](https://show-your.work/en/stable/intro/); Luger). That belongs in **UberSED** (the
proposal/thesis side), pulling figures from hubersed artifacts via Zenodo. Putting it in hubersed would
demand the 276 GB pipeline run in GHA. It won't. Keep the split: **hubersed = plain Snakemake pipeline;
UberSED = showyourwork paper.**

### Phase 6 — Hygiene

- `ruff` (lint+format) + `pre-commit` (incl. `nbstripout` — `nb/` is 2.0 MB, mostly outputs).
- `.github/workflows/tests.yml` — copy prospector's FSPS-in-CI recipe verbatim; it solves your hardest CI
  problem and it's already in your dependency tree.
- `setuptools_scm`/`uv` dynamic version.
- Fix `pyproject.toml`: `description = "Add your description here"`, no license, no classifiers, no URLs.
- Run `pipx run 'sp-repo-review[cli]' .` — take the greens, argue with the reds, don't chase 100%.
- README: what it is, the DAG, how to run one fit, where results go, how to read provenance.

---

## 3. Splitting the log

Current: 1063 lines / 146 KB / **83 sections**, of which **11 are titled** CORRECTION/RETRACT/FALSIFIED/
INVALIDATED, plus **40** inline retraction markers. Append-only, single file, in the *wrong repo*
(describes `bin/prospector/map_chi2.py` and `results/*.pkl`, lives in UberSED).

The retraction density is not a flaw — it is the most valuable property of this document and the thing
that makes your source discipline real. Preserve it. But an append-only file where the answer to
*"is 44.5% still true?"* requires reading 5 non-adjacent sections in order (claim → provisional →
noise-bug → resolution → superseded by 10.9%) has outgrown its format.

**Split (per your call):**

| goes to `hubersed/docs/` | goes to `UberSED/knowledge/` |
|---|---|
| bug reports + retractions with a code cause (index bug, f_nu/f_lambda, MVN, label shift, base≠generator) | science narrative: what the OOD population *is*, bucket taxonomy, physical interpretation |
| verification records (C-2ST, TID round-trips, injection recall) | literature synthesis (Wan+24/25, Burnham+26, spender I/II/III) |
| "Discipline going forward" | proposal/FINESST/issue bodies |
| Key files / artifacts index | μ_α programme design, Goal A/B fork |

Cross-link both ways by anchor. Accept the cost you flagged in the question: two places to look. Mitigate
with a single index in each.

**Two format changes, worth more than the split:**

1. **Status header per claim.** Every finding gets `STATUS: live | superseded-by(§X) | retracted(§X) |
   evidence-lost`. Right now 44.5%, PolyOptCal, and qion all read as live if you land on the wrong
   section. `44.5%` is superseded by `10.9%`. The qion result is retracted. A reader (including future you,
   including me next session) cannot tell without reading everything.
2. **ADRs in `docs/decisions/`.** One file per irreversible choice, with the evidence and the date:
   `0001-alpha-tilt-shipped.md`, `0002-polyoptcal-rejected.md`, `0003-logzsol-widened-for-theta-coverage-
   not-dn4000.md` (that one's justification *changed* mid-log — exactly what an ADR captures and a
   narrative loses).

---

## 4. `tmp/` triage (41 files)

**Rule:** shared logic → package; one-off → `experiments/YYYY-MM-DD_name.py` (tracked, imports package);
dead → delete with a commit message saying why.

The repeated primitives — extract these first, they're duplicated across ~10 scripts each and are where
the bugs lived:

| primitive | currently duplicated in | note |
|---|---|---|
| MAP optimizer wrapper (`_map_optimize`, NSEEDS/MAXFEV, validity gate) | ~10 scripts | unequal budgets caused a retraction |
| model builders (Cue, KC13 dust, LSF, `build_obs`) | ~10 scripts | drift between arms = silent confound |
| `WorkingMVN` | `alpha_tilt_map.py` only | **must** be in package; upstream is broken |
| `dn4000()`, `halpha_ew()` | ≥4 scripts | the f_nu/f_lambda bug's home |
| SFH reconstruction from `logsfr_ratios` | ≥4 scripts | |
| TID → index → spectrum | ≥6 scripts | the 06-10 bug's home |
| `paste_desi_noise` | in `bin/`, reinvented in a GONE script | the reinvention incident |

Cited-in-log count (`grep`), as a keep-priority signal:

- **5–2 cites → keep, promote:** `mock_vs_obs_dn4000.py`, `freesfh_fixedqion_fit.py`, `mualpha_base_set.py`,
  `fit_5_nonoutliers.py`, `fetch_cutouts.py`, `ssp_age_scan_42580.py`, `qion_fixedsfh_fit_94183.py`,
  `outliers_analysis.py`, `burstysfh_fit_94183.py`, `alpha_tilt_map.py`, `outliers_check.ipynb`
- **1 cite → `experiments/`:** `sfh_resolution_parallel_42580.py`, `quiescent_sfh.py`,
  `qion_freesfh_fit_94183.py`, `polycal_fit_42580.py`, `plot_sfh_resolution_42580.py`, `plot_map_fits.py`,
  `mualpha_step1_sensitivity.py`, `mapfit_bestAGB_42580.py`, `inject_dynesty_uniform.py`,
  `imf_mapfit_42580.py`, `freesfh_nopoly_fit.py`, `check_logzsol_response.py`,
  `calibrate_quiescent_prior.py`
- **0 cites → decide individually, don't bulk-delete:** `view_results.ipynb`,
  `uniform_ssfr_fixedqion_fit.py`, `time_emcee.py`, `sfr_time_plot.ipynb`,
  `sfh_resolution_test_42580.py` (superseded by `_parallel`), `plot_worst_spectra.py`,
  `plot_map_models.py`, `make_outlier_showcase.py` (→ UberSED, it's a proposal figure),
  `inject_dynesty_mocks.py`, `dynesty_freesfh_nopoly.py`, `cue_tests.ipynb`,
  `check_latent_space.ipynb`, `agb_pagb_profilemap_engineered.py`

Zero cites ≠ worthless: `inject_dynesty_mocks.py` ("injection-recovery on the REAL mock catalog, not
hand-rolled") is the *correct* pattern the GONE `mualpha_infer.py` violated. Keep it as the reference.

---

## 5. Risks / where I'd push back on myself

- **Restructure while the science is live.** You have open threads (EELG/nebular, 42580 at χ²=4.65,
  Goal B parked). A big move breaks muscle memory and every path in the log. Mitigation: Phase 1 is a
  *pure move*, and Phase 0 is independently valuable even if you stop there.
- **The DAG may not earn its keep.** 276 GB and 40-min regen means you'll run it rarely. If after Phase 5
  you're not invoking Snakemake weekly, it's scaffolding you maintain for nothing. Decide at Phase 5 with
  Phases 0–4 already banked.
- **Phase 2 tests will fail against upstream prospector** (`MultiVariateNormal`). That's correct and it's
  also the log's open TODO — *"worth reporting upstream."* File it; the test then documents *your*
  workaround, not their bug.
- **I am not confident about the log split.** You chose it, and the mechanical/science boundary is real —
  but several entries are genuinely both (the `f_nu` bug *is* the 44.5% result). A third location plus a
  judgement call per entry is real overhead. If it starts hurting, collapse to "all in hubersed, UberSED
  links in" — that's the cheaper failure mode of the two.
- **Unknown to me:** whether `data/` (276 GB) has any off-machine copy. If not, that dwarfs everything
  here as a risk and Phase 0 should include a Zenodo/tape deposit of the irreproducible inputs.

---

## 6. Small things worth fixing while you're in there

- `paths.py` runs `ensure_dirs(...)` **at import time**, creating 6 directories and writing/deleting a
  `.write_test` file in each, as a side effect of `import hubersed.paths`. That makes any import
  filesystem-mutating and untestable in a sandbox. Make it an explicit call.
- `src/hubersed/__init__.py` is `def hello(): return "Hello from hubersed!"` — the uv template stub.
- No `__init__.py` in `prospector/`/`spender/` subpackages.
- `parameter_file.py`: commented `sigma_kms =- ...` typo (log flags it; `=-` is valid Python, so ruff
  won't catch it — a test will).
- `map_chi2.py` TODO from the log: save theta as `{name: theta[theta_index[name]]}`.
- `residual_anatomy.py` + `flow_density.py`: log says *"still positional — migrate to TARGETID before
  use."* Check current state before trusting.
- `.python-version` says 3.13 but `__pycache__` has `cpython-310` artifacts — two interpreters have run
  this tree.

---

## 7. Suggested order (concrete)

**Branching (settled 2026-07-16).** `feature/latent-space-check` is not finished, so it cannot merge to
master yet. Refactor branches off it and merges **back into it**. Consequence, stated so it's a choice
and not an accident: `latent-space-check` is now the de-facto trunk, and master is dead until the science
thread closes. Acceptable — but it means the refactor branch must be **short-lived (days)**, because
every day it lives is a day of live science work that will have to rebase across a file move.

**Correction to an earlier draft of this plan.** It said both "tests first" and "Phase 1 checkpoint =
diff output bit-for-bit". Those are different safety nets and conflating them was an error:

- For a **pure move**, the net is a **golden-output diff** (run a cheap job now → save pkl → rerun after
  → assert identical). Unit tests cannot be the net, because you cannot import what isn't importable.
- **Tests-first survives only for tests that are new modules rather than moves.** `test_dn4000_units`
  qualifies: extracting `dn4000()` into `src/hubersed/features/` is the first brick of the new layout,
  not a relocation — and it collapses a live duplicate (`tmp/mock_vs_obs_dn4000.py:35` has the
  `is_flambda` fix; `tmp/calibrate_quiescent_prior.py:58` `dn4000_narrow` does not).

Order:

1. **Phase 0** on `feature/latent-space-check` directly, no branch. ~1 hour. Un-gitignoring `tmp/` (now
   53 files + 37 figures) is the single highest-value action in this document.
2. **Golden-output baseline.** Run one cheap MAP job, commit the pkl hash. This is what makes step 4
   verifiable.
3. **One test first:** `dn4000()` → `src/hubersed/features/` + `test_dn4000_units`.
4. **`refactor/src-layout`** off `latent-space-check` → merge back in. `__init__.py`s; `bin/` → `src/`;
   delete all **22** `sys.path.insert(0, "bin/prospector")` hacks (cwd-relative — those scripts only run
   from the repo root). Pure move. Verify with step 2's diff.
5. Rest of **Phase 2** tests — they're writable now that imports exist. Then **3**, **4**.
6. Re-decide **Phase 5** with 0–4 banked.
7. **Phase 6** continuously.
8. **§3 log split** — last. Pure loss if the code layout moves underneath it.
