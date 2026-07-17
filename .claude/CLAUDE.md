# hubersed

Amortized SED fitting: normalizing flows on Prospector/FSPS mock spectra, DESI DR1 BGS SV3.
Latent-space OOD detection (spender encoder → NSF flow) + forward-model reach tests (MAP χ²).

Nikhil drives. You assist. **Never commit. Never add yourself as co-author or Co-Authored-By. Never push.**
Stage nothing. Nikhil reviews and commits every change himself.

`gh` is READ-ONLY too. `gh issue view|list`, `gh pr view|diff` — yes. `gh issue create|comment|close`,
`gh pr create|merge`, `gh release` — no. Draft the text, hand it over, Nikhil posts it.

---

## Non-negotiable: source discipline

Verify BEFORE stating. Correctness > speed. Think as long as needed.

- Factual/empirical/literature claim → read primary source first (arxiv TeX, code, data, instrument docs), then state.
- Mark every claim: `verified-from-source/run` | `from-memory-unchecked` | `dont-know`.
- Never invent citations, arxiv IDs, author names, results, file paths, API names.
- Unverified → say so or stay silent. Do NOT write it down.
- A grep hit is not proof of use. A docstring is not proof of behaviour. Read the code.

Applies to this repo too: don't claim a function does X without reading it.

## Non-negotiable: this repo's failure mode

Ten logged bugs, all mechanical, all test-catchable, zero caught (there were no tests).
See `REFACTOR_PLAN.md` §0a for the table. The invariants they map to:

| invariant | why |
|---|---|
| **Latent row → galaxy is ONLY ever by TARGETID.** Never by position. | loader is `sorted(glob)` = lexicographic ≠ numeric `chunk*1024+row`. Diverges from index 2048. Scrambled ~99% of catalogue matching for months. |
| **Band ratios: `f_nu` vs `f_lambda` is not cosmetic.** | Dn4000 is Balogh+99 = ratio of mean F_NU. h5 `fluxes` = maggies (~f_nu). Noised pkls = f_lambda. Skipping ×λ² biases Dn4000 by (3900/4050)² = 0.9273. |
| **`theta` is a dict, never a positional array.** | `logsfr_ratios` is 9-element → `len(theta)=26` vs 18 labels → every label after it shifts. Produced an impossible `gas_logu≈0` readout. |
| **RNG: pass `np.random.Generator`. Never global `np.random.*`.** | SPEC 7. 500k mocks are currently unreseedable. 10 NPY002 hits open. |
| **Compare arms at EQUAL optimizer budget.** | Unequal NSEEDS/MAXFEV produced the "free-SFH ceiling" retraction. |
| **A proxy is not the pipeline.** | The μ_α base set drew from FIT priors + a different template; its Dn4000=3.22 never transferred (generator max 2.300). Validate a fix on the thing it fixes. |

If you touch code near any of these, the change needs a test. Not optional.

## Non-negotiable: artifacts

`results/`, `data/`, `tmp/` are gitignored. **Anything you produce that isn't committed does not exist.**
Six analysis scripts were already lost this way; the log still cites them. See
`REFACTOR_PLAN.md` §0c.

- Never write an artifact without also writing how it was made.
- Never say "TODO: copy this into the repo later." Do it now or say it's throwaway.
- `results/*.pkl` currently carry no git SHA / config. Don't add more of those.

---

## Layout

Current (mid-refactor — check `git status` before assuming):

```
src/hubersed/     the package. 804 lines. importable.
bin/              NOT importable, no __init__.py. 3449 lines. The real pipeline.
tmp/              gitignored. 53 files. Where most logged science actually ran.
results/ data/    gitignored. data/ is 276 GB.
nb/               notebooks.
```

Target — `REFACTOR_PLAN.md` §1. Short version:
`src/hubersed/{io,data,mocks,sps,fitting,detect,features,plotting}/` + `scripts/` (thin CLIs) +
`experiments/` (tracked one-offs) + `tests/` + `config/` + `docs/`. `bin/` and `tmp/` disappear.

**22 scripts contain `sys.path.insert(0, "bin/prospector")`** — cwd-relative, so they only run from
repo root. That's the defect Phase 1 deletes. Don't add a 23rd. Don't patch them individually.

## Commands

```bash
uv run pytest                      # NOT uvx — uvx = isolated env, hubersed not installed
uv run ruff check --select F401,F811,F841,UP,NPY src bin tmp
uv run ruff format
uv run vulture src bin tmp --min-confidence 60
uv run deptry .
```

Always `uv run`. `uvx` is unpinned and skips the project env.

## Dependency source — read it, don't recall it

Three deps are FORKS. Upstream docs describe different code. Never answer a "what does prospect do"
question from memory or from published docs — read these:

| dep | source on disk | note |
|-----|----------------|------|
| `prospect` | `.venv/lib/python3.13/site-packages/prospect/` (46 .py) | fork: `galactic-ai/prospector @ feature/cue-on-v2` |
| `spender` | `.venv/lib/python3.13/site-packages/spender/` (12 .py) | fork: `galactic-ai/spender` |
| `cue` | `.venv/lib/python3.13/site-packages/` | fork: `yi-jia-li/cue @ cuejax` |
| `nflows` | `.venv/lib/python3.13/site-packages/nflows/` (42 .py) | upstream |
| prospector git checkout | `~/Astronomy_Research/prospector/` | has `tests/` + CI worth copying |

No documentation MCP (Context7 etc.) — it indexes UPSTREAM, which is not what is installed here.
The `MultiVariateNormal` bug below exists in no doc anywhere; it was found by reading
`prospect/models/priors.py:269`. That is the standard.

## Traps — do not "fix" these

- **deptry DEP002** (`dynesty`, `fsps`, `corner`, `mpi4py`, `schwimmbad`, `astro-cue`, `astro-sedpy`,
  `umap-learn`) = **runtime-only**, reached through `prospect`, never imported directly (verified:
  zero `import cue|sedpy|umap` in src/bin/tmp). Deleting them breaks runtime and no static tool can
  see it. `dynesty==2.1.5` is pinned exact on purpose. Add to `[tool.deptry] ignore`, don't remove.
- **deptry DEP001** (`fit_config`, `parameter_file`, `map_chi2`, `save_results`, `fit_single`) = the
  `sys.path` hack, not a missing dependency. Phase 1 problem.
- **`parameter_file.py:14`** `np.infty = np.inf` monkeypatches numpy process-wide for a downstream
  caller. Find the caller before ruff `--fix` rewrites it.
- **`prospect.models.priors.MultiVariateNormal` has no `__call__`** → inherits scalar
  `norm.logpdf(x, scale=Sigma)` → 9×9 NaN matrix → `lnprobfn` raises. Upstream bug. Invisible to
  dynesty (uses `unit_transform`), fatal to MAP/emcee. `WorkingMVN` in `tmp/alpha_tilt_map.py` is the
  workaround — it belongs in the package. Don't "fix" by reverting to TopHat.
- **`vulture` "unused function"** at 60% confidence is a guess. `_`-prefixed = internal callers.
  Verify before deleting.

## Working style

- **One file at a time.** Land it, then move.
- **Pure moves stay pure.** Relocate ≠ improve. A move+rewrite is unreviewable. Safety net for a move
  is a golden-output diff (run a cheap job → save pkl → rerun → assert identical), not unit tests.
- **One concern per commit.** `ruff --fix`, `ruff format`, and NPY002 are three commits, not one.
- Push back. If a request has a gap, an unstated assumption, or contradicts something in this file or
  the log, say so before coding. Don't agree by default.
- Don't fill gaps with confident prose. "Not sure, here's how we'd check" beats a polished guess.

## Skills

- `ponytail` — good adversary against the plan (it would kill the DAG, correctly). **Turn OFF while
  executing**: its rung-1 YAGNI and "trivial one-liners need no test" delete exactly the tests this
  repo exists to acquire. Project rules in this file outrank ponytail.
- `caveman` — prose only, and it leaves code alone by design. Marginal here.

## Context

- Refactor plan + evidence: `REFACTOR_PLAN.md`
- Science/bug history: `../../Documents/UberSED/knowledge/outlier_investigation_log.md` (1063 lines,
  83 sections, 11 retraction-titled). Read the STATUS of a claim before relying on it — several
  headline numbers are superseded (44.5% → 10.9%) or retracted (qion, free-SFH ceiling).
- Issues: `galactic-ai/hubersed` #7 (DESI outliers), #10 (encoder comparison). Read-only.
- Branch: `feature/latent-space-check` is the de-facto trunk (73 ahead of master, unmerged).
  Refactor branches off it and merges back into it.
