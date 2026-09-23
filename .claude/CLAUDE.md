# hubersed

Amortized SED fitting with normalizing flows trained on Prospector and FSPS mock spectra,
applied to DESI DR1 BGS SV3. Outliers are found in the spender latent space with a flow
and then checked with forward-model fits.

## Layout

- `src/hubersed/` is the package. Subpackages are `alf`, `detect`, `fitting`, `io`, `mocks`,
  `plotting` and `sps`.
- `scripts/` holds thin command line entry points.
- `experiments/` holds tracked one-off analyses, named `YYYY-MM-DD_name.py` with `# %%` cells.
  Notebooks are not source.
- `tests/` holds the tests. `tests/golden/` holds the golden output check for pure moves.
- `tmp/` and `nb/` are legacy and not linted. Do not add new work there.
- `data/` and `results/` are not tracked.

Import from the `hubersed` package. Never edit `sys.path`.

## Commands

```bash
uv run pytest -m "not slow"                        # quick loop
uv run pytest -m "not fsps and not data and not slow"   # what CI runs
uv run ruff check
uv run ruff format
```

Always use `uv run`, never `uvx`, so tools run in the project environment.
Test markers are `fsps` (needs python-fsps and SPS_HOME), `data` (reads `data/`) and `slow`.

## Working on code

- Read the installed source of a dependency before saying what it does. `prospect`, `spender`
  and `cuejax` are forks, so upstream docs describe different code. `uv.lock` has the commits.
- A grep hit is not proof a function is used, and a docstring is not proof of behaviour.
  Read the code.
- Every artifact needs the code and command that made it, committed next to it.
- Change one file at a time and keep one concern per commit. Formatting, lint fixes and
  behaviour changes go in separate commits.
- A pure move changes nothing but location. Check it with the golden output in
  `tests/golden/`, not with new unit tests.
- Write numpy style docstrings in short plain sentences.
