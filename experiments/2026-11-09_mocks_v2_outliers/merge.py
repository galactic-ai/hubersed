"""Merge the 25 mock parts into one file with the keys of the old 500k mock sets.

Reads the 25 parts results/2026-11-09_mocks_v2_outliers/mocks_<start>_<start + 20000>.h5 in order
and writes data/prospector_model/mocks_v2_500000.h5, next to the old mock sets: fluxes (maggies, chunks of 1024 rows), wavelength, failed,
priors/redshifts, priors/theta, priors/column_ew, priors/rbank_index, with the attributes of the
first part. Checks: the parts tile 0..500,000 with no gap, every part has the same wavelength and
labels, and no mock failed.
COMMAND: from the repository root on the Mac or a compute node,
``uv run python experiments/2026-11-09_mocks_v2_outliers/merge.py`` (post.slurm on stampede3).
"""

import h5py
import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-09_mocks_v2_outliers"
N = 500_000
STEP = 20_000
KEYS = (
    "fluxes",
    "failed",
    "priors/redshifts",
    "priors/theta",
    "priors/column_ew",
    "priors/rbank_index",
)


def main():
    """Merge the parts and print the checks."""
    # The 25 parts of the full run; other files here (the 1,000-mock pilot) are left out.
    parts = [(s, s + STEP, OUT / f"mocks_{s}_{s + STEP}.h5") for s in range(0, N, STEP)]
    missing = [p.name for _, _, p in parts if not p.exists()]
    assert not missing, f"missing parts: {missing}"
    print(f"{len(parts)} parts")

    with h5py.File(parts[0][2], "r") as f0:
        wave = f0["wavelength"][:]
        attrs = dict(f0.attrs)
        shapes = {k: f0[k].shape[1:] for k in KEYS}
        dtypes = {k: f0[k].dtype for k in KEYS}
        sub_attrs = {k: dict(f0[k].attrs) for k in KEYS}

    path = PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5"
    with h5py.File(path, "x") as out:
        for k, v in attrs.items():
            out.attrs[k] = v
        out.attrs.update(start=0, stop=N)
        out.create_dataset("wavelength", data=wave)
        dsets = {}
        for k in KEYS:
            chunks = (1024, *shapes[k]) if k == "fluxes" else None
            dsets[k] = out.create_dataset(
                k,
                (N, *shapes[k]),
                dtypes[k],
                chunks=chunks,
                compression="gzip" if k == "fluxes" else None,
            )
            for a, v in sub_attrs[k].items():
                dsets[k].attrs[a] = v
        nfail = 0
        for start, stop, p in parts:
            with h5py.File(p, "r") as f:
                assert np.array_equal(f["wavelength"][:], wave), p.name
                assert dict(f.attrs).keys() == attrs.keys(), p.name
                assert (f.attrs["start"], f.attrs["stop"]) == (start, stop), p.name
                for a, v in attrs.items():
                    if a not in ("start", "stop"):
                        assert np.array_equal(np.asarray(f.attrs[a]), np.asarray(v)), (p.name, a)
                assert f["fluxes"].shape[0] == stop - start, p.name
                for k in KEYS:
                    if k == "fluxes":
                        for i in range(0, stop - start, 1024):
                            j = min(i + 1024, stop - start)
                            dsets[k][start + i : start + j] = f[k][i:j]
                    else:
                        dsets[k][start:stop] = f[k][:]
                nfail += int(f["failed"][:].sum())
            print(f"  {p.name} done", flush=True)
        assert nfail == 0, f"{nfail} failed mocks"
        z = dsets["priors/redshifts"][:]
        print(f"merged {N} mocks, 0 failed; z {z.min():.4f}-{z.max():.4f}; wrote {path}")


if __name__ == "__main__":
    main()
