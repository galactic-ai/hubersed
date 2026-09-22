"""Check that the golden-output comparison finds a single changed value and ignores run time."""

import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

COMPARE = Path(__file__).parent / "golden" / "compare.py"


def write_outputs(root, latent=0.5, seconds=1.0):
    """Write one small npz, h5 and pickle file like the golden jobs produce."""
    root.mkdir()
    np.savez(root / "priors.npz", mass=np.array([1.0, np.nan]), _seed=0)
    with h5py.File(root / "latents.h5", "w") as f:
        f.create_dataset("latents", data=np.array([[latent, 1.0]], dtype=np.float32))
        f.attrs["tag"] = "chunk1024"
    rec = {"theta": np.array([1.0, 2.0]), "labels": ["a", "b"], "optim": {"seconds": seconds}}
    with open(root / "map.pkl", "wb") as f:
        pickle.dump(rec, f)


def run(base, new):
    """Run compare.py on two folders and return its exit code and output."""
    r = subprocess.run(
        [sys.executable, str(COMPARE), str(base), str(new)], capture_output=True, text=True
    )
    return r.returncode, r.stdout


def test_identical_outputs_pass_even_when_run_time_differs(tmp_path):
    """Identical values with a different wall-clock time and a NaN count as identical."""
    write_outputs(tmp_path / "a", seconds=1.0)
    write_outputs(tmp_path / "b", seconds=99.0)
    assert run(tmp_path / "a", tmp_path / "b")[0] == 0


def test_one_changed_latent_is_reported(tmp_path):
    """A single float32 latent that moves by one part in a million is reported by name."""
    write_outputs(tmp_path / "a", latent=0.5)
    write_outputs(tmp_path / "b", latent=0.5 + 1e-6)
    code, out = run(tmp_path / "a", tmp_path / "b")
    assert code == 1
    assert "latents.h5.latents differs" in out
