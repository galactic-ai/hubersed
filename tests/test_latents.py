"""load_latents returns latents with their TARGETIDs and refuses files without them."""

import h5py
import numpy as np
import pytest

from hubersed.io.latents import load_latents


def write(path, with_tids=True):
    """Write a small latent file with two rows."""
    with h5py.File(path, "w") as f:
        f["latents"] = np.arange(6, dtype=np.float64).reshape(2, 3)
        if with_tids:
            f["target_ids"] = np.array([39627000000000001, 39627000000000002])
        f.attrs["snr_min"] = 3
        f.attrs["checkpoint"] = "enc.pt"


def test_rows_types_and_attrs(tmp_path):
    """Latents come back as float32 next to int64 TARGETIDs, with the file attributes."""
    path = tmp_path / "lat.h5"
    write(path)
    lat, tid, attrs = load_latents(path)
    assert lat.dtype == np.float32 and lat.shape == (2, 3)
    assert tid.dtype == np.int64 and tid.tolist() == [39627000000000001, 39627000000000002]
    assert attrs["snr_min"] == 3 and attrs["checkpoint"] == "enc.pt"


def test_file_without_target_ids_raises(tmp_path):
    """A file from before the TARGETID fix is refused."""
    path = tmp_path / "old.h5"
    write(path, with_tids=False)
    with pytest.raises(KeyError, match="target_ids"):
        load_latents(path)


def test_missing_file_raises(tmp_path):
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_latents(tmp_path / "none.h5")
