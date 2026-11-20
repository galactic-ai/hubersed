"""Find spectra by TARGETID in a Parquet dataset spread over splits, files and row groups."""

import numpy as np
import pytest
from spender.data.dataset import get_data_loader, get_features, write_dataset

from hubersed.io import desi

N = 5000
# DESI-sized TARGETIDs, shuffled so that sorted order differs from write order
TIDS = np.random.default_rng(0).permutation(N) + 39_627_000_000_000_000


@pytest.fixture
def dataset(tmp_path):
    """Write N two-pixel spectra whose first pixel holds their write position.

    Small shards and row groups put the spectra in many files and row groups.
    """
    pos = np.arange(N, dtype=np.float32)
    batches = [
        dict(
            spec=np.stack([pos[k : k + 1000], pos[k : k + 1000]], axis=1),
            w=np.ones((len(pos[k : k + 1000]), 2), np.float32),
            z=np.zeros(len(pos[k : k + 1000]), np.float32),
            zerr=np.zeros(len(pos[k : k + 1000]), np.float32),
            norm=np.ones(len(pos[k : k + 1000]), np.float32),
            target_id=TIDS[k : k + 1000],
        )
        for k in range(0, N, 1000)
    ]
    path = tmp_path / "ds"
    write_dataset(
        str(path), batches, get_features(2, ["target_id"]), ["target_id"], shard_size=512,
        row_group_size=128,
    )  # fmt: skip
    return path


def test_dataset_spans_files_and_row_groups(dataset):
    """The fixture really spreads the spectra over several files."""
    assert len(list((dataset / "data").glob("*.parquet"))) > 3


@pytest.mark.parametrize("i", [0, 1, 777, 2048, N - 1])
def test_tid_round_trip(dataset, i):
    """The spectrum found for a TARGETID is the one written with it."""
    flux, _, _, tid = desi.load_by_targetid(int(TIDS[i]), dataset)
    assert tid == TIDS[i]
    assert flux[0] == i


def test_splits_cover_every_spectrum_once(dataset):
    """Reading train, validation and test returns each TARGETID exactly once."""
    tids = [
        b[0].numpy()
        for split in ("train", "validation", "test")
        for b in get_data_loader(str(dataset), which=split, columns=("target_id",))
    ]
    tids = np.concatenate(tids)
    assert len(tids) == N
    np.testing.assert_array_equal(np.sort(tids), np.sort(TIDS))
