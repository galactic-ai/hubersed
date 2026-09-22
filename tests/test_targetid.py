import numpy as np
import pytest
import torch
from spender.data.desi import DESI

from hubersed.fitting import chi2

ROWS = 1024 
N_CHUNKS = 11  # the fewest chunks for which "10" sorts before "2"
# DESI-sized TARGETIDs, shuffled so that sorted order differs from index order
TIDS = np.random.default_rng(0).permutation(N_CHUNKS * ROWS) + 39_627_000_000_000_000


@pytest.fixture
def chunk_dir(tmp_path, monkeypatch):
    """Write fake chunk pickles and all_target_ids.npy, and point chi2 at them."""
    chunk_dir = tmp_path / "desi_spectra"
    chunk_dir.mkdir()
    for k in range(N_CHUNKS):
        spec = torch.ones(ROWS, 2)  # two pixels are enough
        one = torch.ones(ROWS)
        tid = torch.from_numpy(TIDS[k * ROWS : (k + 1) * ROWS])
        # spec, ivar, z, TARGETID, norm, zerr: the six tensors of a real chunk
        batch = [spec, spec, one, tid, one, one]
        DESI.save_batch(str(chunk_dir), batch, tag="chunk1024", counter=k)
    np.save(tmp_path / "all_target_ids.npy", TIDS)
    monkeypatch.setattr(chi2, "DATA_PATH", tmp_path)
    return chunk_dir


@pytest.mark.usefixtures("chunk_dir")
@pytest.mark.parametrize("gidx", [2048, 10 * ROWS, N_CHUNKS * ROWS - 1])
def test_tid_round_trip(gidx):
    """TARGETID -> tids_to_indices -> load_by_index gives back the same galaxy."""
    tid = TIDS[gidx]
    idx = int(chi2.tids_to_indices(np.array([tid], np.int64))[0])
    _, _, _, loaded_tid = chi2.load_by_index(idx)
    assert idx == gidx
    assert loaded_tid == tid


@pytest.mark.usefixtures("chunk_dir")
def test_unknown_tid_raises():
    """One unknown TARGETID fails the whole call; nothing is dropped or guessed."""
    # above every stored TARGETID: the case the searchsorted clip exists for
    unknown = TIDS.max() + 1
    with pytest.raises(ValueError, match="1/2 TARGETIDs not found"):
        chi2.tids_to_indices(np.array([TIDS[0], unknown], np.int64))


def test_encoder_row_is_not_global_index(chunk_dir):
    """Encoder row 2048 is chunk 10, not global index 2048: spender reads files in string order."""
    loader = DESI.get_data_loader(str(chunk_dir), tag="chunk1024", which="all")
    encoder_tids = torch.cat([batch[3] for batch in loader]).numpy()
    np.testing.assert_array_equal(encoder_tids[:2048], TIDS[:2048])
    assert encoder_tids[2048] == TIDS[10 * ROWS]
