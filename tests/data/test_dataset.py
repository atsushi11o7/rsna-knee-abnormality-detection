import numpy as np
import pandas as pd
import torch

from rsna_knee.data.dataset import (
    KneeStudyDataset,
    SeriesRef,
    allocate,
    collate,
    window_indices,
)
from rsna_knee.data.folds import make_folds


def test_window_uses_physical_offset_on_thin_series():
    pos = np.arange(0, 20, 0.5)  # 0.5 mm spacing
    assert window_indices(pos, 10, 3.5) == [3, 10, 17]


def test_window_steps_out_on_thick_series():
    pos = np.arange(0, 70, 7.0)  # 7 mm spacing: nearest to ±3.5 mm rounds to the centre
    assert window_indices(pos, 4, 3.5) == [3, 4, 5]


def test_window_clamps_at_edges():
    pos = np.arange(0, 10, 1.0)
    assert window_indices(pos, 0, 3.0) == [0, 0, 3]


def test_allocate_sums_to_total():
    counts = allocate(5, 32, np.random.default_rng(0))
    assert counts.sum() == 32 and counts.max() - counts.min() <= 1


def _make_index(tmp_path, n_series=3, n_slices=12):
    refs = []
    for s in range(n_series):
        p = tmp_path / f"s{s}.npy"
        np.save(p, np.full((n_slices, 8, 8), s, np.uint8))
        refs.append(SeriesRef(p, np.arange(n_slices, dtype=np.float32) * 3.0, s % 3, s % 2))
    return {"study": refs}


def test_dataset_train_and_eval_shapes(tmp_path):
    index = _make_index(tmp_path)
    y = pd.DataFrame([[0.5] * 12], index=["study"])
    ds = KneeStudyDataset(["study"], index, y, y, train=True, n_windows=10)
    item = ds[0]
    assert item["images"].shape == (10, 3, 8, 8)
    assert set(item["series_id"].tolist()) == {0, 1, 2}

    ds = KneeStudyDataset(["study"], index, train=False, windows_per_series=4)
    assert ds[0]["images"].shape[0] == 12


def test_collate_pads_and_masks(tmp_path):
    index = _make_index(tmp_path)
    index["small"] = index["study"][:1]
    ds = KneeStudyDataset(["study", "small"], index, train=False, windows_per_series=4)
    batch = collate([ds[0], ds[1]])
    assert batch["images"].shape[:2] == (2, 12)
    assert batch["mask"].sum(1).tolist() == [12, 4]
    assert batch["images"].dtype == torch.uint8


def test_folds_keep_duplicate_reports_together():
    ids = [f"s{i}" for i in range(40)]
    reports = ["same" if i < 4 else f"r{i}" for i in range(40)]
    df = pd.DataFrame({"StudyInstanceUID": ids, "Report": reports})
    y = pd.DataFrame(np.random.default_rng(0).random((40, 12)), index=ids)
    fold = make_folds(df, y, n_splits=4)
    assert fold.loc[ids[:4]].nunique() == 1
    assert (fold >= 0).all()


def test_workers_draw_different_windows(tmp_path):
    index = _make_index(tmp_path, n_series=1, n_slices=200)
    ds = KneeStudyDataset(["study"] * 4, index, train=True, n_windows=8)
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2, collate_fn=collate)
    picks = {tuple(b["rel_pos"][0].tolist()) for b in loader}
    assert len(picks) == 4


def test_missing_studies_raise_for_inference_and_are_reported_for_training(tmp_path):
    import pytest

    index = _make_index(tmp_path)
    with pytest.raises(KeyError):
        KneeStudyDataset(["study", "gone"], index, train=False)
    with pytest.warns(UserWarning):
        ds = KneeStudyDataset(["study", "gone"], index, train=False, drop_missing=True)
    assert ds.missing == ["gone"] and len(ds) == 1


def test_weights_default_to_one(tmp_path):
    index = _make_index(tmp_path)
    y = pd.DataFrame([[0.5] * 12], index=["study"])
    item = KneeStudyDataset(["study"], index, targets=y, train=False)[0]
    assert torch.equal(item["weight"], torch.ones(12))
