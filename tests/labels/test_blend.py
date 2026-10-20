import pandas as pd
import pytest

from rsna_knee.labels.blend import distill, load_oof


def test_distill_mixes_labels_and_oof():
    y = pd.DataFrame({"a": [1.0, 0.0]}, index=["s1", "s2"])
    oof = pd.DataFrame({"a": [0.0, 1.0]}, index=["s2", "s1"])
    out = distill(y, oof, alpha=0.75)
    assert out.loc["s1", "a"] == 1.0 and out.loc["s2", "a"] == 0.0
    assert distill(y, oof).loc["s1", "a"] == 1.0


def test_distill_requires_every_study():
    y = pd.DataFrame({"a": [1.0, 0.0]}, index=["s1", "s2"])
    with pytest.raises(ValueError):
        distill(y, pd.DataFrame({"a": [0.5]}, index=["s1"]))


def test_load_oof_concatenates_folds(tmp_path):
    for k, sid in enumerate(["s1", "s2"]):
        (tmp_path / f"fold{k}").mkdir()
        pd.DataFrame({"a": [0.1]}, index=pd.Index([sid], name="StudyInstanceUID")).to_csv(
            tmp_path / f"fold{k}" / "oof.csv"
        )
    assert sorted(load_oof(tmp_path, 2).index) == ["s1", "s2"]
