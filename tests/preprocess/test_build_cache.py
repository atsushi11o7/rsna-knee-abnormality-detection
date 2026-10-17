import json

import pytest

from rsna_knee.preprocess.build_cache import check_config, index_studies


def test_index_studies_groups_by_study_and_series():
    names = ["test_series/st1/se1/a.dcm", "test_series/st1/se1/b.dcm", "test_series/st1/se2/c.dcm"]
    idx = index_studies(names)
    assert {k: len(v) for k, v in idx["st1"].items()} == {"se1": 2, "se2": 1}


def test_check_config_rejects_mismatch(tmp_path):
    check_config(tmp_path, {"size": 320})
    check_config(tmp_path, {"size": 320})
    with pytest.raises(SystemExit):
        check_config(tmp_path, {"size": 224})
    assert json.loads((tmp_path / "config.json").read_text()) == {"size": 320}


def test_failed_study_is_retried_only_when_asked(tmp_path, monkeypatch):
    from rsna_knee.preprocess import build_cache

    calls = []

    def fake_load(_open, names, *a):
        calls.append(names)
        raise ValueError("broken")

    monkeypatch.setattr(build_cache, "load_series", fake_load)
    cfg = {"crop_mm": 150, "size": 32, "max_slices": 0}
    series = {"se1": ["x.dcm"]}
    rows = build_cache.process_study("st1", series, str(tmp_path), cfg)
    assert rows[0]["ok"] is False and not (tmp_path / "st1" / "meta.json.tmp").exists()
    build_cache.process_study("st1", series, str(tmp_path), cfg)
    assert len(calls) == 1
    build_cache.process_study("st1", series, str(tmp_path), cfg, retry_failed=True)
    assert len(calls) == 2
