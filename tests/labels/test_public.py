import pandas as pd

from rsna_knee.labels.public import (
    SOURCES,
    TARGETS,
    apply_silence_policy,
    load_source,
    loss_weights,
)


def _write(tmp_path, name, values):
    rel, _ = SOURCES[name]
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([values], columns=TARGETS)
    df.insert(0, "StudyInstanceUID", "s0")
    df.to_csv(p, index=False)


def test_marker_becomes_unaddressed(tmp_path):
    _write(tmp_path, "steven_v4", [0.25] * 6 + [0.9] * 6)
    ls = load_source("steven_v4", tmp_path)
    assert ls.addressed.loc["s0"].tolist() == [False] * 6 + [True] * 6


def test_silence_policy_and_weights(tmp_path):
    _write(tmp_path, "steven_v4", [0.25] * 12)
    ls = apply_silence_policy(load_source("steven_v4", tmp_path))
    assert ls.y.loc["s0", "Baker's"] == 0.03
    assert ls.y.loc["s0", "ACL"] == 0.25
    w = loss_weights(ls, silent_weight=0.5)
    assert w.loc["s0", "Baker's"] == 1.0 and w.loc["s0", "ACL"] == 0.5
