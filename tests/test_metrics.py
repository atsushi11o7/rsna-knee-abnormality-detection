import pandas as pd

from rsna_knee.metrics import macro_auc


def test_macro_auc_skips_single_class_columns():
    y = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 0, 0]})
    p = pd.DataFrame({"a": [0.1, 0.9, 0.2, 0.8], "b": [0.5] * 4})
    score, per = macro_auc(y, p)
    assert score == 1.0 and list(per) == ["a"]


def test_macro_auc_binarizes_soft_labels():
    y = pd.DataFrame({"a": [0.25, 0.9, 0.03, 0.7]})
    p = pd.DataFrame({"a": [0.1, 0.9, 0.2, 0.8]})
    assert macro_auc(y, p)[0] == 1.0
