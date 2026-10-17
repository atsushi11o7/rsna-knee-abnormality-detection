"""gold-58 を除いた学習 Study の CV 分割。"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(
    train_csv_df: pd.DataFrame, targets: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> pd.Series:
    """Study ごとの fold 番号を作る。

    レポート本文が同じ Study は同じ fold に入れる。確信度の高い陽性数(上限 5)で層化し、
    正常例と多所見例の比率を fold 間でそろえる。

    Args:
        train_csv_df: コンペの train.csv(`StudyInstanceUID` と `Report` 列を使う)。
        targets: 学習ラベル。index の Study が分割対象。
        n_splits: fold 数。
        seed: 乱数シード。

    Returns:
        index は StudyInstanceUID、値は 0〜n_splits-1。
    """
    reports = train_csv_df.set_index("StudyInstanceUID").loc[targets.index, "Report"]
    groups = reports.factorize()[0]
    strata = (targets > 0.5).sum(axis=1).clip(upper=5)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=targets.index, name="fold")
    for k, (_, val) in enumerate(skf.split(targets, strata, groups)):
        fold.iloc[val] = k
    return fold
