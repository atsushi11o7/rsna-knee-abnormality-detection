"""評価指標。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """コンペの評価指標(12 ラベルの ROC AUC の平均)を計算する。

    Args:
        y_true: 0/1 の正解。0.5 より大きい値を陽性とみなすので、確率ラベルも渡せる。
        y_pred: 予測スコア。`y_true` と同じ index・列。

    Returns:
        (macro AUC, ラベルごとの AUC)。陽性か陰性しかないラベルは除いて平均する。
    """
    y_pred = y_pred.loc[y_true.index, y_true.columns]
    per_label = {}
    for col in y_true.columns:
        t = (y_true[col].to_numpy() > 0.5).astype(int)
        if 0 < t.sum() < len(t):
            per_label[col] = float(roc_auc_score(t, y_pred[col].to_numpy()))
    return float(np.mean(list(per_label.values()))), per_label
