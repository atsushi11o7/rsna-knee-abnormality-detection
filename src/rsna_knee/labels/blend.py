"""疑似ラベルの作り直し(自己蒸留)。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_oof(run_dir: Path, n_folds: int) -> pd.DataFrame:
    """`<run_dir>/fold{k}/oof.csv` を全 fold 分つなげる。

    Args:
        run_dir: 実験の出力先(例: outputs/<exp_name>)。
        n_folds: fold 数。全 fold の OOF がそろっている必要がある。

    Returns:
        index は StudyInstanceUID、列は 12 ラベルの OOF 確率。
    """
    parts = [pd.read_csv(run_dir / f"fold{k}" / "oof.csv", index_col=0) for k in range(n_folds)]
    oof = pd.concat(parts)
    if oof.index.duplicated().any():
        raise ValueError(f"{run_dir}: a study appears in more than one fold")
    return oof


def distill(y: pd.DataFrame, oof: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """元のラベルと OOF 予測を混ぜた新しいラベルを返す。

    画像モデルの OOF 予測は、レポート由来のラベルより放射線科医の正解に近くなることがあるので、
    両者を混ぜてラベルのノイズを減らす。予測で完全に置き換えると悪化するという報告があり
    (#735304)、既定は半々。

    Args:
        y: 元のラベル(index は StudyInstanceUID)。
        oof: OOF 予測。`y` の全 Study を含む必要がある。
        alpha: 元のラベルの比率。
    """
    missing = y.index.difference(oof.index)
    if len(missing):
        raise ValueError(f"{len(missing)} studies have no OOF prediction")
    return alpha * y + (1 - alpha) * oof.loc[y.index, y.columns]
