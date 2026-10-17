"""公開されているレポート由来ラベル(すべて CC0)と gold-58 を、共通の形式で読み込む。

各ソースは「レポートに記載なし」を独自の値で表すので、その値から記載有無のマスクを復元する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

LABEL_ROOT = Path("data/labels")

# LABEL_ROOT からの相対パスと、そのソースが「記載なし」に書く値。
# steven_v4 はブレンド済みのため、0.25 による記載なし判定は近似。v4 の 0.25 は 10,093 セルで、
# うち 10,085 が v2 の記載なし(0.5)と一致し、8 は不一致、逆に v2 が記載なしで v4 が 0.25 でない
# セルが 82 ある(2026-10-17 確認)。nartaa/rsna-knee-hpo-assets の説明も 0.25 を記載なしとしている。
SOURCES: dict[str, tuple[str, float | None]] = {
    "steven_v4": ("stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", 0.25),
    "steven_v2": ("stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv", 0.5),
    "jev": ("riadmohamed42_jev-knee-labels/HYBRID_labels_scores.csv", None),
    "consensus": (
        "karttikjangid05_rsna-knee-public-llm-label-sets-compared/consensus_labels.csv",
        None,
    ),
}

# レポートが記載なしのときの gold-58 での陽性率(#733932)。これらの所見では記載なしが
# 実質的に陰性を意味するため、中立値のままにせず、この値に置き換える。
SILENT_MEANS_ABSENT = {"Baker's": 0.03, "Medial OA": 0.0}


@dataclass
class LabelSet:
    """1 つのラベルソース。

    Attributes:
        y: 確率ラベル。index は StudyInstanceUID、列は TARGETS。
        addressed: レポートに記載があれば True。`y` と同じ形状。
    """

    y: pd.DataFrame
    addressed: pd.DataFrame


def load_source(name: str, root: Path = LABEL_ROOT) -> LabelSet:
    """ラベルソースを読み込む。

    Args:
        name: SOURCES のキー。
        root: 公開ラベルの保存先。

    Returns:
        欠損値は記載なし扱いとし、そのソースの記載なしの値(なければ 0.5)で埋めたもの。
    """
    rel, marker = SOURCES[name]
    df = pd.read_csv(root / rel).set_index("StudyInstanceUID")[TARGETS].astype(float)
    if marker is None:
        addressed = df.notna()
    else:
        addressed = df.notna() & (df - marker).abs().gt(1e-9)
    return LabelSet(df.fillna(0.5 if marker is None else marker), addressed)


def apply_silence_policy(ls: LabelSet) -> LabelSet:
    """SILENT_MEANS_ABSENT の所見について、記載なしのセルを陰性寄りの値に置き換える。"""
    y = ls.y.copy()
    for col, prior in SILENT_MEANS_ABSENT.items():
        y.loc[~ls.addressed[col], col] = prior
    return LabelSet(y, ls.addressed)


def loss_weights(ls: LabelSet, silent_weight: float = 0.5) -> pd.DataFrame:
    """損失の重みを作る。

    Args:
        ls: ラベルソース。
        silent_weight: 記載なしのセルの重み。

    Returns:
        記載ありは 1、記載なしは `silent_weight`。SILENT_MEANS_ABSENT の所見は
        記載なし自体が情報を持つので常に 1。
    """
    w = ls.addressed.astype(float)
    w[~ls.addressed] = silent_weight
    for col in SILENT_MEANS_ABSENT:
        w[col] = 1.0
    return w


def load_gold(train_csv: Path = Path("data/csv/train.csv")) -> pd.DataFrame:
    """放射線科医の正解ラベル(58 件)を読み込む。値は 0/1。"""
    df = pd.read_csv(train_csv).set_index("StudyInstanceUID")
    return df.loc[df[TARGETS[0]].notna(), TARGETS].astype(int)


def load_train_labels(
    name: str = "steven_v4",
    silent_weight: float = 0.5,
    root: Path = LABEL_ROOT,
    train_csv: Path = Path("data/csv/train.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習用のラベルと損失の重みを返す。gold-58 は除外する。

    記載なしの値を持たないソース(jev、consensus)は全セルが記載あり扱いになるため、
    記載なしの処理と `silent_weight` は効かない。

    Args:
        name: SOURCES のキー。
        silent_weight: 記載なしのセルの重み。
        root: 公開ラベルの保存先。
        train_csv: コンペの train.csv。

    Returns:
        (ラベル, 重み)。どちらも index は StudyInstanceUID、列は TARGETS。
    """
    ls = apply_silence_policy(load_source(name, root))
    gold = load_gold(train_csv).index
    keep = ~ls.y.index.isin(gold)
    return ls.y[keep], loss_weights(ls, silent_weight)[keep]
