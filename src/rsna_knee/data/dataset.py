"""Study 単位のデータセット。キャッシュ済みシリーズから、属性付きの 2.5D 窓の集合を作る。

窓は 3 スライスで、中心スライスと、その前後で指定の実寸オフセットに最も近いスライスからなる
(単純な隣接スライスではない)。スライス間隔がオフセットより広いシリーズでは、最も近いのが
中心自身になるため、隣接スライスを使う。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PLANES = ["Sagittal", "Coronal", "Axial"]


@dataclass
class SeriesRef:
    """キャッシュ済みシリーズへの参照。

    Attributes:
        path: 画像の npy(uint8、形状 (S, H, W))。
        positions: 各スライスの法線方向の位置(mm)。長さ S。
        plane: PLANES の添字。
        fat_sup: 脂肪抑制なら 1。
    """

    path: Path
    positions: np.ndarray
    plane: int
    fat_sup: int


def build_index(cache_dir: Path, series_csv: Path, split: str) -> dict[str, list[SeriesRef]]:
    """キャッシュから、Study ごとの使用可能なシリーズ一覧を作る。

    Args:
        cache_dir: build_cache の出力先。
        series_csv: コンペの `{split}_series.csv`(断面と脂肪抑制の列を使う)。
        split: "train" または "test"。

    Returns:
        StudyInstanceUID → SeriesRef のリスト。失敗・空のシリーズは含まない。
    """
    meta = pd.read_parquet(cache_dir / f"{split}_series_meta.parquet")
    if "n_slices" not in meta:  # 全シリーズで前処理に失敗すると、この列自体が作られない
        return {}
    meta = meta[meta.ok & (meta.n_slices > 0)]
    info = pd.read_csv(series_csv).set_index("SeriesInstanceUID")
    index: dict[str, list[SeriesRef]] = {}
    for r in meta.itertuples():
        row = info.loc[r.SeriesInstanceUID]
        d = cache_dir / r.StudyInstanceUID
        index.setdefault(r.StudyInstanceUID, []).append(
            SeriesRef(
                path=d / f"{r.SeriesInstanceUID}.npy",
                positions=np.load(d / f"{r.SeriesInstanceUID}_pos.npy"),
                plane=PLANES.index(row.Anatomical_Plane),
                fat_sup=int(row.Fat_Suppression),
            )
        )
    return index


def window_indices(positions: np.ndarray, center: int, offset_mm: float) -> list[int]:
    """窓を構成する 3 スライスの添字を返す。

    Args:
        positions: シリーズの各スライス位置(mm、昇順)。
        center: 中心スライスの添字。
        offset_mm: 前後のスライスを探す実寸オフセット。

    Returns:
        [前, 中心, 後]。端では中心と同じ添字になることがある。
    """
    c = positions[center]
    lo = int(np.abs(positions - (c - offset_mm)).argmin())
    hi = int(np.abs(positions - (c + offset_mm)).argmin())
    # 間隔の広いシリーズでは最も近いのが中心自身になるので、1 枚外側を使う
    if lo == center and center > 0:
        lo = center - 1
    if hi == center and center < len(positions) - 1:
        hi = center + 1
    return [lo, center, hi]


def allocate(n_series: int, total: int, rng: np.random.Generator) -> np.ndarray:
    """`total` 個の窓をシリーズにできるだけ均等に割り振る。"""
    base = np.full(n_series, total // n_series)
    base[rng.choice(n_series, total % n_series, replace=False)] += 1
    return base


def pick_centers(n: int, k: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """窓の中心スライスを選ぶ。学習時はランダム、評価時は等間隔(最大 n 個)。"""
    if train:
        return np.sort(rng.choice(n, k, replace=n < k))
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))


class KneeStudyDataset(Dataset):
    """1 Study を、属性付きの 2.5D 窓の集合として返すデータセット。

    Args:
        study_ids: 対象の StudyInstanceUID。
        index: build_index の結果。
        targets: 学習ラベル(index は StudyInstanceUID)。推論時は None。
        weights: 損失の重み。省略時は全要素 1。
        train: True なら窓をランダムに選ぶ。
        n_windows: 学習時の 1 Study あたりの窓数。
        windows_per_series: 評価時の 1 シリーズあたりの最大窓数。
        offset_mm: 窓の前後スライスを探す実寸オフセット。
        drop_missing: True なら、キャッシュに無い Study を警告付きで除外し、`missing` に残す
            (学習用)。既定の False ではエラーにする(推論で行が黙って欠けるのを防ぐ)。
    """

    def __init__(
        self,
        study_ids: list[str],
        index: dict[str, list[SeriesRef]],
        targets: pd.DataFrame | None = None,
        weights: pd.DataFrame | None = None,
        train: bool = True,
        n_windows: int = 32,
        windows_per_series: int = 16,
        offset_mm: float = 3.5,
        drop_missing: bool = False,
    ):
        self.missing = [s for s in study_ids if s not in index]
        if self.missing:
            if not drop_missing:
                raise KeyError(f"{len(self.missing)} studies have no cached series")
            warnings.warn(
                f"dropping {len(self.missing)} studies with no cached series", stacklevel=2
            )
        self.study_ids = [s for s in study_ids if s in index]
        self.index = index
        self.targets = targets
        if targets is not None and weights is None:
            weights = pd.DataFrame(1.0, index=targets.index, columns=targets.columns)
        self.weights = weights
        self.train = train
        self.n_windows = n_windows
        self.windows_per_series = windows_per_series
        self.offset_mm = offset_mm

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, i: int) -> dict:
        """1 Study 分の窓を返す。

        Returns:
            images (K, 3, H, W) uint8、plane / fat_sup / series_id (K,) int、
            rel_pos (K,) float(シリーズ内の相対位置 0〜1)、study_id。
            ラベルがあれば target / weight (12,) float も含む。
        """
        sid = self.study_ids[i]
        series = self.index[sid]
        # DataLoader が worker・エポックごとに設定する torch の乱数から作る。self に numpy の
        # Generator を持たせると全 worker にコピーされ、同じ窓を選んでしまう
        rng = np.random.default_rng(int(torch.randint(2**62, (1,))))
        if self.train:
            counts = allocate(len(series), self.n_windows, rng)
        else:
            counts = np.full(len(series), self.windows_per_series)

        imgs, plane, fat_sup, rel_pos, series_id = [], [], [], [], []
        for s_idx, (ref, k) in enumerate(zip(series, counts, strict=True)):
            if k == 0:
                continue
            vol = np.load(ref.path, mmap_mode="r")
            pos = ref.positions
            span = max(float(pos[-1] - pos[0]), 1e-6)
            for c in pick_centers(len(pos), int(k), self.train, rng):
                imgs.append(np.asarray(vol[window_indices(pos, int(c), self.offset_mm)]))
                plane.append(ref.plane)
                fat_sup.append(ref.fat_sup)
                rel_pos.append((pos[c] - pos[0]) / span)
                series_id.append(s_idx)

        item = {
            "study_id": sid,
            "images": torch.from_numpy(np.stack(imgs)),
            "plane": torch.tensor(plane),
            "fat_sup": torch.tensor(fat_sup),
            "rel_pos": torch.tensor(rel_pos, dtype=torch.float32),
            "series_id": torch.tensor(series_id),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets.loc[sid].values, dtype=torch.float32)
            item["weight"] = torch.tensor(self.weights.loc[sid].values, dtype=torch.float32)
        return item


def collate(batch: list[dict]) -> dict:
    """窓数の違う Study をバッチにまとめる。

    最長の Study に合わせて 0 でパディングし、実在する窓を True とする `mask` (B, K) を付ける。
    """
    k = max(b["images"].shape[0] for b in batch)
    out: dict = {"study_id": [b["study_id"] for b in batch]}
    mask = torch.zeros(len(batch), k, dtype=torch.bool)
    for key in ["images", "plane", "fat_sup", "rel_pos", "series_id"]:
        first = batch[0][key]
        padded = first.new_zeros((len(batch), k, *first.shape[1:]))
        for i, b in enumerate(batch):
            n = b[key].shape[0]
            padded[i, :n] = b[key]
            mask[i, :n] = True
        out[key] = padded
    out["mask"] = mask
    for key in ["target", "weight"]:
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    return out
