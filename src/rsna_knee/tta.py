"""推論時のデータ拡張(TTA)。"""

from __future__ import annotations

import torch

from rsna_knee.data.dataset import PLANES

SAGITTAL = PLANES.index("Sagittal")


def mirror_batch(batch: dict) -> dict:
    """膝を左右反転したバッチを返す(解剖学的ミラー)。

    矢状断は左右方向がスライスの並びなので、窓内のスライス順(チャネル)とシリーズ内の
    相対位置を反転する。冠状断・横断は画像の横方向が左右なので、画像を左右反転する。
    ラベルの内側・外側は解剖学的な呼び方なので、反転しても入れ替えない。
    """
    images = batch["images"]
    sag = (batch["plane"] == SAGITTAL)[..., None, None, None]
    mirrored = torch.where(sag, images.flip(2), images.flip(-1))
    rel_pos = torch.where(batch["plane"] == SAGITTAL, 1 - batch["rel_pos"], batch["rel_pos"])
    return batch | {"images": mirrored, "rel_pos": rel_pos}
