"""学習と推論で共通の小さな処理。"""

from __future__ import annotations

import torch


def to_device(batch: dict, device: torch.device) -> dict:
    """バッチ内のテンソルを `device` に移す(テンソル以外はそのまま)。"""
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }
