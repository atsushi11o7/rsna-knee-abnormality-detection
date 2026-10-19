"""2.5D 窓(3 スライス)を特徴ベクトルに変換する 2D エンコーダ。"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class WindowEncoder(nn.Module):
    """timm の ViT で窓ごとの特徴を出す。

    uint8 の窓を受け取り、GPU 上でリサイズと正規化をしてからエンコーダに通す。
    出力は CLS とパッチ平均の連結を `out_dim` に射影したもの。

    Args:
        backbone: timm のモデル名。
        img_size: エンコーダへの入力サイズ。DINOv2 はパッチ 14 なので 14 の倍数にする。
        out_dim: 出力の次元。
        pretrained: 事前学習済みの重みを使うか。
        pretrained_file: 重みファイルのパス。オフライン環境(Kaggle 提出時)で使う。
        grad_checkpointing: メモリ節約のため、ブロックごとに勾配チェックポイントを使う。
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch14_dinov2.lvd142m",
        img_size: int = 224,
        out_dim: int = 384,
        pretrained: bool = True,
        pretrained_file: str | None = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        overlay = {"file": pretrained_file} if pretrained_file else None
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            pretrained_cfg_overlay=overlay,
            num_classes=0,
            img_size=img_size,
        )
        self.backbone.set_grad_checkpointing(grad_checkpointing)
        self.img_size = img_size
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        dim = self.backbone.num_features
        self.proj = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, out_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """窓をエンコードする。

        Args:
            images: (N, 3, H, W) uint8。

        Returns:
            (N, out_dim) の特徴。
        """
        x = images.float() / 255.0
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        tokens = self.backbone.forward_features(x)
        n_prefix = self.backbone.num_prefix_tokens
        # ACL や半月板などの局所所見は CLS に残りにくいので、パッチ平均も併用する
        feat = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(1)], dim=-1)
        return self.proj(feat)
