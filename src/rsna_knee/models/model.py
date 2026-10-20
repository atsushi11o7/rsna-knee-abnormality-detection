"""窓エンコーダと集約 Transformer をつないだ Study 単位のモデル。"""

from __future__ import annotations

import torch
from torch import nn

from rsna_knee.models.aggregator import MILAggregator, StudyAggregator
from rsna_knee.models.encoder import WindowEncoder


class KneeModel(nn.Module):
    """`collate` のバッチを受け取り、12 ラベルのロジットを返す。

    Args:
        encoder: 窓エンコーダの設定(WindowEncoder の引数)。
        aggregator: 集約部分の設定。`type` が "transformer"(既定、StudyAggregator)か
            "mil"(MILAggregator)を選び、残りはそのクラスの引数。`dim` は自動で合わせる。
    """

    def __init__(self, encoder: dict | None = None, aggregator: dict | None = None):
        super().__init__()
        self.encoder = WindowEncoder(**(encoder or {}))
        dim = self.encoder.proj[-1].out_features
        agg = dict(aggregator or {})
        cls = {"transformer": StudyAggregator, "mil": MILAggregator}[agg.pop("type", "transformer")]
        self.aggregator = cls(dim=dim, **agg)
        self.eval_chunk = 256  # 推論時に一度にエンコードする窓の数

    def forward(self, batch: dict) -> torch.Tensor:
        """推論する。

        Args:
            batch: `collate` の出力。images (B, K, 3, H, W) uint8、plane / fat_sup / rel_pos /
                mask (B, K) を使う。

        Returns:
            (B, 12) のロジット。
        """
        images, mask = batch["images"], batch["mask"]
        b, k = mask.shape
        # パディングの窓はエンコーダに通さない(評価時は Study ごとの窓数の差が大きい)
        windows = images[mask]
        if self.training:
            valid = self.encoder(windows)
        else:
            # 推論では Study によって窓数が大きく違うので、分割してエンコードしメモリを抑える
            valid = torch.cat([self.encoder(c) for c in windows.split(self.eval_chunk)])
        feats = valid.new_zeros(b, k, valid.shape[-1])
        feats[mask] = valid
        return self.aggregator(feats, batch["plane"], batch["fat_sup"], batch["rel_pos"], mask)

    def param_groups(self, lr: float, encoder_lr: float, weight_decay: float) -> list[dict]:
        """エンコーダと集約部分で学習率を分けたパラメータグループを返す。

        バイアスと LayerNorm、埋め込み、ラベルクエリには weight decay をかけない。
        """
        groups = []
        for module, group_lr in [(self.encoder.backbone, encoder_lr), (self, lr)]:
            decay, no_decay = [], []
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if module is self and name.startswith("encoder.backbone."):
                    continue
                if p.ndim < 2 or "emb" in name or "queries" in name or "token" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups.append({"params": decay, "lr": group_lr, "weight_decay": weight_decay})
            groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})
        return groups
