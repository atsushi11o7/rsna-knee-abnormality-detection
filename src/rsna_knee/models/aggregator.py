"""窓特徴の集合を Study 単位の 12 ラベルにまとめる Transformer。"""

from __future__ import annotations

import math

import torch
from torch import nn

N_PLANES = 3


class PositionEmbedding(nn.Module):
    """シリーズ内の相対位置(0〜1)を、正弦波特徴と線形層で埋め込む。"""

    def __init__(self, dim: int, n_freqs: int = 8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freqs)) * math.pi, persistent=False)
        self.proj = nn.Linear(2 * n_freqs, dim)

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        x = rel_pos.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([x.sin(), x.cos()], dim=-1))


class StudyAggregator(nn.Module):
    """窓特徴に属性埋め込みを足し、Transformer で集約して 12 ラベルを出す。

    集約は「全窓に self-attention → ラベルごとのクエリから cross-attention」の 2 段。
    所見ごとに、注目するシリーズやスライスを別々に学習させる。

    Args:
        dim: 特徴の次元。
        n_labels: 出力ラベル数。
        n_layers: self-attention の層数。
        n_heads: attention のヘッド数。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        dim: int = 384,
        n_labels: int = 12,
        n_layers: int = 2,
        n_heads: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.plane_emb = nn.Embedding(N_PLANES, dim)
        self.fat_emb = nn.Embedding(2, dim)
        self.pos_emb = PositionEmbedding(dim)
        layer = nn.TransformerEncoderLayer(
            dim, n_heads, 4 * dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, n_heads, dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        # ラベルごとに別の出力重みを持つ(クエリ i の出力 → ラベル i のロジット)
        self.head_w = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_labels))
        # nn.Embedding の既定初期化(標準偏差 1)だと、画像特徴(約 0.6)より大きくなり、
        # 全 Study のトークンが断面などの属性で決まって出力が一定値に潰れた
        for emb in (self.plane_emb, self.fat_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(
        self,
        feats: torch.Tensor,
        plane: torch.Tensor,
        fat_sup: torch.Tensor,
        rel_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """集約してロジットを返す。

        Args:
            feats: (B, K, dim) 窓特徴。
            plane: (B, K) 断面の添字。
            fat_sup: (B, K) 脂肪抑制なら 1。
            rel_pos: (B, K) シリーズ内の相対位置。
            mask: (B, K) 実在する窓が True。

        Returns:
            (B, n_labels) のロジット。
        """
        x = feats + self.plane_emb(plane) + self.fat_emb(fat_sup) + self.pos_emb(rel_pos)
        pad = ~mask
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        out, _ = self.cross(q, x, x, key_padding_mask=pad, need_weights=False)
        out = self.query_norm(out + q)
        return (out * self.head_w).sum(-1) + self.head_b


class MILAggregator(nn.Module):
    """所見ごとの attention プーリング(MIL)で窓特徴をまとめる。

    StudyAggregator から Transformer とラベルクエリを除き、所見ごとに「どの窓を重視するか」
    の重みだけを学習する。入力(窓特徴 + 属性埋め込み)は StudyAggregator と同じにして、
    集約方法だけを比較できるようにしている。

    Args:
        dim: 特徴の次元。
        n_labels: 出力ラベル数。
        hidden: attention スコアを出す中間層の次元。
        dropout: ドロップアウト率。
    """

    def __init__(self, dim: int = 384, n_labels: int = 12, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.plane_emb = nn.Embedding(N_PLANES, dim)
        self.fat_emb = nn.Embedding(2, dim)
        self.pos_emb = PositionEmbedding(dim)
        for emb in (self.plane_emb, self.fat_emb):
            nn.init.normal_(emb.weight, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, n_labels)
        )
        self.head_w = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_labels))

    def forward(
        self,
        feats: torch.Tensor,
        plane: torch.Tensor,
        fat_sup: torch.Tensor,
        rel_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """集約してロジットを返す。引数は StudyAggregator.forward と同じ。"""
        x = feats + self.plane_emb(plane) + self.fat_emb(fat_sup) + self.pos_emb(rel_pos)
        h = self.norm(x)
        a = self.att(h).masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(dim=1)
        pooled = torch.einsum("bkn,bkd->bnd", a, h)
        return (pooled * self.head_w).sum(-1) + self.head_b
