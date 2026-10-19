# models

2.5D 窓の集合から、Study 単位の 12 ラベルを予測するモデル。

## 構成

```
images (B, K, 3, H, W) uint8
  │  mask で実在する窓だけを取り出す
  ▼
WindowEncoder    : GPU 上でリサイズ・正規化 → DINOv2-S/14 → [CLS, パッチ平均] → 射影  → (N, 384)
  │  (B, K, 384) に戻す(パディング位置は 0)
  ▼
StudyAggregator  : + 断面 / 脂肪抑制 / 相対位置の埋め込み
                   → Transformer encoder(全窓の self-attention、パディングはマスク)
                   → 12 個のラベルクエリから cross-attention → ラベルごとの線形出力
  ▼
logits (B, 12)
```

## ファイル

| ファイル | 名前 | 役割 |
|---|---|---|
| `encoder.py` | `WindowEncoder` | 窓 → 特徴。timm のバックボーン、入力サイズ、オフライン用の重みファイルを指定できる |
| `aggregator.py` | `StudyAggregator` | 窓特徴の集合 → 12 ラベルのロジット |
| | `PositionEmbedding` | シリーズ内の相対位置(0〜1)の正弦波埋め込み |
| `model.py` | `KneeModel` | 上の 2 つをつなぐ。`collate` のバッチをそのまま受け取る |
| | `KneeModel.param_groups` | エンコーダと集約部分で学習率を分けた optimizer 用のグループ |

## 設計上の判断

- **バックボーン**: DINOv2-S/14(timm `vit_small_patch14_dinov2.lvd142m`、Apache 2.0)。重みは timm 経由で Hugging Face Hub から取得する。全層をファインチューンし、学習率は集約部分より小さくする
- **入力サイズ**: パッチ 14 の倍数にする(既定 224)。キャッシュは 320px のままで、GPU 上でリサイズする
- **窓特徴**: CLS とパッチ平均の連結。CLS だけだと局所的な所見(ACL、半月板)を落とす恐れがあるため
- **パディング**: パディング位置の窓はエンコーダに通さず、attention でもマスクする。パディングの中身が結果に影響しないことをテストで確認している
- **出力**: ラベルごとに別のクエリと出力重みを持つ。所見ごとに注目する断面・スライスを別々に学習させるため

## Kaggle でのオフライン読み込み

提出時はインターネットが使えないため、事前学習の重みはファイルから読む。

```python
KneeModel({"pretrained": True, "pretrained_file": "/kaggle/input/<dataset>/model.safetensors"})
```

学習済みのチェックポイントにはエンコーダの重みも含まれるので、それを読み込む場合は `pretrained=False` でよい。

## 計測(RTX 5070 Ti、bf16、224px、学習時 32 窓 / Study)

| バッチ(Study 数) | 勾配チェックポイント | 速度 | ピークメモリ |
|---|---|---|---|
| 4 | なし | 29 Study/s | 6.4 GiB |
| 8 | なし | 30 Study/s | 12.3 GiB |
| 8 | あり | 24 Study/s | 2.8 GiB |
| 16 | あり | 25 Study/s | 5.2 GiB |

推論(fp16): 8 Study / 752 窓で 0.23 秒。
