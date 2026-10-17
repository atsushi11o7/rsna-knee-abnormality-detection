# rsna-knee-abnormality-detection

[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — Kaggleの膝MRI異常検出コンペ用リポジトリ。

## 概要

DICOM形式の膝MRI画像から、12種類の異常(ACL / MCL / 内側・外側半月板 / 内側・外側・膝蓋大腿 OA / 関節液貯留 / 滑膜炎 / Baker 嚢胞 / 骨挫傷 / 骨折)を Study 単位で予測する。

- 評価指標: 12 ラベルの macro ROC AUC
- 提出形式: Kaggle Notebook によるコード提出。Notebook が隠しテストを推論して `submission.csv` を書き出す
  - CPU / GPU とも実行 9 時間以内、インターネット無効
  - 学習済み重みやコードは Kaggle Dataset として Notebook に添付する
  - Kaggle の GPU(T4 / P100)は bf16 非対応なので、推論は fp16 で行う
- 締切: 2026-10-22 23:59 UTC

## セットアップ

```bash
uv sync
```

### データ

```bash
kaggle competitions download rsna-knee-abnormality-detection -p data
```

zip は 265GB、展開すると 570GB になる。展開せず、zip から直接読んで前処理キャッシュを作る。

```bash
python -m rsna_knee.preprocess.build_cache --split train --out data/cache/v2 --workers 12
```

## ディレクトリ構成

```
src/rsna_knee/
├── preprocess/          # DICOM → キャッシュ(一度だけ回す重い処理)
│   ├── dicom.py         # 読み込み、スライスのソート、mm 単位クロップ、正規化
│   └── build_cache.py   # zip から npy キャッシュを作る CLI
├── data/                # 学習時のデータ読み込み(予定)
│   ├── dataset.py       # キャッシュ → 2.5D の窓、シリーズのスロット割り当て
│   └── folds.py         # CV 分割
├── labels/              # 疑似ラベル(予定)
│   ├── public.py        # 公開ラベルの読み込みと「記載なし」の値の統一
│   └── blend.py         # ブレンド、OOF との混合
├── models/              # (予定)
│   ├── encoder.py       # 2D エンコーダ(DINOv2 / ConvNeXt)
│   └── aggregator.py    # Transformer 集約、ラベルクエリ
├── train.py             # 学習のエントリポイント(予定)
└── infer.py             # 推論。提出ノートブックからも呼ぶ(予定)
configs/                 # Hydra 設定(予定)
tests/                   # src と同じ構成で配置
data/                    # コンペ配布物、公開ラベル、キャッシュ(gitignore)
docs/                    # ローカル専用メモ(gitignore)
submission/              # 提出物(gitignore)
```
