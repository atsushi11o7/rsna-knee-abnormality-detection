# data

前処理済みキャッシュを、学習・推論で使うバッチに変換する。学習中に毎回呼ばれる軽い処理を置く。

## 責務

- キャッシュとコンペの CSV から、Study ごとのシリーズ一覧(断面・脂肪抑制付き)を作る
- 1 Study を「属性付きの 2.5D 窓の集合」として返す
- 窓数の違う Study をパディングしてバッチにする
- CV の fold を作る

## ファイル

### `dataset.py`

| 名前 | 役割 |
|---|---|
| `build_index(cache_dir, series_csv, split)` | Study → `SeriesRef` のリスト。失敗・空のシリーズは除く |
| `KneeStudyDataset` | 1 Study を窓の集合として返す Dataset |
| `collate(batch)` | パディングとマスクを付けてバッチにする |
| `window_indices(positions, center, offset_mm)` | 窓の 3 スライスの添字 |
| `allocate(n_series, total, rng)` | 学習時の窓数をシリーズに均等配分 |
| `pick_centers(n, k, train, rng)` | 窓の中心スライスを選ぶ |
| `PLANES` | 断面の並び(`plane` の値はこの添字) |

### `folds.py`

| 名前 | 役割 |
|---|---|
| `make_folds(train_csv_df, targets, n_splits, seed)` | Study ごとの fold 番号。同じレポート本文の Study は同じ fold、陽性数で層化 |

## 窓の作り方

窓 = 中心スライスと、その前後で `offset_mm`(既定 3.5mm)に最も近いスライスの 3 枚。これを 3 チャネル画像として 2D エンコーダに入れる。

- 単純な隣接スライスではなく実寸で選ぶので、0.4mm 間隔の薄いシリーズでも、窓の厚みが極端に薄くならない
- スライス間隔がオフセットより広いシリーズでは、隣接スライスを使う(この場合、窓の厚みは間隔に依存する)
- 端では、中心と同じスライスが重複することがある

| | 窓の数 | 中心の選び方 |
|---|---|---|
| 学習(`train=True`) | 1 Study で `n_windows`(既定 32)。シリーズに均等配分 | ランダム(エポックごとに変わる) |
| 評価・推論(`train=False`) | 1 シリーズで最大 `windows_per_series`(既定 16) | 等間隔(決定的) |

乱数は torch の乱数から作る。DataLoader が worker・エポックごとに設定するため、worker 間で同じ窓を選ぶことはない。

## データ形式

`KneeStudyDataset[i]`:

| キー | 形状・型 | 内容 |
|---|---|---|
| `study_id` | str | StudyInstanceUID |
| `images` | (K, 3, H, W) uint8 | 窓の画像。H, W はキャッシュの解像度(320)。モデルの入力サイズへのリサイズは GPU 側で行う |
| `plane` | (K,) int64 | 断面(`PLANES` の添字) |
| `fat_sup` | (K,) int64 | 脂肪抑制なら 1 |
| `rel_pos` | (K,) float32 | シリーズ内の相対位置(0〜1) |
| `series_id` | (K,) int64 | Study 内のシリーズ番号 |
| `target` | (12,) float32 | ラベル(`targets` 指定時のみ) |
| `weight` | (12,) float32 | 損失の重み(`targets` 指定時のみ。`weights` 省略時は 1) |

`collate` の出力は、上記に先頭のバッチ次元 B を足し、K を最長の Study に合わせて 0 でパディングしたもの。`mask` (B, K) bool は、実在する窓が True。

## キャッシュに無い Study の扱い

| `drop_missing` | 動作 | 用途 |
|---|---|---|
| False(既定) | KeyError | 推論。呼び出し側で事前予測などのフォールバックを明示的に行う |
| True | 警告を出して除外し、`dataset.missing` に残す | 学習(明示的に指定する) |
