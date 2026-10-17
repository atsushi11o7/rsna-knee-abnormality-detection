# preprocess

DICOM を学習・推論で使えるキャッシュ(npy)に変換する。重い処理なので、データごとに一度だけ実行する。

## 責務

- DICOM の読み込み(zip メンバー / 展開済みディレクトリの両方)
- スライスを解剖学的な位置順に並べる
- 実寸(mm)基準でクロップし、解像度をそろえる
- シリーズ単位で輝度を正規化し、uint8 で保存する

学習時は zip から直接読む(展開すると 570GB で zip と同居できないため)。提出時は、展開済みの隠しテストに同じ処理をかける。同じ関数を通るので、学習と推論で前処理がずれない。

## ファイル

### `dicom.py`: 1 シリーズの前処理

| 名前 | 役割 |
|---|---|
| `load_series(open_file, names, crop_mm, size, max_slices)` | 1 シリーズを読み込み、並べ替え・クロップ・正規化して `SeriesVolume` を返す |
| `slice_normal(orientation)` | ImageOrientationPatient からスライス法線を求める |
| `sort_slices(headers)` | 法線方向の位置順に並べる(同位置なら InstanceNumber 順) |
| `crop_resize(img, pixel_spacing, crop_mm, size)` | 画像中心から実寸 `crop_mm` 四方を切り出してリサイズ。撮影範囲が足りなければ 0 埋め |
| `to_uint8(stack)` | シリーズ全体の 0.5〜99.5 パーセンタイルで 0〜255 に正規化 |
| `pick_indices(n, max_slices)` | スライス数の上限処理(0 なら全スライス) |
| `Opener` | 相対パスから DICOM を開く関数の型 |

### `build_cache.py`: キャッシュ作成の CLI

| 名前 | 役割 |
|---|---|
| `main()` | CLI。Study 単位でプロセス並列に処理し、最後にメタ情報を parquet にまとめる |
| `list_dicoms(src, split)` / `index_studies(names)` | DICOM の一覧を Study → Series → ファイルにまとめる |
| `make_opener(src)` | `.zip` なら zip メンバー、それ以外はディレクトリから開く関数を返す |
| `process_study(...)` | 1 Study を処理して保存する。`meta.json` があればスキップ(再開用) |
| `check_config(out, cfg)` | 既存キャッシュと設定が違えば止める |

```bash
# 学習データ(zip から)
python -m rsna_knee.preprocess.build_cache --src data/rsna-knee-abnormality-detection.zip \
    --split train --out data/cache/v2 --workers 12
# 提出時(展開済みディレクトリから)
python -m rsna_knee.preprocess.build_cache --src /kaggle/input/rsna-knee-abnormality-detection \
    --split test --out /tmp/cache
```

主なオプションは、`--crop-mm`(既定 150)、`--size`(既定 320)、`--max-slices`(既定 0 = 全スライス)、`--limit`(先頭 N Study だけ処理)、`--retry-failed`(失敗シリーズを含む Study をやり直す)。

## 出力形式

```
<out>/
├── config.json                     # {"crop_mm", "size", "max_slices"}
├── {split}_series_meta.parquet     # 全シリーズのメタ情報(1 行 = 1 シリーズ)
└── <StudyInstanceUID>/
    ├── meta.json                   # この Study のシリーズ情報(parquet と同じ内容)
    ├── <SeriesInstanceUID>.npy     # uint8、形状 (S, size, size)。位置順
    └── <SeriesInstanceUID>_pos.npy # float32、形状 (S,)。各スライスの法線方向の位置(mm)
```

メタ情報の列:

| 列 | 内容 |
|---|---|
| `StudyInstanceUID`, `SeriesInstanceUID` | ID |
| `ok`, `error` | 成否と、失敗時の例外 |
| `n_slices_orig`, `n_slices`, `n_bad` | 元のスライス数、保存数、読めなかった数 |
| `pixel_spacing`, `rows`, `cols` | 元画像の解像度 |
| `slice_thickness`, `slice_gap` | スライス厚と、位置の差分の中央値(mm) |
| `series_description`, `manufacturer`, `field_strength` | 撮像法の名前、メーカー、磁場強度 |
| `patient_sex`, `laterality` | 性別、左右タグ(約半数が空) |
| `orientation`, `first_position` | 先頭スライスの ImageOrientationPatient / ImagePositionPatient |

断面と脂肪抑制の有無は含まない。コンペの `{split}_series.csv` と結合して使う(`data/dataset.py` の `build_index`)。

## 確認済みの前提(train、2026-10-17)

- 4,407 Study / 24,371 シリーズ / 819,078 スライスがすべて成功(9.4 分、79GB)
- 全ファイルが非圧縮・1 ファイル 1 枚・MONOCHROME2
- スライスの向き(ImageOrientationPatient)は、断面ごとに全シリーズで同一。向きの正規化はしていない
- RescaleSlope はシリーズ内で一定。シリーズ単位の正規化で吸収されるので適用していない
- zip から読んだ場合と展開済みディレクトリから読んだ場合で、出力は完全に一致する

## 未対応

- 左右の判定(タグは約半数が空。ミラー TTA や左右反転の拡張で必要になったら、位置情報から求める)
