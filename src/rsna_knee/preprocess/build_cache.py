"""コンペの zip または展開済みディレクトリから、シリーズ単位の uint8 キャッシュを作る。

学習時は zip から直接読む(展開すると 570GB で zip と同居できないため)。Kaggle の提出時は
隠しテストが展開済みディレクトリで渡されるので、同じ処理をディレクトリに対して行う。

Usage:
    python -m rsna_knee.preprocess.build_cache --src data/rsna-knee-abnormality-detection.zip \
        --out data/cache/v2 --split train --workers 12
    python -m rsna_knee.preprocess.build_cache \
        --src /kaggle/input/rsna-knee-abnormality-detection --out /tmp/cache --split test
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.preprocess.dicom import Opener, load_series

_open: Opener | None = None


def make_opener(src: str) -> Opener:
    """`src` が .zip なら zip メンバーを、それ以外はディレクトリ内のファイルを開く関数を返す。"""
    if src.endswith(".zip"):
        return zipfile.ZipFile(src).open
    return partial(_open_file, Path(src))


def _open_file(root: Path, name: str):
    return open(root / name, "rb")


def _init_worker(src: str) -> None:
    global _open
    _open = make_opener(src)


def list_dicoms(src: str, split: str) -> list[str]:
    """DICOM の相対パス一覧を返す。

    Args:
        src: zip のパス、または展開済みデータセットのルート。
        split: "train" または "test"。

    Returns:
        `{split}_series/<study>/<series>/<sop>.dcm` 形式の相対パス。
    """
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
        return [n for n in names if n.startswith(f"{split}_series/") and n.endswith(".dcm")]
    root = Path(src)
    return [p.relative_to(root).as_posix() for p in root.glob(f"{split}_series/*/*/*.dcm")]


def index_studies(names: list[str]) -> dict[str, dict[str, list[str]]]:
    """相対パスを Study → Series → ファイル一覧にまとめる。"""
    studies: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name in names:
        _, study, series, _ = name.split("/")
        studies[study][series].append(name)
    return studies


def _write_json_atomic(path: Path, obj) -> None:
    # 書き込み途中で止まっても壊れた JSON が残らないよう、一時ファイルから置き換える
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def process_study(
    study: str, series: dict[str, list[str]], out: str, cfg: dict, retry_failed: bool = False
) -> list[dict]:
    """1 Study の全シリーズを処理し、npy とメタ情報を保存する。

    `meta.json` がある Study は処理済みとしてスキップする(中断からの再開用)。

    Args:
        study: StudyInstanceUID。
        series: SeriesInstanceUID → DICOM の相対パス一覧。
        out: キャッシュの出力先。
        cfg: `crop_mm`、`size`、`max_slices`。
        retry_failed: True なら、失敗シリーズを含む処理済み Study をやり直す。

    Returns:
        シリーズごとのメタ情報(失敗時は `ok=False` と `error`)。
    """
    study_dir = Path(out) / study
    meta_path = study_dir / "meta.json"
    if meta_path.exists():
        rows = json.loads(meta_path.read_text())
        if not (retry_failed and not all(r["ok"] for r in rows)):
            return rows

    study_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for series_uid, names in series.items():
        row = {"StudyInstanceUID": study, "SeriesInstanceUID": series_uid}
        try:
            vol = load_series(_open, names, cfg["crop_mm"], cfg["size"], cfg["max_slices"])
            np.save(study_dir / f"{series_uid}.npy", vol.pixels)
            np.save(study_dir / f"{series_uid}_pos.npy", np.asarray(vol.positions, np.float32))
            row.update(vol.meta, ok=True)
        except Exception as e:  # noqa: BLE001 - 1 シリーズの失敗で全体を止めない
            row.update(ok=False, error=repr(e))
        rows.append(row)
    # 最後に書くことで、途中で止まった Study は再開時にやり直される
    _write_json_atomic(meta_path, rows)
    return rows


def check_config(out: Path, cfg: dict) -> None:
    """既存キャッシュと設定が違えば止める(異なる設定のキャッシュが混ざるのを防ぐ)。"""
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise SystemExit(f"{path} differs from the requested settings; use a new --out")
    _write_json_atomic(path, cfg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/rsna-knee-abnormality-detection.zip")
    p.add_argument("--out", default="data/cache/v2")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--crop-mm", type=float, default=150.0)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--max-slices", type=int, default=0, help="0 keeps all slices")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="process only the first N studies")
    p.add_argument("--retry-failed", action="store_true", help="redo studies with failed series")
    args = p.parse_args()

    cfg = {"crop_mm": args.crop_mm, "size": args.size, "max_slices": args.max_slices}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_config(out, cfg)

    studies = index_studies(list_dicoms(args.src, args.split))
    keys = sorted(studies)[: args.limit or None]
    print(f"{len(keys)} studies, {sum(len(studies[k]) for k in keys)} series", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.src,)) as ex:
        futs = [
            ex.submit(process_study, k, dict(studies[k]), str(out), cfg, args.retry_failed)
            for k in keys
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 50 == 0 or i == len(keys):
                el = time.time() - t0
                print(
                    f"{i}/{len(keys)} studies  {el / 60:.1f} min  "
                    f"eta {el / i * (len(keys) - i) / 60:.1f} min",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{args.split}_series_meta.parquet", index=False)
    print(f"done: {df.ok.sum()}/{len(df)} series ok", flush=True)


if __name__ == "__main__":
    main()
