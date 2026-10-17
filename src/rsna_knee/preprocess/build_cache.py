"""Build a per-series uint8 slice cache straight from the competition zip.

The unpacked data (570 GB) does not fit next to the zip, so DICOMs are read from the
archive member by member and only the cropped/resized stacks are written to disk.

Usage:
    python -m rsna_knee.preprocess.build_cache --zip data/rsna-knee-abnormality-detection.zip \
        --out data/cache/v1 --split train --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.preprocess.dicom import load_series

_zf: zipfile.ZipFile | None = None


def _init_worker(zip_path: str) -> None:
    global _zf
    _zf = zipfile.ZipFile(zip_path)


def index_zip(zip_path: str, split: str) -> dict[str, dict[str, list[str]]]:
    studies: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".dcm") or not name.startswith(f"{split}_series/"):
                continue
            _, study, series, _ = name.split("/")
            studies[study][series].append(name)
    return studies


def process_study(study: str, series: dict[str, list[str]], out: str, cfg: dict) -> list[dict]:
    study_dir = Path(out) / study
    meta_path = study_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    study_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for series_uid, names in series.items():
        row = {"StudyInstanceUID": study, "SeriesInstanceUID": series_uid}
        try:
            vol = load_series(_zf, names, cfg["crop_mm"], cfg["size"], cfg["max_slices"])
            np.save(study_dir / f"{series_uid}.npy", vol.pixels)
            np.save(study_dir / f"{series_uid}_pos.npy", np.asarray(vol.positions, np.float32))
            row.update(vol.meta, ok=True)
        except Exception as e:  # noqa: BLE001 - one broken series must not stop the run
            row.update(ok=False, error=repr(e))
        rows.append(row)
    # Written last so an interrupted study is redone on resume.
    meta_path.write_text(json.dumps(rows))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="data/rsna-knee-abnormality-detection.zip")
    p.add_argument("--out", default="data/cache/v1")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--crop-mm", type=float, default=150.0)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--max-slices", type=int, default=64)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="process only the first N studies")
    args = p.parse_args()

    cfg = {"crop_mm": args.crop_mm, "size": args.size, "max_slices": args.max_slices}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg))

    studies = index_zip(args.zip, args.split)
    keys = sorted(studies)[: args.limit or None]
    print(f"{len(keys)} studies, {sum(len(studies[k]) for k in keys)} series", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.zip,)) as ex:
        futs = [ex.submit(process_study, k, dict(studies[k]), str(out), cfg) for k in keys]
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
