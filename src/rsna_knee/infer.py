"""隠しテストを前処理して推論し、submission.csv を書く。Kaggle の提出 Notebook から呼ぶ。

モデルのディレクトリには、学習時の出力(`model.pt` と `config.yaml`(学習時は `.hydra/` 内))と、
キャッシュの設定 `cache_config.json` を置く。前処理・窓選択・モデル設定は学習時のものを使う。

Usage:
    python -m rsna_knee.infer --data-dir /kaggle/input/rsna-knee-abnormality-detection \
        --model-dirs /kaggle/input/<weights>/fold0 --out submission.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from rsna_knee.data.dataset import KneeStudyDataset, build_index, collate
from rsna_knee.labels.public import TARGETS
from rsna_knee.models.model import KneeModel
from rsna_knee.preprocess.build_cache import build_cache
from rsna_knee.utils import to_device


def load_model(model_dir: Path, device: torch.device) -> tuple[KneeModel, dict]:
    """学習済みモデルと、その学習設定を読み込む。

    事前学習の重みはチェックポイントに含まれるので、ダウンロードはしない。
    """
    # Kaggle の環境に Hydra / OmegaConf が無くても読めるよう、PyYAML で読む
    # Kaggle Dataset ではドット始まりのフォルダが落ちることがあるので、直下の config.yaml を優先する
    path = model_dir / "config.yaml"
    if not path.exists():
        path = model_dir / ".hydra" / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    enc = cfg["model"]["encoder"] | {"pretrained": False}
    model = KneeModel(enc, cfg["model"]["aggregator"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict(
    model: KneeModel, loader: DataLoader, device: torch.device, dtype: torch.dtype
) -> pd.DataFrame:
    """確率(sigmoid 後)を Study ごとに返す。"""
    ids, probs = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(batch)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(batch["study_id"])
    return pd.DataFrame(np.concatenate(probs), index=ids, columns=TARGETS)


def _native_bf16() -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # 古い torch には including_emulation が無い
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


def run(
    data_dir: Path,
    model_dirs: list[Path],
    out: Path,
    cache_dir: Path,
    split: str = "test",
    workers: int = 4,
    batch_size: int = 4,
    limit: int = 0,
    src: str | None = None,
) -> pd.DataFrame:
    """前処理 → 推論 → 複数モデルの平均 → submission の書き出しまでを行う。

    Args:
        data_dir: コンペのデータのルート(`{split}_series/` と `{split}.csv` を含む)。
        model_dirs: 学習済みモデルのディレクトリ(複数なら確率を平均する)。
        out: 出力する submission.csv。
        cache_dir: 前処理キャッシュの出力先(書き込み可能な場所)。
        split: "test"(提出)または "train"(手元での時間計測・検証用)。
        workers: 前処理と DataLoader の並列数。
        batch_size: 推論のバッチサイズ(Study 数)。
        limit: 0 以外なら先頭 N Study だけ処理する(動作確認用)。
        src: DICOM の読み込み元(zip など)。省略時は `data_dir`。手元で train を zip から読む検証用。

    Returns:
        submission と同じ内容の DataFrame。
    """
    t0 = time.time()
    device = torch.device("cuda")
    # T4 / P100 は bf16 を持たない。エミュレーションだと極端に遅くなるので fp16 を使う
    dtype = torch.bfloat16 if _native_bf16() else torch.float16

    cache_cfg = json.loads((model_dirs[0] / "cache_config.json").read_text())
    for d in model_dirs[1:]:
        if json.loads((d / "cache_config.json").read_text()) != cache_cfg:
            raise ValueError(f"{d} was trained on a different cache configuration")
    build_cache(src or str(data_dir), cache_dir, split, cache_cfg, workers=workers, limit=limit)
    print(f"preprocess done: {time.time() - t0:.0f}s", flush=True)

    study_ids = pd.read_csv(data_dir / f"{split}.csv")["StudyInstanceUID"].astype(str).tolist()
    if limit:
        study_ids = sorted(study_ids)[:limit]
    index = build_index(cache_dir, data_dir / f"{split}_series.csv", split)
    present = [s for s in study_ids if s in index]
    missing = [s for s in study_ids if s not in index]
    if missing:
        print(f"WARNING: {len(missing)} studies have no readable series; using 0.5", flush=True)

    preds = []
    for d in model_dirs:
        model, cfg = load_model(d, device)
        ds = KneeStudyDataset(
            present,
            index,
            train=False,
            windows_per_series=cfg["eval"]["windows_per_series"],
        )
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=workers, collate_fn=collate, pin_memory=True
        )
        preds.append(predict(model, loader, device, dtype))
        print(f"model {d}: {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    pred = sum(preds) / len(preds)
    sub = pd.DataFrame(0.5, index=study_ids, columns=TARGETS)
    sub.loc[present] = pred.loc[present].values
    sub.index.name = "StudyInstanceUID"
    sub.to_csv(out)
    print(f"wrote {out}: {len(sub)} rows, {len(missing)} fallback, {time.time() - t0:.0f}s total")
    return sub


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/rsna_cache"))
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--src", default=None)
    args = p.parse_args()
    run(
        args.data_dir,
        args.model_dirs,
        args.out,
        args.cache_dir,
        args.split,
        args.workers,
        args.batch_size,
        args.limit,
        args.src,
    )


if __name__ == "__main__":
    main()
