"""1 fold 分の学習。OOF 予測と gold-58 への予測を保存する。

Usage:
    python -m rsna_knee.train fold=0
    python -m rsna_knee.train fold=-1 exp_name=full   # 全データで学習
    python -m rsna_knee.train fold=0 data.limit=64 train.epochs=1   # 動作確認
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from rsna_knee.data.dataset import KneeStudyDataset, build_index, collate
from rsna_knee.data.folds import make_folds
from rsna_knee.labels.public import TARGETS, load_gold, load_train_labels
from rsna_knee.metrics import macro_auc
from rsna_knee.models.model import KneeModel


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def jitter_intensity(images: torch.Tensor, strength: float) -> torch.Tensor:
    """窓ごとにコントラストと明るさをランダムに変える(uint8 のまま返す)。"""
    if strength <= 0:
        return images
    shape = (*images.shape[:2], 1, 1, 1)
    gain = 1 + (torch.rand(shape, device=images.device) * 2 - 1) * strength
    bias = (torch.rand(shape, device=images.device) * 2 - 1) * strength * 255
    return (images.float() * gain + bias).clamp(0, 255).to(torch.uint8)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """確率ラベルに対する、セルごとの重み付き BCE。"""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def cosine_with_warmup(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    return f


@torch.no_grad()
def predict(model: KneeModel, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """確率(sigmoid 後)を Study ごとに返す。"""
    model.eval()
    ids, probs = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(batch["study_id"])
    return pd.DataFrame(np.concatenate(probs), index=ids, columns=TARGETS)


def eval_loader(ids: list[str], index: dict, cfg: DictConfig) -> DataLoader:
    ds = KneeStudyDataset(ids, index, train=False, windows_per_series=cfg.eval.windows_per_series)
    return DataLoader(
        ds,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    out = Path(HydraConfig.get().runtime.output_dir)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    y, w = load_train_labels(
        cfg.data.label_source, cfg.data.silent_weight, train_csv=Path(cfg.data.train_csv)
    )
    if cfg.data.limit:
        y, w = y.iloc[: cfg.data.limit], w.iloc[: cfg.data.limit]
    folds = make_folds(pd.read_csv(cfg.data.train_csv), y, cfg.data.n_folds, cfg.seed)
    folds.to_csv(out / "folds.csv")
    gold = load_gold(Path(cfg.data.train_csv))
    index = build_index(Path(cfg.data.cache_dir), Path(cfg.data.series_csv), "train")

    train_ids = list(folds.index[folds != cfg.fold])
    valid_ids = list(folds.index[folds == cfg.fold])
    train_ds = KneeStudyDataset(
        train_ids, index, y, w, train=True, n_windows=cfg.train.n_windows, drop_missing=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    valid_loader = eval_loader(valid_ids, index, cfg) if valid_ids else None
    gold_loader = eval_loader(list(gold.index), index, cfg)

    model = KneeModel(
        OmegaConf.to_container(cfg.model.encoder), OmegaConf.to_container(cfg.model.aggregator)
    ).to(device)
    opt = torch.optim.AdamW(
        model.param_groups(cfg.train.lr, cfg.train.encoder_lr, cfg.train.weight_decay)
    )
    total = cfg.train.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(total, int(total * cfg.train.warmup_ratio))
    )
    print(f"fold {cfg.fold}: train {len(train_ds)}, valid {len(valid_ids)}, steps {total}")

    for epoch in range(cfg.train.epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in train_loader:
            batch = to_device(batch, device)
            batch["images"] = jitter_intensity(batch["images"], cfg.train.intensity_jitter)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            loss = weighted_bce(logits, batch["target"], batch["weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())

        log = {"epoch": epoch, "train_loss": float(np.mean(losses)), "time": time.time() - t0}
        gold_pred = predict(model, gold_loader, device)
        log["gold_auc"], gold_per = macro_auc(gold, gold_pred)
        if valid_loader is not None:
            oof = predict(model, valid_loader, device)
            log["valid_auc"], valid_per = macro_auc(y.loc[oof.index], oof)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in log.items()}))
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

    # 選択は行わず最終エポックを使う(gold-58 は小さすぎて選択に使うと過学習する)
    torch.save(model.state_dict(), out / "model.pt")
    gold_pred.to_csv(out / "gold_pred.csv", index_label="StudyInstanceUID")
    summary = {"gold_auc": log["gold_auc"], "gold_per_label": gold_per}
    if valid_loader is not None:
        oof.to_csv(out / "oof.csv", index_label="StudyInstanceUID")
        summary |= {"valid_auc": log["valid_auc"], "valid_per_label": valid_per}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
