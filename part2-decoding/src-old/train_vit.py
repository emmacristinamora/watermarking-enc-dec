"""Train a ViT-B/16 watermark decoder with 8 binary heads.

Mirrors Person A's train.py for the shared-backbone ResNet-50 baseline, but
swaps in ViTWatermarkDecoder and forces 224x224 input (ViT-B/16 patch grid).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
sys.path.insert(0, str(DECODING_ROOT))
sys.path.insert(0, str(HERE))

from src.dataloader import WatermarkDataset
from src.model_vit import ViTWatermarkDecoder
from src.utils import compute_metrics, print_metrics
from _smoke_utils import default_data_paths, pick_device, resolve_paths


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((ViTWatermarkDecoder.INPUT_SIZE, ViTWatermarkDecoder.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_loaders(metadata, images, splits, batch_size, num_workers, smoke_max=None):
    transform = build_transform()
    full = WatermarkDataset(metadata_path=metadata, image_dir=images, transform=transform)
    with open(splits) as f:
        split_idx = json.load(f)
    train_idx, val_idx = split_idx["train"], split_idx["val"]
    if smoke_max is not None:
        train_idx = train_idx[: max(2, smoke_max)]
        val_idx = val_idx[: max(2, smoke_max // 4)]
    train_loader = DataLoader(
        Subset(full, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(full, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        targets = batch["bits"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        targets = batch["bits"].to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        n += 1
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = total_loss / max(1, n)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ViT-B/16 watermark decoder.")
    p.add_argument("--smoke", action="store_true", help="64-image smoke run, 2 epochs, CPU only.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--splits", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=str(DECODING_ROOT / "checkpoints" / "vit_best.pth"))
    p.add_argument("--smoke-root", type=str, default=str(DECODING_ROOT / ".smoke"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 4
        args.num_workers = 0
        args.pretrained = False
        args.checkpoint = str(DECODING_ROOT / ".smoke" / "checkpoints" / "vit_best.pth")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults = default_data_paths(DECODING_ROOT.parent)
    metadata, images, splits, _, is_smoke = resolve_paths(
        args,
        default_metadata=defaults["metadata"],
        default_images=defaults["images"],
        default_splits=defaults["splits"],
    )

    device = pick_device(force_cpu=args.smoke)
    print(f"=== train_vit ===")
    print(f"device: {device} | smoke: {is_smoke}")
    print(f"data: metadata={metadata}\n      images={images}\n      splits={splits}")

    train_loader, val_loader = make_loaders(
        metadata, images, splits, args.batch_size, args.num_workers,
        smoke_max=64 if is_smoke else None,
    )
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = ViTWatermarkDecoder(pretrained=args.pretrained).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    best_exact = -1.0
    best_metrics = None
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        print(f"\nepoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  train_loss={train_loss:.4f}")
        print_metrics(val_metrics, prefix="  Val ")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "mean_bit_accuracy": val_metrics["mean_bit_accuracy"],
                "exact_match_rate": val_metrics["exact_match_rate"],
                "per_bit_accuracy": val_metrics["per_bit_accuracy"],
            }
        )
        if val_metrics["exact_match_rate"] > best_exact:
            best_exact = val_metrics["exact_match_rate"]
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "architecture": "vit_b_16",
                },
                args.checkpoint,
            )
            print(f"  saved {args.checkpoint} (exact_match={best_exact:.4f})")

    elapsed = time.time() - t_start
    print(f"\ntraining complete in {elapsed:.1f}s")
    if best_metrics is not None:
        print_metrics(best_metrics, prefix="Best Val ")

    summary_path = Path(args.checkpoint).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "history": history,
                "best_exact_match": best_exact,
                "elapsed_seconds": elapsed,
                "device": str(device),
                "smoke": is_smoke,
            },
            f,
            indent=2,
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
