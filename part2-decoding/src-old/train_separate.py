"""Train 8 independent ResNet-50 binary classifiers (one per watermark bit).

Reuses Person A's WatermarkDataset, splits.json, and metric helpers. Each bit
is trained with its own optimizer, cosine LR schedule, and best-checkpoint
selection on validation accuracy. The full ensemble is then evaluated for
mean-bit accuracy and exact match.
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
from src.model_separate import SeparateBitClassifier
from src.utils import compute_metrics, print_metrics
from _smoke_utils import default_data_paths, pick_device, resolve_paths


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_loaders(metadata, images, splits, image_size, batch_size, num_workers, smoke_max=None):
    transform = build_transform(image_size)
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


def train_one_bit(
    bit_idx: int,
    bit_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    output_dir: Path,
) -> dict:
    bit_model = bit_model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(bit_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    best_acc = -1.0
    best_state = None
    history = []

    for epoch in range(num_epochs):
        bit_model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"bit {bit_idx} epoch {epoch+1}/{num_epochs}", leave=False):
            images = batch["image"].to(device)
            target = batch["bits"][:, bit_idx].to(device).float()
            optimizer.zero_grad()
            logit = bit_model(images).squeeze(-1)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        bit_model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                target = batch["bits"][:, bit_idx].to(device).float()
                logit = bit_model(images).squeeze(-1)
                pred = (torch.sigmoid(logit) > 0.5).float()
                preds.append(pred.cpu())
                targets.append(target.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        acc = float((preds == targets).float().mean().item())
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_acc": acc})
        print(f"  bit {bit_idx} epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in bit_model.state_dict().items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"bit_{bit_idx}_best.pth"
    torch.save(
        {
            "bit_index": bit_idx,
            "model_state_dict": best_state if best_state is not None else bit_model.state_dict(),
            "best_val_acc": best_acc,
            "history": history,
        },
        ckpt_path,
    )
    print(f"  saved {ckpt_path} (best val_acc={best_acc:.4f})")
    return {"bit_index": bit_idx, "best_val_acc": best_acc, "checkpoint": str(ckpt_path), "history": history}


@torch.no_grad()
def evaluate_ensemble(model: SeparateBitClassifier, val_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    for batch in val_loader:
        images = batch["image"].to(device)
        targets = batch["bits"].to(device)
        probs = model(images)
        preds = (probs > 0.5).float()
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 8 separate ResNet-50 bit classifiers.")
    p.add_argument("--smoke", action="store_true", help="64-image smoke run, 2 epochs, CPU only.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--splits", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=str(DECODING_ROOT / "checkpoints" / "separate"))
    p.add_argument("--smoke-root", type=str, default=str(DECODING_ROOT / ".smoke"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 4
        args.image_size = 96
        args.num_workers = 0
        args.pretrained = False
        args.checkpoint_dir = str(DECODING_ROOT / ".smoke" / "checkpoints" / "separate")

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
    print(f"=== train_separate ===")
    print(f"device: {device} | smoke: {is_smoke}")
    print(f"data: metadata={metadata}\n      images={images}\n      splits={splits}")

    train_loader, val_loader = make_loaders(
        metadata, images, splits, args.image_size, args.batch_size, args.num_workers,
        smoke_max=64 if is_smoke else None,
    )
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = SeparateBitClassifier(pretrained=args.pretrained)
    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    t_start = time.time()
    for bit_idx in range(SeparateBitClassifier.NUM_BITS):
        print(f"\n--- training bit {bit_idx} ---")
        info = train_one_bit(
            bit_idx=bit_idx,
            bit_model=model.bit_models[bit_idx],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            output_dir=output_dir,
        )
        summary.append(info)
        model.bit_models[bit_idx].to("cpu")

    elapsed = time.time() - t_start
    print(f"\nall 8 bits trained in {elapsed:.1f}s")

    model.load_all(str(output_dir), map_location="cpu")
    model.to(device)
    metrics = evaluate_ensemble(model, val_loader, device)
    print_metrics(metrics, prefix="Val (ensemble) ")

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "per_bit": summary,
                "ensemble_val_metrics": {
                    "mean_bit_accuracy": metrics["mean_bit_accuracy"],
                    "exact_match_rate": metrics["exact_match_rate"],
                    "per_bit_accuracy": metrics["per_bit_accuracy"],
                },
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
