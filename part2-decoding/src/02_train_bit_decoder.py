# part2-decoding/src/02_train_bit_decoder.py

"""
Train 8 independent ResNet-50 binary classifiers, one per watermark bit.

Each bit has its own ResNet-50 backbone, optimizer, and cosine LR schedule.
Best checkpoint per bit is selected by validation accuracy. Bits are trained
sequentially; each model is offloaded to CPU after its training run to keep
GPU memory bounded. After all 8 bits are trained, the ensemble is evaluated
on the val and test splits.

Results are written to data/separate-bit-decoder/training_summary.json,
matching the format of existing training runs.

Usage:
    python part2-decoding/src/02_train_bit_decoder.py \
        [--config part2-decoding/config/separate_bit_decoder.yaml]
"""


# === IMPORTS ===

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from utils import WatermarkDataset, compute_metrics, make_transform


# === HELPERS ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 8 separate ResNet-50 bit classifiers.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("part2-decoding/config/separate_bit_decoder.yaml"),
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)["separate_bit_decoder"]


def build_resnet50_binary(pretrained: bool) -> nn.Module:
    """ResNet-50 with the final fc replaced by a single-logit head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = models.resnet50(weights=weights)
    backbone.fc = nn.Linear(backbone.fc.in_features, 1)
    return backbone


def train_one_bit(
    bit_idx: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """
    Train one binary classifier for a single watermark bit.

    Args:
        bit_idx: which bit (0–7) this model decodes
        model: single-logit ResNet-50 (not yet on device)
        train_loader: full training DataLoader (all 8-bit labels present)
        val_loader: validation DataLoader
        device: training device
        num_epochs: number of epochs
        lr: initial learning rate (cosine-annealed to 0)
        checkpoint_dir: directory where bit_{bit_idx}_best.pth is saved

    Returns:
        Dict with bit_index, best_val_acc, checkpoint path, and per-epoch history.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = -1.0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"bit {bit_idx} epoch {epoch + 1}/{num_epochs}", leave=False):
            images = batch["image"].to(device)
            target = batch["bits"][:, bit_idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(-1), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / len(train_loader)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                target = batch["bits"][:, bit_idx].to(device)
                pred = (torch.sigmoid(model(images).squeeze(-1)) > 0.5).float()
                preds.append(pred.cpu())
                targets.append(target.cpu())
        acc = float((torch.cat(preds) == torch.cat(targets)).float().mean())
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_acc": acc})
        print(f"  bit {bit_idx} epoch {epoch + 1}: train_loss={train_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"bit_{bit_idx}_best.pth"
    torch.save(
        {
            "bit_index": bit_idx,
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "best_val_acc": best_acc,
        },
        ckpt_path,
    )
    print(f"  -> saved {ckpt_path.name} (best val_acc={best_acc:.4f})")
    return {
        "bit_index": bit_idx,
        "best_val_acc": best_acc,
        "checkpoint": str(ckpt_path),
        "history": history,
    }


@torch.no_grad()
def evaluate_ensemble(
    checkpoint_dir: Path,
    num_bits: int,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the full ensemble by running each bit model in turn.

    Loads each best checkpoint to GPU, collects predictions, then offloads.
    This keeps peak GPU memory equal to one ResNet-50 at inference time.
    """
    all_preds = []
    for bit_idx in range(num_bits):
        model = build_resnet50_binary(pretrained=False).to(device)
        ckpt = torch.load(checkpoint_dir / f"bit_{bit_idx}_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        bit_preds = []
        for batch in loader:
            images = batch["image"].to(device)
            pred = (torch.sigmoid(model(images).squeeze(-1)) > 0.5).float()
            bit_preds.append(pred.cpu())
        all_preds.append(torch.cat(bit_preds))
        model.to("cpu")

    # stack into (N, num_bits) and collect targets in a second pass
    preds_tensor = torch.stack(all_preds, dim=1)
    targets_list = []
    for batch in loader:
        targets_list.append(batch["bits"])
    targets_tensor = torch.cat(targets_list)

    return compute_metrics(preds_tensor, targets_tensor)


# === MAIN ===

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}\n")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    images_dir = Path(data_cfg["images_dir"])
    metadata_path = images_dir / data_cfg["metadata_file"]
    splits_path = images_dir / data_cfg["splits_file"]

    with open(splits_path) as f:
        splits = json.load(f)

    transform = make_transform(model_cfg["image_size"])
    dataset = WatermarkDataset(metadata_path, images_dir, splits, transform)

    train_loader = DataLoader(
        Subset(dataset, splits["train"]),
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        Subset(dataset, splits["val"]),
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    test_loader = DataLoader(
        Subset(dataset, splits["test"]),
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    num_bits = data_cfg["num_bits"]

    per_bit_summary: list[dict] = []
    t_start = time.time()

    for bit_idx in range(num_bits):
        print(f"\n--- training bit {bit_idx} ---")
        model = build_resnet50_binary(pretrained=model_cfg["pretrained"])
        info = train_one_bit(
            bit_idx=bit_idx,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=train_cfg["num_epochs"],
            lr=train_cfg["learning_rate"],
            checkpoint_dir=checkpoint_dir,
        )
        per_bit_summary.append(info)

    elapsed = time.time() - t_start
    print(f"\nall {num_bits} bits trained in {elapsed:.1f}s")

    print("evaluating ensemble on val split...")
    val_metrics = evaluate_ensemble(checkpoint_dir, num_bits, val_loader, device)
    print(f"val  mean_bit_acc={val_metrics['mean_bit_accuracy']:.4f}  exact_match={val_metrics['exact_match_rate']:.4f}")

    print("evaluating ensemble on test split...")
    test_metrics = evaluate_ensemble(checkpoint_dir, num_bits, test_loader, device)
    print(f"test mean_bit_acc={test_metrics['mean_bit_accuracy']:.4f}  exact_match={test_metrics['exact_match_rate']:.4f}")

    results: dict[str, Any] = {
        "per_bit": per_bit_summary,
        "ensemble_val_metrics": {
            "mean_bit_accuracy": val_metrics["mean_bit_accuracy"],
            "exact_match_rate": val_metrics["exact_match_rate"],
            "per_bit_accuracy": val_metrics["per_bit_accuracy"],
        },
        "ensemble_test_metrics": {
            "mean_bit_accuracy": test_metrics["mean_bit_accuracy"],
            "exact_match_rate": test_metrics["exact_match_rate"],
            "per_bit_accuracy": test_metrics["per_bit_accuracy"],
        },
        "elapsed_seconds": elapsed,
        "device": str(device),
    }

    results_path = Path(cfg["output"]["results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {results_path}")


if __name__ == "__main__":
    main()
