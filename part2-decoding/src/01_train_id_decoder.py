# part2-decoding/src/01_train_id_decoder.py

"""
Train a shared-backbone watermark ID decoder.

Supports three experiments selected via --experiment:
  baseline_resnet50          ResNet-50 at 1024px (reference)
  ablation1_efficientnet_b0  EfficientNet-B0 at 1024px (backbone ablation)
  ablation2_resolution_512   ResNet-50 at 512px (resolution ablation)

All hyperparameters and paths come from config/id_decoder.yaml.
After training, the best checkpoint (by val exact-match) is evaluated on
the test split and results are written to data/id-decoders/<experiment>.json.

Usage:
    python part2-decoding/src/01_train_id_decoder.py \
        --experiment baseline_resnet50 \
        [--config part2-decoding/config/id_decoder.yaml]
"""


# === IMPORTS ===

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils import WatermarkDataset, build_model, compute_metrics, make_transform


# === HELPERS ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a shared-backbone ID decoder.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("part2-decoding/config/id_decoder.yaml"),
    )
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["baseline_resnet50", "ablation1_efficientnet_b0", "ablation2_resolution_512"],
        help="Which experiment to run.",
    )
    return p.parse_args()


def load_config(path: Path, experiment: str) -> dict[str, Any]:
    """
    Load id_decoder.yaml and merge shared + experiment-specific settings.

    Args:
        path: path to id_decoder.yaml
        experiment: key under id_decoder.experiments

    Returns:
        Dict with keys 'data', 'training', 'model' (the experiment block).
    """
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        root = yaml.safe_load(f)["id_decoder"]
    experiments = root.get("experiments", {})
    if experiment not in experiments:
        raise ValueError(
            f"experiment {experiment!r} not in config. Available: {list(experiments)}"
        )
    return {
        "data": root["data"],
        "training": root["training"],
        "model": experiments[experiment],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        targets = batch["bits"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        targets = batch["bits"].to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds)
        all_targets.append(targets)
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = total_loss / len(loader)
    return metrics


# === MAIN ===

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.experiment)

    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"experiment : {args.experiment}")
    print(f"device     : {device}\n")

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

    model = build_model(
        architecture=model_cfg["architecture"],
        num_outputs=data_cfg["num_bits"],
        pretrained=model_cfg["pretrained"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"architecture : {model_cfg['architecture']}")
    print(f"image size   : {model_cfg['image_size']}px")
    print(f"parameters   : {n_params:.1f}M\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    checkpoint_path = Path(model_cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_exact_match = 0.0
    best_epoch = 0

    for epoch in range(train_cfg["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch {epoch + 1:02d}/{train_cfg['num_epochs']}  "
            f"train_loss={train_loss:.4f}  "
            f"val_mean_acc={val_metrics['mean_bit_accuracy']:.4f}  "
            f"val_exact={val_metrics['exact_match_rate']:.4f}"
        )

        if val_metrics["exact_match_rate"] > best_exact_match:
            best_exact_match = val_metrics["exact_match_rate"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"  -> saved checkpoint (val exact-match: {best_exact_match:.4f})")

    # load best checkpoint and evaluate on test split
    print(f"\nloading best checkpoint (epoch {best_epoch + 1})...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"test mean_bit_accuracy : {test_metrics['mean_bit_accuracy']:.4f}")
    print(f"test exact_match_rate  : {test_metrics['exact_match_rate']:.4f}")

    results: dict[str, Any] = {
        "experiment": args.experiment,
        "architecture": model_cfg["architecture"],
        "checkpoint_epoch": best_epoch,
        "val_exact_match": round(best_exact_match, 4),
        "test_metrics": {
            "mean_bit_accuracy": round(test_metrics["mean_bit_accuracy"], 4),
            "exact_match_rate": round(test_metrics["exact_match_rate"], 4),
            "per_bit_accuracy": [round(x, 4) for x in test_metrics["per_bit_accuracy"]],
        },
    }

    results_path = Path(model_cfg["results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nresults saved to {results_path}")
    print(f"checkpoint  saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
