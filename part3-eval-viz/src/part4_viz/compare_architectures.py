"""Side-by-side test-set comparison of the three decoder architectures.

Loads ResNet-50 (Person A), 8x separate ResNet-50 (this person), and ViT-B/16
(this person), evaluates each on Person A's held-out test split, writes a
markdown comparison table, and saves a grouped-bar PNG.

Architectures whose checkpoints are missing are evaluated with random weights
and clearly labelled as ``(random init)`` in the report — useful in the smoke
path before any GPU training has happened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
sys.path.insert(0, str(DECODING_ROOT))
sys.path.insert(0, str(HERE))

from src.dataloader import WatermarkDataset
from src.models import get_model
from src.utils import compute_metrics
from _smoke_utils import default_data_paths, pick_device, resolve_paths


@torch.no_grad()
def _evaluate_logits(model, loader: DataLoader, device, force_input_size: int | None = None) -> dict:
    """Evaluate ``model`` on ``loader``.

    ``force_input_size`` resizes incoming batches before the forward pass —
    needed for ViT-B/16, which torchvision hard-locks at 224x224 regardless of
    what resolution the rest of the pipeline runs at.
    """
    model.eval().to(device)
    all_preds, all_targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["bits"]
        if force_input_size is not None and images.shape[-1] != force_input_size:
            # bilinear so this works on MPS too (bicubic+antialias is unimplemented there).
            # this is a model-fitting downsample, not the spec'd resize attack — quality
            # difference at 1024->224 scale is negligible.
            images = F.interpolate(
                images,
                size=(force_input_size, force_input_size),
                mode="bilinear",
                align_corners=False,
            )
        if hasattr(model, "forward_logits"):
            logits = model.forward_logits(images)
        else:
            logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        all_preds.append(preds)
        all_targets.append(targets)
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))


def _resolve_resnet_checkpoint(explicit: str | None) -> str | None:
    """Pick the first existing ResNet-50 checkpoint from the canonical names."""
    candidates = [explicit] if explicit else []
    candidates += [
        str(DECODING_ROOT / "checkpoints" / "baseline_resnet50.pth"),
        str(DECODING_ROOT / "checkpoints" / "best_model.pth"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None


def _load_resnet(checkpoint_path: str | None):
    model = get_model("resnet50", num_outputs=8, pretrained=False)
    resolved = _resolve_resnet_checkpoint(checkpoint_path)
    if resolved is not None:
        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        return model, True, resolved
    return model, False, checkpoint_path


def _load_separate(ckpt_dir: str | None):
    from src.model_separate import SeparateBitClassifier
    model = SeparateBitClassifier(pretrained=False)
    available = bool(ckpt_dir and Path(ckpt_dir).exists() and any(Path(ckpt_dir).glob("bit_*_best.pth")))
    if available:
        model.load_all(ckpt_dir, map_location="cpu")
    return model, available


def _load_vit(checkpoint_path: str | None):
    from src.model_vit import ViTWatermarkDecoder
    model = ViTWatermarkDecoder(pretrained=False)
    available = bool(checkpoint_path and Path(checkpoint_path).exists())
    if available:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    return model, available


def render_markdown(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Architecture"] + [f"Bit {i}" for i in range(8)] + ["Mean", "Exact"]
    lines = ["# Architecture comparison (test set)\n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        per_bit = [f"{x:.4f}" for x in r["metrics"]["per_bit_accuracy"]]
        name = r["name"] + ("" if r["available"] else " (random init)")
        cells = [name] + per_bit + [f"{r['metrics']['mean_bit_accuracy']:.4f}", f"{r['metrics']['exact_match_rate']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Checkpoints used:")
    for r in rows:
        status = "loaded" if r["available"] else "MISSING — random init"
        lines.append(f"- **{r['name']}**: `{r['checkpoint']}` ({status})")
    output_path.write_text("\n".join(lines) + "\n")


def render_bar_chart(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arch_names = [r["name"] for r in rows]
    bits = np.arange(8)
    width = 0.8 / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, r in enumerate(rows):
        label = r["name"] + ("" if r["available"] else " (rand)")
        ax.bar(bits + j * width, r["metrics"]["per_bit_accuracy"], width=width, label=label)
    ax.set_xticks(bits + (len(rows) - 1) * width / 2)
    ax.set_xticklabels([f"bit {i}" for i in range(8)])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-bit test accuracy by architecture")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ResNet-50, separate, and ViT decoders.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--splits", type=str, default=None)
    p.add_argument(
        "--resnet-checkpoint",
        type=str,
        default=str(DECODING_ROOT / "checkpoints" / "baseline_resnet50.pth"),
        help="ResNet-50 baseline checkpoint. Falls back to best_model.pth if missing.",
    )
    p.add_argument("--separate-checkpoint-dir", type=str, default=str(DECODING_ROOT / "checkpoints" / "separate"))
    p.add_argument("--vit-checkpoint", type=str, default=str(DECODING_ROOT / "checkpoints" / "vit_best.pth"))
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--report-md", type=str, default=str(DECODING_ROOT / "results" / "architecture_comparison.md"))
    p.add_argument("--chart-png", type=str, default=str(DECODING_ROOT / "results" / "figures" / "architecture_comparison.png"))
    p.add_argument("--smoke-root", type=str, default=str(DECODING_ROOT / ".smoke"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.batch_size = 4
        args.num_workers = 0
        # Smoke uses a small resolution to keep CPU runs fast; ViT downsamples
        # internally regardless of what the rest of the pipeline runs at.
        args.image_size = 96
        args.resnet_checkpoint = str(DECODING_ROOT / ".smoke" / "checkpoints" / "baseline_resnet50.pth")
        args.separate_checkpoint_dir = str(DECODING_ROOT / ".smoke" / "checkpoints" / "separate")
        args.vit_checkpoint = str(DECODING_ROOT / ".smoke" / "checkpoints" / "vit_best.pth")
        args.report_md = str(DECODING_ROOT / ".smoke" / "architecture_comparison.md")
        args.chart_png = str(DECODING_ROOT / ".smoke" / "figures" / "architecture_comparison.png")

    defaults = default_data_paths(DECODING_ROOT.parent)
    metadata, images, splits, _, is_smoke = resolve_paths(
        args,
        default_metadata=defaults["metadata"],
        default_images=defaults["images"],
        default_splits=defaults["splits"],
    )

    device = pick_device(force_cpu=args.smoke)
    print(f"=== compare_architectures ===")
    print(f"  device={device}  smoke={is_smoke}")

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    full = WatermarkDataset(metadata_path=metadata, image_dir=images, transform=transform)
    with open(splits) as f:
        split_idx = json.load(f)
    test_idx = split_idx["test"]
    if is_smoke:
        test_idx = test_idx[: min(8, len(test_idx))]
    loader = DataLoader(Subset(full, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"  test samples: {len(test_idx)}")

    # ViT-B/16 is hard-locked at 224 by torchvision; it always sees a
    # downsampled batch regardless of the requested --image-size.
    arch_specs = [
        ("ResNet-50 (shared backbone)", "resnet",   args.resnet_checkpoint,        None),
        ("8x ResNet-50 (separate)",     "separate", args.separate_checkpoint_dir,  None),
        ("ViT-B/16",                    "vit",      args.vit_checkpoint,           224),
    ]

    rows = []
    for name, kind, ckpt, force_size in arch_specs:
        print(f"\n  evaluating {name} ...")
        if kind == "resnet":
            model, available, ckpt_used = _load_resnet(ckpt)
        elif kind == "separate":
            model, available = _load_separate(ckpt)
            ckpt_used = ckpt
        else:  # vit
            model, available = _load_vit(ckpt)
            ckpt_used = ckpt
        metrics = _evaluate_logits(model, loader, device, force_input_size=force_size)
        print(
            f"    mean={metrics['mean_bit_accuracy']:.4f}  "
            f"exact={metrics['exact_match_rate']:.4f}  loaded={available}  "
            f"ckpt={Path(ckpt_used).name if ckpt_used else 'none'}  "
            f"input_size={force_size or args.image_size}"
        )
        rows.append({
            "name": name,
            "checkpoint": ckpt_used,
            "available": available,
            "metrics": {
                "per_bit_accuracy": [float(x) for x in metrics["per_bit_accuracy"]],
                "mean_bit_accuracy": float(metrics["mean_bit_accuracy"]),
                "exact_match_rate": float(metrics["exact_match_rate"]),
            },
        })
        del model

    render_markdown(rows, Path(args.report_md))
    render_bar_chart(rows, Path(args.chart_png))
    print(f"\n  wrote {args.report_md}")
    print(f"  wrote {args.chart_png}")


if __name__ == "__main__":
    main()
