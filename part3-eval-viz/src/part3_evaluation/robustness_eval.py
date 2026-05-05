"""Robustness evaluation: JPEG, resize, crop attacks against the decoder.

Attacks operate on normalized image tensors. Each attack is implemented as a
``Callable[[Tensor], Tensor]`` so the suite is easy to extend. ``main`` loads
one of the three architectures (resnet | separate | vit), runs every attack
over the held-out test set, prints a table, and saves two diagnostic figures.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------- attack primitives ---------------------------- #

def _denormalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def _normalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch - IMAGENET_MEAN) / IMAGENET_STD


def apply_jpeg(image_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip every image through PIL's JPEG encoder at ``quality``."""
    raw = _denormalize(image_tensor)
    out = torch.empty_like(raw)
    for i in range(raw.shape[0]):
        arr = (raw[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        out[i] = torch.from_numpy(decoded).permute(2, 0, 1)
    return _normalize(out)


def apply_resize(image_tensor: torch.Tensor, down_res: int = 512, up_res: int | None = None) -> torch.Tensor:
    raw = _denormalize(image_tensor)
    target_up = up_res if up_res is not None else raw.shape[-1]
    down = F.interpolate(raw, size=(down_res, down_res), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    up = F.interpolate(down, size=(target_up, target_up), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return _normalize(up)


def apply_random_crop(image_tensor: torch.Tensor, crop_ratio: float = 0.75, seed: int = 0) -> torch.Tensor:
    raw = _denormalize(image_tensor)
    B, C, H, W = raw.shape
    side_h = max(1, int(round(H * (crop_ratio ** 0.5))))
    side_w = max(1, int(round(W * (crop_ratio ** 0.5))))
    g = torch.Generator().manual_seed(seed)
    out = torch.empty_like(raw)
    for i in range(B):
        top = int(torch.randint(0, max(1, H - side_h + 1), (1,), generator=g).item())
        left = int(torch.randint(0, max(1, W - side_w + 1), (1,), generator=g).item())
        crop = raw[i : i + 1, :, top : top + side_h, left : left + side_w]
        resized = F.interpolate(crop, size=(H, W), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        out[i] = resized[0]
    return _normalize(out)


def default_attacks() -> "OrderedDict[str, Callable[[torch.Tensor], torch.Tensor]]":
    attacks: "OrderedDict[str, Callable[[torch.Tensor], torch.Tensor]]" = OrderedDict()
    attacks["clean"] = lambda x: x
    attacks["jpeg_q90"] = lambda x: apply_jpeg(x, 90)
    attacks["jpeg_q75"] = lambda x: apply_jpeg(x, 75)
    attacks["jpeg_q50"] = lambda x: apply_jpeg(x, 50)
    attacks["resize_512"] = lambda x: apply_resize(x, down_res=512, up_res=x.shape[-1])
    attacks["random_crop_75"] = lambda x: apply_random_crop(x, crop_ratio=0.75)
    return attacks


# ----------------------------- model evaluation ----------------------------- #

@torch.no_grad()
def _predict(
    model,
    attacked_batch: torch.Tensor,
    device: torch.device,
    force_input_size: int | None = None,
) -> torch.Tensor:
    """Forward pass at the model's required input size.

    Attacks are applied at the dataset resolution (e.g. 1024 - the spec's
    canonical native size). For models that only accept a fixed input (ViT),
    we downsample post-attack so the attack's degradation is at the dataset
    resolution but the forward pass is at the model's required resolution.
    """
    x = attacked_batch.to(device)
    if force_input_size is not None and x.shape[-1] != force_input_size:
        # bilinear so this works on MPS too. The bicubic spec applies only to
        # the resize *attack* (apply_resize), not this model-fitting downsample.
        x = F.interpolate(
            x,
            size=(force_input_size, force_input_size),
            mode="bilinear",
            align_corners=False,
        )
    if hasattr(model, "forward_logits"):
        logits = model.forward_logits(x)
    else:
        logits = model(x)
    probs = torch.sigmoid(logits)
    return (probs > 0.5).float().cpu()


def evaluate_robustness(
    model,
    test_loader: DataLoader,
    attacks,
    device: torch.device,
    force_input_size: int | None = None,
) -> dict:
    model.eval().to(device)
    results = {}
    for name, fn in attacks.items():
        all_preds, all_targets = [], []
        for batch in test_loader:
            images = batch["image"]
            targets = batch["bits"]
            attacked = fn(images)
            preds = _predict(model, attacked, device, force_input_size=force_input_size)
            all_preds.append(preds)
            all_targets.append(targets)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(preds, targets)
        results[name] = {
            "per_bit_accuracy": [float(x) for x in metrics["per_bit_accuracy"]],
            "mean_bit_accuracy": float(metrics["mean_bit_accuracy"]),
            "exact_match_rate": float(metrics["exact_match_rate"]),
        }
        print(f"  {name:>16}: mean={metrics['mean_bit_accuracy']:.4f}  exact={metrics['exact_match_rate']:.4f}")
    return results


# ----------------------------- reporting ------------------------------------ #

def print_robustness_table(results: dict) -> None:
    headers = ["attack"] + [f"b{i}" for i in range(8)] + ["mean", "exact"]
    rows = []
    for name, m in results.items():
        row = [name]
        row += [f"{x:.3f}" for x in m["per_bit_accuracy"]]
        row.append(f"{m['mean_bit_accuracy']:.3f}")
        row.append(f"{m['exact_match_rate']:.3f}")
        rows.append(row)
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print()
    print(line)
    print(sep)
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def save_robustness_plots(results: dict, output_dir: str) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    attack_names = list(results.keys())
    n_attacks = len(attack_names)
    bits = np.arange(8)
    width = 0.8 / max(1, n_attacks)
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, name in enumerate(attack_names):
        ax.bar(bits + j * width, results[name]["per_bit_accuracy"], width=width, label=name)
    ax.set_xticks(bits + (n_attacks - 1) * width / 2)
    ax.set_xticklabels([f"bit {i}" for i in range(8)])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-bit accuracy across attacks")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = out_dir / "robustness_per_bit.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    saved.append(out)
    print(f"  wrote {out}")

    jpeg_qualities = []
    jpeg_means = []
    for name, m in results.items():
        if name.startswith("jpeg_q"):
            try:
                q = int(name.split("q")[-1])
            except ValueError:
                continue
            jpeg_qualities.append(q)
            jpeg_means.append(m["mean_bit_accuracy"])
    if "clean" in results:
        jpeg_qualities.insert(0, 100)
        jpeg_means.insert(0, results["clean"]["mean_bit_accuracy"])
    if jpeg_qualities:
        order = np.argsort(jpeg_qualities)[::-1]
        xs = [jpeg_qualities[i] for i in order]
        ys = [jpeg_means[i] for i in order]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("JPEG quality (100 = clean)")
        ax.set_ylabel("mean bit accuracy")
        ax.set_title("Degradation under JPEG compression")
        ax.set_ylim(0, 1.05)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / "robustness_jpeg_curve.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(out)
        print(f"  wrote {out}")
    return saved


# ----------------------------- model loading -------------------------------- #

def _resolve_resnet_checkpoint(explicit: str | None) -> str | None:
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
        print(f"  loaded ResNet-50 from {Path(resolved).name}")
    else:
        print("  WARNING: no ResNet-50 checkpoint found, running with random init")
    return model


def _load_separate(checkpoint_dir: str | None):
    from src.model_separate import SeparateBitClassifier
    model = SeparateBitClassifier(pretrained=False)
    if checkpoint_dir and Path(checkpoint_dir).exists() and any(Path(checkpoint_dir).glob("bit_*_best.pth")):
        model.load_all(checkpoint_dir, map_location="cpu")
    return model


def _load_vit(checkpoint_path: str | None):
    from src.model_vit import ViTWatermarkDecoder
    model = ViTWatermarkDecoder(pretrained=False)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness eval across JPEG/resize/crop attacks.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", choices=["resnet", "separate", "vit"], default="resnet")
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
    p.add_argument("--output-dir", type=str, default=str(DECODING_ROOT / "results" / "figures"))
    p.add_argument("--results-json", type=str, default=str(DECODING_ROOT / "results" / "robustness.json"))
    p.add_argument("--smoke-root", type=str, default=str(DECODING_ROOT / ".smoke"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.batch_size = 4
        args.num_workers = 0
        # Smoke uses a small resolution to keep CPU runs fast; ViT downsamples
        # internally to its required input size regardless.
        args.image_size = 96
        args.output_dir = str(DECODING_ROOT / ".smoke" / "figures")
        args.results_json = str(DECODING_ROOT / ".smoke" / "robustness.json")

    defaults = default_data_paths(DECODING_ROOT.parent)
    metadata, images, splits, _, is_smoke = resolve_paths(
        args,
        default_metadata=defaults["metadata"],
        default_images=defaults["images"],
        default_splits=defaults["splits"],
    )

    device = pick_device(force_cpu=args.smoke)
    print(f"=== robustness_eval ===")
    print(f"  model={args.model}  device={device}  smoke={is_smoke}")

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
    test_loader = DataLoader(Subset(full, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model == "resnet":
        model = _load_resnet(args.resnet_checkpoint)
    elif args.model == "separate":
        model = _load_separate(args.separate_checkpoint_dir)
    else:
        model = _load_vit(args.vit_checkpoint)

    attacks = default_attacks()
    if is_smoke:
        attacks = OrderedDict(
            (k, v) for k, v in attacks.items() if k in {"clean", "jpeg_q75", "resize_512", "random_crop_75"}
        )

    # ViT-B/16 is hard-locked at 224 by torchvision. Apply attacks at the
    # dataset resolution (e.g. 1024 for the spec'd resize attack) but
    # downsample inside the forward pass.
    force_input = 224 if args.model == "vit" else None
    if force_input is not None and args.image_size != force_input:
        print(f"  note: ViT input forced to {force_input} (attacks still happen at {args.image_size})")

    print("\nrunning attacks...")
    results = evaluate_robustness(model, test_loader, attacks, device, force_input_size=force_input)
    print_robustness_table(results)
    save_robustness_plots(results, args.output_dir)

    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(
            {"model": args.model, "smoke": is_smoke, "results": results},
            f,
            indent=2,
        )
    print(f"\nresults: {args.results_json}")


if __name__ == "__main__":
    main()
