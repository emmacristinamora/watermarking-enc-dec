"""Signal analysis of LoRA watermark perturbations.

Four diagnostics, all writing PNGs into ``decoding/results/figures/``:

  1. ``visualize_comparison`` — baseline vs bit=1 vs bit=0 for each slider.
  2. ``difference_images``   — (watermarked - baseline) * 10 to surface signal.
  3. ``fft_analysis``        — log-magnitude 2D FFT side-by-side.
  4. ``gradcam_analysis``    — Grad-CAM heatmaps per bit on Person A's
                               ResNet-50 (or any model passed in).
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
import torch.nn as nn
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
sys.path.insert(0, str(DECODING_ROOT))
sys.path.insert(0, str(HERE))

from src.dataloader import WatermarkDataset
from src.models import get_model
from _smoke_utils import default_data_paths, pick_device, resolve_paths, ensure_smoke_fixture


SLIDER_NAMES = [
    "S1 warm/cool",
    "S2 sharp/soft",
    "S3 grainy/clean",
    "S4 bright/dark",
    "S5 contrast",
    "S6 saturation",
    "S7 detail",
    "S8 vintage/modern",
]


def _load_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path) as f:
        return json.load(f)


def _index_by_id_prompt(metadata: list[dict]) -> dict[tuple[int, int], dict]:
    """Map (id_int, prompt_idx) -> entry. Prompt idx is parsed from the filename."""
    index = {}
    for entry in metadata:
        fname = entry["file"]
        # filename format: id{ID:03d}_p{P:02d}.png
        try:
            p_idx = int(Path(fname).stem.split("_p")[-1])
        except (ValueError, IndexError):
            continue
        index[(entry["id_int"], p_idx)] = entry
    return index


def _baseline_path_for_prompt(baseline_dir: str, prompt_idx: int) -> Path:
    return Path(baseline_dir) / f"baseline_p{prompt_idx:02d}.png"


def _find_pair_for_slider(
    metadata_index: dict[tuple[int, int], dict], slider_idx: int, prompt_idx: int
) -> tuple[dict | None, dict | None]:
    """Return (entry_with_bit_set, entry_with_bit_unset) at the same prompt.

    Uses id = 2**(7-slider_idx) for bit-set, id=0 for bit-unset (everything
    else identical). Falls back to any pair that differs only on that slider.
    """
    bit_set_id = 2 ** (7 - slider_idx)
    bit_unset_id = 0
    set_entry = metadata_index.get((bit_set_id, prompt_idx))
    unset_entry = metadata_index.get((bit_unset_id, prompt_idx))
    if set_entry is not None and unset_entry is not None:
        return set_entry, unset_entry
    # Fallback: scan for any pair differing in only this bit.
    for (id_int, p_idx), entry in metadata_index.items():
        if p_idx != prompt_idx or entry["bits"][slider_idx] != 1:
            continue
        partner_bits = list(entry["bits"])
        partner_bits[slider_idx] = 0
        partner_id = int("".join(str(b) for b in partner_bits), 2)
        partner = metadata_index.get((partner_id, prompt_idx))
        if partner is not None:
            return entry, partner
    return None, None


def _open_rgb(path: str | os.PathLike) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def visualize_comparison(
    baseline_dir: str, images_dir: str, output_dir: str, metadata_path: str,
    prompt_indices: list[int] | None = None, slider_indices: list[int] | None = None,
) -> list[Path]:
    metadata = _load_metadata(metadata_path)
    index = _index_by_id_prompt(metadata)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompt_indices is None:
        prompt_indices = sorted({p for (_, p) in index})[:10]
    if slider_indices is None:
        slider_indices = list(range(8))

    saved = []
    for p_idx in prompt_indices:
        baseline_path = _baseline_path_for_prompt(baseline_dir, p_idx)
        if not baseline_path.exists():
            print(f"[visualize_comparison] missing baseline for prompt {p_idx}, skipping")
            continue
        baseline = _open_rgb(baseline_path)

        rows = len(slider_indices)
        fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
        for row, s_idx in enumerate(slider_indices):
            set_entry, unset_entry = _find_pair_for_slider(index, s_idx, p_idx)
            axes[row][0].imshow(baseline)
            axes[row][0].set_title(f"baseline (p{p_idx})" if row == 0 else "baseline")
            axes[row][0].axis("off")
            for col, entry, label in (
                (1, set_entry, f"{SLIDER_NAMES[s_idx]} bit=1"),
                (2, unset_entry, f"{SLIDER_NAMES[s_idx]} bit=0"),
            ):
                ax = axes[row][col]
                if entry is None:
                    ax.text(0.5, 0.5, "no sample", ha="center", va="center")
                else:
                    img = _open_rgb(Path(images_dir) / entry["file"])
                    ax.imshow(img)
                ax.set_title(label)
                ax.axis("off")
        fig.suptitle(f"Prompt {p_idx}: baseline vs slider perturbations", y = 0.995)
        fig.tight_layout(rect = [0,0,1,0.98])
        out = out_dir / f"comparison_prompt_{p_idx:02d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(out)
        print(f"  wrote {out}")
    return saved


def difference_images(
    baseline_dir: str, images_dir: str, output_dir: str, metadata_path: str,
    amplification: float = 10.0, prompt_indices: list[int] | None = None,
    slider_indices: list[int] | None = None,
) -> list[Path]:
    metadata = _load_metadata(metadata_path)
    index = _index_by_id_prompt(metadata)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompt_indices is None:
        prompt_indices = sorted({p for (_, p) in index})[:3]
    if slider_indices is None:
        slider_indices = list(range(8))

    saved = []
    for s_idx in slider_indices:
        n = len(prompt_indices)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
        for row, p_idx in enumerate(prompt_indices):
            baseline_path = _baseline_path_for_prompt(baseline_dir, p_idx)
            set_entry, unset_entry = _find_pair_for_slider(index, s_idx, p_idx)
            if not baseline_path.exists() or set_entry is None or unset_entry is None:
                for ax in axes[row]:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                    ax.axis("off")
                continue
            baseline = _open_rgb(baseline_path)
            wm_set = _open_rgb(Path(images_dir) / set_entry["file"])
            wm_unset = _open_rgb(Path(images_dir) / unset_entry["file"])
            if wm_set.shape != baseline.shape:
                wm_set = np.asarray(
                    Image.fromarray((wm_set * 255).astype(np.uint8)).resize(
                        (baseline.shape[1], baseline.shape[0])
                    )
                ) / 255.0
                wm_unset = np.asarray(
                    Image.fromarray((wm_unset * 255).astype(np.uint8)).resize(
                        (baseline.shape[1], baseline.shape[0])
                    )
                ) / 255.0

            diff_set = np.clip(0.5 + amplification * (wm_set - baseline), 0, 1)
            diff_unset = np.clip(0.5 + amplification * (wm_unset - baseline), 0, 1)
            cross = np.clip(0.5 + amplification * (wm_set - wm_unset), 0, 1)

            axes[row][0].imshow(diff_set); axes[row][0].set_title(f"p{p_idx} (bit=1) - baseline ×{amplification:.0f}"); axes[row][0].axis("off")
            axes[row][1].imshow(diff_unset); axes[row][1].set_title(f"p{p_idx} (bit=0) - baseline ×{amplification:.0f}"); axes[row][1].axis("off")
            axes[row][2].imshow(cross); axes[row][2].set_title(f"p{p_idx} bit=1 - bit=0 ×{amplification:.0f}"); axes[row][2].axis("off")

        fig.suptitle(f"Slider {SLIDER_NAMES[s_idx]} difference maps")
        fig.tight_layout()
        out = out_dir / f"diff_slider_{s_idx}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(out)
        print(f"  wrote {out}")
    return saved


def _log_fft_magnitude(image: np.ndarray) -> np.ndarray:
    grey = image.mean(axis=-1)
    fft = np.fft.fftshift(np.fft.fft2(grey))
    mag = np.log1p(np.abs(fft))
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def fft_analysis(
    baseline_dir: str, images_dir: str, output_dir: str, metadata_path: str,
    prompt_indices: list[int] | None = None, slider_indices: list[int] | None = None,
) -> list[Path]:
    metadata = _load_metadata(metadata_path)
    index = _index_by_id_prompt(metadata)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompt_indices is None:
        prompt_indices = sorted({p for (_, p) in index})[:1]
    if slider_indices is None:
        slider_indices = list(range(8))

    saved = []
    for p_idx in prompt_indices:
        baseline_path = _baseline_path_for_prompt(baseline_dir, p_idx)
        if not baseline_path.exists():
            print(f"[fft_analysis] missing baseline for prompt {p_idx}, skipping")
            continue
        baseline = _open_rgb(baseline_path)
        baseline_fft = _log_fft_magnitude(baseline)

        rows = len(slider_indices)
        fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
        for row, s_idx in enumerate(slider_indices):
            set_entry, _ = _find_pair_for_slider(index, s_idx, p_idx)
            axes[row][0].imshow(baseline_fft, cmap="magma")
            axes[row][0].set_title("baseline FFT" if row == 0 else "baseline")
            axes[row][0].axis("off")
            if set_entry is None:
                for col in (1, 2):
                    axes[row][col].text(0.5, 0.5, "no sample", ha="center", va="center")
                    axes[row][col].axis("off")
                continue
            wm = _open_rgb(Path(images_dir) / set_entry["file"])
            if wm.shape != baseline.shape:
                wm = np.asarray(
                    Image.fromarray((wm * 255).astype(np.uint8)).resize(
                        (baseline.shape[1], baseline.shape[0])
                    )
                ) / 255.0
            wm_fft = _log_fft_magnitude(wm)
            diff = wm_fft - baseline_fft
            axes[row][1].imshow(wm_fft, cmap="magma"); axes[row][1].set_title(f"{SLIDER_NAMES[s_idx]} FFT"); axes[row][1].axis("off")
            axes[row][2].imshow(diff, cmap="seismic", vmin=-np.abs(diff).max() or 1e-6, vmax=np.abs(diff).max() or 1e-6)
            axes[row][2].set_title("FFT diff"); axes[row][2].axis("off")
        fig.suptitle(f"FFT magnitude (prompt {p_idx})")
        fig.tight_layout()
        out = out_dir / f"fft_prompt_{p_idx:02d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(out)
        print(f"  wrote {out}")
    return saved


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _resnet_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4
    if hasattr(model, "layer4"):
        return model.layer4
    raise AttributeError("Could not find ResNet target layer for Grad-CAM")


def gradcam_analysis(model, dataloader, output_dir: str, device, max_samples: int = 4) -> list[Path]:
    """Grad-CAM heatmap per bit, overlaid on a sample image.

    For each bit we find the strongest-activating sample in the loader, then
    save an 8-panel figure of class-activation maps. Works with any model
    exposing a (batch, 8) logits tensor and a ResNet-style ``layer4`` block.
    """
    from torchcam.methods import GradCAM
    from torchcam.utils import overlay_mask
    from PIL import Image as PILImage

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)

    target_layer = _resnet_target_layer(model)
    cam = GradCAM(model, target_layer=target_layer)

    samples = []
    for batch in dataloader:
        for i in range(batch["image"].shape[0]):
            samples.append(
                {
                    "image": batch["image"][i],
                    "bits": batch["bits"][i],
                    "filename": batch["filename"][i] if isinstance(batch["filename"], list) else batch["filename"][i],
                }
            )
            if len(samples) >= max_samples * 4:
                break
        if len(samples) >= max_samples * 4:
            break

    if not samples:
        print("[gradcam_analysis] no samples, skipping")
        cam.remove_hooks()
        return []

    saved = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for bit_idx in range(8):
        chosen = max(samples, key=lambda s: float(s["bits"][bit_idx].item()))
        img = chosen["image"].unsqueeze(0).to(device).requires_grad_(True)
        model.zero_grad()
        logits = model(img)
        score = logits[0, bit_idx]
        activation_maps = cam(class_idx=bit_idx, scores=logits)
        heatmap = activation_maps[0][0].detach().cpu().numpy()
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        rgb = _denormalize(chosen["image"])
        pil_rgb = PILImage.fromarray((rgb * 255).astype(np.uint8))
        pil_mask = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_rgb.size, PILImage.BILINEAR)
        overlay = overlay_mask(pil_rgb, pil_mask, alpha=0.5)

        ax = axes[bit_idx // 4][bit_idx % 4]
        ax.imshow(overlay)
        ax.set_title(f"{SLIDER_NAMES[bit_idx]}\nbit={int(chosen['bits'][bit_idx].item())} score={score.item():.2f}")
        ax.axis("off")

    fig.suptitle("Grad-CAM per bit (ResNet-50 layer4)")
    fig.tight_layout()
    out = out_dir / "gradcam_per_bit.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    cam.remove_hooks()
    saved.append(out)
    print(f"  wrote {out}")
    return saved


def _build_resnet_for_gradcam(checkpoint_path: str | None, device) -> nn.Module | None:
    """Load Person A's ResNet-50 if a checkpoint exists; else random init.

    Returns None if ``torchvision`` cannot construct it (defensive)."""
    model = get_model("resnet50", num_outputs=8, pretrained=False)
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            print(f"  loaded ResNet-50 from {checkpoint_path}")
        except Exception as exc:
            print(f"  could not load {checkpoint_path}: {exc}; using random init")
    else:
        print("  no ResNet-50 checkpoint found; using random init for Grad-CAM demo")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Signal analysis of LoRA watermark perturbations.")
    p.add_argument("--smoke", action="store_true", help="2 prompts × 2 sliders, low-res, CPU only.")
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--splits", type=str, default=None)
    p.add_argument("--baseline-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=str(DECODING_ROOT / "results" / "figures"))
    p.add_argument("--checkpoint", type=str, default=str(DECODING_ROOT / "checkpoints" / "best_model.pth"))
    p.add_argument("--smoke-root", type=str, default=str(DECODING_ROOT / ".smoke"))
    p.add_argument("--skip-gradcam", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    defaults = default_data_paths(DECODING_ROOT.parent)
    metadata, images, splits, baseline, is_smoke = resolve_paths(
        args,
        default_metadata=defaults["metadata"],
        default_images=defaults["images"],
        default_splits=defaults["splits"],
        default_baseline=defaults["baseline"],
    )
    if baseline is None or not Path(baseline).exists():
        # Real data may not have a baseline folder mirrored locally; fall back to smoke baselines.
        fx = ensure_smoke_fixture(root=args.smoke_root)
        baseline = str(fx.baseline)
        print(f"  no baseline dir at default location; using {baseline}")
    if args.smoke:
        prompt_indices = [0, 1]
        slider_indices = [0, 1]
    else:
        prompt_indices = None
        slider_indices = None

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=== signal_analysis ===")
    print(f"  metadata={metadata}\n  images={images}\n  baseline={baseline}\n  out={out_dir}\n  smoke={is_smoke}")

    print("\n[1/4] visualize_comparison")
    visualize_comparison(baseline, images, out_dir, metadata, prompt_indices, slider_indices)
    print("\n[2/4] difference_images")
    difference_images(baseline, images, out_dir, metadata,
                      prompt_indices=(prompt_indices if is_smoke else None),
                      slider_indices=slider_indices)
    print("\n[3/4] fft_analysis")
    fft_analysis(baseline, images, out_dir, metadata, prompt_indices, slider_indices)

    if args.skip_gradcam:
        print("\n[4/4] gradcam_analysis (skipped)")
        return

    print("\n[4/4] gradcam_analysis")
    device = pick_device(force_cpu=is_smoke)
    image_size = 96 if is_smoke else 1024
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = WatermarkDataset(metadata_path=metadata, image_dir=images, transform=transform)
    from torch.utils.data import DataLoader, Subset
    n = min(16 if is_smoke else 32, len(dataset))
    loader = DataLoader(Subset(dataset, list(range(n))), batch_size=4, shuffle=False)
    model = _build_resnet_for_gradcam(args.checkpoint, device)
    gradcam_analysis(model, loader, out_dir, device, max_samples=4)


if __name__ == "__main__":
    main()
