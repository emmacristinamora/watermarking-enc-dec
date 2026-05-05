"""Helpers shared by the decoding pipeline scripts.

Three responsibilities:

1. ``pick_device(force_cpu)`` — the canonical CUDA → MPS → CPU fallback.
2. ``default_data_paths(repo_root)`` — points at the canonical dataset
   location (``watermark_encoding/data/{images,baseline,metadata.json}``) with
   a fallback to ``encoding/data/metadata.json`` for environments where only
   the version-controlled label copy is present.
3. ``ensure_smoke_fixture(...)`` — synthesizes a tiny self-contained dataset
   (64 watermarked images + 10 baselines + metadata + splits) so every script
   can be smoke-tested locally without the full 2560-image SDXL output. The
   fixture is deterministic and idempotent: re-running just returns the cached
   paths.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROMPTS = [
    "a mountain landscape at sunset",
    "a beach with calm waves",
    "a forest path in autumn",
    "a snowy village at night",
    "a city street at noon",
    "a modern kitchen interior",
    "a field of flowers in sunlight",
    "a lighthouse on a rocky coast",
    "a desert landscape at dawn",
    "a cobblestone street in a old town",
]


@dataclass
class FixturePaths:
    root: Path
    metadata: Path
    images: Path
    baseline: Path
    splits: Path


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _render_image(rng: np.random.Generator, bits, prompt_idx: int, size: int = 64) -> Image.Image:
    """Synthesize a 'watermarked' RGB image whose statistics depend on bits.

    Each bit nudges a different image statistic so that a downstream classifier
    can actually learn to recover the bits even on the smoke fixture. The
    perturbations stack onto a per-prompt base pattern shared across IDs.
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / float(size)
    base = np.stack(
        [
            0.5 + 0.4 * np.sin(2 * np.pi * (xx + 0.1 * prompt_idx)),
            0.5 + 0.4 * np.sin(2 * np.pi * (yy + 0.2 * prompt_idx)),
            0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy + 0.3 * prompt_idx)),
        ],
        axis=-1,
    )
    base = base + 0.05 * rng.standard_normal(base.shape)

    b = [int(v) for v in bits]
    sign = lambda i: 1.0 if b[i] == 1 else -1.0
    base[..., 0] += 0.10 * sign(0)
    base[..., 2] -= 0.10 * sign(0)
    if b[1] == 1:
        base = 0.5 * (base + np.roll(base, 1, axis=0))
    if b[2] == 1:
        base = base + 0.05 * rng.standard_normal(base.shape)
    base += 0.10 * sign(3)
    mean = base.mean(axis=(0, 1), keepdims=True)
    base = mean + (1.0 + 0.30 * sign(4)) * (base - mean)
    grey = base.mean(axis=-1, keepdims=True)
    base = grey + (1.0 + 0.30 * sign(5)) * (base - grey)
    if b[6] == 1:
        stripe = 0.05 * np.sin(20 * np.pi * xx)[..., None]
        base = base + stripe
    base[..., 0] += 0.05 * sign(7)
    base[..., 2] -= 0.05 * sign(7)

    arr = np.clip(base, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _render_baseline(prompt_idx: int, size: int = 64) -> Image.Image:
    rng = _seeded_rng(1000 + prompt_idx)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    base = np.stack(
        [
            0.5 + 0.4 * np.sin(2 * np.pi * (xx + 0.1 * prompt_idx)),
            0.5 + 0.4 * np.sin(2 * np.pi * (yy + 0.2 * prompt_idx)),
            0.5 + 0.4 * np.sin(2 * np.pi * (xx + yy + 0.3 * prompt_idx)),
        ],
        axis=-1,
    )
    base = base + 0.02 * rng.standard_normal(base.shape)
    arr = np.clip(base, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def ensure_smoke_fixture(
    root: str | os.PathLike = ".smoke",
    num_ids: int = 32,
    prompts_per_id: int = 2,
    image_size: int = 64,
    seed: int = 0,
) -> FixturePaths:
    """Create (or reuse) a tiny synthetic dataset for smoke tests.

    Total images = num_ids * prompts_per_id (default 32*2 = 64). Bits are
    derived from the lower 8 bits of the ID so the model can learn a real
    mapping. Splits use 80/10/10 with at least one sample per split.
    """
    root = Path(root).resolve()
    images_dir = root / "images"
    baseline_dir = root / "baseline"
    metadata_path = root / "metadata.json"
    splits_path = root / "splits.json"

    expected = num_ids * prompts_per_id
    already_built = (
        metadata_path.exists()
        and splits_path.exists()
        and images_dir.exists()
        and len(list(images_dir.glob("*.png"))) == expected
        and baseline_dir.exists()
        and len(list(baseline_dir.glob("*.png"))) >= prompts_per_id
    )
    if already_built:
        return FixturePaths(root, metadata_path, images_dir, baseline_dir, splits_path)

    images_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for id_int in range(num_ids):
        bits = [int(b) for b in f"{id_int % 256:08b}"]
        for p_idx in range(prompts_per_id):
            rng = _seeded_rng(seed + id_int * 100 + p_idx)
            img = _render_image(rng, bits, p_idx, size=image_size)
            filename = f"id{id_int:03d}_p{p_idx:02d}.png"
            img.save(images_dir / filename)
            metadata.append(
                {
                    "file": filename,
                    "id_int": id_int,
                    "bits": bits,
                    "prompt": PROMPTS[p_idx % len(PROMPTS)],
                }
            )

    for p_idx in range(max(prompts_per_id, 10)):
        img = _render_baseline(p_idx, size=image_size)
        img.save(baseline_dir / f"baseline_p{p_idx:02d}.png")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    rng = _seeded_rng(seed)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)
    n = len(indices)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    test = indices[:n_test]
    val = indices[n_test : n_test + n_val]
    train = indices[n_test + n_val :]
    with open(splits_path, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)

    return FixturePaths(root, metadata_path, images_dir, baseline_dir, splits_path)


def default_data_paths(repo_root: str | os.PathLike) -> dict[str, str]:
    """Canonical dataset locations.

    The encoder writes to ``watermark_encoding/data/`` (see
    ``encoding/scripts/generate_dataset.py``), and that path is what gets
    synced to/from S3. ``encoding/data/metadata.json`` is a version-controlled
    copy of the labels used as a fallback when only the labels are present
    locally (e.g. a fresh clone before any S3 download).
    """
    repo_root = Path(repo_root)
    canonical_metadata = repo_root / "watermark_encoding" / "data" / "metadata.json"
    fallback_metadata = repo_root / "encoding" / "data" / "metadata.json"
    metadata = canonical_metadata if canonical_metadata.exists() else fallback_metadata
    return {
        "metadata": str(metadata),
        "images": str(repo_root / "watermark_encoding" / "data" / "images"),
        "baseline": str(repo_root / "watermark_encoding" / "data" / "baseline"),
        "splits": str(repo_root / "decoding" / "data" / "splits.json"),
    }


def resolve_paths(args, default_metadata: str, default_images: str, default_splits: str, default_baseline: str | None = None):
    """Resolve data paths from CLI args, falling back to smoke fixture if needed.

    Returns a tuple ``(metadata, images, splits, baseline_or_None, is_smoke)``.
    """
    is_smoke = bool(getattr(args, "smoke", False))

    if is_smoke and not (getattr(args, "metadata", None) and getattr(args, "images", None) and getattr(args, "splits", None)):
        smoke_root = getattr(args, "smoke_root", None) or ".smoke"
        fx = ensure_smoke_fixture(root=smoke_root)
        return (
            str(fx.metadata),
            str(fx.images),
            str(fx.splits),
            str(fx.baseline),
            True,
        )

    metadata = getattr(args, "metadata", None) or default_metadata
    images = getattr(args, "images", None) or default_images
    splits = getattr(args, "splits", None) or default_splits
    baseline = getattr(args, "baseline_dir", None) or default_baseline
    return metadata, images, splits, baseline, is_smoke


if __name__ == "__main__":
    fx = ensure_smoke_fixture()
    print(f"fixture root: {fx.root}")
    print(f"images: {len(list(fx.images.glob('*.png')))}")
    print(f"baselines: {len(list(fx.baseline.glob('*.png')))}")
    with open(fx.splits) as f:
        s = json.load(f)
    print(f"split sizes: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")
