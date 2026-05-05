# part1-encoding/src/02_generate_decoder_dataset.py

"""
Generate the watermarked image dataset and create a stratified train/val/test split.

Loads SDXL and 8 LoRA style-slider adapters, then generates 256 IDs × 10 prompts
= 2560 images. Images are written directly into train/, val/, or test/ based on a
per-ID stratified split (8/1/1 prompts per ID). Outputs:
  - metadata.json   JSON array, one record per image: file, id_int, bits, prompt
  - splits.json     {"train": [...indices...], "val": [...], "test": [...]}
  - train/          2048 watermarked images
  - val/            256 watermarked images
  - test/           256 watermarked images

Usage:
    python part1-encoding/src/02_generate_decoder_dataset.py \
        [--config part1-encoding/config/generate_decoder_data.yaml] \
        [--sanity-only]
"""


# === IMPORTS ===

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from diffusers import DiffusionPipeline


# === HELPERS ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate watermarked decoder dataset.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("part1-encoding/config/generate_decoder_data.yaml"),
    )
    p.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run sanity checks only; skip full dataset generation.",
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)["generate_decoder_dataset"]


def resolve_lora_paths(cfg: dict[str, Any]) -> list[Path]:
    """
    Build the safetensors path for each of the 8 LoRA adapters.

    Args:
        cfg: the generate_decoder_dataset config dict

    Returns:
        List of 8 Paths in adapter order (S1 → S8).
    """
    loras = cfg["loras"]
    base = Path(loras["dir"])
    suffix = "last" if loras["checkpoint"] == "last" else f"{loras['checkpoint']}steps"
    alpha, rank, method = loras["alpha"], loras["rank"], loras["training_method"]

    paths = []
    for i in range(1, cfg["watermark"]["num_bits"] + 1):
        folder_name = f"watermark_s{i}_alpha{alpha}_rank{rank}_{method}"
        file_name = f"{folder_name}_{suffix}.safetensors"
        lora_path = base / folder_name / file_name
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
        paths.append(lora_path)
    return paths


def compute_split(
    num_ids: int,
    num_prompts: int,
    train_per_id: int,
    val_per_id: int,
    seed: int,
) -> dict[tuple[int, int], str]:
    """
    Assign each (id_int, prompt_idx) pair to 'train', 'val', or 'test'.

    Stratified by ID: for each ID the 10 prompt indices are shuffled, the
    first train_per_id go to train, the next val_per_id to val, the rest to test.

    Args:
        num_ids: number of watermark IDs (256)
        num_prompts: number of prompts per ID (10)
        train_per_id: prompts assigned to train per ID (8)
        val_per_id: prompts assigned to val per ID (1)
        seed: RNG seed for reproducibility

    Returns:
        Dict mapping (id_int, prompt_idx) → "train" | "val" | "test".
    """
    rng = np.random.default_rng(seed)
    split_map: dict[tuple[int, int], str] = {}
    for id_int in range(num_ids):
        indices = list(range(num_prompts))
        rng.shuffle(indices)
        for rank, p_idx in enumerate(indices):
            if rank < train_per_id:
                split = "train"
            elif rank < train_per_id + val_per_id:
                split = "val"
            else:
                split = "test"
            split_map[(id_int, p_idx)] = split
    return split_map


def make_step_callback(
    target_alphas: list[float],
    adapter_names: list[str],
    activate_at_step: int,
) -> Callable:
    """Return a pipeline callback that injects slider weights at activate_at_step."""
    def callback(pipe, step: int, _timestep, kwargs: dict) -> dict:
        if step == activate_at_step:
            pipe.set_adapters(adapter_names, adapter_weights=target_alphas)
        return kwargs
    return callback


def save_json(data: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_sanity_checks(
    pipe: DiffusionPipeline,
    lora_paths: list[Path],
    adapter_names: list[str],
    cfg: dict[str, Any],
) -> None:
    """
    Validate setup before committing to the full generation run.

    Checks: (1) all LoRA files exist, (2) two test images generate and save
    without error, (3) bit-encoding logic is correct for boundary IDs.
    """
    wm = cfg["watermark"]
    inf = cfg["inference"]
    scale = wm["scale"]
    activate_at = wm["activate_at_step"]
    prompts = cfg["prompts"]
    device = cfg["model"]["device"]

    print("check 1: LoRA files exist...")
    for p in lora_paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
    print("  passed")

    print("check 2: test generation (IDs 0 and 255)...")
    tmp_dir = Path(cfg["output"]["dataset_dir"]) / ".sanity"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for test_id in [0, 255]:
        bits = [int(b) for b in f"{test_id:08b}"]
        alphas = [scale if b == 1 else -scale for b in bits]
        pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)
        cb = make_step_callback(alphas, adapter_names, activate_at)
        gen = torch.Generator(device=device).manual_seed(inf["seed"])
        img = pipe(
            prompts[0],
            num_inference_steps=inf["num_steps"],
            generator=gen,
            callback_on_step_end=cb,
        ).images[0]
        out = tmp_dir / f"sanity_id{test_id:03d}.png"
        img.save(out)
        assert out.stat().st_size > 0, f"empty test image: {out}"
    print("  passed")

    print("check 3: bit-encoding logic...")
    for test_id in [0, 127, 255]:
        bits = [int(b) for b in f"{test_id:08b}"]
        assert len(bits) == 8 and all(b in (0, 1) for b in bits)
        alphas = [scale if b == 1 else -scale for b in bits]
        assert all(a in (scale, -scale) for a in alphas)
    print("  passed\n")


# === MAIN ===

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_dir = Path(cfg["output"]["dataset_dir"])
    split_dirs = {
        "train": dataset_dir / "train",
        "val":   dataset_dir / "val",
        "test":  dataset_dir / "test",
    }
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / cfg["output"]["metadata_file"]
    splits_path = dataset_dir / cfg["output"]["splits_file"]

    loras = cfg["loras"]
    adapter_names = [f"{loras['adapter_prefix']}{i}" for i in range(1, cfg["watermark"]["num_bits"] + 1)]
    lora_paths = resolve_lora_paths(cfg)

    model_cfg = cfg["model"]
    dtype = getattr(torch, model_cfg["dtype"])
    device = model_cfg["device"]

    print("loading SDXL pipeline...")
    pipe = DiffusionPipeline.from_pretrained(model_cfg["name_or_path"], torch_dtype=dtype).to(device)

    print("loading LoRA adapters...")
    for path, name in zip(lora_paths, adapter_names):
        pipe.load_lora_weights(str(path), adapter_name=name)
        print(f"  {name} loaded")

    print("\nrunning sanity checks...")
    run_sanity_checks(pipe, lora_paths, adapter_names, cfg)

    if args.sanity_only:
        print("sanity-only run complete.")
        return

    split_cfg = cfg["split"]
    prompts = cfg["prompts"]
    split_map = compute_split(
        num_ids=cfg["watermark"]["num_ids"],
        num_prompts=len(prompts),
        train_per_id=split_cfg["train_prompts_per_id"],
        val_per_id=split_cfg["val_prompts_per_id"],
        seed=split_cfg["seed"],
    )

    wm = cfg["watermark"]
    inf = cfg["inference"]
    scale = wm["scale"]
    activate_at = wm["activate_at_step"]
    total = wm["num_ids"] * len(prompts)

    metadata: list[dict[str, Any]] = []
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    print(f"generating {total} images → {dataset_dir}\n")
    count = 0
    for id_int in range(wm["num_ids"]):
        bits = [int(b) for b in f"{id_int:08b}"]
        alphas = [scale if b == 1 else -scale for b in bits]

        for prompt_idx, prompt in enumerate(prompts):
            split = split_map[(id_int, prompt_idx)]

            pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)
            cb = make_step_callback(alphas, adapter_names, activate_at)
            gen = torch.Generator(device=device).manual_seed(inf["seed"] + prompt_idx)

            image = pipe(
                prompt,
                num_inference_steps=inf["num_steps"],
                generator=gen,
                callback_on_step_end=cb,
            ).images[0]

            filename = f"id{id_int:03d}_p{prompt_idx:02d}.png"
            image.save(split_dirs[split] / filename)

            idx = len(metadata)
            metadata.append({"file": filename, "id_int": id_int, "bits": bits, "prompt": prompt})
            split_indices[split].append(idx)

            count += 1
            if count % 256 == 0:
                print(f"  {count}/{total} ({100 * count // total}%)")

    save_json(metadata, metadata_path)
    save_json(split_indices, splits_path)

    train_n, val_n, test_n = (len(split_indices[k]) for k in ("train", "val", "test"))
    print(f"\ndone. train={train_n}  val={val_n}  test={test_n}")
    print(f"metadata : {metadata_path}")
    print(f"splits   : {splits_path}")


if __name__ == "__main__":
    main()
