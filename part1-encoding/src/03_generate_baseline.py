# part1-encoding/src/03_generate_baseline.py

"""
Generate unwatermarked baseline images for signal analysis.

Loads SDXL without any LoRA adapters and generates one image per prompt
(10 total) using the same seeds as the decoder dataset. These baselines
are used in part3 for visual comparison, pixel-difference analysis, and
FFT signal inspection.

Usage:
    python part1-encoding/src/03_generate_baseline.py \
        [--config part1-encoding/config/generate_decoder_data.yaml]
"""


# === IMPORTS ===

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import DiffusionPipeline


# === HELPERS ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate unwatermarked baseline images.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("part1-encoding/config/generate_decoder_data.yaml"),
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)["generate_decoder_dataset"]


# === MAIN ===

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    inf = cfg["inference"]
    prompts = cfg["prompts"]
    baseline_dir = Path(cfg["output"]["baseline_dir"])
    baseline_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, model_cfg["dtype"])
    device = model_cfg["device"]

    print("loading SDXL pipeline...")
    pipe = DiffusionPipeline.from_pretrained(model_cfg["name_or_path"], torch_dtype=dtype).to(device)

    print(f"generating {len(prompts)} baseline images → {baseline_dir}\n")
    for prompt_idx, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(inf["seed"] + prompt_idx)
        image = pipe(
            prompt,
            num_inference_steps=inf["num_steps"],
            generator=gen,
        ).images[0]
        filename = f"baseline_p{prompt_idx:02d}.png"
        image.save(baseline_dir / filename)
        print(f"  saved {filename}")

    print(f"\ndone. {len(prompts)} baseline images saved to {baseline_dir}")


if __name__ == "__main__":
    main()
