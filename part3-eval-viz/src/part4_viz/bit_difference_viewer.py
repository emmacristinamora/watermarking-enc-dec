#!/usr/bin/env python3
"""Generate per-prompt bit-difference figures with similarity metrics.

By default this script resolves local workspace paths the same way as other
`decoding/scripts/*.py` tools, so it can run without explicit path flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
DECODING_ROOT = HERE.parents[1]
if str(DECODING_ROOT) not in sys.path:
    sys.path.insert(0, str(DECODING_ROOT))

from _smoke_utils import default_data_paths, ensure_smoke_fixture  # type: ignore  # noqa: E402

SLIDER_NAMES = [
    "S1 warm/cool", "S2 sharp/soft", "S3 grainy/clean", "S4 bright/dark",
    "S5 contrast", "S6 saturation", "S7 detail", "S8 vintage/modern",
]


def _open_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _baseline_path_for_prompt(baseline_dir: Path, prompt_idx: int) -> Path:
    candidates = [
        baseline_dir / f"prompt_{prompt_idx:02d}" / "baseline.png",
        baseline_dir / f"baseline_p{prompt_idx:02d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return preferred modern path for clearer error text if none exists.
    return candidates[0]


def _prompt_idx_from_entry(entry: dict[str, Any]) -> int:
    if "prompt_idx" in entry:
        return int(entry["prompt_idx"])

    # Legacy metadata format: derive prompt index from filename like id012_p03.png
    f = str(entry.get("file", ""))
    if "_p" in f:
        tail = f.split("_p", 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)

    raise KeyError("Could not resolve prompt_idx (missing prompt_idx and parseable file pattern)")


def _entry_id(entry: dict[str, Any]) -> str:
    # Newer metadata has `id`; legacy metadata has `id_int` only.
    if "id" in entry:
        return str(entry["id"])
    if "id_int" in entry:
        return f"id{int(entry['id_int']):03d}"
    raise KeyError("Metadata entry missing both 'id' and 'id_int'")


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected metadata list in {metadata_path}")
    return data


def _index_by_id_prompt(metadata: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for e in metadata:
        out[(_entry_id(e), _prompt_idx_from_entry(e))] = e
    return out


def _find_pair_for_slider(index: dict[tuple[str, int], dict[str, Any]], slider_idx: int, prompt_idx: int):
    key_suffix = f"_s{slider_idx}_"
    candidates = [e for (k_id, p), e in index.items() if p == prompt_idx and key_suffix in k_id]
    set_entry = next((e for e in candidates if int(e["bits"][slider_idx]) == 1), None)
    unset_entry = next((e for e in candidates if int(e["bits"][slider_idx]) == 0), None)
    return set_entry, unset_entry


def _find_pair_for_slider_from_prompt_entries(entries: list[dict[str, Any]], slider_idx: int):
    """Fallback pairing for legacy metadata without single-slider IDs.

    Finds two samples that differ only at slider_idx (all other bits equal),
    one with bit=1 and one with bit=0.
    """
    if not entries:
        return None, None

    by_key: dict[tuple[int, ...], dict[int, dict[str, Any]]] = {}
    for e in entries:
        bits = tuple(int(b) for b in e["bits"])
        key = bits[:slider_idx] + bits[slider_idx + 1 :]
        bucket = by_key.setdefault(key, {})
        bucket[bits[slider_idx]] = e

    for bucket in by_key.values():
        if 1 in bucket and 0 in bucket:
            return bucket[1], bucket[0]
    return None, None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12
    return float(np.dot(av, bv) / denom)


def _metrics(img_a: np.ndarray, img_b: np.ndarray) -> dict[str, float]:
    diff = img_a - img_b
    mse = float(np.mean(diff ** 2))
    mad = float(np.mean(np.abs(diff)))
    cos = _cosine_similarity(img_a, img_b)
    return {"mse": mse, "mad": mad, "cosine": cos}


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    defaults = default_data_paths(DECODING_ROOT.parent)
    metadata = Path(args.metadata) if args.metadata else Path(defaults["metadata"])
    images = Path(args.images) if args.images else Path(defaults["images"])
    baseline = Path(args.baseline_dir) if args.baseline_dir else Path(defaults["baseline"])
    output = Path(args.output_dir) if args.output_dir else (DECODING_ROOT / "results" / "figures")

    if not baseline.exists():
        fx = ensure_smoke_fixture(root=str(DECODING_ROOT / ".smoke"))
        baseline = Path(fx.baseline)
        print(f"[info] baseline dir not found, using smoke baseline: {baseline}")

    return metadata, images, baseline, output


def generate_bit_difference_figures(
    metadata_path: Path,
    images_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    prompt_indices: list[int] | None = None,
    amplification: float = 8.0,
) -> list[Path]:
    metadata = _load_metadata(metadata_path)
    index = _index_by_id_prompt(metadata)
    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for e in metadata:
        by_prompt.setdefault(_prompt_idx_from_entry(e), []).append(e)

    output_dir.mkdir(parents=True, exist_ok=True)

    if prompt_indices is None:
        prompt_indices = sorted({_prompt_idx_from_entry(e) for e in metadata})

    written: list[Path] = []

    for p_idx in prompt_indices:
        base_path = _baseline_path_for_prompt(baseline_dir, p_idx)
        if not base_path.exists():
            print(f"[skip] missing baseline for prompt {p_idx}: {base_path}")
            continue

        baseline = _open_rgb(base_path)

        fig, axes = plt.subplots(8, 4, figsize=(16, 32), squeeze=False)
        fig.suptitle(f"Prompt {p_idx}: per-bit similarity and difference localization", y=0.995)

        prompt_metrics: dict[str, dict[str, float]] = {}

        for bit_idx in range(8):
            set_entry, unset_entry = _find_pair_for_slider(index, bit_idx, p_idx)
            if set_entry is None or unset_entry is None:
                set_entry, unset_entry = _find_pair_for_slider_from_prompt_entries(
                    by_prompt.get(p_idx, []), bit_idx
                )
            row = axes[bit_idx]

            row[0].imshow(baseline)
            row[0].set_title("baseline")
            row[0].axis("off")

            if set_entry is None or unset_entry is None:
                for c in [1, 2, 3]:
                    row[c].text(0.5, 0.5, "missing pair", ha="center", va="center")
                    row[c].axis("off")
                continue

            img_set = _open_rgb(images_dir / set_entry["file"])
            img_unset = _open_rgb(images_dir / unset_entry["file"])

            mm = _metrics(img_set, img_unset)
            prompt_metrics[f"bit_{bit_idx}"] = mm

            row[1].imshow(img_set)
            row[1].set_title(f"{SLIDER_NAMES[bit_idx]} bit=1")
            row[1].axis("off")

            row[2].imshow(img_unset)
            row[2].set_title(f"{SLIDER_NAMES[bit_idx]} bit=0")
            row[2].axis("off")

            delta = np.mean(np.abs(img_set - img_unset), axis=2)
            delta = np.clip(delta * amplification, 0.0, 1.0)
            im = row[3].imshow(delta, cmap="inferno", vmin=0.0, vmax=1.0)
            row[3].set_title(
                f"abs diff ×{amplification:g}\n"
                f"cos={mm['cosine']:.5f} mse={mm['mse']:.6f} mad={mm['mad']:.6f}"
            )
            row[3].axis("off")
            fig.colorbar(im, ax=row[3], fraction=0.046, pad=0.02)

        fig.tight_layout(rect=[0, 0, 1, 0.982])
        out_png = output_dir / f"bitdiff_prompt_{p_idx:02d}.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)

        out_json = output_dir / f"bitdiff_prompt_{p_idx:02d}.json"
        out_json.write_text(json.dumps(prompt_metrics, indent=2), encoding="utf-8")

        written.extend([out_png, out_json])
        print(f"wrote {out_png}")
        print(f"wrote {out_json}")

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-prompt bit difference visualizer")
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--baseline-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=str(DECODING_ROOT / "results" / "figures"))
    p.add_argument("--prompts", type=int, nargs="*", default=None, help="Optional prompt indices (e.g. 0 1 2)")
    p.add_argument("--amplification", type=float, default=8.0, help="Multiplier for abs-diff heatmaps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata, images, baseline, out = _resolve_paths(args)
    print(f"metadata={metadata}\nimages={images}\nbaseline={baseline}\nout={out}")
    generate_bit_difference_figures(
        metadata_path=metadata,
        images_dir=images,
        baseline_dir=baseline,
        output_dir=out,
        prompt_indices=args.prompts,
        amplification=args.amplification,
    )
