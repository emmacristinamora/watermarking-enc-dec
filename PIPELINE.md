# Decoding pipeline — end-to-end flow

This document walks through every step of the decoder pipeline as it runs at the **canonical native resolution of 1024 × 1024**, and where each instruction from the original task brief maps onto a specific script and CLI flag.

---

## 0 · The watermark and the data

- **Encoder.** SDXL base (`stabilityai/stable-diffusion-xl-base-1.0`), 30 inference steps. From step 25 onward, 8 LoRA "style sliders" S1–S8 are activated with weights `+0.3` (bit = 1) or `-0.3` (bit = 0). The 8 weights together encode a single 8-bit ID in `[0, 255]`. Source: [encoding/scripts/generate_dataset.py](encoding/scripts/generate_dataset.py).
- **Dataset.** 256 IDs × 10 prompts = **2560 PNGs at 1024 × 1024**, plus 10 unwatermarked baselines (one per prompt). All files live in `watermark_encoding/data/{images,baseline}/`. Labels are in `watermark_encoding/data/metadata.json` (and a version-controlled mirror at [encoding/data/metadata.json](encoding/data/metadata.json)).
- **Splits.** Stratified 80/10/10 by ID — every ID appears in train (8 prompts), val (1 prompt), and test (1 prompt). 2048 / 256 / 256 samples. Source: [decoding/data/splits.json](decoding/data/splits.json), generator [decoding/scripts/split_data.py](decoding/scripts/split_data.py).
- **Decoder framing.** 8 independent binary classifications, evaluated by per-bit accuracy and exact-match (all 8 bits correct simultaneously).

---

## 1 · Workspace setup on the GPU VM

The Runpod / cloud workflow assumes three folders under `/workspace/`:

```
/workspace/
├── computer-vision-watermarking/   ← git clone
├── watermark_encoding/             ← from S3 (data/ + models/)
└── decoding/                       ← from S3 (checkpoints/ + results/)
```

Run [decoding/scripts/setup_workspace.sh](decoding/scripts/setup_workspace.sh) to merge the two S3 staging folders into the cloned repo:

```bash
bash computer-vision-watermarking/decoding/scripts/setup_workspace.sh --install-deps
```

After this, the data is at `computer-vision-watermarking/watermark_encoding/data/` and any pre-existing checkpoints are at `computer-vision-watermarking/decoding/checkpoints/`. Code under `decoding/{src,scripts,configs}/` is **never overwritten** by the merge — those live in git.

---

## 2 · The three architectures

| ID | Architecture | File | Input | Forward returns |
| --- | --- | --- | --- | --- |
| A1 | Shared-backbone ResNet-50 (Person A) | [src/models/resnet.py](decoding/src/models/resnet.py) | flexible | `(batch, 8)` logits |
| A2 | 8 × ResNet-50, separate per-bit backbones | [src/model_separate.py](decoding/src/model_separate.py) | flexible | `(batch, 8)` probs (`forward`) or logits (`forward_logits`) |
| A3 | ViT-B/16 + 8 binary heads | [src/model_vit.py](decoding/src/model_vit.py) | **224 × 224 only** | `(batch, 8)` logits |

A1 and A2 train and evaluate at the dataset's **native 1024 × 1024**. A3's input size is hard-locked by `torchvision`'s `vision_transformer.py` and is the principal reason ViT-B/16 fails on this task — see §3.3 and §6.

---

## 3 · Training — the three runs

All three optimisers use the same hyperparameters (same as Person A's published baseline):

| Hyperparameter | Value |
| --- | --- |
| Loss | `BCEWithLogitsLoss` |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Schedule | Cosine annealing over `num_epochs` |
| Batch size | 4 |
| Epochs | 25 (separate / ViT) — 30 (Person A's baseline) |
| Pretrained backbone | ImageNet (`torchvision` default weights) |
| Image transform | `Resize((H, H))` → `ToTensor` → ImageNet `Normalize` |

The only knob that differs across runs is the input resolution `H`.

### 3.1 Shared-backbone ResNet-50 (Person A's baseline) — 1024 × 1024

```bash
python scripts/train.py --config configs/baseline_resnet50.yaml
```

Already trained, already evaluated. Reference numbers from [results/baseline_resnet50.md](decoding/results/baseline_resnet50.md): val 93.90 % mean / 62.11 % exact, test 94.04 % mean / 60.94 % exact.

### 3.2 Eight separate ResNet-50s — **1024 × 1024**

```bash
python scripts/train_separate.py \
    --epochs 25 --batch-size 4 --image-size 1024
```

For each of 8 bits, a fresh ImageNet-pretrained ResNet-50 with a single-logit head is trained on that bit alone. The training loop in [scripts/train_separate.py](decoding/scripts/train_separate.py) reads `bits[:, bit_idx]` as the binary target, applies BCE-with-logits, runs the cosine schedule, and saves the best validation-accuracy checkpoint to `checkpoints/separate/bit_{i}_best.pth`. After all eight bits finish, the script reassembles the ensemble and reports mean / exact-match on the validation split.

### 3.3 ViT-B/16 (architecturally constrained)

```bash
python scripts/train_vit.py --epochs 25 --batch-size 16
```

Same loss / optimiser / schedule as above, but on a single ViT-B/16 backbone with 8 independent linear heads on the CLS token's 768-dim feature. Saves `checkpoints/vit_best.pth` whenever validation exact-match improves. Source: [scripts/train_vit.py](decoding/scripts/train_vit.py).

> **Expected outcome — a negative result.** `torchvision`'s ViT-B/16 hard-asserts a fixed input size, so the watermarked image must be downsampled before the forward pass. That ~4.6× linear downsample destroys most of the high-frequency LoRA perturbation, and the 16 × 16 patch tokenizer then bins what's left into receptive fields too coarse for bit-level identity. The model converges to ~ln 2 BCE loss (chance) and exact match stays around 1 %. ViT-B/16 is documented as **architecturally unfit** for this task; it remains in the comparison only as evidence that the watermark signal lives in the high-frequency band a fixed-input-size transformer cannot represent.

---

## 4 · Signal analysis

```bash
python scripts/signal_analysis.py
```

Source: [scripts/signal_analysis.py](decoding/scripts/signal_analysis.py). Operates on **full 1024 × 1024 images** (no resize) so the diagnostic figures faithfully represent the watermark perturbation. Four passes, all writing into `results/figures/`:

| Pass | Function | Output | Purpose |
| --- | --- | --- | --- |
| 1 | `visualize_comparison` | `comparison_prompt_{00..09}.png` | For each of the 10 prompts, an 8 × 3 grid: rows are sliders S1–S8, columns are baseline / bit=1 / bit=0. Pairs are matched on prompt + ID-with-only-this-bit-flipped so visual differences isolate that single slider. |
| 2 | `difference_images` | `diff_slider_{0..7}.png` | For each slider, three panels per prompt: (watermarked bit=1) − baseline, (bit=0) − baseline, and (bit=1) − (bit=0), each amplified ×10 and clipped to `[0, 1]`. Surfaces where each slider deposits its perturbation in pixel space. |
| 3 | `fft_analysis` | `fft_prompt_{00..N}.png` | Log-magnitude 2D FFT of baseline vs watermarked image, plus their difference (`seismic` colormap). Bands of energy at characteristic radii indicate the slider operates in mid- or high-frequency regions; flat differences indicate luminance shifts only. |
| 4 | `gradcam_analysis` | `gradcam_per_bit.png` | Grad-CAM (via `torchcam`) on Person A's ResNet-50 `layer4`, one panel per bit. Shows where the trained classifier attends when predicting each bit. |

Defaults: 10 prompts, all 8 sliders for passes 1 and 2; 1 prompt for pass 3 (FFT); strongest-activating sample per bit for pass 4. CLI flags let you narrow the prompt / slider sweep.

---

## 5 · Robustness evaluation

```bash
# Run once per architecture you trained
python scripts/robustness_eval.py --model resnet   --batch-size 4
python scripts/robustness_eval.py --model separate --batch-size 4
python scripts/robustness_eval.py --model vit      --batch-size 16   # ViT downsamples internally
```

Source: [scripts/robustness_eval.py](decoding/scripts/robustness_eval.py). Iterates the held-out test set (256 samples) through 6 attack channels and a clean baseline. Each attack operates on **denormalised image tensors at the model's native input resolution**, then re-normalises before the forward pass.

| Attack key | Implementation | Spec correspondence |
| --- | --- | --- |
| `clean` | identity | upper bound for that model |
| `jpeg_q90` | PIL JPEG round-trip at quality 90 | "JPEG compression at quality 90" |
| `jpeg_q75` | quality 75 | "quality 75" |
| `jpeg_q50` | quality 50 | "quality 50" |
| `resize_512` | `bicubic` downscale to 512 × 512, `bicubic` upscale back to 1024 × 1024 | The spec'd resize attack. Always evaluated at the dataset's native 1024. |
| `random_crop_75` | crop 75 % of area at random offset, resize back to input resolution | "random crop to 75 % area, resize back" |

For each attack, predictions are compared against ground-truth bits and three numbers are recorded: per-bit accuracy (8 numbers), mean across bits, exact-match. The script prints a formatted ASCII table to stdout, writes the full table to `results/robustness.json`, and saves two figures into `results/figures/`:

- `robustness_per_bit.png` — grouped bar chart, x = bit index, hue = attack.
- `robustness_jpeg_curve.png` — degradation curve, x = JPEG quality (with `clean` plotted at quality = 100), y = mean bit accuracy.

> The dataset is 1024 × 1024 native; all attacks are applied at that resolution. For ViT only, the attacked tensor is downsampled to ViT's required input size *just before* the forward pass, so the attack itself still runs at 1024.

---

## 6 · Architecture comparison

```bash
python scripts/compare_architectures.py --batch-size 4
```

Source: [scripts/compare_architectures.py](decoding/scripts/compare_architectures.py). Loads all three architectures from their canonical checkpoint paths:

| Architecture | Checkpoint |
| --- | --- |
| Shared ResNet-50 | `checkpoints/baseline_resnet50.pth` (or `best_model.pth` for the older Person A baseline) |
| 8 × separate ResNet-50 | `checkpoints/separate/bit_{0..7}_best.pth` |
| ViT-B/16 | `checkpoints/vit_best.pth` |

Runs each on the held-out **test split** (`splits.json` test indices) and writes:

- `results/architecture_comparison.md` — markdown table with per-bit accuracy + mean + exact-match for all three architectures, plus a "Checkpoints used" footer that flags any architecture loaded with random weights (clearly labelled `(random init)` if a `.pth` was missing).
- `results/figures/architecture_comparison.png` — grouped bar chart of per-bit accuracies.

> The script uses one global `--image-size` for the data loader (default **1024**). The ViT row is evaluated by downsampling each batch to ViT's required input size inside `_evaluate_logits`, so all three architectures are scored against the same set of 1024 × 1024 test images even though ViT physically sees a downsampled view.

---

## 7 · Final report

[results/evaluation_report_template.md](decoding/results/evaluation_report_template.md) is the paper-section scaffold. Replace each `TODO` with the matching number from `results/architecture_comparison.md`, `results/robustness*.json`, and the figures generated in §4. The discussion section already includes a limitations paragraph noting that diffusion-based purification attacks (Zhao et al., NeurIPS 2024) reduce detection of LoRA-style watermarks to under 5 % and are out of scope for this work.

---

## 8 · Saving back to S3

```bash
cd /workspace
bash computer-vision-watermarking/decoding/scripts/save_workspace.sh --strip-optimizer
```

[decoding/scripts/save_workspace.sh](decoding/scripts/save_workspace.sh) hardlinks the run outputs into a clean tree under `/workspace/{watermark_encoding,decoding}/`, ready to drag into S3 via the Runpod UI:

```
/workspace/decoding/
├── checkpoints/
│   ├── baseline_resnet50.pth
│   ├── vit_best.pth
│   ├── vit_best.summary.json
│   └── separate/
│       ├── bit_{0..7}_best.pth
│       └── training_summary.json
└── results/
    ├── architecture_comparison.md
    ├── baseline_resnet50.md
    ├── ablation*.md
    ├── robustness_*.json
    ├── evaluation_report_template.md
    └── figures/
        ├── comparison_prompt_*.png
        ├── diff_slider_*.png
        ├── fft_prompt_*.png
        ├── gradcam_per_bit.png
        ├── architecture_comparison.png
        └── robustness_*/  (per-model robustness plots)
```

Code under `decoding/{src,scripts,configs}/` is **never** included — that lives in git. `--strip-optimizer` halves the `.pth` sizes by dropping `optimizer_state_dict` (use it for archive uploads, skip if you intend to resume training from those checkpoints).

---

## 9 · End-to-end timing on a single RTX PRO 6000 Blackwell

| Stage | Approx. wallclock | Notes |
| --- | --- | --- |
| `setup_workspace.sh` | < 1 min | mostly disk shuffles; rsync at ~ 1 GB/s |
| `train_separate` @ 1024, 25 epochs, batch 4 | ~ 4 h | 8 backbones × 2048 train samples × 25 epochs |
| `train_vit`, 25 epochs, batch 16 | ~ 30 min | 1 backbone × 2048 × 25 (downsampled internally) |
| `signal_analysis` | ~ 1 min | mostly I/O + matplotlib |
| `robustness_eval` per model @ 1024 | ~ 5 min | 256 samples × 6 attacks; JPEG via PIL on CPU |
| `compare_architectures` @ 1024 | < 2 min | 256 samples × 3 models |
| `save_workspace.sh` | < 1 min | hardlinks |

Total walltime for a fresh end-to-end run: **~ 5 hours**, dominated by the 8-way separate training.

---

## 10 · Resolution invariant

The dataset is **1024 × 1024 native** end-to-end:

- ResNet-50 baseline trains and evaluates at 1024.
- 8 × separate ResNet-50 trains and evaluates at 1024.
- Robustness attacks (`apply_jpeg`, `apply_resize`, `apply_random_crop`) all run at 1024 — including the spec'd `1024 → 512 → 1024` resize.
- Signal analysis figures are computed on the full 1024 × 1024 source images.
- ViT-B/16 is the only architectural exception: `torchvision`'s implementation hard-asserts a fixed input size, so each batch is downsampled inside `_evaluate_logits` / `_predict` immediately before the ViT forward pass. The attack itself still happens at 1024; only the model's window is smaller.

Every script's `--image-size` flag now defaults to **1024**. There is no longer a 224-trained variant of ResNet or separate; the earlier 224 path was an error in our reading of the brief and has been removed.
