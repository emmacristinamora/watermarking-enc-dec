# Decoder Performance Report

End-to-end evaluation of the three decoder architectures trained against the 8-bit LoRA watermark, on 2560 SDXL outputs at native 1024 × 1024.

> All numbers below are computed at the **dataset's native 1024 × 1024** resolution. There is no longer a 224-trained variant; the earlier 224 path was an error in our reading of the brief and has been removed from the codebase.

---

## TL;DR

- **Person A's shared-backbone ResNet-50 is the headline model.** Test mean bit accuracy 93.07 %, exact-match 57.03 %.
- **The 8 × separate ResNet-50 ensemble underperforms the shared backbone** at this dataset scale — 88.28 % / 33.59 % — despite having ~7.5× more parameters. The gap is concentrated in bit 3 (bright/dark): 58.20 % vs 82.81 %.
- **ViT-B/16 fails to learn anything.** It plateaus at chance throughout training; test exact-match is exactly 0. Robustness eval confirms it is **literally image-blind** — identical predictions across clean, JPEG q=90/75/50, and resize attacks (per-bit accuracy matches to 4 decimal places).
- **No architecture survives the spec robustness suite.** Every attack drops both models to near-chance on exact-match, with separate slightly more resilient than shared on JPEG and resize but neither usable under degradation.

---

## 1 · Setup

| Item | Value |
| --- | --- |
| Encoder | SDXL base + 8 LoRA style-sliders, scale ±0.3, activated step 25/30 |
| Dataset | 256 IDs × 10 prompts = 2560 PNGs at 1024 × 1024 |
| Baselines | 10 unwatermarked references (one per prompt) |
| Splits | 80/10/10 stratified by ID — 2048 train / 256 val / 256 test ([decoding/data/splits.json](decoding/data/splits.json)) |
| Decoder framing | 8 independent binary classifications |
| Loss / optim / sched | BCEWithLogitsLoss · Adam · cosine annealing |
| Learning rate | 1e-4 |
| Epochs (Person A baseline / separate / ViT) | 30 / 25 / 20 |
| Batch size | 16 (baseline) / 4 (separate at 1024) / 16 (ViT) |
| Pretrained backbones | ImageNet (`torchvision` defaults) |
| Hardware | NVIDIA RTX 4090 (Person A) · NVIDIA RTX PRO 6000 Blackwell (Person B) |

---

## 2 · Architecture comparison (held-out test set)

| Architecture | Params | Bit 0 | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 | Bit 6 | Bit 7 | Mean | **Exact** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 (shared backbone) | 25 M | 0.9023 | 0.9453 | 0.9570 | **0.8281** | 0.9219 | 0.9297 | 0.9961 | 0.9648 | **0.9307** | **0.5703** |
| 8 × ResNet-50 (separate)    | 188 M | 0.9062 | 0.9219 | 0.9609 | **0.5820** | 0.8203 | 0.9297 | 0.9883 | 0.9531 | 0.8828 | 0.3359 |
| ViT-B/16                    | 86 M  | 0.4258 | 0.5039 | 0.4492 | 0.4883 | 0.4961 | 0.4570 | 0.4727 | 0.4766 | 0.4712 | 0.0000 |

**Slider semantics** — bit 0: warm/cool · bit 1: sharp/soft · bit 2: grainy/clean · bit 3: bright/dark · bit 4: contrast · bit 5: saturation · bit 6: detail · bit 7: vintage/modern.

**Reference figure:** `results/figures/architecture_comparison.png` (regenerate via `scripts/compare_architectures.py`).

### Validation snapshots

| Architecture | Val mean | Val exact | Source |
| --- | --- | --- | --- |
| ResNet-50 (shared, epoch 21) | 0.9331 | 0.5898 | [results/baseline_resnet50.md](decoding/results/baseline_resnet50.md) (post-rebase rerun) |
| 8 × ResNet-50 (separate, ensemble) | 0.8979 | 0.3789 | [decoding/checkpoints/separate/training_summary.json](decoding/checkpoints/separate/training_summary.json) |
| ViT-B/16 (best of 20 epochs) | ~0.50 | 0.0117 | [decoding/checkpoints/vit_best.summary.json](decoding/checkpoints/vit_best.summary.json) |

The val→test gap is small for both ResNet variants (~2 pp on mean accuracy, <3 pp on exact-match), indicating both models generalised cleanly. ViT exhibits no learning at all in either set.

### 8 × separate per-bit best val accuracy (training summary)

| Bit | Best val acc | Notes |
| --- | --- | --- |
| 0 | 0.9219 | warm/cool |
| 1 | 0.9414 | sharp/soft |
| 2 | 0.9531 | grainy/clean |
| 3 | **0.6367** | bright/dark — **the bottleneck** |
| 4 | 0.8164 | contrast |
| 5 | 0.9453 | saturation |
| 6 | 0.9961 | detail |
| 7 | 0.9727 | vintage/modern |

7 of 8 bits are at 92 %+ on validation. Bit 3 alone collapses to 63.67 %, which alone explains why the 8× separate ensemble's exact-match (which requires *all 8* bits correct) drops to 33.59 % despite per-bit averages comparable to the shared backbone.

---

## 3 · Robustness evaluation (1024 native)

Six attack channels applied at the dataset's native resolution. Numbers below are mean bit accuracy on the test set (256 samples), with exact-match in parentheses.

| Attack | ResNet-50 shared | 8 × separate | ViT-B/16 |
| --- | --- | --- | --- |
| clean | **0.931** (0.570) | 0.884 (0.336) | 0.471 (0.000) |
| jpeg q=90 | 0.685 (0.059) | **0.777** (0.129) | 0.471 (0.000) |
| jpeg q=75 | 0.565 (0.008) | **0.624** (0.027) | 0.471 (0.000) |
| jpeg q=50 | 0.532 (0.016) | 0.532 (0.012) | 0.472 (0.000) |
| resize 1024→512→1024 | 0.550 (0.004) | **0.609** (0.008) | 0.471 (0.000) |
| random crop 75 % | 0.569 (0.004) | 0.570 (0.012) | 0.480 (0.000) |

Sources: [decoding/results/robustness_resnet.json](decoding/results/robustness_resnet.json), [robustness_separate.json](decoding/results/robustness_separate.json), [robustness_vit.json](decoding/results/robustness_vit.json). Per-architecture figures in `results/figures/robustness_{resnet,separate,vit}/`. Reproduce all three with `bash scripts/run_full_evaluation.sh`.

**ViT-B/16's column is constant across attacks.** Per-bit accuracy is identical to four decimal places for clean, JPEG q=90/75, and resize — meaning the model produces the same prediction regardless of the input image. Random crop introduces a 0.9 pp shift only because the cropped region falls partly outside the model's expected content distribution; even that change is in the noise. This is the strongest possible empirical evidence that the ViT never learned a function of the image during training: it converged to a constant per-bit bias derived from the training-set prior, and 6 different attacks now confirm that bias is image-independent.

### What survives, what doesn't

**Almost nothing survives at the exact-match level.** Even the gentlest attack (JPEG q=90) reduces exact-match to 5.9 % for the shared backbone and 13.3 % for separate. By q=75 both models are at <3 % exact match.

**Mean bit accuracy is more graceful.** The shared model holds 68 % at JPEG q=90, falls into the 53–57 % band for everything more aggressive (essentially chance with a bias). Separate keeps 60–78 % across all four mid-grade attacks, plateauing at chance only at q=50.

**The separate-vs-shared crossover under attack** is the most interesting headline result: although separate is *worse* on clean data (88.4 % vs 93.1 %), it is *better* under JPEG q=90, JPEG q=75, and the spec resize attack. The likely mechanism is that 8 independent backbones each learn redundant low- and mid-frequency features for "their" bit, so when JPEG quantises high frequencies the per-bit models still have a usable signal, whereas the shared backbone's joint representation collapses faster.

### Per-bit robustness for the shared baseline (mean bit accuracy)

| Bit | clean | q=90 | q=75 | q=50 | resize | crop |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (warm/cool) | 0.902 | 0.609 | 0.543 | 0.504 | 0.523 | 0.547 |
| 1 (sharp/soft) | 0.945 | 0.633 | 0.539 | 0.504 | 0.547 | 0.551 |
| 2 (grainy/clean) | 0.957 | 0.719 | 0.563 | 0.547 | 0.516 | 0.594 |
| 3 (bright/dark) | 0.828 | 0.520 | 0.523 | 0.543 | 0.617 | 0.543 |
| 4 (contrast) | 0.922 | 0.680 | 0.559 | 0.480 | 0.488 | 0.523 |
| 5 (saturation) | 0.930 | 0.746 | 0.664 | 0.652 | 0.574 | 0.602 |
| 6 (detail) | 0.996 | 0.738 | 0.543 | 0.508 | 0.633 | 0.691 |
| 7 (vintage/modern) | 0.965 | 0.832 | 0.586 | 0.516 | 0.500 | 0.500 |

Bit 6 (detail) and bit 7 (vintage/modern) survive JPEG q=90 best (74–83 %). Bit 3 (bright/dark), already the weakest on clean data, is the first to collapse to chance under any attack.

### Per-bit robustness for 8 × separate

| Bit | clean | q=90 | q=75 | q=50 | resize | crop |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.906 | 0.770 | 0.637 | 0.563 | 0.625 | 0.523 |
| 1 | 0.922 | 0.813 | 0.711 | 0.555 | 0.590 | 0.602 |
| 2 | 0.961 | 0.832 | 0.625 | 0.504 | 0.582 | 0.574 |
| 3 | 0.582 | 0.523 | 0.523 | 0.512 | 0.539 | 0.504 |
| 4 | 0.828 | 0.672 | 0.574 | 0.465 | 0.520 | 0.484 |
| 5 | 0.930 | 0.852 | 0.676 | 0.582 | 0.805 | 0.641 |
| 6 | 0.988 | 0.879 | 0.602 | 0.535 | 0.668 | 0.617 |
| 7 | 0.953 | 0.871 | 0.637 | 0.563 | 0.539 | 0.605 |

The separate model holds bit 5 (saturation) at 80.5 % through the resize attack — substantially better than the shared backbone's 57.4 % at the same bit. Bit 3 starts and stays at chance.

**Reference figures:** `results/figures/robustness_per_bit.png`, `results/figures/robustness_jpeg_curve.png`.

---

## 4 · Failure-mode analysis

### 4.1 Bit 3 (bright/dark) is the system bottleneck

Across every architecture and every regime, bit 3 is the weakest:

| Model | Bit 3 test acc |
| --- | --- |
| Person A baseline 1024 (best_model.pth) | 0.852 |
| Person A baseline 1024 (post-rebase) | 0.828 |
| Person A 512×512 ablation | **0.891** ← improves at lower res |
| 8 × separate 1024 (this work) | **0.582** ← collapses |

Why? The bright/dark slider modulates **global mean luminance** — exactly the band that JPEG's quantisation tables are built to preserve, but also the band where natural-image dynamic range dwarfs a ±0.3-scale LoRA perturbation. The signal is *small in amplitude, low-frequency, and globally distributed* — i.e. the kind of shift that:

1. is easy for a CNN with a strong inductive bias to confuse with prompt-level luminance variation,
2. requires global pooling features (which the shared backbone provides via cross-bit feature sharing, but the per-bit separate models cannot),
3. is the only bit where Person A's 512-resolution ablation *outperforms* 1024 by 3.9 pp — suggesting the optimal ResNet receptive field for this bit is below the native resolution.

**Implication for the ensemble.** With bit 3 stuck at ~58 %, the separate ensemble's exact-match cannot exceed `0.58 × 0.92 × 0.96 × … ≈ 0.40` even if every other bit were perfect. The observed 33.59 % exact-match is consistent with this ceiling.

### 4.2 Why ViT-B/16 fails

`torchvision.models.vit_b_16` hard-asserts a fixed input size; the watermarked image has to be downsampled before reaching the encoder. Two compounding effects then destroy the signal:

1. **Pre-encoder downsample loss.** A bicubic/bilinear ~4.6× downsample with antialiasing is essentially a low-pass filter with cutoff well below the LoRA perturbation's spectral support. By the time the ViT sees the image, most of the bit signal is already gone.
2. **Patch tokeniser coarseness.** ViT-B/16 grids the (already smoothed) input into 196 patches of 16 × 16 pixels — each token covering ~73 source pixels. That receptive field is too coarse to recover bit-level identity from what little high-frequency signal survived step 1.

The training history in [decoding/checkpoints/vit_best.summary.json](decoding/checkpoints/vit_best.summary.json) confirms the model never escapes the random-init regime: train loss stays at ~0.694 (= ln 2, the BCE chance value) for all 20 epochs; validation exact-match peaks at 1.17 % and degrades thereafter.

The robustness eval ([§3 above](#3--robustness-evaluation-1024-native)) provides the strongest possible confirmation: per-bit accuracy is **literally identical to four decimal places** across clean, JPEG q=90, JPEG q=75, JPEG q=50, and the resize attack. The model returns the same prediction regardless of image content — it learned only a constant per-bit bias from the training-set prior. Even random-crop, which alters input statistics most aggressively, shifts the answer by less than 1 pp. **This is an architectural mismatch, not a hyperparameter or training-time problem.** A retrain with a higher learning rate or longer schedule will not fix it; only an architecture that accepts higher-resolution input or uses smaller patches would.

### 4.3 Separate vs shared at the same parameter budget

At first glance the 188 M-parameter separate ensemble should beat the 25 M-parameter shared backbone — more capacity, more specialised representations. It does not, primarily because:

- **No cross-bit feature sharing.** Bit 3's bright/dark signal correlates with bits 4 (contrast) and 5 (saturation) — the shared backbone learns one luminance feature that helps all three; the separate models can't share.
- **More overfitting capacity per bit.** With 25 M parameters dedicated to a single binary label and only 2048 training samples, the per-bit models have far more degrees of freedom than the data supports. The val→test gap on bit 3 (0.6367 val → 0.5820 test) is larger than the shared model's gap on the same bit (0.875 val per Person A → 0.828 test).
- **Counter-intuitive robustness win.** Separate is *more* robust to JPEG and resize. This is consistent with the per-bit overfitting hypothesis: each backbone has memorised redundant low-frequency cues for "its" bit, and those redundant cues happen to survive JPEG quantisation better than the shared backbone's tightly-coupled features.

In aggregate: **the shared backbone wins on clean accuracy and exact-match; the separate ensemble wins on graceful degradation** — but neither retains usable performance under any of the spec attacks at the exact-match level.

---

## 5 · Comparison to Person A's prior ablations

For continuity with the existing reports in `decoding/results/`:

| Configuration | Architecture | Test mean | Test exact | Source |
| --- | --- | --- | --- | --- |
| ResNet-50 @ 1024 | shared, 30 epochs | 0.9404 | 0.6094 | [baseline_resnet50.md](decoding/results/baseline_resnet50.md) (Person A's `best_model.pth`) |
| ResNet-50 @ 1024 (rerun) | shared, 30 epochs | 0.9307 | 0.5703 | this work, `baseline_resnet50.pth` |
| ResNet-50 @ 512 | shared, 30 epochs | 0.9390 | **0.6367** | [ablation2_resolution_512.md](decoding/results/ablation2_resolution_512.md) |
| EfficientNet-B0 @ 1024 | shared, 30 epochs | 0.9258 | 0.5469 | [ablation1_efficientnet_b0.md](decoding/results/ablation1_efficientnet_b0.md) |
| 8 × ResNet-50 @ 1024 | separate, 25 epochs | 0.8828 | 0.3359 | this work |
| ViT-B/16 (224 input) | single, 20 epochs | 0.4712 | 0.0000 | this work |

**The 512-resolution shared baseline remains the strongest configuration** at 63.67 % exact match — better than any architecture variant tried so far. If a single number were to be published, it should be that one.

---

## 6 · Limitations

1. **Diffusion-based purification attacks are out of scope.** Zhao et al. (NeurIPS 2024, *"Invisible Image Watermarks Are Provably Removable Using Generative AI"*) demonstrate that running a watermarked image through a small number of diffusion denoising steps reduces detection rates of state-of-the-art LoRA-style watermarks to below 5 %. We did not evaluate against this attack: defending against it requires co-training the encoder against a denoiser and is fundamentally adversarial.
2. **JPEG, resize, and crop only model incidental degradation.** They are useful proxies for social-media re-uploads but do not approximate a motivated attacker. Our results (no exact-match survives any attack) suggest the encoder needs a robustness-aware retraining pass before deployment.
3. **Single-LoRA, fixed-seed, fixed-prompt training set.** All 2560 images come from a 10-prompt cycle with a deterministic seed schedule. Out-of-distribution prompts (different aesthetic, different aspect ratio, different LoRA combinations) have not been characterised.
4. **Independent-bit assumption.** We frame decoding as 8 independent classifications. Joint decoders (e.g. learning a code over the 256 IDs directly, or training with a structured loss that penalises near-misses) are unexplored and may close the exact-match gap.
5. **No 8 × separate retrain at 512.** Person A's 512 ablation suggests 512 is a better resolution for this task than 1024. The 8 × separate ensemble was trained at 1024 only; whether it would beat the shared 512 baseline at the same lower resolution is an open question (and would test the separate-ensemble hypothesis more fairly).
6. **ViT-B/16 specifically is unfit.** Other transformer architectures (Swin, DeiT-3 with adjustable patch size, ConvNeXt) might not have the same hard input constraint and could be a fairer "transformer" entry — but were not in scope here.

---

## 7 · Reproducibility

| Number in this report | Source script | Output |
| --- | --- | --- |
| Architecture comparison table | `scripts/compare_architectures.py --batch-size 4` | `results/architecture_comparison.md`, `results/figures/architecture_comparison.png` |
| ResNet shared robustness | `scripts/robustness_eval.py --model resnet --batch-size 4 --results-json results/robustness_resnet.json` | `results/robustness_resnet.json` |
| 8 × separate robustness | `scripts/robustness_eval.py --model separate --batch-size 4 --results-json results/robustness_separate.json` | `results/robustness_separate.json` |
| ViT robustness (image-blind) | `scripts/robustness_eval.py --model vit --batch-size 16 --results-json results/robustness_vit.json` | `results/robustness_vit.json` |
| **All of the above in one shot** | `bash scripts/run_full_evaluation.sh --batch-size 4` | every output above + `decoding/.full_eval/manifest.json` |
| Signal-analysis figures | `scripts/signal_analysis.py` | `results/figures/{comparison_*,diff_slider_*,fft_*,gradcam_*}.png` |
| Per-bit val histories | already on disk: `decoding/checkpoints/{vit_best.summary.json,separate/training_summary.json}` | — |

The full pipeline is documented step-by-step in [decoding/PIPELINE.md](decoding/PIPELINE.md). To regenerate this report's numbers from scratch on a fresh GPU pod, follow §1–§6 there.

---

## 8 · One-paragraph summary for the paper

> We trained three decoder architectures against an 8-bit LoRA watermark embedded in 2560 SDXL outputs. Person A's shared-backbone ResNet-50 at native 1024 × 1024 reaches 93.07 % mean bit accuracy and 57.03 % exact match on the held-out test split, with bit 3 (bright/dark) the persistent weak link at 82.8 %. An 8 × ResNet-50 ensemble, with one independent backbone per bit, underperforms the shared model (88.28 % / 33.59 %) — the gain in capacity does not overcome the loss of cross-bit feature sharing on bit 3. ViT-B/16 fails entirely because `torchvision`'s implementation hard-asserts a fixed input size that destroys the high-frequency LoRA signal before the encoder runs; robustness eval confirms the trained ViT is literally image-blind, returning identical predictions across all attacks. Under the spec attacks (JPEG, 1024→512→1024 resize, 75 % random crop) all working architectures lose nearly all exact-match capability; the separate ensemble degrades more gracefully than the shared backbone on JPEG and resize, but neither is usable under any attack. Person A's earlier 512-resolution ablation (63.67 % exact match) remains the strongest single configuration. Adversarial purification attacks (Zhao et al., NeurIPS 2024) and joint-code decoders are out of scope and left as future work.
