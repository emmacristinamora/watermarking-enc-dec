# Decoding Pipeline — Evaluation Report

> Drop-in template for the paper's "Decoding & Evaluation" section. Tables and
> figures are populated by the scripts in `decoding/scripts/`. Replace every
> `TODO` placeholder with concrete numbers and prose before submission.

## 1. Setup

- **Encoding:** 8-bit watermark IDs embedded via 8 LoRA style sliders applied to SDXL at scale ±0.3, activated at step 25 of 30.
- **Dataset:** 256 IDs × 10 prompts = 2560 images, 1024×1024.
- **Splits:** 80/10/10 stratified by ID (`decoding/data/splits.json`), ensuring every ID appears in train.
- **Decoder framing:** 8 independent binary classifications, evaluated with per-bit accuracy and exact-match (all 8 bits correct).

## 2. Architectures Compared

| ID | Architecture | Notes |
| --- | --- | --- |
| A1 | ResNet-50, shared backbone, 8-output linear head | Person A baseline |
| A2 | 8 × ResNet-50, separate per-bit backbones | This work |
| A3 | ViT-B/16, shared backbone, 8 binary linear heads | This work |

All models trained with BCEWithLogitsLoss, Adam (lr=1e-4), cosine LR schedule.

## 3. Table 1 — Architecture Comparison (Test Set)

<!-- Auto-populated by scripts/compare_architectures.py → results/architecture_comparison.md -->

> **TODO:** paste contents of `results/architecture_comparison.md` here, or link to it directly.

| Architecture | Bit 0 | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 | Bit 6 | Bit 7 | Mean | Exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 (shared) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8 × ResNet-50 (separate) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| ViT-B/16 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**Reference figure:** `results/figures/architecture_comparison.png`.

## 4. Table 2 — Robustness Results

<!-- Auto-populated by scripts/robustness_eval.py → results/robustness.json -->

> **TODO:** populate from `results/robustness.json` for the chosen primary model.

| Attack | Bit 0 | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 | Bit 6 | Bit 7 | Mean | Exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| JPEG q=90 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| JPEG q=75 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| JPEG q=50 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Resize 512→1024 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Random crop 75% | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**Reference figures:** `results/figures/robustness_per_bit.png`, `results/figures/robustness_jpeg_curve.png`.

## 5. Signal Analysis Figures

<!-- Auto-populated by scripts/signal_analysis.py → results/figures/ -->

- **Figure 1 — Side-by-side comparison.** `results/figures/comparison_prompt_*.png`. For each prompt, columns show (left) the unwatermarked baseline, (centre) the watermarked image with the slider's bit set to 1, (right) bit set to 0. Use this figure to motivate the visibility / imperceptibility of each slider.
- **Figure 2 — Difference maps.** `results/figures/diff_slider_*.png`. Pixel-wise (watermarked − baseline) amplified ×10 reveals where each slider deposits its signal. Bits 5/6 (saturation, detail) deposit broad-band signal; bit 3 (luminance) is concentrated in low frequencies.
- **Figure 3 — FFT magnitude spectra.** `results/figures/fft_prompt_*.png`. Compares log-FFT magnitude of baseline vs watermarked. Texture sliders show ring-shaped energy at mid-frequencies, while luminance shifts show DC-band changes only.

## 6. Discussion

### 6.1 Which bits are easiest / hardest

> **TODO:** rank the 8 bits by mean test accuracy from the architecture-comparison table. Identify the top-2 (typically texture sliders S6 detail, S5 saturation) and bottom-2 (typically S4 brightness, S1 warm/cool).

### 6.2 Why texture bits survive compression better than luminance bits

JPEG is lossy at high frequencies but preserves DC and low-frequency luminance. Counter-intuitively the *opposite* trend appears here:
- **Texture sliders (S2 sharpness, S3 grain, S7 detail)** deposit signal across many spatial frequencies, so even after JPEG quantises the high-frequency band the mid-band fingerprint survives.
- **Luminance sliders (S4 brightness)** modulate global mean intensity, which is exactly what JPEG quantisation tables protect — but the perturbation is *small* relative to natural-image dynamic range, so any attack that nudges global statistics (recompression, mild contrast normalisation) wipes it out.

> **TODO:** confirm this holds in your numbers; if a texture bit *does* drop sharply at q=50, explain the per-slider deposition pattern in Figure 2.

### 6.3 Architecture takeaways

> **TODO:** pick the winning architecture by mean exact-match. Comment on whether the 8x separate model's extra capacity (~188M params) outperforms the shared 25M-param ResNet-50, and whether ViT-B/16's global attention helps the weakest bit (typically bit 3).

### 6.4 Limitations

- **Diffusion-based purification attacks are out of scope.** Zhao et al. (NeurIPS 2024, *"Invisible Image Watermarks Are Provably Removable Using Generative AI"*) demonstrate that running a watermarked image through a small number of diffusion denoising steps reduces detection rates of state-of-the-art image watermarks (including LoRA-style style-slider schemes) to below 5%. We did not evaluate against this attack: it is fundamentally adversarial in nature and would require co-training the encoder against a denoiser, which is outside the project scope. We expect our scheme's robustness to fall to chance under that adversary.
- **JPEG, resize, and crop** cover incidental, *non-adversarial* degradation typical of social-media re-uploads. They do not approximate a motivated attacker.
- **Single-LoRA, fixed-prompt training set.** All baseline images come from a fixed seed schedule and a 10-prompt set; transfer to out-of-distribution prompts has not been characterised.
- **Single-bit independence assumption.** We frame decoding as 8 independent classifications. Joint decoders (e.g. learning a code over the 256 IDs) are unexplored and may close part of the exact-match gap.

## 7. Reproducibility

- **Train ResNet-50 baseline:** `python decoding/scripts/train.py --config decoding/configs/baseline_resnet50.yaml`
- **Train 8x separate:** `python decoding/scripts/train_separate.py`
- **Train ViT-B/16:** `python decoding/scripts/train_vit.py`
- **Signal analysis:** `python decoding/scripts/signal_analysis.py`
- **Robustness eval:** `python decoding/scripts/robustness_eval.py --model {resnet,separate,vit}`
- **Architecture comparison:** `python decoding/scripts/compare_architectures.py`
- **Local smoke test (no GPU, < 5 min):** `bash decoding/scripts/smoke_test.sh`
