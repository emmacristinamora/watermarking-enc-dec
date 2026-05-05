# Ablation 1: EfficientNet-B0 Architecture

## Experiment
Test if EfficientNet-B0 provides better accuracy than ResNet-50, particularly for weak bits (Bit 0, Bit 3).

## Configuration
- **Architecture:** EfficientNet-B0 (pretrained ImageNet)
- **Batch size:** 12 (reduced from 16 due to memory)
- **Epochs:** 30
- **Learning rate:** 0.0001
- **Resolution:** 1024×1024
- **Parameters:** 4.0M (vs ResNet-50's 25.6M)

## Results

### Test Set Performance
- **Mean bit accuracy:** 92.58%
- **Exact match rate:** 54.69%

### Per-Bit Accuracy
| Bit | Slider | Accuracy |
|-----|--------|----------|
| 0 | warm/cool | 92.19% |
| 1 | sharp/soft | 93.36% |
| 2 | grainy/clean | 95.70% |
| 3 | bright/dark | 85.94% |
| 4 | contrast | 87.11% |
| 5 | saturation | 91.02% |
| 6 | detail | **100.00%** |
| 7 | vintage/modern | 95.31% |

### Comparison to ResNet-50

| Metric | ResNet-50 | EfficientNet-B0 | Difference |
|--------|-----------|-----------------|------------|
| Mean bit accuracy | 94.04% | 92.58% | **-1.46%** |
| Exact match | 60.94% | 54.69% | **-6.25%** |
| Bit 0 accuracy | 91.41% | 92.19% | +0.78% |
| Bit 3 accuracy | 85.16% | 85.94% | +0.78% |
| Val-test gap | 1.9% | 7.9% | +6.0% |

## Analysis

**EfficientNet-B0 underperforms ResNet-50:**
- 1.5% lower mean accuracy
- 6.3% lower exact match rate (major drop)
- Higher overfitting (7.9% gap vs 1.9%)

**Why it failed:**
- Model too small (4M params) to capture subtle watermark signals
- Slight improvement on weak bits (0, 3) but major degradation on mid-strength bits (4, 5)
- Net result: worse overall performance

**What worked:**
- Perfect detection on Bit 6 (detail/smooth)
- Marginal improvement on weakest bits

## Conclusion

**Verdict:** ResNet-50 is the better baseline.

EfficientNet-B0 is too lightweight for this task. The watermark signals at scale 0.3 are subtle enough that they require the additional capacity of a larger model.

**Recommendation:** Stick with ResNet-50 (94.04%, 60.94%) as baseline for Person B.

**Future work:** If trying smaller models, EfficientNet-B3 (12M params) might be a better middle ground than B0.

