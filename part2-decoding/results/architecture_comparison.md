# Architecture comparison (test set)

| Architecture | Bit 0 | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 | Bit 6 | Bit 7 | Mean | Exact |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 (shared backbone) | 0.9023 | 0.9453 | 0.9531 | 0.8359 | 0.9219 | 0.9297 | 0.9961 | 0.9648 | 0.9312 | 0.5742 |
| 8x ResNet-50 (separate) | 0.9062 | 0.9219 | 0.9609 | 0.5820 | 0.8281 | 0.9297 | 0.9883 | 0.9531 | 0.8838 | 0.3359 |
| ViT-B/16 | 0.4258 | 0.5039 | 0.4336 | 0.4883 | 0.4922 | 0.4570 | 0.4727 | 0.4531 | 0.4658 | 0.0000 |

Checkpoints used:
- **ResNet-50 (shared backbone)**: `/workspace/computer-vision-watermarking/decoding/checkpoints/baseline_resnet50.pth` (loaded)
- **8x ResNet-50 (separate)**: `/workspace/computer-vision-watermarking/decoding/checkpoints/separate` (loaded)
- **ViT-B/16**: `/workspace/computer-vision-watermarking/decoding/checkpoints/vit_best.pth` (loaded)
