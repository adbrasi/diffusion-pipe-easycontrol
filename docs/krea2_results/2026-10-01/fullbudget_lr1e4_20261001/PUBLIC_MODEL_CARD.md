---
tags:
- lora
- image-to-image
- krea2
- next-scene
---
# Krea2 A native — LR 0.0001

Ongoing next-scene LoRA training with the native reference contract, initialized from scratch. Target: 5,000 steps, 6,138 scene-pair/caption samples selected across four datasets by available cache storage.

Learning rate 0.0001, 50-step warmup, rank 64, real micro batch 2, accumulation 1, 512-pixel buckets; scaled FP8 base with BF16 computation. Base checkpoint: `krea2_raw_fp8_scaled.safetensors`.

This public repository contains adapter exports, complete optimizer/resume states, and four held-out Turbo samples per 250-step milestone. Uploads are verified before old local checkpoints are removed. Training is ongoing; sample grids are the evidence for quality at each milestone.

The previous LR0.0004 run is separate. This LR0.0001 run is continuing from its own step500, not from the previous run.
