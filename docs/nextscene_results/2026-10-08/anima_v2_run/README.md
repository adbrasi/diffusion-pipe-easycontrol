# Anima A/aligned 1024 v2 — run of 2026-10-08

- **Data:** `AdwolfCzar/ofificial_next_scene_v11`, which contains `dataset-1-anima.zip` and `dataset-2-natural.zip`.
  - The two zips hold the same 1,736 pairs (A = images_A, B = images_B) and differ only in caption style.
  - Both are trained together with equal weight: 1,726 pairs × 2 captions = 3,452 samples per epoch.
  - Held-out set: 8 pairs (2 per source ds1–ds4), removed from both caption sets.
  - 6 more ds1 pairs that share a video with a held-out pair were also excluded.
- **Recipe:** identical to 2026-10-01 (r64, lr 1e-4, batch 4, BF16, 1024 buckets, aligned, ref_dropout 0.1, high_noise 0.2).
  - Additions: experimental `compile_blocks` + `uncheckpointed_blocks = 2`.
  - Run length: 5 epochs = **4,325 steps**, at 3.63 s/step.
- **Evaluation:** every 500 steps, 3 held-out pairs, each with A + anima prompt, A + natural prompt and shuffled A + natural prompt. The format was changed at step 500 at the user's request; no steps were lost.
- **Backups:** `AdwolfCzar/anima-nextscene-a-aligned-1024-v2`, under `checkpoints/20261008_18-01-16/step500 … step4325`. Each upload was verified with a remote SHA256.
- **Incidents:**
  - The user's ComfyUI held 25 GB of VRAM and caused an OOM at the start; the user freed the GPU.
  - One HF upload was slow at step 3000.
  - The adapter files were saved with mode 600 (root). Local read access was granted.

## ComfyUI node parity (comfyui_nextscene vs training inference code)

Setup: ComfyUI 0.39.0 with native Anima plus the repo nodes, run headless without installing them. Both sides received the same pre-resized A, prompt, negative prompt, initial noise, 30 shifted sigmas, Euler sampler and CFG 4. Checkpoint: step4000.

| measure | value |
| --- | --- |
| reference latent (VAE + normalization), relative difference | 0.42% |
| step-0 CFG denoised, relative difference | 3.2% |
| final image PSNR | 20.7 dB: same composition, characters, pose and framing; fine details differ |

Conclusion: the node reproduces the training contract. The remaining differences come from ComfyUI's own Qwen3/DiT kernels and are amplified over 30 CFG steps. The user's workflow was also reviewed:
- `CLIPLoader type=qwen_image` has no effect for qwen_3_06b.
- `ModelSamplingAuraFlow` with shift 3 equals the Anima default.
