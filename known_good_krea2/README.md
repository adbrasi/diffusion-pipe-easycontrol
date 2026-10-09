# Krea 2 — known-good training state (A1000)

This branch freezes the exact code that produced **A1000**, the Krea 2 next-scene adapter the
user validated visually on 2026-09-30 (screenshot 20:44 BRT). Return here if a later experiment fails.

- Code: commit `6355674` (pre 2026-10-01 changes: no text-fusion padding mask, per-token
  timestep MLP, legacy `bf16` FP8 matmul only). At batch 1 the later code is mathematically the
  same; this branch is kept for bit-faithful reproduction.
- ComfyUI submodule: `fb2315f1` (recorded in this branch). Its Qwen3-VL text encoder matched the
  user's ComfyUI 0.39 (`f856877e`) exactly on 2026-10-08 (`tools/krea2_native_parity.py te-fork`, relL2 0.0).
- Adapter: HF `AdwolfCzar/krea2-ab-runs` (private) →
  `checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000/` (sha256 `8db5282a…`).
- Weights: `Comfy-Org/Krea-2` @ `eb1eddd3` — `krea2_raw_fp8_scaled`, `qwen3vl_4b_bf16`, `qwen_image_vae`,
  `krea2_turbo_lora_rank_64_bf16` (eval only).

## Recipe (A1000_original_512.toml)
krea2_native, FP8-scaled base with bf16 matmul, 512 px, micro-batch 2, LR 1e-4 constant, warmup 50,
rank/alpha 64, AdamW8bitKahan, `reference_timestep=zero`, `vl_grounding=true`,
`reference_pixels=target`, 1000 steps. Data was a seeded 1,500-pair subset (legacy
`tools/k2ab_prepare_data.py`).

`A1000code_1024_dataset2.toml` is the same recipe at 1024 / batch 1 on dataset v11-natural
(started 2026-10-08, stopped at step ~136 to try improvements).

## Reproduce
    git worktree add /workspace/k2a1000 krea2-known-good-a1000
    git -C /workspace/k2a1000 submodule update --init submodules/ComfyUI
    cd /workspace/k2a1000 && deepspeed --num_gpus=1 train.py --deepspeed --config known_good_krea2/<config>.toml

Untrained adapters (< ~125 steps) show 16 px patch-grid speckle with `index_timestep_zero`; this is
the base model's reaction to t=0 reference tokens, not a training bug (see KREA2_AB_RUN_LOG 2026-09-30).
