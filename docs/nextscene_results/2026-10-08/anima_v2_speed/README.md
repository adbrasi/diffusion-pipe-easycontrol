# Anima A/aligned 1024 v2 — speed audit (2026-10-08)

Goal: faster 1024 training **without changing training quality**. Recipe (LR, rank 64, batch 4,
BF16, buckets, A/aligned contract) is identical to the 2026-10-01 campaign. Only options that keep
the math were tried; FP8/W8A8, smaller reference and fused QKV (changes LoRA keys) were excluded.

Bench: ds4 of the old dataset (1251 pairs, 2 AR buckets), RTX 5090 at 575 W, torch 2.11+cu128,
14 steps, mean of last 8 steps (steady state).

| variant | s/step | samples/s | peak VRAM |
| --- | ---: | ---: | ---: |
| baseline (as 2026-10-01) | 5.22 | 0.766 | 20.3 GB |
| last 2 blocks uncheckpointed | 5.10 | 0.784 | 26.2 GB |
| last 4 blocks uncheckpointed | OOM | | |
| **per-block torch.compile** | **3.66** | **1.09** | 18.2 GB |
| compile + 3 uncheckpointed | 3.55 | 1.13 | 26.5 GB |
| compile + 5 uncheckpointed | OOM | | |
| **chosen: compile + 2 uncheckpointed (smoke)** | **3.58** | **1.12** | — |

The previous run measured 5.61 s/step at a 400 W power cap. Compile adds a one-time ~15–20 s per new
bucket shape (at most 7 shapes) plus ~2 s on each stage's first step.

## Quality check (compile_accuracy.py)

Real Anima block 5 + PEFT LoRA r64 (non-zero B), 1024 shapes (B4, T2, 64×64), A/aligned RoPE.
The table gives the relative L2 error against an FP32 reference.

| quantity | eager BF16 | compiled BF16 |
| --- | ---: | ---: |
| block output | 6.47e-3 | 5.55e-3 |
| input grad | 2.21e-2 | 1.99e-2 |
| LoRA grads (20 tensors) | 1.98e-2 | 1.74e-2 |

Compiled is slightly *closer* to FP32 than eager on every quantity, including the worst LoRA tensor,
because fused kernels skip intermediate BF16 roundings. Uncheckpointed blocks are bit-identical math.
Loss curves track the baseline at every step (`*.steps.txt`).

SDPA backends at the training shape (attn_bench.py) give identical error. Default and flash take
36–42 ms and cuDNN 37 ms, so nothing was gained and nothing was changed.

## Smoke of the final recipe

Ran 10 steps, saved, and resumed to 12 with LR 1e-4 kept. The adapter has 560 keys (no
`_orig_mod`) and the contract is aligned/ref t=0. Held-out eval at 1024 produced the grid
`smoke_step12_grid.png` (step 12, not a quality signal).

Estimate for a dataset like the old one (≈6.6k steps/epoch, 5 epochs ≈ 33k steps): ≈33 h of
compute plus about 9 h of evals every 500 steps, ≈44 h in total versus 61.5 h before.
