# Krea 2 next-scene — "native on Raw" run log (v3, from 2026-10-11)

Session: GPU box RTX 5090 32 GB (Vast), follows `docs/HANDOFF_GPU_2026-10-11.md` and
`docs/DIAGNOSTICO_KREA2_2026-10-11.md`. Newest entries at the bottom.

- **Code branch:** `krea2-raw-v3`, based on `krea2-probe-512` (the A1000 code lineage) plus the tools listed below.
- **HF (public):** `AdwolfCzar/krea2-nextscene-raw-v3`: adapters, grids, configs, README.

## Phase 0: setup and gates (2026-10-11)

**Environment**
- Own venv at `/workspace/venv`. The image's `/venv/main` is root-owned and read-only for this user.
- torch 2.11.0+cu128, deepspeed 0.18.4, transformers 5.19.
- Training ComfyUI: submodule `fb2315f1`.
- Eval ComfyUI: a separate clone at `f856877e` (the user's v0.39), run headless with `--disable-all-custom-nodes`.

**Models** (`Comfy-Org/Krea-2`):
- `krea2_raw_fp8_scaled` (training base and eval base);
- `krea2_turbo_lora_rank_64_bf16` (eval Turbo = Raw fp8 + this LoRA);
- `krea2_turbo_fp8_scaled` (T2 only);
- `qwen_image_vae`, `qwen3vl_4b_bf16`.
- The ostris adapter is not used.

**Dataset:** `AdwolfCzar/ofificial_next_scene_v11` @ `0708d6c0`, `dataset-2-natural.zip`. This is still the newest next-scene dataset under AdwolfCzar as of 2026-10-11; the others are loop/jiggle/etc. sets.
- 1,736 pairs, each `images_A/<f>` + `images_B/<f>` + `.txt`.
- Prepared with `tools/krea2_v3_prepare.py` into `/workspace/k2data/{train,heldout}`. Report: `/workspace/k2data/report.json`.
- **Dropped:**
  - 19 junk captions: "test", "Decline", "...", "TEXT_PLACEHOLDER";
  - 35 pairs where crop-fitting A to B's bucket loses more than 20% of A. All are ds3 manga/pin-ups with a different A/B AR (up to 83%).
- **Held-out:** 12 pairs, stratified by source prefix (ds1/ds2/ds3/ds4).
  - Whole video groups are excluded from training.
  - ds2/3/4 stems are sequential frame numbers without a video id, so ±8 neighbours are also excluded (41 pairs).
- **Train:** 1,629 pairs.
  - Buckets: 16:9 1277 (78%), 2:3 162, 3:4 88, 7:3 56, 4:3 30, 1:1 28, 3:2 13, 9:16 12, 2:1 4.
  - Mean crop: B 0.9%, A 1.0%. 153 B and 160 A lose more than 5%: these are the ~0.70 ARs between the 2:3 and 3:4 buckets, max ~5–6%.

**TE parity gate: PASSED.** `tools/krea2_native_parity.py te-stock` (ComfyUI `f856877e`) versus `te-fork` (`fb2315f1`), 3 real refs from the dataset with different prompts: **relL2 = 0.00e+00 on all 3**. The text cache is safe to build.

**Eval tool:** `tools/krea2_v3_eval.py` (new; replaces `krea2_v2_eval.py`).
- **Graph:** the stock graph the user runs: `UNETLoader(raw fp8)` + Turbo LoRA + our LoRA (`LoraLoaderModelOnly`, any strength) → `TextEncodeQwenImageEditPlus(vae, image1)` → `FluxKontextMultiReferenceLatentMethod(index_timestep_zero)` → euler, Krea2 sigmas mu 1.15, 8 steps, CFG 1.
  - The reference uses the node's own ~1 MP rule.
  - Variants: `Turbo` (default), `Raw` (28 steps, CFG 4), `TurboOfficial` (full `krea2_turbo_fp8_scaled`).
- **Output size:** the **training AR bucket** (the same 10-AR list as the dataset config, including 16:9) at ~1 MP, sides /16. The old `bucket_dims` without 16:9 is gone.
- **Columns:** `native` (the same graph without our LoRA, the user's "traditional method"), any adapter@strength, and `shuffle=1` (reference of the next case with a *different* A).
- **Rows** (`/workspace/k2eval/cases_v1.json`, 21 rows):
  - 4 train pairs with their caption, plus 4 rows with an **altered** caption (different action/pose/camera);
  - 8 held-out pairs with their caption;
  - 5 **short user-style prompts** on held-out/train refs.
  - Explicit pairs stay in the val loss but are kept out of the public grids.
- **Outputs:** one grid per row kind, plus `results.json` with a cheap copy metric: correlation of the 64-px grey generated image with A and with B.
  - corrA ≫ corrB on held-out rows means it copies A.
  - It is a coarse heuristic, not a verdict.

**Training code changes** (branch `krea2-raw-v3`):
- `train.py` now prints `EVAL step N heldout avg … | q0.1=… q0.9=…` to the log. Eval was already seeded (`isolate_rng` + seed), so these numbers are comparable across checkpoints.
- Config: `examples/krea2_raw_v3/raw512.toml`. Diagnosis §7.1 recipe:
  - plain Raw fp8_scaled, no `merge_adapters`;
  - `timestep_law="krea2_apex"`;
  - micro 1 × accum 2, r32, LR 1e-4 cosine, 50 warmup;
  - `max_steps` 1,220 ≈ 1.5 epochs;
  - save and eval every 125 steps;
  - real AR buckets, `reference_pixels="target"`, vl_grounding, ref t=0, caption_dropout 0.
