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

## Phase 1: decisive tests on the existing adapters (2026-10-11)

Grids: `/workspace/k2eval/phase1_turbo/grid_{train_caption,train_altered,heldout,short}.jpg` (uploaded to HF `evidence/phase1/`).
- Seed 76, Turbo (Raw fp8 + Turbo LoRA), 8 steps, CFG 1, user ComfyUI `f856877e`, node_1mp reference, training AR bucket at ~1 MP.
- Columns: native (no LoRA) | A1000@1 | probe p250@1 | p1000@1 | p1000@0.7 | t1024 s500@1 | s1500@1 | s1500@0.7 | s3000@1 | s3000@0.6 | p1000 with shuffled ref.

### T1 (Turbo half), T3, T4: visual verdict (mine; the user's verdict is pending)

- **A1000 (Raw-trained) is clearly the best on Turbo.** Crisp, saturated, clean lines. It follows the **short user-style prompts** and the **altered captions**:
  - the cat-eared girl runs down a night city street;
  - three kids on a park bench at sunset;
  - the man in uniform turns to the camera;
  - the girl eats ramen;
  - the squirrel moves to a branch holding an acorn;
  - the manga boy points and shouts "ENOUGH!".
  - Identity and outfit are kept.
- **Probe p1000 and t1024 s500 (Turbo+adapter-trained) are already grey/hazy.** The haze and loss of contrast grow monotonically to s1500 and s3000, together with anatomy deformation (short prompts: archer, cat girl). Their compositions stay closer to A:
  - the squirrel stays in the tree hollow under the altered caption;
  - "kids on a bench at sunset" stays in A's corridor.
  - This is the "copies / re-imagines A, ignores prompt" failure.
- **Strength 0.6–0.7 (T3)** reduces the haze of s1500/s3000 a little. It does not restore crispness, and it does not fix the composition copy. Magnitude is not the main lever.
- **native (no LoRA)** shows the expected 16-px speckle grid everywhere. It roughly follows the prompt and sometimes duplicates the subject.
- **One quirk:** A1000 and native both drew the running girl **twice** (side by side) on the altered tr0 caption. Watch for subject duplication.

### LoRA size audit (`tools/lora_delta_audit.py`, spectral |ΔW|₂/|W|₂, base `krea2_raw_bf16`)

| module | Turbo distill | **A1000 s1000 (Raw, r64)** | probe s250 | probe s1000 | t1024 s3000 (diag.) |
|---|---|---|---|---|---|
| b0 attn.wq | 0.019 | 0.100 | 0.044 | 0.071 | 0.107 |
| b0 mlp.down | 0.059 | **0.349** | 0.092 | 0.177 | 0.355 |
| b14 attn.wk | 0.014 | 0.156 | 0.038 | 0.074 | — |
| b14 mlp.down | 0.071 | **0.282** | 0.057 | 0.126 | 0.202 |
| b27 attn.wq | 0.245 | 0.111 | 0.044 | 0.092 | 0.157 |

**Finding:** the crisp, Raw-trained A1000 moves the weights **as much as or more than** the "burnt" t1024 s3000. LoRA magnitude does **not** explain the haze. What differs is the base it was trained against.
- This strongly supports diagnosis §2.3: train on plain Raw.
- It drops "block-0 mlp.down ≤ 0.18" as a guardrail. I keep the audit only as a trend monitor within a run.

### T6
The train logs on HF contain no `lora key not loaded` / `NOT LOADED` lines. They also contain no merge logging at all, so the result is inconclusive. It is moot now, because v3 does not merge adapters.

### T2: official Turbo vs Raw fp8 + Turbo LoRA r64
`evidence/phase1/T2.jpg`: 8 cases, with and without p1000.
- Same composition, colours and look. Only small detail differences.
- The p1000 haze is identical on both.
- **Conclusion:** evaluating (and using) "Raw fp8 + Turbo LoRA" is representative of Turbo.

### T1: Raw 28 steps, CFG 4 vs Turbo, same seeds
`evidence/phase1/T1_raw.jpg`: 5 cases, columns native, A1000, p1000, t500, t1500, t3000.
- **On Raw, all Turbo+adapter-trained adapters (p1000, t500, t1500, t3000) are crisp and saturated, with no haze.** On Turbo the same adapters are grey and washed out.
- **Confirmed:** the haze is the Turbo distillation breaking under a LoRA trained against the "Raw + Turbo LoRA + ostris adapter" base. The decision to **train on plain Raw** is closed.
- Even on Raw, the Turbo-trained adapters keep A's composition more than A1000 does. Example: "kids on a park bench at sunset" stays in A's corridor, while A1000 goes to a park.
- Raw duplication quirk again: A1000 on Raw drew the archer prompt as a two-figure "character sheet" (sh0).

**Phase 1 decision:** go to Phase 2 (native on Raw, diagnosis §7.1) as planned. The A1000 look on Turbo is the quality bar to beat, together with better prompt-following and fewer duplications.
