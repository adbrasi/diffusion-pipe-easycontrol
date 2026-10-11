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

## Phase 2: native on Raw, 512 (2026-10-11)

### Smoke run (10 steps, `examples/krea2_raw_v3/smoke.toml`)
- Needed `git submodule update --init --recursive` (HunyuanVideo import in `utils/patches.py`).
- Cache: buckets correct (7:3 → 784×336, etc.). 814 steps/epoch (micro 1 × accum 2).
- `Cosine LR: T_max=…` prints. **Fixed a trainer bug:** cosine `T_max` was `epochs × steps_per_epoch` and ignored `max_steps`. The previous runs' LR would not have reached ~0 at `max_steps`. It now decays over `min(epochs·spe, max_steps) − warmup`.
- 5.9 s/step at 512 (2 forwards per step), 20.4 GB peak. No NaN.
- **Val loss (12 held-out, 9 fixed quantiles, seeded):** base 0.2043, step10 0.2008. Base per quantile: q0.1 0.405 … q0.5 0.160 … q0.9 0.157.
- Stock ComfyUI `f856877e` loads the step10 adapter with **0 `lora key not loaded`**.

### Full run `raw512_r32` launched 2026-10-11 ~14:50 UTC
- Config `examples/krea2_raw_v3/raw512.toml`, branch `krea2-raw-v3`. 1,220 steps ≈ 1.5 epochs, ETA ~2h10m.
- Each checkpoint is auto-uploaded and verified to HF `AdwolfCzar/krea2-nextscene-raw-v3/checkpoints/raw512_r32/stepN`.

### raw512_r32 finished (1,220 steps, ~2h10m, 5.9 s/step)
Held-out val loss, avg over 9 quantiles:

| step | 0 | 125 | 250 | 375 | 500 | 625 | 750 | 875 | 1000 | 1125 |
|---|---|---|---|---|---|---|---|---|---|---|
| avg | 0.2043 | 0.1162 | 0.1147 | 0.1143 | 0.1139 | 0.1134 | 0.1134 | 0.1133 | 0.1133 | 0.1132 |

- **Monotonic, no held-out overfit within 1.5 epochs.**
- The step0 → 125 drop is mostly in low-noise quantiles (q0.1 0.405 → 0.200). That is the base learning t=0 reference tokens: the speckle fix.
- After step 250 the gains are tiny and spread over all t.
- Val loss cannot see "copies A / ignores prompt" (it is measured with the true B caption), so the choice is made on the grids.
- Delta audit, b0 `mlp.down`: s125 0.095, s250 0.138. A1000@1000 was 0.349.

### raw512_r32 visual eval
Grids: HF `evidence/raw512_eval/`. Turbo 8 steps, seed 76. Columns: native(index_timestep_zero) | A1000 | s250 | s500 | s750 | s1000 | s1220.
- The user saw `/workspace/k2eval/raw512_eval` and called it "muito interessantes".
- **Quality: solved.** Every checkpoint is crisp and saturated on Turbo, with no haze (contrast with t1024 / probe).
- **Reference consistency: strong, and better than A1000.**
  - Kept: character design, outfit details (fire-force boy's neck scar, glasses, robot design), drawing style, **watermarks** (@SUOIRESNU, DERPIXON, PATREON), the robot companion.
  - The user's tip, adopted: watermark fidelity is an effort-to-be-consistent signal. It is kept at all steps.
- **Prompt following:**
  - Good for action, pose and camera: the archer aims at the viewer; the man turns his back.
  - The manga boy points and shouts "ENOUGH!", clearly better at s750+.
  - **Weak for big location changes:** "kids on a park bench at sunset" stays in A's corridor; the squirrel stays in the hollow; A's HUD overlay is carried over.
- **Cause, found in the data:** 1,440 of 1,629 training captions (88%) say "The same background/place/room…", and none describe a new place. The pairs are consecutive frames, so the LoRA learned that the next scene happens in the same place.
  - A1000 (older data) changes scenes more but loses style and watermarks.
  - **To get location changes, the dataset needs pairs with a real location change.** Training alone cannot invent them.
- s1000 ≈ s1220. Picked s1220 (end of cosine) as the init for the 1024 finish.
- **Duplication quirk:** "the same girl runs…" (tr0alt) gives two girls in *all* columns including native. It is a base/prompt quirk, not the LoRA.

### User feedback 2026-10-11 (evening) and answers
1. **"native has noise and weird stuff, but in ComfyUI plain TextEncodeQwenImageEditPlus works with good quality, only imprecise."**
   - Explained by code: Krea2 in stock ComfyUI has `default_ref_method=None` (`comfy/ldm/krea2/model.py:312`).
   - **Without `FluxKontextMultiReferenceLatentMethod`, the reference latent is silently ignored.** The model sees the image only through the Qwen3-VL tokens of the prompt. That gives exactly "a blonde girl, not Barbie" and no speckles.
   - My `native` column used `index_timestep_zero` (our LoRA's contract), which the base never learned, hence the speckles.
   - The eval tool now also has **`native_vl`**: no method node, which is the user's plain graph. It is used in all later grids.
   - Our LoRA must keep using `index_timestep_zero`.
2. "t3000_s06 has incredible results on grid_train_caption". Those rows are training images with their own captions, i.e. memorised. On held-out and short prompts the t-series is hazy on Turbo (Phase 1). So the long Raw run (below) tests "more training" on the right base.
3. The user is sleeping; keep the GPU busy. Plan: 1024 finish → full eval (with native_vl, strength 0.7, shuffled, internet images) → 3-epoch Raw run → eval.

### finish1024 (240 steps, 1024, batch 1, LR 5e-5 cosine, init raw512 s1220)
- 1024 val loss is flat: 0.07995 (step 0, i.e. the 512 adapter) → 0.07977 (step 240). The 512 adapter already transfers to 1024.
- ~6 s/step, 28.2 GB.
- Eval `evidence/finish1024_eval/`: 29 rows including 8 **internet images** (safebooru SFW: Gura, Aya, Cirno, Koishi, a penguin-hood girl, a winter girl). Columns: native_vl | A1000 | r512 s1220 | f120 | f240 | strength 0.7 | shuffled.

**User verdict (2026-10-11 night): "these latest tests don't hold the reference strongly, they don't do the same style correctly."** Confirmed on the web rows:
- Every column, including `native_vl` (no reference latent at all), redraws the character in **Krea's own polished anime house style**.
- Our LoRA keeps identity, outfit and even A's pose (the penguin hood). It does **not** keep A's rendering style (linework, colouring, shading).
- r512 vs f240 is almost the same. f240 is not a clear win.
- So the reference signal in our LoRA is too weak for **style**. In-domain (anime frames) it looked fine because Krea's default anime look is close to those frames.

**Hypothesis** ("two poles", `AUDITORIA_2026-08-06.md`): the v11 captions are long and describe B fully (pose, clothes, hair, background, text). The text path alone can satisfy the loss, so the LoRA never needs to read *style* from A.

**Tests queued overnight**, one variable each against raw512_r32:
1. **Strength 1.4 / 1.8 + shuffled ref** on 4 web rows. If more LoRA gives more of A's style, the signal exists but is weak.
2. **raw512_long3ep**: same recipe, 3 epochs (length only; checkpoints every 250).
3. **raw512_shorttier**: `captions.json` with [full, short] per pair. Same 1,220 updates, i.e. the same image views.
   - Short = framing sentence + the action sentence with appearance clauses removed + "The same background.". Appearance, clothing, text and watermark sentences are dropped, so they must come from A.
   - Pairs that introduce a new character keep the full caption: 1,231/1,629 get a short one.
   - Tool: `tools/krea2_v3_short_captions.py`.

**User request:** generate far fewer eval images. Every later eval uses `cases_small.json`: 8 rows (4 web, 2 held-out, 1 altered train caption, 1 short), with only the columns needed for the question.

## Correction of the objective (user, 2026-10-11 night): READ THIS FIRST

> "The purpose is to generate the next image from a previous reference **consistently**, so we can reuse the character, the environment, the colour palette and the theme. The dataset is entirely about that."

- Do **not** score "moves to a completely new location" as a goal. A character on a white background should stay in that world.
- Score **fidelity to A's specifics**, using the user's web_winter example:
  - hair colour (slightly purple);
  - glove colours (pink vs lilac);
  - the *same* bag;
  - outfit details, the letter, the dog;
  - also palette and theme.
- The user's verdict on raw512 / finish1024 on web images: "the same universe but not the same character". That is about what **native** plain TextEncodeQwenImageEditPlus already gives.
- The user's favourite so far: **t1024 s3000 @0.6**. It looked like it was really learning the next scene, despite the Turbo haze.

**Diagnosis of the gap:**
- My Raw run fixed the base (no haze), but I also cut the **training budget**: 1.5 epochs at 512, LR mass ~1,000, LoRA b0 `mlp.down` 0.138.
- The runs with strong reference fidelity were much bigger: t1024 s3000 had 3,000+ updates at 1024 (LR mass ~2,400, `mlp.down` 0.355), and A1000 had 0.349.
- The val loss plateau did not show this: it measures B with its full caption.

**Decision:**
- Train the **t1024 budget on the correct base**: `examples/krea2_raw_v3/raw1024_long.toml`.
  - Raw fp8_scaled, 1024, batch 1, LR 1e-4 cosine to 3,000 steps, apex law, r32.
  - Save and val every 250. Checkpoints auto-uploaded to HF `checkpoints/raw1024_long/`.
- The short-tier 512 run stays queued after it.
- **Eval criterion from now on:** reference fidelity on `cases_small.json` (web refs plus in-domain held-out). Compare native_vl, our LoRA, and t3000@0.6 as the user's favourite bar, with few images.

**Infra incident:** the queued script waited with `pgrep -f finish1024_eval`, which matched my own shell command, so the GPU sat idle for ~25 min. The queue is now a plain sequential script (`tools/k2v3_ops/queue2.sh`).

### Strength test on web refs (`evidence/strength_web.jpg`; r512 s1220 at 1.0 / 1.4 / 1.8 + shuffled)
- 1.4 and 1.8 barely change anything. The output stays in Krea's house style, with the same generic rendering. **Scaling the LoRA is not the lever.**
- The shuffled ref swaps identity (Aya's hat and wings show up on the Gura prompt; Gura on the Cirno prompt). The **ref latent is read for identity**, but not for rendering style and fine details.
- **Hypothesis H-turbo-style (to test at the next GPU break, ~4 images):**
  - A Raw-trained LoRA applied on Turbo lets the **Turbo distillation's house aesthetic** override the reference style.
  - t3000 (the user's favourite) was trained *against Turbo*, so it learned to counter it.
  - Test: r512 s1220 and native_vl on Raw 28 steps vs Turbo, on 2 web refs.
  - If Raw keeps A's style clearly better, consider training against Turbo in a way that does not break the distillation (e.g. the real `krea2_turbo` checkpoint, short), or ask whether the user accepts Raw/hybrid sampling.
- **H-undertrained** is being tested now by `raw1024_long` (t1024 budget on Raw).

### Style test (`evidence/styletest.jpg`, 2 web refs, seed 76)
Columns: Turbo native_vl | Turbo r512 s1220 | **Raw** native_vl | **Raw** r512 (28 steps, CFG 4) | Turbo **t3000@0.6** (the user's favourite).
- **t3000@0.6 is by far the closest to the reference's real art style:** Aya's soft painterly rendering, face shape and expression; Gura's rendering and pose. It has a mild haze but **high fidelity**.
- r512 does not get there on Turbo **or on Raw**. On Raw both native and r512 drift further away (Aya even comes out letterboxed). **H-turbo-style is not supported.** Raw sampling is not the cure.
- **H-undertrained is supported:** the 1024 × ~3,000-update LoRA learned real style/identity transfer that the 1.5-epoch 512 LoRA did not. The fidelity comes from **training budget and resolution**. The haze comes from the Turbo+adapter base.
- **Next:** `raw1024_long` (t1024 budget, Raw base) relaunched right after this test (2026-10-11 ~22:05 UTC). ETA ~5.5 h (6 s/step + val every 250).
  - Expectation: t3000-like fidelity without the haze.
  - Then `raw512_shorttier`.
