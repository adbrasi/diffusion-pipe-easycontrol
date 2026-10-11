# HANDOFF → GPU session (written 2026-10-11 by the no-GPU analysis session)

Audience: a fresh Claude Code session on a rented RTX 5090 (32 GB). You have no prior context; this file plus the
files listed in §3 are your context. Write in whatever language you like in your notes; the user speaks Portuguese.

## 1. Mission

Produce a **Krea 2 "next scene" reference-edit LoRA** that works in **stock ComfyUI**:
- `TextEncodeQwenImageEditPlus` + `FluxKontextMultiReferenceLatentMethod(index_timestep_zero)`;
- our LoRA via `LoraLoaderModelOnly`, run on **Turbo** (8 steps, CFG 1).

Given reference image A + prompt, it must generate B: same characters, world and style, a different moment, pose, framing or place. It must:
- **respect the reference** (identity, outfit, style);
- **respect the prompt** (do the requested action, not a copy of A);
- **keep Krea 2's image quality** (no haze/wash-out, no anatomy collapse);
- stack with other LoRAs reasonably.

Krea 2 is the priority. Anima next-scene is secondary (§8).

**Autonomy:** you have full permission to use the GPU however you want: run any tests, probes, trainings and alternative methods. Do not wait for the user's approval. Decide, execute, record.

## 2. Where things stand (one paragraph per run; details in docs)

- **beta1** (`AdwolfCzar/k2-context-rush-ofc-beta1`, 13,445 steps). Routed LoRA (reference rows only), `width_shift`, fp8 base re-quantized *without* its scale, dirty 30k-pair data, caption_dropout 0.1.
  - Always follows the reference, but often returns the input image, ignores prompts and degrades when stacked.
  - Needs the custom `ctxrush-edit` node.
- **A1000** (2026-09-30, branch `krea2-known-good-a1000`). `krea2_native` on **Raw** `fp8_scaled` (correct scale), 512, batch 2, r64, LR 1e-4 constant, flux_shift.
  - Crisp on Turbo; copies the reference in ~3/10 cases.
- **probe512 Turbo** (2026-10-09, branch `krea2-probe-512`). Same code, but the training base was Raw + `krea2_turbo_lora_rank_64` + **ostris `krea2_turbo_training_adapter_v1` merged**. Also: shift 3.1582 + 15% of timesteps in U[0.8,1], r32, cosine, real AR buckets, new dataset.
  - The user judged it better than A1000 at step1000.
- **Turbo 1024** (same recipe at 1024 / batch 1). User stopped it at step 3362: "starting to burn".
  - Ignores the prompt, copies or "re-imagines" A, and looks hazy/washed-out from ~step1500 on.
  - Adapters and grids: HF `AdwolfCzar/krea2-turbo-1024`.

**Diagnosis** (`docs/DIAGNOSTICO_KREA2_2026-10-11.md`; read it fully). The training code has no math bug. The failure comes from method choices:
1. **Training on Turbo + ostris adapter.**
   - Krea officially says to train on **Raw** and apply on Turbo. The adapter card says it is a hack for short style runs, and that long runs break the distillation.
   - Measured: at step3000 the LoRA moves block-0/14 weights **4–6× more than the entire Raw→Turbo distillation delta**.
   - The grids show haze starting already at probe step500–750, while the Raw-trained A1000 stays crisp.
2. **Timestep law.** 50% of samples at t>0.8 and only 1.9% at t<0.3: the detail band is starved and the run is pushed toward copying.
3. **Over-training.** Batch 1 at the same LR gives 2.8× the LR mass of the approved probe.
4. **Blind evaluation.** 3 cases, 1 seed, prompt = the training caption, Turbo only, no baseline column.

Things to KEEP:
- the native stock contract;
- Raw `fp8_scaled` with scale;
- text encoder parity (`fb2315f1` / `f856877e`);
- real AR buckets (16:9 is ~71% of the data);
- the new dataset;
- cosine LR, rank 32–64;
- frequent checkpoints.

## 3. Read first (in this order)

1. `docs/DIAGNOSTICO_KREA2_2026-10-11.md`: the full diagnosis, recurring-error table, proposed recipe (§7.1) and GPU tests (§6).
2. `docs/RELATORIO_2026-10-08_09.md`: last GPU day (dataset, TE parity, probe/1024 setup, ComfyUI version, HF repos).
3. `docs/KREA2_ANALISE_METODO_2026-09.md`: why the native contract; the fp8/TE/contract pitfalls of beta1.
4. `docs/AUDITORIA_2026-08-06.md`: the "two poles" (copy versus ignore-ref) theory and the epoch-discipline lesson.
5. `docs/KREA2_AB_RUN_LOG.md` (skim): how A/B runs were operated, plus eval/metric caveats.
6. Branch `krea2-known-good-a1000`: `known_good_krea2/README.md` and the configs there.
7. Code: on branch `krea2-probe-512`, read `models/krea2_native.py`, `models/krea2.py` (`prepare_inputs`: shift, `high_noise_prob`, `timestep_law='krea2_apex'`) and `models/base.py` (`base_quant='fp8_scaled'`, `merge_adapters`). The eval tools are on the docs branch: `tools/krea2_v2_eval.py`, `tools/k2ab_eval_stock.py`, `tools/krea2_native_parity.py`, `tools/lora_delta_audit.py`.

## 4. Setup

- **Repo:** `https://github.com/adbrasi/diffusion-pipe-easycontrol`. Docs and tools live on branch `claude/elegant-ptolemy-8kywqy`; commit your notes to whatever branch your session designates, branched from it.
- **Training code:** use branch `krea2-probe-512` (the A1000 code, the fastest known-good lineage). Put it in a worktree and copy into it `tools/krea2_v2_eval.py`, `tools/krea2_v2_campaign.py` and `tools/lora_delta_audit.py` from the docs branch.
  - The docs branch head also contains newer model code: txtfusion padding-mask fix, `fp8_scaled_matmul='comfy'` and `reference_pixels='node_1mp'`. It is ~25% slower and has identical math at batch 1. With micro_batch 1 the padding issue does not exist, so the old code is fine.
- **Submodules:** `git submodule update --init submodules/ComfyUI` (the pinned `fb2315f1` on `krea2-probe-512`).
- **Gate before any caching:** run the TE parity gate, `tools/krea2_native_parity.py te-fork` (see RELATORIO §2). The bundled June ComfyUI `0ba903bd` diverges ~100%, so never cache text with it.
- **Models:**
  - Training base: `Comfy-Org/Krea-2`, `diffusion_models/krea2_raw_fp8_scaled.safetensors`.
  - Eval: `krea2_turbo_lora_rank_64_bf16.safetensors` (what the user runs: Raw fp8 + this LoRA = Turbo). Also download `krea2_turbo_fp8_scaled.safetensors` for test T2.
  - `vae/qwen_image_vae.safetensors` and `text_encoders/qwen3vl_4b_bf16.safetensors`.
  - The ostris adapter is **not** needed.
- **Stock ComfyUI for eval:** a headless ComfyUI ≥ `c9602625`; the user runs `f856877e`, v0.39. `tools/krea2_v2_eval.py` launches it with `--disable-all-custom-nodes`. Paths default to `/workspace/...`; adjust them to this machine.
- **Dataset:** the user's new curated next-scene set (~2k real pairs, nothing synthetic; the user says it is very good).
  - Last known location: HF dataset `AdwolfCzar/ofificial_next_scene_v11`, which has `dataset-2-natural.zip` (natural-language captions, used for Krea) and `dataset-1-anima.zip` (tag captions, Anima). They hold the same 1,736 pairs.
  - The user mentioned a "new" set. List `AdwolfCzar`'s HF datasets, use the newest next-scene one, and record which (name + revision) in your notes.
  - Layout for training: `target/*.png|jpg` + `target/*.txt` and `control/*` with the same stems. Hold out ≥8 pairs; avoid neighbouring frames of the same video in train versus held-out (see `tools/k2ab_prepare_heldout.py`).
- **HF uploads:** **always upload to PUBLIC repos under `AdwolfCzar`.** The private storage is full. Upload adapters, eval grids, configs and a short README per run (e.g. a new public repo `AdwolfCzar/krea2-nextscene-raw-v3`). Verify uploads (size/sha).
- **Notes:** after every meaningful test or finding, append to a run log `.md` in `docs/` (e.g. `docs/KREA2_RAW_RUN_LOG.md`) and **commit + push to GitHub**. Record what you ran (config, commit, seed), what you saw and what you concluded. This is how the next session avoids redoing work.

## 5. Evaluation protocol (user-defined, use it everywhere)

Sample on **Turbo**: 8 steps, CFG 1, mu 1.15, Raw fp8 + Turbo LoRA. **Do not use Raw for routine samples**: it is slow, and the user's experience is that Raw-trained LoRAs judged on Turbo are what matters. Krea itself recommends Turbo for inference. Use Raw only for a specific diagnostic that needs it (test T1 below, a few images).

**Columns per row** (same seed per row; 2 seeds where cheap):
1. A (reference) and B (ground truth);
2. **native baseline**: the same stock graph **without our LoRA** (`adapter_name=None` in `k2ab_eval_stock.graph`). This is the user's "traditional method": the model sees the image through Qwen3-VL but imprecisely;
3. **our LoRA @ 1.0**;
4. **our LoRA @ reduced strength** (~0.6–0.75). `graph()` hard-codes `strength_model=1.`; add a parameter;
5. optional control: our LoRA @ 1.0 with a **shuffled reference** (tells "uses the ref" from "memorised").

**Rows:**
- a few **training pairs** with (a) the original caption and (b) a **slightly altered caption**, e.g. a different action, pose or camera, to see whether the prompt is obeyed rather than A/B being copied;
- **held-out pairs**;
- a few **short user-style prompts**.

Never judge only with the dataset caption: that hid the "ignores prompt" failure last time.

**Resolution / AR:** sample at the training bucket of the target, ~1 MP.
- `tools/krea2_v2_eval.py::bucket_dims` still uses the OLD 7 geometric buckets (0.5–2.0, **no 16:9**). Switch it to the real AR list used in training: `[1,2],[9,16],[2,3],[3,4],[1,1],[4,3],[3,2],[16,9],[2,1],[7,3]`.
- Check every pair: A and B must have compatible AR. `reference_pixels="target"` crop-fits A to B's bucket, so log how much of A gets cropped. Nothing may silently center-crop 10% of the frame again.

**Known artefact:** with `index_timestep_zero`, an untrained or very young adapter, and even the base with no LoRA, produce a 16-px grid of speckles ("manchas"). The base model never saw t=0 reference tokens. It disappears after ~125–250 training steps. Speckles at step 5–100 are therefore expected. Speckles that persist past ~250–500 steps, or any NaN/black/saturated output, mean something is wrong.

Also use objective checks:
- **validation loss** on held-out at fixed t-quantiles, via `eval_datasets` + `eval_every_n_steps` in the diffusion-pipe config. It was never used before; see `examples/main_example.toml`.
- **LoRA size:** `tools/lora_delta_audit.py` on each checkpoint. The user-approved probe512@1000 had block-0 `mlp.down` |ΔW|₂/|W|₂ ≈ 0.18; the "burnt" 1024@3000 had 0.355. This is a heuristic, not a law.

## 6. Work plan (adapt freely; record why when you deviate)

**Phase 0: setup and gates.** Models, dataset, ComfyUI, TE parity gate, held-out split, fixed eval script (§5).

**Phase 1: decisive tests on existing adapters (~1–1.5 h).** From `docs/DIAGNOSTICO_KREA2_2026-10-11.md` §6.
- Adapters:
  - `AdwolfCzar/krea2-turbo-1024`: `checkpoints/20261009_02-44-34/step{500..3000}`;
  - `AdwolfCzar/krea2-probe512-turbo`: `checkpoints/20261009_01-03-48/step{250..1000}`;
  - A1000: private `AdwolfCzar/krea2-ab-runs`, `checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000`.
- Tests:
  - **T1 (the one Raw test):** the same few images with 1024@500/1500/3000, probe@1000 and A1000, on Turbo *and* on Raw (28 steps, CFG ~4). If Raw is crisp where Turbo is hazy, "train on Raw" is confirmed.
  - **T2:** official Turbo checkpoint versus Raw + Turbo LoRA (no task LoRA), same seeds.
  - **T3:** strength sweep.
  - **T4:** caption versus short/user prompts.
  - **T5:** quantile validation loss per checkpoint.
  - **T6:** grep the old run log for `lora key not loaded`, if available.

**Phase 2: main method ("native on Raw", diagnosis §7.1).** Config derived from `examples/krea2_probe512_turbo/probe.toml`:
- **remove `merge_adapters`**; base `krea2_raw_fp8_scaled` with `base_quant="fp8_scaled"`;
- `timestep_law = "krea2_apex"`, and remove `shift` and `high_noise_prob`: the law is exclusive with shift and covers both ends;
- micro_batch 1 × `gradient_accumulation_steps` 2 at 512;
- r32 (or r64), LR 1e-4, cosine sized for ~1.5 epochs;
- real AR buckets, `reference_pixels="target"`, `vl_grounding=true`, `reference_timestep="zero"`, `caption_dropout=0`;
- `save_every_n_steps` ~125, plus `eval_datasets`.

Sequence:
1. Smoke for 5–10 steps: crash, VRAM, t-distribution and NaN check, not quality.
2. 200–500-step probe: direction is usually visible by 200–500 steps (§5 grid).
3. Full 512 run with eval every ~125–250 steps; stop on the eval window, not on a step budget. Previous approved results all lived at 0.5–1.3 epochs.
4. Short 1024 finishing pass from the best 512 checkpoint: ~10–20% of the steps, LR ~5e-5, batch 1. Optionally try `reference_pixels="node_1mp"` here.
5. Optionally average 2–3 neighbouring good checkpoints.

**Phase 3, only if Phase 2 copies the reference under shuffled-ref / altered-prompt tests.** Options, one variable at a time:
- a short-instruction caption tier;
- reference position offset (arm C in `docs/HANDOFF_KREA2_AB.md`);
- the "beta1 corrected" routed method (`examples/krea2_ab/B_beta1_fixed.toml` + `docs/AUDITORIA_2026-08-06.md` Fase 1). Caveat: a routed LoRA cannot load in a stock loader and needs the `ctxrush-edit` node.

You may test other methods you find promising; change one variable at a time against a control.

**Rules learned the hard way:**
- one variable at a time versus a control;
- never extend or resume a run without checking that the LR in the log is what the config says;
- never use the base's inference shift as the training timestep law;
- never train on a base that differs from the inference base;
- never trust 3 cases with the training caption;
- short probes before long runs.

## 7. Throughput references (same GPU class)

- 512, batch 2, A1000 code: ~2.8–3.2 s/step.
- 1024, batch 1: ~6.5 s/step, peak 26.6 GB.
- The training adapter was merged into weights in those runs, so its removal should not change speed.

## 8. Secondary: Anima next-scene (only after Krea is on track, or in GPU idle gaps)

Diagnosis: `docs/DIAGNOSTICO_KREA2_2026-10-11.md` §5. The v2 1024 run (`AdwolfCzar/anima-nextscene-a-aligned-1024-v2`) achieves next-scene but lowers general quality. Suspects:
- constant LR 1e-4 for 5 epochs;
- captions without quality/score/artist tags while the LoRA covers cross-attn;
- no short caption tier.

Cheap test: a fixed-seed grid of base, step500/1000/2000/4325, prompts with and without quality tags, and strength 0.6/0.8.
