# Next-scene: MOSS-Image × Krea 2 (2026-10-26 → 10-27)

RTX 5090 32 GB (Vast). Goal: given reference image **A** (previous scene) + prompt, generate the **next scene B**,
keeping characters, outfit, style, palette and setting consistent.

All adapters, samples, grids and eval images are on HF (public):
**https://huggingface.co/AdwolfCzar/moss-image-nextscene-lora**. Each run is a subfolder there.

## TL;DR

1. **The dataset captions were the problem, not the models.** Long captions describe B completely, so the text
   alone satisfies the loss and the adapter never learns to read the reference. **Delta captions fix it.** They
   describe only what changes ("the same girl…" with no appearance, "a new …" with a short appearance,
   "Same place." or the new location). Same model and config, only the captions changed: MOSS v1 → v3 went
   from "identity drifts with short prompts" to "identity/outfit/style read from the image".
2. **Krea 2 has much higher reference fidelity than MOSS** (blue hair streak, freckles, scarf, eyepatch,
   photo stays photo). With `krea2_native` (raw-v3 / A1000 recipes) it **copies the reference composition**
   and rarely changes location, especially on Turbo (cfg 1).
3. **The Ostris o_edit method (ai-toolkit `krea2`, `edit: true`) is the only Krea method that obeys scene and
   camera changes.** It tends to output **split panels / storyboard pages**. The prefix
   `Next scene of Picture 1.` makes it **write text** ("Picture 1", "Next scene 2") into the image, and removing
   the prefix removes the text. The panels come from the base model's own tendency plus multi-panel manga
   pages in the data, and were decreasing with training.
4. **MOSS: 576 vision tokens (`vlm_max_pixels` 768²) > 144 default** for small details (eyepatch, beard,
   patterned dress, skull print). It is uneven elsewhere. Costs about +13% step time.

## Data

- Source: `AdwolfCzar/ofificial_next_scene_v11` → `dataset-2-natural.zip`. All 3 zips of that repo contain the
  same 1,736 image pairs; only the captions differ.
- 6 held-out validation pairs (stems in `captions/val6.jsonl`), always excluded from training.
- **recap1000**: 1,000 random train pairs (seed 1234, `captions/pick1000.txt`) re-captioned with OpenRouter
  `meta/muse-spark-1.3-contributor` (reasoning effort low, temp 0.3, `max_tokens` 3000 because the model
  reasons first). Exact template: `scripts/recaption.py`. 1000/1000 OK, **US$0.39 total**, median 50 words.
  816 captions use "the same", 411 "a new", 842 end in "Same place.". Captions: `captions/cap1000.jsonl`.
  The images are the v11 images, so the dataset can be rebuilt with `scripts/build_cap1000.py`.
- `AdwolfCzar/proxima_cena_grounded_original_dataset` (12,455 pairs, 4 subsets) was prepared
  (`scripts/prep_big.py`) but not trained. Use git LFS to download it; the HF API returns 429 on that many files.

## Runs

| run | model / method | data | res | steps | result |
|---|---|---|---|---|---|
| MOSS v1 | MOSS Base, ai-toolkit edit+kv_cache, r32, lr 1e-4, convrot8 | v11 1711 long captions | 768 | 1500 | perfect with the dataset caption; identity drifts with short prompts; no location change |
| MOSS v2 | same | same | 1024 | 2500 (stopped) | sharper, same fidelity pattern; step-250 samples were flat texture (weights fine) |
| **MOSS v3** | same | **recap1000 (delta)** | 768 | 1000 | **big jump**: outfit/identity from the image with no appearance in the prompt; photo/manga OOD OK, watercolor lost |
| **MOSS v4** | v3 + `vlm_max_pixels` 768² (576 tokens) | recap1000 | 768 | 1000 | better small details (eyepatch, beard, dress pattern); some regressions (photo face, fox) |
| Krea raw-v3 | diffusion-pipe `krea2_native`, branch `krea2-raw-v3`, apex timesteps, r32, cosine | recap1000 | 768 | 1000 | best identity/style fidelity; copies composition; Turbo ignores location changes |
| Krea A1000 recipe | branch `krea2-known-good-a1000` (the 2026-09-30 20:44 BRT screenshot adapter): logit_normal+flux_shift, r64, batch 2, constant lr | recap1000 | 768 | 1000 | ≈ raw-v3 at step 250; slowly better prompt obedience (Deku over-shoulder at 1000) |
| **Krea Ostris o_edit** | ai-toolkit `arch: krea2`, `edit/kv_cache/match_target_res`, Raw, r32, linear timesteps | recap1000 | 768 | 1000 | **obeys location/camera changes**; split-panel/storyboard tendency; prefix → written text |

Configs: `configs/moss_ai_toolkit/`, `configs/krea2_diffusion_pipe/` (diffusion-pipe TOMLs; method B/beta1_fixed
and a-native were set up but cancelled), `configs/krea2_ostris_ai_toolkit/`.

### How the A1000 was identified
`known_good_krea2/README.md` on branch `krea2-known-good-a1000` names it as the adapter in the user's 2026-09-30
20:44 BRT screenshot. HF `AdwolfCzar/krea2-ab-runs` has the commit "Review A1000: ten Turbo and ten Raw renders"
at 20:36 BRT that day. Its conditioning is the same as raw-v3 (`TextEncodeQwenImageEditPlus` +
`index_timestep_zero`); only the recipe differs.

## Evaluation

- MOSS: ai-toolkit inference engine (`eval/run_tests.py`) on 14 refs (6 val, 2 train/unseen, Fumika and Terra
  from `7err4/MOSS-Image-CharRef`, and 4 OOD refs generated by MOSS Base: photo, 3D fox, watercolor knight,
  B&W manga, via `eval/gen_refs.py`) × prompts identical / similar / similar-in-delta-format / different. Seed 42.
- Krea `krea2_native`: `tools/krea2_v3_eval.py` (stock ComfyUI `f856877e`), Turbo 8 steps cfg 1, same 45 cases
  (`eval/krea_cases45.json`). A1000: 6 cases every 250 steps via save_quit → eval → resume (`eval/supervisor_a1000.sh`).
- Ostris: in-training samples (Raw, cfg 4, 25 steps) every 250, plus a with/without-prefix test (`eval/grid_prefix.py`).

## Insights / next run

1. Keep delta captions. Consider **dropping the `Next scene of Picture 1.` prefix** for Krea Ostris (it causes
   written text). For MOSS it was harmless.
2. Most delta captions end in "Same place.", so the data teaches "next scene = same location".
   **Add or upweight pairs with a real location or camera change** (captions not ending in "Same place.").
3. Multi-panel manga pages (ds3-like) teach panel layouts. Drop them or caption them explicitly ("a manga page with N panels").
4. Krea Ostris: train longer (2000–3000 steps); panels were decreasing. Use a negative like "split screen, comic
   panels, collage, text". Evaluate in ComfyUI with `ComfyUI-Krea2-Ostris-Edit`: training samples and ai-toolkit
   engine renders differed for the same seed.
5. MOSS: keep `vlm_max_pixels` 768² (it must match at inference). LoRA strength 1.0–1.3; 1.6 distorts.
6. Ops notes: `pkill -f <pattern>` kills your own shell if the pattern is on the command line (use `[x]` tricks).
   diffusion-pipe writes `cache/` inside the dataset dir (27 GB here). Pause with `touch <run>/save_quit`, never SIGTERM.
