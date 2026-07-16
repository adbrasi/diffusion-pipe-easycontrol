# Reference adapters for Ideogram 4 and Krea 2

This fork contains three clean-reference adapter families for each model:

| Model type | Adapter behavior |
|---|---|
| `ideogram4_ic_lora` | Global task LoRA over `[text, noisy target, clean reference]` |
| `ideogram4_ominicontrol` | OminiControl v1; LoRA runs only on reference rows |
| `ideogram4_ominicontrol2` | v1 plus compact reference encoding and independent condition attention |
| `krea2_ic_lora` | Global task LoRA over `[text, noisy target, clean reference]` |
| `krea2_edit` | `krea2_ic_lora` plus Qwen3-VL image grounding of the reference (public Krea Edit dual contract) |
| `krea2_ominicontrol` | OminiControl v1; LoRA runs only on reference rows |
| `krea2_ominicontrol2` | v1 plus compact reference encoding and independent condition attention |

All variants supervise and decode target tokens only. Ideogram uses internal
reference timestep `1.0` (clean); Krea uses reference timestep `0.0` (clean).
OminiControl adapters preserve PEFT checkpoint keys while routing each LoRA
delta only to reference rows. Target and text rows execute the frozen base
linear path.

## Dataset contract

Target and reference files must have matching stems:

```text
dataset/
  target_images/
    shot_0001.png
    shot_0001.txt
  reference_images/
    shot_0001.png
```

The caption describes the target, including the intended change relative to
the reference. Each dataset directory must define `control_path`; see
`examples/ideogram4_reference_dataset.toml` and
`examples/krea2_reference_dataset.toml`.

For next-shot consistency, avoid a dataset made only from adjacent frames.
Mix temporal gaps so copying the reference is not the easiest solution.

OminiControl2 downsamples reference pixels before VAE encoding. Do not reuse a
latent cache made with IC-LoRA/OminiControl v1; regenerate it after selecting a
v2 config. The saved checkpoint records `condition_encode`, stride, position
scale, attention independence and condition-only routing.

## Krea 2 Edit dual conditioning

`krea2_edit` reproduces the public Krea Edit contract (Krea2OstrisEdit /
ai-toolkit `edit=true` / ComfyUI grounded encode). Each reference conditions
the model twice:

- clean VAE latents appended after the noisy target at timestep `0.0`, RoPE
  frame `1` (unchanged from `krea2_ic_lora`);
- an image-grounded Qwen3-VL prompt: the reference is serialized as
  `Picture 1: <|vision_start|><|image_pad|><|vision_end|>` ahead of the
  caption inside the standard Krea 2 conditioning template. The VL copy is
  downscaled (aspect preserved, never upscaled) to `vl_image_max_pixels`
  (default `147456` = 384*384).

Operational notes:

- The text encoder checkpoint must include the Qwen3-VL vision tower
  (`visual.*` weights). `tools/preflight_krea2_edit.py` verifies this from the
  safetensors header without loading torch.
- Text embeddings now depend on the reference image. Expect roughly
  `(caption + ~144 vision tokens) x 12 layers x 2560 x 2 bytes ~= 8-14 MB` of
  cache per pair, and pass `--regenerate_cache` whenever reference images
  change in place.
- `condition_dropout` must be `0.0` (enforced): the public edit training never
  drops references — CFG contrasts the prompt only, with the reference
  grounded in both conditional and unconditional embeddings — and dropping
  only the VAE branch while Qwen3-VL keeps seeing the reference would be an
  inconsistent partial dropout.
- `condition_token_stride` is fixed at 1; compact-stride encoding belongs to
  the OminiControl variants only.
- LoRA coverage matches the public dual-conditioning checkpoints (verified
  from the ostris style-reference and conradlocke identity-edit headers):
  every linear in the 28 `SingleStreamBlock`s **plus** the 4
  `TextFusionTransformer` blocks (`txtfusion.layerwise_blocks` and
  `txtfusion.refiner_blocks`, 512 tensors total), excluding the layer
  projector and `txtmlp`. The text fusion consumes the Qwen3-VL stack —
  including the reference's vision tokens — so it must adapt. `krea2_ic_lora`
  (VAE-only) keeps its blocks-only coverage.
- The diffusion-pipe path currently crops the reference and target to the same
  training bucket. This is internally consistent and matches the paired
  dataset used by this fork, but differs from AI Toolkit/Krea2OstrisEdit,
  which can preserve an independent reference aspect ratio under a separate
  pixel budget.

Runtime validation on an RTX 5090 covered a 5-step smoke, a 50-step pilot and
a resumed 50-to-60-step run. The saved adapter contained the expected 512
LoRA tensors (448 in the DiT blocks and 64 in `txtfusion`), and loaded without
key conversion when compared with the public rank-64 identity-edit layout.
The training and inference paths both use `to_layers()`, and their Krea
resolution-dependent schedule was numerically checked against the Ostris
pipeline.

```bash
python tools/preflight_krea2_edit.py --config examples/krea2_edit.toml
```

## Safe pilot order

Start with 512px and one GPU:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/ideogram4_ominicontrol.toml \
  --cache_only --regenerate_cache
```

Then run a short training pilot by lowering `max_steps` to 20-50 in a copied
config. Verify that a checkpoint contains `adapter_model.safetensors` and no
`ADAPTER_AUDIT_FAILED.txt` before starting a long run.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config /workspace/configs/ideogram4_ominicontrol_smoke.toml
```

Use the equivalent Krea examples for Krea 2. OminiControl condition-only
routing can leave the final block's reference-only output/MLP LoRA parameters
unused. The supplied examples intentionally use a single GPU, avoiding DDP's
unused-parameter constraint.

For the one-reference Krea 2 IC-LoRA pilot, run the dependency-free preflight
before allocating the GPU:

```bash
python tools/preflight_krea2_ic_lora.py \
  --config examples/krea2_ic_lora.toml
```

The expected dataset pair is `control/reference image -> target image`. The
caption belongs to the target and should state the intended change. The Krea
preflight validates one control per target, matching stems, captions, model
paths, one-frame buckets and the 512px token count.

## Inference without ComfyUI

`tools/infer_reference_adapter.py` uses the training pipeline's own
`to_layers()` objects. It does not reproduce packing separately. This keeps
training and inference identical for role masks, positions, timesteps,
compact-reference transforms and output slicing.

First validate checkpoint metadata without loading a model:

```bash
python tools/infer_reference_adapter.py \
  --config /workspace/configs/ideogram4_ominicontrol_smoke.toml \
  --adapter /workspace/output/20260714_000000/step20 \
  --validate-only
```

Generate an image:

```bash
python tools/infer_reference_adapter.py \
  --config /workspace/configs/ideogram4_ominicontrol_smoke.toml \
  --adapter /workspace/output/20260714_000000/step20 \
  --reference /workspace/test/reference.png \
  --prompt '{"description":"same character walking through the room"}' \
  --width 512 --height 512 \
  --steps 20 --seed 123 \
  --text-guidance 1.0 \
  --reference-guidance 1.0 \
  --output /workspace/test/result.png
```

### Krea 2 sampling profiles

Krea 2 Raw is not a CFG-free model. The runner now resolves model-specific
defaults from the diffusion checkpoint name:

| Checkpoint | Steps | `--text-guidance` | Timestep shift |
|---|---:|---:|---|
| Krea 2 Raw | 28 | 5.5 | resolution-dependent `mu` |
| Krea 2 Turbo | 8 | 1.0 | fixed `mu=1.15` |

This runner uses standard CFG,
`uncond + scale * (cond - uncond)`. Krea2OstrisEdit exposes
`cond + scale * (cond - uncond)`, so its Raw guidance `4.5` is `5.5` here.
Use `--krea-variant raw` or `--krea-variant turbo` when the checkpoint path
does not identify the variant. Explicit `--steps` and `--text-guidance` values
always override the profile.

Infer at the same resolution/aspect bucket used by the evaluated pair before
testing generalization. In particular, using Raw with the old generic defaults
of 20 steps and guidance 1.0 can leave a noisy or mosaic-like result that looks
like broken reference packing even when the checkpoint is valid.

`--adapter-scale 0` is **not** a vanilla Krea baseline: it disables the LoRA
weights but still sends clean reference tokens and image-grounded Qwen3-VL
conditioning through the custom edit sequence. Use a true text-to-image Krea
pipeline without a reference to validate the base model. Public edit LoRAs
without this fork's contract metadata can be tested deliberately with
`--allow-contract-mismatch` after their architecture and rank are verified.

The adapter argument may be either the safetensors file or a directory that
contains exactly one safetensors file. The runner offloads the diffusion model
before VAE decode and honors `blocks_to_swap` from the training config.

Independent guidance is available because reference dropout is part of the
training recipe:

```text
v = v_unconditional
  + reference_guidance * (v_reference - v_unconditional)
  + text_guidance * (v_full - v_reference)
```

Increase reference guidance only after a scale of `1.0` works. High values can
turn consistency into copying. `--reference-fit exact` is recommended for
spatial controls; it rejects accidental resize/crop misalignment.

Krea 2 inference follows its official resolution-dependent flow schedule. The
runner derives `mu` from the image-token count using the official
`y1=0.5`/`y2=1.15`, 256px/1280px endpoints. Use `--mu 1.15` only when an
experiment deliberately needs a fixed Krea schedule; `--shift` is reserved for
the non-Krea schedulers.

OminiControl2 inference currently uses its compact and independent condition
contract without KV caching. This is correct but not yet the paper's optimized
runtime. A later cache optimization can be added without changing checkpoints.
