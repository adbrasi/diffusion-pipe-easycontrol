# Ideogram 4 IC-LoRA RunPod checklist

This path trains one reference-conditioned Ideogram 4 adapter with the packing
contract:

```text
[text | noisy target | clean reference]
```

The first RunPod job should prove the complete cache, forward/backward, and save
path at 512px. Do not begin a large 1024px run before the ten-step smoke test
saves a valid adapter.

## 1. Environment

From the repository root:

```bash
git submodule update --init --recursive
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Launch all commands from this repository root. Like `train.py`, the preflight
tool resolves relative TOML paths from the current working directory.

Place the three ComfyUI-format components at the paths used by the example
configs, or edit both Ideogram config files:

```text
/workspace/models/ideogram4_fp8_scaled.safetensors
/workspace/models/flux2-vae.safetensors
/workspace/models/qwen3vl_8b_fp8_scaled.safetensors
```

## 2. Paired dataset

Targets and references are paired by filename stem:

```text
/workspace/dataset/
├── target_images/
│   ├── shot_0008.png
│   ├── shot_0008.txt
│   ├── shot_0020.png
│   └── shot_0020.txt
└── reference_images/
    ├── shot_0008.png
    └── shot_0020.png
```

The image in `reference_images` is the previous/source frame. The image in
`target_images` is what the model must generate. The target caption should
describe the desired target and the transition from the reference. Use varied
temporal gaps; a dataset made only from adjacent near-identical frames creates
an easy copy shortcut.

Run the dependency and pair check before using the GPU:

```bash
python tools/preflight_ideogram4_ic_lora.py \
    --config examples/ideogram4_ic_lora_smoke.toml
```

## 3. Cache-only validation

```bash
deepspeed --num_gpus=1 train.py --deepspeed \
    --config examples/ideogram4_ic_lora_smoke.toml \
    --cache_only
```

The cache must contain both `latents` and `control_latents`. A missing
`control_path`, unmatched filename stem, or mismatched latent grid is a hard
error rather than a silent text-to-image fallback.

## 4. Ten-step smoke training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=1 train.py --deepspeed \
    --config examples/ideogram4_ic_lora_smoke.toml \
    --trust_cache
```

Success means the run reaches step 10 and writes an
`adapter_model.safetensors`. Inspect its metadata:

```bash
python - <<'PY'
from safetensors import safe_open

path = '/workspace/ideogram4_ic_lora_smoke_output/REPLACE_RUN/step10/adapter_model.safetensors'
with safe_open(path, framework='pt', device='cpu') as handle:
    print(handle.metadata())
PY
```

The metadata must report:

```text
model_type: ideogram4_ic_lora
reference_contract: ideogram4_reference_conditioning_v1
sequence_layout: text,target,reference
reference_indicator: 4
reference_position_offset: 1
reference_model_timestep: 1.0
```

If 512px runs out of memory on a 32GB card, raise `blocks_to_swap` from 8 to
12, then 16. If 1024px later runs out of memory, block swap can reduce weight
residency but cannot remove the quadratic attention cost of the doubled image
sequence; reduce resolution before changing the architecture.

## 5. Pilot training

After the smoke checkpoint is valid:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=1 train.py --deepspeed \
    --config examples/ideogram4_ic_lora.toml \
    --trust_cache
```

The pilot config uses rank 64, learning rate `5e-5`, condition dropout `0.1`,
and 7,000 maximum steps. Treat these as a reproducible starting point, not a
claim that step 7,000 is optimal for a new dataset.

## 6. Inference contract

The stock ComfyUI Ideogram 4 path at the submodule commit does not accept
reference tokens. Loading the LoRA alone is insufficient. The current working
inference implementation is:

- `BitPoet/ComfyUI`, branch `dev-ideogram4-inpaint`
- `BitPoet/ComfyUI-bitpoet-IG4Inpaint`

Those projects implement the same indicator `4`, MRoPE `+1`, clean internal
timestep `1.0`, and target-only output slice used here. In the custom node,
select `center_crop`; diffusion-pipe uses `ImageOps.fit` when it preprocesses
both target and reference images.

Use the normal ComfyUI LoRA loader for `adapter_model.safetensors`, then attach
the reference only to positive conditioning with `Ideogram 4 Reference
Conditioning`. The target latent fixes the output grid, and the node must encode
the reference to exactly the same 128-channel latent shape.

Ideogram model weights and derivatives have license restrictions independent
of this repository's code license. Confirm that the license covering the
weights permits the intended use before using a trained adapter commercially.
