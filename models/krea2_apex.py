"""krea2_apex — the node-matched reference contract (docs/KREA2_APEX_SPEC.md).

Everything here exists to make ONE thing true: a LoRA trained by this pipeline
runs under a stock ``LoraLoaderModelOnly`` + ``comfyui-krea2edit`` (>= 1.2.4,
``fit_mode='fit'``, ``grounding_px=768``) with byte-identical geometry. The
deltas against ``krea2_edit`` are:

1. **Geometry** — ``position_mode='frame_fit'``: the reference sits on the RoPE
   frame axis (``pos[...,0] = 1``) and is CENTERED inside the target grid with
   FRACTIONAL offsets (``max(0,(G_tgt-G_ref)/2)``). No ``width_shift``, no
   position rescaling: the pixels were already resampled to the target's grid
   density, so the position grid is stride-1 by construction.
2. **Reference fit** — the control image is preprocessed with the node's own
   ``fit`` branch (AR-preserving, CROP_TOL 8% fill, /16 floor snap capped at the
   target, crop-to-grid, bicubic + antialias) instead of ``ImageOps.fit``
   (center-crop + stretch to the full grid), which is the v1 geometry the node
   abandoned after the seam-doubling RCA.
3. **Reference timestep** — forced to ``'target'``: the node computes a single
   ``tvec`` for the whole sequence, so a ``t=0`` per-token LoRA degrades there
   SILENTLY.
4. **LoRA scope** — blocks + txtfusion + ``txtmlp`` (258 modules). ``txtmlp`` is
   the only 2560->6144 projection through which the Qwen3-VL vision tokens
   enter the DiT width; it was frozen in every previous arm.
5. **Text encoding** — live (``live_text_encoding = true``): no text cache, so
   the grounding-resolution jitter and the caption clause dropout are resampled
   PER STEP instead of being frozen per file at cache time.
"""

import random

import torch
from torch import nn
import torch.nn.functional as F
import peft
from PIL import Image

from models.krea2_edit import Krea2EditPipeline
from utils.common import is_main_process, round_to_nearest_multiple


# Near-matched AR tolerance of the node's fit branch: within 8% the reference is
# center-cropped to fill the target grid EXACTLY, because 1-2 token margins are
# not harmless (target edge columns with no reference correspondence get filled
# by repeating adjacent reference content).
CROP_TOL = 0.08
# 224 block linears + 32 txtfusion linears (projector excluded) + 2 txtmlp.
EXPECTED_ADAPTER_TARGETS = 258


def load_rgb_chw(path):
    """PIL -> float32 CHW in [0,1], white background under transparency."""
    image = Image.open(path)
    if image.mode == 'RGBA' or ('transparency' in image.info and image.mode != 'RGB'):
        rgba = image.convert('RGBA')
        canvas = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        image = canvas.convert('RGB')
    else:
        image = image.convert('RGB')
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(image.height, image.width, 3).to(torch.float32) / 255.0
    return pixels.permute(2, 0, 1).contiguous()


def fit_reference_pixels(image_chw, target_h_px, target_w_px):
    """Port of comfyui-krea2edit ``_fit_encode_image`` fit branch (__init__.py:88-126).

    Input/output are CHW float tensors in [0,1]. The returned size is <= the
    target on both axes and a multiple of 16, which is exactly what the
    ``frame_fit`` centered-fractional placement expects.
    """
    img = image_chw.unsqueeze(0)
    ih, iw = img.shape[-2:]
    px_h, px_w = int(target_h_px), int(target_w_px)
    sc = min(px_h / ih, px_w / iw)
    if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
        # Near-matched AR: minimal center-crop, then fill the grid exactly.
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0:y0 + ch, x0:x0 + cw]
        nh, nw = px_h, px_w
    else:
        # Genuine AR mismatch: /16 floor snap capped at the target's /16 floor,
        # then CROP-TO-GRID so the fitted axis lands on the /16 grid at scale sc
        # EXACTLY (resizing ih*sc -> floor16 squashes content by up to 15px and
        # the misregistration peaks at the reference band edge = the seam).
        nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
        nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
        ch2 = min(ih, max(1, int(round(nh / sc))))
        cw2 = min(iw, max(1, int(round(nw / sc))))
        y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
        img = img[..., y0:y0 + ch2, x0:x0 + cw2]
    img = F.interpolate(img.float(), size=(nh, nw), mode='bicubic', antialias=True)
    return img.squeeze(0).clamp(0.0, 1.0)


def fit_reference_file(path, target_h_px, target_w_px):
    """``fit_reference_pixels`` from a path, returned in the trainer's [-1,1] range."""
    pixels = fit_reference_pixels(load_rgb_chw(path), target_h_px, target_w_px)
    return pixels * 2.0 - 1.0


class PreprocessFitControlFile:
    """Drop-in replacement for ``PreprocessMediaFile`` on CONTROL files only.

    Same call signature and same return contract (a list of ``(tensor, mask)``
    with the tensor shaped ``(C, frames, H, W)``), so ``latents_map_fn`` needs a
    one-line swap. A plain class (not a closure) because the caching flow runs
    it inside forked ``datasets.map`` workers.
    """

    def __init__(self, round_height=16, round_width=16):
        self.round_height = round_height
        self.round_width = round_width

    def __call__(self, spec, mask_filepath=None, size_bucket=None):
        path = spec[1] if isinstance(spec, (list, tuple)) else spec
        if size_bucket is None:
            raise ValueError('krea2_apex reference fit needs the target size bucket')
        bucket_w, bucket_h = size_bucket[0], size_bucket[1]
        target_h = round_to_nearest_multiple(bucket_h, self.round_height)
        target_w = round_to_nearest_multiple(bucket_w, self.round_width)
        pixels = fit_reference_file(path, target_h, target_w)
        return [(pixels.unsqueeze(1), None)]


class Krea2ApexPipeline(Krea2EditPipeline):
    name = 'krea2_apex'
    config_section = 'krea2_apex'
    # NOTE: apex does NOT subclass the initial layer — the frame_fit branch was
    # added additively to Krea2ReferenceInitialLayer, so the checkpointable
    # class name stays the parent's (DeepSpeed matches by exact class name; a
    # wrong name here silently disables activation checkpointing on the layer
    # that owns the trainable txtfusion/txtmlp).
    checkpointable_layers = ['Krea2ReferenceInitialLayer', 'TransformerLayer']
    adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.', '.txtmlp.')

    def __init__(self, config):
        super().__init__(config)
        section = config.get(self.config_section, {})

        # --- Geometry: not configurable, it IS the contract. ---
        self.position_mode = 'frame_fit'
        self.reference_position_offset = 1.0
        self.reference_position_scale = 1.0
        self.independent_condition = False

        # --- Reference timestep: 'target' unless explicitly opted out. ---
        requested_timestep = section.get('reference_timestep', 'target')
        allow_node_incompatible = bool(section.get('allow_node_incompatible', False))
        if requested_timestep != 'target' and not allow_node_incompatible:
            raise ValueError(
                "krea2_apex requires reference_timestep='target' (comfyui-krea2edit computes a "
                "single tvec for the whole sequence, so a 'zero' adapter degrades there in "
                "silence). Set allow_node_incompatible = true to run the declared A/B."
            )
        self.reference_timestep_mode = requested_timestep

        # --- Live text encoding (see §2.3 of the spec). ---
        self.live_text_encoding = bool(config['model'].get('live_text_encoding', False))

        # --- Caption regime, resampled per step in encode_text_live. ---
        self.clause_dropout_prob = float(section.get('clause_dropout_prob', 0.0))
        self.clause_dropout_rate = float(section.get('clause_dropout_rate', 0.60))
        self.keep_clauses = int(section.get('keep_clauses', 1))
        if not 0.0 <= self.clause_dropout_prob <= 1.0:
            raise ValueError('clause_dropout_prob must be between 0 and 1')
        if not 0.0 <= self.clause_dropout_rate <= 1.0:
            raise ValueError('clause_dropout_rate must be between 0 and 1')
        if self.keep_clauses < 0:
            raise ValueError('keep_clauses must be >= 0')
        if self.caption_dropout + self.clause_dropout_prob > 1.0:
            raise ValueError('caption_dropout + clause_dropout_prob must be <= 1')

        # --- Batch: frame_fit produces per-sample reference shapes. ---
        micro_batch = config.get('micro_batch_size_per_gpu', 1)
        if isinstance(micro_batch, list):
            micro_batch = max(entry[1] for entry in micro_batch)
        if int(micro_batch) != 1:
            raise ValueError(
                'krea2_apex requires micro_batch_size_per_gpu = 1: the fit preprocess gives '
                'references of different shapes per sample, which cannot be collated into a '
                'single reference tensor.'
            )
        if int(config.get('caching_batch_size', 1)) != 1:
            raise ValueError('krea2_apex requires caching_batch_size = 1 (see above)')
        if int(config.get('pipeline_stages', 1)) != 1 and self.live_text_encoding:
            raise ValueError(
                'live_text_encoding assumes pipeline_stages = 1: with more stages both the '
                'first and the last stage pull from the dataloader and would each run the TE'
            )

    # ------------------------------------------------------------------
    # Reference pixels
    # ------------------------------------------------------------------
    def get_preprocess_control_file_fn(self):
        return PreprocessFitControlFile()

    # ------------------------------------------------------------------
    # LoRA scope
    # ------------------------------------------------------------------
    def _promote_scoped_linears(self, prefixes):
        """Make comfy's non-``nn.Linear`` linears adaptable by peft.

        The fp8_scaled checkpoint is loaded through ``comfy.ops`` mixed-precision
        ops, whose ``Linear`` derives from plain ``nn.Module``. ``models/base.py``
        ``dequantize()`` swaps those back to a real ``nn.Linear`` only when the
        weight was a ``QuantizedTensor`` — and ``txtmlp`` is stored in high
        precision, so it is left as the comfy class. peft dispatches on
        ``isinstance(target, nn.Linear)``, so without this the module is simply
        invisible (256 targets instead of 258, silently missing exactly the
        projection we came here to train).

        Same swap as ``dequantize()``: keep the Parameters, change the wrapper.
        """
        import accelerate
        import comfy.ops
        operations = comfy.ops.disable_weight_init
        model = self.diffusion_model
        promoted = []
        for name, module in list(model.named_modules()):
            if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
                continue
            if isinstance(module, nn.Linear) or not hasattr(module, 'in_features'):
                continue
            weight = getattr(module, 'weight', None)
            if weight is None or weight.__class__.__name__ == 'QuantizedTensor':
                raise RuntimeError(
                    f'Cannot make {name} ({type(module).__name__}) trainable: unexpected weight '
                    f'{type(weight).__name__}. Refusing to guess.'
                )
            bias = getattr(module, 'bias', None)
            with accelerate.init_empty_weights():
                new_linear = operations.Linear(
                    module.in_features, module.out_features, bias=bias is not None
                )
            new_linear.comfy_cast_weights = True
            new_linear.weight = weight
            if bias is not None:
                new_linear.bias = bias
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._modules[child] = new_linear
            promoted.append(name)
        if promoted and is_main_process():
            print(f'[{self.name}] promoted to nn.Linear for LoRA: {promoted}')
        return promoted

    def configure_adapter(self, adapter_config):
        """Blocks + text fusion + txtmlp = 258 linears / 516 tensors.

        ``txtmlp.1`` / ``txtmlp.3`` are the ONLY projection carrying the fused
        Qwen3-VL stack (including the reference's vision tokens) into the DiT
        width. ``txtfusion.projector`` (Linear(12,1), the Qwen3-VL depth
        selector) stays excluded: peft would save it as a bare ``.weight``,
        which ``comfy/lora.py`` ignores in silence (it consumes ``.diff``).
        """
        target_model = self.diffusion_model
        self._promote_scoped_linears(('txtmlp',))
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            in_scope = (
                module.__class__.__name__ in ('SingleStreamBlock', 'TextFusionTransformer')
                or name == 'txtmlp' or name.startswith('txtmlp.')
            )
            if not in_scope:
                continue
            for full_name, submodule in module.named_modules(prefix=name):
                if not isinstance(submodule, nn.Linear):
                    continue
                if full_name.endswith('projector'):
                    continue
                target_linear_modules.add(full_name)
        targets = sorted(target_linear_modules)
        if not targets:
            raise RuntimeError('No Krea2 DiT linear modules found for the apex adapter')

        if len(targets) != EXPECTED_ADAPTER_TARGETS:
            blocks = [t for t in targets if t.startswith('blocks.')]
            fusion = [t for t in targets if 'txtfusion' in t]
            txtmlp = [t for t in targets if t.startswith('txtmlp.')]
            other = [t for t in targets if t not in set(blocks) | set(fusion) | set(txtmlp)]
            present = [
                f'{name}:{type(module).__module__}.{type(module).__name__}'
                f':linear={isinstance(module, nn.Linear)}'
                for name, module in target_model.named_modules() if 'txtmlp' in name
            ]
            raise RuntimeError(
                f'krea2_apex expected {EXPECTED_ADAPTER_TARGETS} LoRA targets '
                f'(224 blocks + 32 txtfusion + 2 txtmlp) but found {len(targets)}: '
                f'blocks={len(blocks)}, txtfusion={len(fusion)}, txtmlp={len(txtmlp)}, '
                f'unclassified={len(other)}.\nUnclassified: {other[:20]}\n'
                f'txtmlp targets: {txtmlp}\ntxtmlp modules in the model: {present}\n'
                f'Verify the scope against the model before changing this number.'
            )

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=targets,
            )
        elif adapter_type == 'lokr':
            peft_config = peft.LoKrConfig(
                r=adapter_config['rank'],
                decompose_factor=adapter_config['decompose_factor'],
                alpha=adapter_config['alpha'],
                rank_dropout=adapter_config['rank_dropout'],
                target_modules=targets,
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)
        if is_main_process():
            fusion = sum(1 for name in targets if 'txtfusion' in name)
            txtmlp = sum(1 for name in targets if name.startswith('txtmlp.'))
            print(
                f'[{self.name}] apex LoRA targets: {len(targets)} linears '
                f'({len(targets) - fusion - txtmlp} in DiT blocks, {fusion} in text fusion, '
                f'{txtmlp} in txtmlp)'
            )
            self.lora_model.print_trainable_parameters()
        for name, parameter in target_model.named_parameters():
            parameter.original_name = name
            if parameter.requires_grad:
                parameter.data = parameter.data.to(adapter_config['dtype'])

    # ------------------------------------------------------------------
    # Live text encoding + per-step caption regime
    # ------------------------------------------------------------------
    def apply_caption_regime(self, caption):
        """Per-step caption mixture (spec §1 item 23 / §2.5).

        0.05 empty (the grounded unconditional CFG uses at inference) ·
        0.40 clause dropout (keep the first clause, drop each remaining one
        with p=0.60) · the rest untouched. The first clause is shot type in
        practically every caption of this dataset, and shot type is exactly
        what CHANGES between A and B — it is C_delta, never C_shared.
        """
        r = random.random()
        if r < self.caption_dropout:
            return ''
        if r < self.caption_dropout + self.clause_dropout_prob:
            parts = caption.split(', ')
            kept = parts[:self.keep_clauses] + [
                part for part in parts[self.keep_clauses:]
                if random.random() >= self.clause_dropout_rate
            ]
            return ', '.join(kept)
        return caption

    def encode_text_live(self, batch):
        """Encode this step's captions with the resident Qwen3-VL.

        Runs in the TRAINING process (the dataloader generator is pulled by the
        training loop, not by a forked worker), which is what makes CUDA legal
        here.
        """
        captions = batch.get('caption', None)
        if captions is None:
            raise RuntimeError('live_text_encoding needs raw captions in the batch')
        if isinstance(captions, str):
            captions = [captions]
        control_files = batch.get('control_file', None)
        if isinstance(control_files, str):
            control_files = [control_files]
        captions = [self.apply_caption_regime(caption) for caption in captions]

        text_encoder = self.text_encoders[0]
        if isinstance(text_encoder, nn.Module):
            text_encoder = text_encoder.to('cuda')
        else:
            text_encoder.load_model_if_needed()
        outputs = self.get_call_text_encoder_fn(text_encoder)(
            captions, [False] * len(captions), control_files
        )
        # get_call_text_encoder_fn is @torch.inference_mode(); its tensors CANNOT
        # enter an autograd graph, and with LoRA on txtfusion/txtmlp the context
        # does exactly that. Clone OUTSIDE the inference region (here) to get
        # ordinary tensors back.
        for key, value in outputs.items():
            batch[key] = [tensor.clone() for tensor in value]
        return batch

    # ------------------------------------------------------------------
    # Contract metadata
    # ------------------------------------------------------------------
    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        n_ref = '1024'
        metadata.update({
            'control_family': 'krea2_apex',
            'geometry_contract': 'krea2_frame_fit_v1',
            'ref_axis': 'frame',
            'ref_frame_index': str(int(self.reference_position_offset)),
            'ref_offsets': 'centered_fractional',
            'ref_fit': 'ar_preserve_crop_to_grid_16_bicubic',
            'px_per_token': '16',
            'node_compat': 'comfyui-krea2edit>=1.2.4 ; fit_mode=fit ; grounding_px=768',
            'grounding': (
                f'qwen3vl_native_pre_fit_longest{self.vl_longest_side or 0}'
                + (
                    f'_jitter{self.vl_grounding_jitter[0]}-{self.vl_grounding_jitter[1]}'
                    if self.vl_grounding_jitter else ''
                )
                + ('_perstep' if self.live_text_encoding else '_perfile')
            ),
            'timestep_law': (
                f'krea2_apex m=0.337+0.335*ln(N/{n_ref}) s=1.538+0.130*ln(N/{n_ref})'
                if self.model_config.get('timestep_law', None) == 'krea2_apex' else 'none'
            ),
            'caption_regime': (
                f'{1.0 - self.caption_dropout - self.clause_dropout_prob:.2f} full | '
                f'{self.clause_dropout_prob:.2f} clause_dropout('
                f'{self.clause_dropout_rate:.2f},keep{self.keep_clauses}) | '
                f'{self.caption_dropout:.2f} empty'
            ),
            'live_text_encoding': str(self.live_text_encoding).lower(),
            'lora_targets': 'blocks+txtfusion+txtmlp',
        })
        return metadata
