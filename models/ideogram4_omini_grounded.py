"""Omini-Grounded for Ideogram 4 — the Krea 2 winning recipe ported.

Three channels, each with its own adapter surface:

* VAE reference tokens with a condition-only ROUTED LoRA on the DiT blocks
  (inherited from OminiControl v1: zero delta on text/target rows).
* The reference image shown to the Qwen3-VL VISUAL TOWER during text encoding
  (grounding): captions resolve entities against the actual reference pixels.
* A GLOBAL LoRA on ``llm_cond_proj`` — the Ideogram analog of Krea's txtfusion:
  it projects the 13 concatenated VL taps (including vision tokens) into the
  DiT, so it must be trainable for the grounding to adapt.

Coupled reference dropout: the text cache stores BOTH the grounded and the
text-only embedding per sample; at train time a single Bernoulli draw selects
the text-only variant AND zeroes the VAE reference latents together, so the two
channels never disagree about the reference's presence. (CFG note: the
inference unconditional should drop the CAPTION and keep the reference in both
channels — that contrast is handled by the runner, not here.)
"""

import hashlib

import numpy as np
import torch
from PIL import Image
from torch import nn

from models.ideogram4_ic_lora import LORA_FORBIDDEN_MODULE_PATTERNS
from models.ideogram4_ominicontrol import Ideogram4OminiControlPipeline
from models.ideogram4_reference_contract import split_lora_target_modules

GROUNDED_EXTRA_TARGET = 'llm_cond_proj'


def prepare_vl_image_longest_side(path, longest_side):
    """Load reference pixels for the visual tower: batch-1 BHWC in [0, 1]."""
    image = Image.open(path).convert('RGB')
    width, height = image.size
    scale = longest_side / max(width, height)
    if scale < 1.0:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.LANCZOS,
        )
    pixels = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
    return pixels.unsqueeze(0)


class Ideogram4OminiGroundedPipeline(Ideogram4OminiControlPipeline):
    name = 'ideogram4_omini_grounded'

    def __init__(self, config):
        super().__init__(config)
        section = config.get('omini_grounded', {})
        self.vl_longest_side = int(section.get('vl_longest_side', 512))
        if self.vl_longest_side < 28:
            raise ValueError('vl_longest_side must be at least 28')
        # Codex-endorsed default: the semantic channel starts small (rank 16);
        # raise only on evidence of undercapacity.
        self.grounded_rank = int(section.get('grounded_rank', 16))

    # ------------------------------------------------------------------ LoRA
    def configure_adapter(self, adapter_config):
        """Blocks coverage (adaln per train_adaln_modulation) + global llm_cond_proj."""
        import peft
        from utils.common import is_main_process

        target_model = self.diffusion_model
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        if not self.train_adaln_modulation:
            target_linear_modules, excluded = split_lora_target_modules(
                sorted(target_linear_modules), LORA_FORBIDDEN_MODULE_PATTERNS
            )
        else:
            target_linear_modules, excluded = sorted(target_linear_modules), []

        grounded_targets = [
            name
            for name, module in target_model.named_modules()
            if isinstance(module, nn.Linear) and name.split('.')[-1] == GROUNDED_EXTRA_TARGET
        ]
        if not grounded_targets:
            raise RuntimeError(f'No {GROUNDED_EXTRA_TARGET} linear found on the diffusion model')
        target_linear_modules = list(target_linear_modules) + grounded_targets

        if is_main_process():
            print(
                f'[{self.name}] LoRA targets: {len(target_linear_modules)} linears '
                f'({len(grounded_targets)} global {GROUNDED_EXTRA_TARGET}), '
                f'excluded {len(excluded)} adaln'
            )

        if adapter_config['type'] != 'lora':
            raise NotImplementedError('ideogram4_omini_grounded only supports type = lora')
        peft_config = peft.LoraConfig(
            r=adapter_config['rank'],
            lora_alpha=adapter_config['alpha'],
            lora_dropout=adapter_config['dropout'],
            bias='none',
            target_modules=target_linear_modules,
            rank_pattern={name: self.grounded_rank for name in grounded_targets},
            alpha_pattern={name: self.grounded_rank for name in grounded_targets},
        )
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)
        if self.condition_only_lora:
            installed = self.condition_lora_router.install(self.diffusion_model.layers)
            if is_main_process():
                print(f'[{self.name}] condition-only routing on {installed} block linears '
                      f'({GROUNDED_EXTRA_TARGET} LoRA stays global)')

    def _audit_adapter_keys(self, keys, save_dir):
        blocks_keys = [k for k in keys if GROUNDED_EXTRA_TARGET not in k]
        grounded_keys = [k for k in keys if GROUNDED_EXTRA_TARGET in k]
        if not grounded_keys:
            print(f'[{self.name}] WARNING: no {GROUNDED_EXTRA_TARGET} keys in checkpoint '
                  '(grounding channel untrained?)')
        super()._audit_adapter_keys(blocks_keys, save_dir)

    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        metadata.update({
            'control_family': 'omini_grounded_v1',
            'vl_grounding': 'qwen3vl_visual',
            'vl_longest_side': str(self.vl_longest_side),
            'grounded_extra_target': GROUNDED_EXTRA_TARGET,
        })
        return metadata

    # ------------------------------------------------------- text encoding
    def get_call_text_encoder_fn(self, text_encoder):
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        def encode_one(text, images):
            tokens = text_encoder.tokenize(text, images=images)
            o = text_encoder.encode_from_tokens_scheduled(tokens)
            text_embeds = o[0][0].to(self.dtype)
            extra = o[0][1]
            if 'attention_mask' in extra:
                attention_mask = extra['attention_mask'].to(torch.int64)
            else:
                attention_mask = torch.ones(
                    text_embeds.shape[:2], dtype=torch.int64, device=text_embeds.device
                )
            if text_embeds.shape[1] != attention_mask.shape[1]:
                raise RuntimeError(
                    f'attention_mask length {attention_mask.shape[1]} does not match '
                    f'embedding length {text_embeds.shape[1]}'
                )
            return text_embeds[0], attention_mask[0]

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_files):
            assert not any(is_video)
            if control_files is None:
                control_files = [None] * len(captions)
            if len(control_files) != len(captions):
                raise ValueError(
                    f'Got {len(captions)} captions but {len(control_files)} control files'
                )

            need_plain = self.condition_dropout > 0
            grounded_embeds, grounded_masks = [], []
            plain_embeds, plain_masks = [], []
            for caption, control_file in zip(captions, control_files):
                if control_file is None:
                    # Amostras internas sem referência (ex.: dataset auxiliar de 1
                    # item do diffusion-pipe): degrada para text-only nos dois slots.
                    embed, mask = encode_one(caption, [])
                    grounded_embeds.append(embed)
                    grounded_masks.append(mask)
                    if need_plain:
                        plain_embeds.append(embed.clone())
                        plain_masks.append(mask.clone())
                    continue
                files = control_file if isinstance(control_file, (list, tuple)) else [control_file]
                images = [prepare_vl_image_longest_side(f, self.vl_longest_side) for f in files]
                g_embed, g_mask = encode_one(caption, images)
                grounded_embeds.append(g_embed)
                grounded_masks.append(g_mask)
                if need_plain:
                    p_embed, p_mask = encode_one(caption, [])
                    if torch.equal(g_mask, p_mask) and g_embed.shape == p_embed.shape:
                        raise RuntimeError(
                            'Grounded and text-only embeddings are identical in shape/mask: '
                            'the visual tower is not receiving the reference image'
                        )
                    plain_embeds.append(p_embed)
                    plain_masks.append(p_mask)

            result = {
                f'text_embeds_{te_idx}': grounded_embeds,
                f'attention_mask_{te_idx}': grounded_masks,
            }
            if need_plain:
                # Dual cache (~2x disco): só quando o dropout acoplado está ativo.
                # Com o CFG dual-model (uncond = transformer incondicional
                # dedicado), o adapter não precisa cobrir o caso sem referência.
                result[f'text_embeds_noref_{te_idx}'] = plain_embeds
                result[f'attention_mask_noref_{te_idx}'] = plain_masks
            return result

        return fn

    # ----------------------------------------------------- sample/inference
    @torch.no_grad()
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=1, control_files=None):
        """Grounded CFG: the unconditional keeps the reference grounding and
        contrasts the caption only (canonical dual-conditioning contract)."""
        from comfy import model_management

        inputs = {}
        inputs_uncond = {}
        for te in self.get_text_encoders():
            if isinstance(te, nn.Module):
                te = te.to('cuda')
            else:
                te.load_model_if_needed()
            call_text_encoder_fn = self.get_call_text_encoder_fn(te)
            inputs.update(call_text_encoder_fn([prompt], [False], [control_files]))
            if cfg > 1:
                inputs_uncond.update(
                    call_text_encoder_fn([negative_prompt], [False], [control_files])
                )
            if isinstance(te, nn.Module):
                te = te.to('cpu')
            else:
                model_management.unload_all_models()
        self.conds = tuple(tensor.cuda() for tensor in self.get_conds(inputs))
        if cfg > 1:
            self.unconds = tuple(tensor.cuda() for tensor in self.get_conds(inputs_uncond))
        self.sample_cfg = cfg

    # --------------------------------------------------- coupled dropout
    def prepare_inputs(self, inputs, timestep_quantile=None):
        apply_dropout = (
            timestep_quantile is None
            and self.condition_dropout > 0
            and 'text_embeds_noref_0' in inputs
        )
        if apply_dropout:
            batch_size = inputs['latents'].shape[0]
            drop = torch.rand(batch_size) < self.condition_dropout
            if drop.any():
                embeds = list(inputs['text_embeds_0'])
                masks = list(inputs['attention_mask_0'])
                for i in torch.nonzero(drop).flatten().tolist():
                    embeds[i] = inputs['text_embeds_noref_0'][i]
                    masks[i] = inputs['attention_mask_noref_0'][i]
                inputs['text_embeds_0'] = embeds
                inputs['attention_mask_0'] = masks
                control = inputs['control_latents'].float()
                keep = (~drop).view(-1, *([1] * (control.ndim - 1))).to(control.dtype)
                inputs['control_latents'] = control * keep

        # The single coupled draw above replaces the VAE-only dropout below.
        saved_dropout = self.condition_dropout
        self.condition_dropout = 0.0
        try:
            return super().prepare_inputs(inputs, timestep_quantile=timestep_quantile)
        finally:
            self.condition_dropout = saved_dropout
