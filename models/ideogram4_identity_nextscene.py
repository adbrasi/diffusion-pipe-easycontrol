"""FIRA — Factorized Identity-Reference Adapter for Ideogram 4 next-scene.

Sequence: [identity capsule (8) | delta text | noisy target | compact VAE ref].

Channels are factorized so no single channel carries the whole reference scene:
* identity capsule: last-8 hidden states of a Qwen3-VL encode of the ENTITY CROP
  under a fixed identity template (face/silhouette/costume/style anchors) — the
  causal tail has attended the crop, so it is a query-conditioned identity code;
* delta text: change-only caption through the stock Ideogram text path (no LoRA);
* compact VAE ref: the SAME entity crop at stride 2, amplitude-scaled and
  VAE-only-dropped — texture/palette memory without scene layout;
* target: gains a trained reader (Q + attention.o) via RoleAwareLoRARouter.

Loss: change-weighted via dataset mask_path (floor baked into the mask images).
"""

import torch

from models.ideogram4_ic_lora import LORA_FORBIDDEN_MODULE_PATTERNS  # noqa: F401 (doc parity)
from models.ideogram4_ominicontrol import Ideogram4OminiTransformerLayer
from models.ideogram4_ominicontrol2 import (
    Ideogram4OminiControl2InitialLayer,
    Ideogram4OminiControl2Pipeline,
)
from models.ideogram4_ic_lora import Ideogram4ReferenceFinalLayer
from models.ideogram4_omini_grounded import prepare_vl_image_longest_side
from models.role_routed_lora import LlmProjCapsuleRouter, RoleAwareLoRARouter
from models.base import make_contiguous
from torch import nn

IDENTITY_TEMPLATE = (
    'Reference R1. Encode only persistent character identity: face, body '
    'proportions, hair, costume motifs and rendering style. Ignore pose, '
    'action, camera, background, lighting and composition. Identity anchors: '
    'face; silhouette; costume; rendering style.'
)


class Ideogram4IdentityNextScenePipeline(Ideogram4OminiControl2Pipeline):
    name = 'ideogram4_identity_nextscene'

    def __init__(self, config):
        super().__init__(config)
        section = config.get('identity_nextscene', {})
        self.capsule_tokens = int(section.get('identity_capsule_tokens', 8))
        self.vl_longest_side = int(section.get('vl_longest_side', 224))
        self.vae_reference_scale = float(section.get('vae_reference_scale', 0.5))
        self.vae_reference_dropout = float(section.get('vae_reference_dropout', 0.5))
        self.identity_rank = int(section.get('identity_rank', 8))
        self.adaln_rank = int(section.get('adaln_rank', 8))
        # FIRA replaces both legacy dropouts by design.
        self.condition_dropout = 0.0
        self.role_router = RoleAwareLoRARouter(True)
        self.capsule_router = LlmProjCapsuleRouter(self.capsule_tokens)

    # ------------------------------------------------------------------ LoRA
    def configure_adapter(self, adapter_config):
        import peft
        from utils.common import is_main_process

        target_model = self.diffusion_model
        targets = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for sub_name, sub in module.named_modules(prefix=name):
                if isinstance(sub, nn.Linear):
                    targets.add(sub_name)
        grounded = [n for n, m in target_model.named_modules()
                    if isinstance(m, nn.Linear) and n.split('.')[-1] == 'llm_cond_proj']
        if not grounded:
            raise RuntimeError('llm_cond_proj not found')
        targets = sorted(targets) + grounded

        rank_pattern = {n: self.adaln_rank for n in targets if 'adaln_modulation' in n}
        rank_pattern.update({n: self.identity_rank for n in grounded})
        peft_config = peft.LoraConfig(
            r=adapter_config['rank'],
            lora_alpha=adapter_config['alpha'],
            lora_dropout=adapter_config['dropout'],
            bias='none',
            target_modules=targets,
            rank_pattern=rank_pattern,
            alpha_pattern=dict(rank_pattern),
        )
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)

        block_installed = self.role_router.install(
            (n, m) for n, m in self.diffusion_model.named_modules() if '.layers.' in n or n.startswith('layers')
        )
        capsule_installed = self.capsule_router.install(self.diffusion_model.named_modules())
        if is_main_process():
            print(f'[{self.name}] role-routed LoRA: {block_installed} block linears; '
                  f'capsule router on {capsule_installed} llm_cond_proj')

    def _audit_adapter_keys(self, keys, save_dir):
        blocks_keys = [k for k in keys if 'llm_cond_proj' not in k]
        if not any('llm_cond_proj' in k for k in keys):
            print(f'[{self.name}] WARNING: no llm_cond_proj keys (identity channel untrained?)')
        super()._audit_adapter_keys(blocks_keys, save_dir)

    # ------------------------------------------------------- text + capsule
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
            embeds = o[0][0].to(self.dtype)
            extra = o[0][1]
            if 'attention_mask' in extra:
                mask = extra['attention_mask'].to(torch.int64)
            else:
                mask = torch.ones(embeds.shape[:2], dtype=torch.int64, device=embeds.device)
            return embeds[0], mask[0]

        @torch.inference_mode()
        def fn(captions, is_video, control_files):
            assert not any(is_video)
            if control_files is None:
                control_files = [None] * len(captions)
            embeds_list, mask_list = [], []
            for caption, control_file in zip(captions, control_files):
                text_embeds, text_mask = encode_one(caption, [])
                if control_file is None:
                    capsule = text_embeds.new_zeros(self.capsule_tokens, text_embeds.shape[-1])
                else:
                    files = control_file if isinstance(control_file, (list, tuple)) else [control_file]
                    crop = prepare_vl_image_longest_side(files[0], self.vl_longest_side)
                    id_embeds, _ = encode_one(IDENTITY_TEMPLATE, [crop])
                    if id_embeds.shape[0] < self.capsule_tokens:
                        raise RuntimeError('identity encode shorter than capsule size')
                    capsule = id_embeds[-self.capsule_tokens:]
                combined = torch.cat([capsule, text_embeds], dim=0)
                combined_mask = torch.cat([
                    torch.ones(self.capsule_tokens, dtype=torch.int64, device=text_mask.device),
                    text_mask,
                ], dim=0)
                embeds_list.append(combined)
                mask_list.append(combined_mask)
            return {
                f'text_embeds_{te_idx}': embeds_list,
                f'attention_mask_{te_idx}': mask_list,
            }

        return fn

    # ------------------------------------------------------------ VAE memory
    def prepare_reference_latents(self, reference_latents, noisy_target, timestep_quantile=None):
        latents = super().prepare_reference_latents(
            reference_latents, noisy_target, timestep_quantile=timestep_quantile
        )
        latents = latents * self.vae_reference_scale
        if timestep_quantile is None and self.vae_reference_dropout > 0:
            keep = (
                torch.rand(latents.shape[0], device=latents.device)
                >= self.vae_reference_dropout
            ).view(-1, *([1] * (latents.ndim - 1)))
            latents = latents * keep.to(latents.dtype)
        return latents

    # ---------------------------------------------------------------- layers
    def to_layers(self):
        model = self.diffusion_model
        layers = [
            Ideogram4OminiControl2InitialLayer(
                model,
                reference_position_offset=self.reference_position_offset,
                reference_model_timestep=self.reference_model_timestep,
                reference_position_scale=self.reference_position_scale,
                independent_condition=self.independent_condition,
            )
        ]
        layers.extend(
            FiraTransformerLayer(block, index, self.offloader, self.role_router, self.capsule_tokens)
            for index, block in enumerate(model.layers)
        )
        layers.append(Ideogram4ReferenceFinalLayer(model))
        return layers

    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        metadata.update({
            'control_family': 'fira_v1',
            'identity_capsule_tokens': str(self.capsule_tokens),
            'vae_reference_scale': str(self.vae_reference_scale),
            'vae_reference_dropout': str(self.vae_reference_dropout),
            'identity_template_sha': 'fira_identity_v1',
        })
        return metadata

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=1, control_files=None):
        from comfy import model_management

        inputs = {}
        for te in self.get_text_encoders():
            if isinstance(te, nn.Module):
                te = te.to('cuda')
            else:
                te.load_model_if_needed()
            call_fn = self.get_call_text_encoder_fn(te)
            inputs.update(call_fn([prompt], [False], [control_files]))
            if isinstance(te, nn.Module):
                te = te.to('cpu')
            else:
                model_management.unload_all_models()
        self.conds = tuple(tensor.cuda() for tensor in self.get_conds(inputs))
        self.sample_cfg = cfg


class FiraTransformerLayer(Ideogram4OminiTransformerLayer):
    def __init__(self, layer, block_idx, offloader, role_router, capsule_tokens):
        # parent stores .router and calls set_reference_span; we bypass that.
        super().__init__(layer, block_idx, offloader, router=_NullRouter())
        self.role_router = role_router
        self.capsule_tokens = capsule_tokens

    def forward(self, inputs):
        hidden_states, attention_mask, adaln_input, sizes, *freqs_cis = inputs
        text_length, gh, gw = (int(v) for v in sizes)
        target_end = text_length + gh * gw
        self.role_router.set_spans(
            capsule_end=self.capsule_tokens,
            text_end=text_length,
            target_end=target_end,
            sequence_length=hidden_states.shape[1],
        )
        return super().forward(make_contiguous(hidden_states, attention_mask, adaln_input, sizes, *freqs_cis))


class _NullRouter:
    def set_reference_span(self, start, end):
        pass
