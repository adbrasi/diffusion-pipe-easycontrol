"""CtxRush Anima — next-scene control nodes (projeto CONTEXTO).

Um node all-in-one por adapter treinado no fork (dataset contexto_rush):
- ic_lora_v2      : ref_first [ref t=0 | target t=sigma], LoRA global
- ic_lora_routed  : mesmo contrato, LoRA MASCARADO às rows da referência
                    (nunca fundir nos pesos — o merge aplicaria o delta ao
                    target e quebraria o contrato zero-drift)
- omini_subject   : [target t=sigma | ref t=0], LoRA global

Lições dos audits embutidas:
- referência entra como IMAGE + VAE com crop-fit em PIXEL para o tamanho da
  geração e encode nativo (nunca redimensionar latente — borra o sinal);
- referência NUNCA é escalada em latente (sem peso reduzido no frame limpo);
- CFG: o uncond treinado (condition_dropout) tem a condição ZERADA — o node
  zera o frame de referência nos chunks uncond via cond_or_uncond;
- LoRA aplicado em runtime (delta bf16 por forward), com escala ajustável.

Recomendações de sampling (report Anima): sampler er_sde/res_multistep,
scheduler simple, CFG 4, shift 3.0 (default do ComfyUI para Anima), 30 steps.
"""

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import folder_paths
from safetensors import safe_open


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_single_image(image):
    if image.ndim != 4:
        raise ValueError('Expected an IMAGE batch tensor')
    if image.shape[0] != 1:
        image = image[:1]
    return image


def _crop_fit(image, width, height):
    """Center crop-fit em pixel (contrato de treino: mesmo bucket do target)."""
    samples = image.movedim(-1, 1)
    source_height, source_width = samples.shape[-2:]
    scale = max(width / source_width, height / source_height)
    resized_width = max(round(source_width * scale), width)
    resized_height = max(round(source_height * scale), height)
    samples = comfy.utils.common_upscale(
        samples, resized_width, resized_height, 'lanczos', 'disabled'
    )
    top = (resized_height - height) // 2
    left = (resized_width - width) // 2
    return samples[:, :, top: top + height, left: left + width].movedim(1, -1)


def _load_lora_pairs(lora_path):
    pairs = {}
    with safe_open(lora_path, framework='pt') as f:
        keys = list(f.keys())
        for k in keys:
            if 'lora_A' not in k:
                continue
            key_b = k.replace('lora_A', 'lora_B')
            if key_b not in keys:
                continue
            base = k.replace('.lora_A.weight', '').replace('diffusion_model.', '')
            base = base.replace('base_model.model.', '')
            pairs[base] = (f.get_tensor(k), f.get_tensor(key_b))
    if not pairs:
        raise ValueError(f'No LoRA A/B pairs found in {lora_path}')
    return pairs


def _match_entries(dit, pairs, device, dtype):
    named = dict(dit.named_modules())
    entries, missing = [], []
    for path, (a, b) in pairs.items():
        module = named.get(path)
        if module is None:
            missing.append(path)
        else:
            entries.append((module, a.to(device, dtype), b.to(device, dtype)))
    if not entries:
        raise ValueError(f'No LoRA modules matched the Anima DiT (e.g. {list(pairs)[:3]})')
    if missing:
        print(f'[CtxRushAnima] WARNING: {len(missing)} LoRA keys not matched (e.g. {missing[:3]})')
    return entries


class _LoraScope:
    """Aplica o delta LoRA em runtime durante o forward interno.

    masked=None      -> delta em TODAS as rows (adapters globais)
    masked='first'   -> delta só no frame de referência quando ref é o frame 0
    masked='last'    -> idem quando ref é o último frame
    Layouts (Anima blocks): 3D (B, T*H*W, D) no self_attn; 5D (B,T,H,W,D) no mlp.
    """

    def __init__(self, entries, scale, masked=None):
        self.entries = entries
        self.scale = scale
        self.masked = masked
        self._originals = []

    def __enter__(self):
        if self.scale == 0:
            return self
        for module, a, b in self.entries:
            orig = module.forward

            def wrapped(x, *args, _orig=orig, _a=a, _b=b, **kwargs):
                result = _orig(x, *args, **kwargs)
                delta = F.linear(F.linear(x.to(_a.dtype), _a), _b) * self.scale
                if self.masked is None:
                    return result + delta.to(result.dtype)
                if result.ndim == 3 and result.shape[1] % 2 == 0:
                    hw = result.shape[1] // 2
                    mask = result.new_zeros(1, result.shape[1], 1)
                    if self.masked == 'first':
                        mask[:, :hw] = 1
                    else:
                        mask[:, hw:] = 1
                elif result.ndim == 5 and result.shape[1] == 2:
                    mask = result.new_zeros(1, 2, 1, 1, 1)
                    mask[:, 0 if self.masked == 'first' else 1] = 1
                else:
                    return result
                return result + delta.to(result.dtype) * mask

            module.forward = wrapped
            self._originals.append((module, orig))
        return self

    def __exit__(self, *exc):
        for module, orig in self._originals:
            module.forward = orig
        self._originals = []
        return False


MODE_INFO = {
    # mode: (ref_first, masked)
    'ic_lora_v2': (True, None),
    'ic_lora_routed': (True, 'first'),
    'omini_subject': (False, None),
    'routed_targetfirst': (False, 'last'),
}


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class CtxRushAnimaNextScene:
    """All-in-one: liga a referência (frame limpo a t=0), aplica o adapter no
    contrato certo e devolve model+latent prontos para o KSampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Anima (base v1.0) SEM Load LoRA — o node aplica o adapter em runtime.'}),
                'vae': ('VAE', {'tooltip': 'qwen_image_vae (Wan 2.1).'}),
                'image': ('IMAGE', {'tooltip': 'Referência (cena anterior).'}),
                'lora_name': (folder_paths.get_filename_list('loras'),),
                'mode': (list(MODE_INFO), {'default': 'ic_lora_routed',
                         'tooltip': 'Contrato do adapter: ic_lora_v2 (global), ic_lora_routed (mascarado, zero-drift), omini_subject.'}),
                'lora_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                                  'tooltip': '0 = base + referência (baseline honesto).'}),
                'width': ('INT', {'default': 672, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 400, 'min': 64, 'max': 4096, 'step': 16}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 16}),
            },
            'optional': {
                'ref_cfg': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.1,
                    'tooltip': 'Guidance da REFERÊNCIA (3 branches, independente do CFG do texto). '
                               '0 = sem guidance de ref. Requer expected_cfg = CFG do KSampler.'}),
                'expected_cfg': ('FLOAT', {'default': 4.0, 'min': 1.0, 'max': 12.0, 'step': 0.5,
                    'tooltip': 'DEVE ser igual ao CFG do KSampler — usado para desacoplar ref_cfg do CFG do texto.'}),
            },
        }

    RETURN_TYPES = ('MODEL', 'LATENT')
    RETURN_NAMES = ('model', 'latent')
    FUNCTION = 'apply'
    CATEGORY = 'CtxRush/Anima'
    DESCRIPTION = ('Next-scene Anima: referência como frame temporal limpo (t=0), '
                   'LoRA em runtime no contrato do treino, CFG com uncond zerado. '
                   'Use CFG 4, shift 3, sampler er_sde/res_multistep + scheduler simple.')

    def apply(self, model, vae, image, lora_name, mode='ic_lora_routed',
              lora_strength=1.0, width=672, height=400, batch_size=1,
              ref_cfg=1.0, expected_cfg=4.0):
        ref_first, masked = MODE_INFO[mode]

        image = _require_single_image(image)
        ref_pixels = _crop_fit(image, width, height)
        ref_latent = vae.encode(ref_pixels)
        if ref_latent.ndim == 4:
            ref_latent = ref_latent.unsqueeze(2)  # (B,16,1,h,w)

        lora_path = folder_paths.get_full_path('loras', lora_name)
        pairs = _load_lora_pairs(lora_path)

        patched = model.clone()
        dit = patched.get_model_object('diffusion_model')
        device = comfy.model_management.get_torch_device()
        entries = _match_entries(dit, pairs, device, torch.bfloat16)
        print(f'[CtxRushAnima] mode={mode} ref_first={ref_first} masked={masked} '
              f'linears={len(entries)} strength={lora_strength}')

        src = patched.model.process_latent_in(ref_latent)

        # Guidance de 3 branches por cima do CFG de 2 branches do KSampler:
        #   uncond -> ref zerada (u);  cond -> t + (ref_cfg/expected_cfg)*(c - t)
        # Total (com cfg do KSampler == expected_cfg):
        #   u + cfg*(t-u) + ref_cfg*(c-t)   [InstructPix2Pix-style]
        ref_ratio = float(ref_cfg) / float(expected_cfg)

        def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
            orig_4d = x.ndim == 4
            if orig_4d:
                x = x.unsqueeze(2)
            bs = x.shape[0]

            ref = src.to(x.device, x.dtype)
            if ref.shape[0] != bs:
                ref = ref.expand(bs, -1, -1, -1, -1)
            if ref.shape[-2:] != x.shape[-2:]:
                raise RuntimeError(
                    f'Reference latent {tuple(ref.shape[-2:])} != generation {tuple(x.shape[-2:])}. '
                    'Gere no mesmo width/height configurado no node.'
                )
            ref = ref.clone()
            ref_zero = torch.zeros_like(ref)

            to = kwargs.get('transformer_options') or {}
            c_or_u = to.get('cond_or_uncond')
            cond_mask = torch.ones(bs, dtype=torch.bool, device=x.device)
            if c_or_u and bs % len(c_or_u) == 0:
                chunk = bs // len(c_or_u)
                for i, flag in enumerate(c_or_u):
                    if flag == 1:  # uncond chunk: ref sempre zerada (branch u)
                        ref[i * chunk:(i + 1) * chunk] = 0
                        cond_mask[i * chunk:(i + 1) * chunk] = False

            t = timesteps if timesteps.ndim == 1 else timesteps[:, 0]
            t_zero = torch.zeros_like(t)

            def run(ref_frames):
                if ref_first:
                    x_cat = torch.cat([ref_frames, x], dim=2)
                    t_cat = torch.stack([t_zero, t], dim=1)
                else:
                    x_cat = torch.cat([x, ref_frames], dim=2)
                    t_cat = torch.stack([t, t_zero], dim=1)
                with _LoraScope(entries, lora_strength, masked=masked):
                    out = executor(x_cat, t_cat, context, fps, padding_mask, **kwargs)
                return out[:, :, -1:, :, :] if ref_first else out[:, :, :1, :, :]

            out = run(ref)  # c nos chunks cond, u nos chunks uncond
            if abs(ref_ratio - 1.0) > 1e-6 and cond_mask.any():
                out_no_ref = run(ref_zero)  # t nos chunks cond
                mixed = out_no_ref + ref_ratio * (out - out_no_ref)
                out = torch.where(cond_mask.view(-1, 1, 1, 1, 1), mixed, out)

            if orig_4d:
                out = out.squeeze(2)
            return out

        to = patched.model_options.setdefault('transformer_options', {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, 'ctxrush_anima', wrapper, to
        )

        latent = torch.zeros(
            [batch_size, 16, 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (patched, {'samples': latent})


NODE_CLASS_MAPPINGS = {
    'CtxRushAnimaNextScene': CtxRushAnimaNextScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'CtxRushAnimaNextScene': 'CtxRush - Anima Next-Scene (ic_lora / routed / omini)',
}
