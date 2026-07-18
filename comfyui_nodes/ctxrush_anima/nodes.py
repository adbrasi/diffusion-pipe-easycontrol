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
- guidance de texto e referência são independentes: o Dual Guider calcula
  explicitamente negativo/sem-ref, positivo/sem-ref e positivo/com-ref;
- LoRA aplicado em runtime (delta bf16 por forward), com escala ajustável.

Recomendações de sampling (report Anima): sampler er_sde/res_multistep,
scheduler simple, CFG 4, shift 3.0 (default do ComfyUI para Anima), 30 steps.
"""

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
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


def _mix_reference_guidance(out_no_ref, out_with_ref, cond_mask, ref_ratio):
    """Mistura somente os chunks positivos antes do CFG do KSampler."""
    mixed = out_no_ref + ref_ratio * (out_with_ref - out_no_ref)
    mask_shape = (cond_mask.shape[0],) + (1,) * (out_no_ref.ndim - 1)
    return torch.where(cond_mask.reshape(mask_shape), mixed, out_no_ref)


def _add_reference_guidance(text_prediction, text_no_ref, text_with_ref, ref_cfg):
    return text_prediction + ref_cfg * (text_with_ref - text_no_ref)


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
    """Retorna (path, module, A, B). O path decide o canal na inferência:
    llm_adapter.* roda ANTES do forward do DiT (preprocess_text_embeds) e
    precisa de object patch próprio; cross_attn é canal semântico (sempre
    global); o resto é canal de aparência (mascarável nos modos routed)."""
    named = dict(dit.named_modules())
    entries, missing = [], []
    for path, (a, b) in pairs.items():
        module = named.get(path)
        if module is None:
            missing.append(path)
        else:
            entries.append((path, module, a.to(device, dtype), b.to(device, dtype)))
    if not entries:
        sample = list(pairs)[:3]
        if any(('attn.wk' in p) or ('txtfusion' in p) or ('attn.gate' in p) for p in pairs):
            raise ValueError(
                f'Este arquivo é um LoRA do KREA 2, não do Anima (keys: {sample}). '
                'Use os adapters ctxrush_ctrl_* ou ctxrush_anima_* com este node.'
            )
        raise ValueError(f'No LoRA modules matched the Anima DiT (e.g. {sample})')
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
    # mode: (ref_first, masked) — masked se aplica SÓ ao canal de aparência
    # (self_attn+mlp); cross_attn e llm_adapter, quando presentes no adapter,
    # são sempre globais (canal semântico dos braços de escopo largo).
    'ic_lora_v2': (True, None),
    'ic_lora_routed': (True, 'first'),
    'omini_subject': (False, None),
    'routed_targetfirst': (False, 'last'),
    'broad_targetfirst': (False, None),   # ic_lora_v3 / ominicontrol_broad ⭐
    'dual_targetfirst': (False, 'last'),  # ic_lora_dual
}

# Casar adapter com o contrato errado NÃO dá erro — só mata o efeito.
# O modo 'auto' resolve o contrato pelo nome do arquivo (ordem importa:
# padrões mais específicos primeiro).
_NAME_HINTS = (
    ('v3', 'broad_targetfirst'),
    ('dual', 'dual_targetfirst'),
    ('broad', 'broad_targetfirst'),
    ('routedtf', 'routed_targetfirst'),
    ('targetfirst', 'routed_targetfirst'),
    ('routedrf', 'ic_lora_routed'),
    ('reffirst', 'ic_lora_routed'),
    ('iclora_routed', 'ic_lora_routed'),
    ('routed', 'ic_lora_routed'),
    ('iclora', 'ic_lora_v2'),
    ('ic_lora', 'ic_lora_v2'),
    ('globaltf', 'omini_subject'),
    ('omini', 'omini_subject'),
)


def _parse_block_range(spec, num_blocks=28):
    """'all' -> None (todos); '4-24' ou '0-13,20-27' -> set de índices de block."""
    spec = (spec or 'all').strip().lower()
    if spec in ('all', ''):
        return None
    keep = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            keep.update(range(int(a), int(b) + 1))
        else:
            keep.add(int(part))
    bad = [i for i in keep if i < 0 or i >= num_blocks]
    if bad:
        raise ValueError(f'block_range fora de 0-{num_blocks - 1}: {bad}')
    return keep


def _block_index(path):
    parts = path.split('.')
    if parts and parts[0] == 'blocks' and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None


def _resolve_mode(mode, lora_name):
    if mode != 'auto':
        return mode
    low = lora_name.lower()
    for hint, resolved in _NAME_HINTS:
        if hint in low:
            return resolved
    raise ValueError(
        f'mode=auto não reconhece o contrato pelo nome "{lora_name}". '
        'Escolha o mode manualmente (ic_lora_v2 / ic_lora_routed / omini_subject / routed_targetfirst).'
    )

_GUIDER_REF_BRANCHES = (False, False, True)
_GUIDER_BRANCH_OPTION = 'ctxrush_anima_ref_branches'


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class CtxRushAnimaNextScene:
    """All-in-one: liga a referência (frame limpo a t=0), aplica o adapter no
    contrato certo e devolve model+latent para o sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Anima (base v1.0) SEM Load LoRA — o node aplica o adapter em runtime.'}),
                'vae': ('VAE', {'tooltip': 'qwen_image_vae (Wan 2.1).'}),
                'image': ('IMAGE', {'tooltip': 'Referência (cena anterior).'}),
                'lora_name': (folder_paths.get_filename_list('loras'),),
                'mode': (['auto'] + list(MODE_INFO), {'default': 'auto',
                         'tooltip': 'auto = resolve o contrato pelo nome do arquivo (recomendado). '
                                    'Manual: TEM que casar com o braço do adapter, senão o efeito some sem erro.'}),
                'lora_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                                  'tooltip': '0 = base + referência (baseline honesto).'}),
                'width': ('INT', {'default': 672, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 400, 'min': 64, 'max': 4096, 'step': 16}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 16}),
            },
            'optional': {
                'ref_cfg': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.1,
                    'tooltip': 'Guidance da REFERÊNCIA (3 branches, independente do CFG do texto). '
                               '0 = sem guidance de ref. Ignorado ao usar o CtxRush Anima Dual Guider.'}),
                'expected_cfg': ('FLOAT', {'default': 4.0, 'min': 1.0, 'max': 12.0, 'step': 0.5,
                    'tooltip': 'KSampler clássico: DEVE ser igual ao CFG. Ignorado pelo CtxRush Anima Dual Guider.'}),
                'appearance_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                    'tooltip': 'Canal de APARÊNCIA (self_attn+mlp) × lora_strength. 0 = desliga.'}),
                'cross_attn_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                    'tooltip': 'Canal texto↔visual (cross_attn) × lora_strength. Só age em adapters broad/dual.'}),
                'llm_adapter_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                    'tooltip': 'Ponte texto→DiT (llm_adapter) × lora_strength. O canal do eureka. 0 = mede o quanto ele importa.'}),
                'block_range': ('STRING', {'default': 'all',
                    'tooltip': 'Quais blocks (0-27) recebem o LoRA de aparência/cross_attn. '
                               'Ex.: "4-24" ou "0-13,20-27". llm_adapter não é afetado.'}),
            },
        }

    RETURN_TYPES = ('MODEL', 'LATENT')
    RETURN_NAMES = ('model', 'latent')
    FUNCTION = 'apply'
    CATEGORY = 'CtxRush/Anima'
    DESCRIPTION = ('Next-scene Anima: referência como frame temporal limpo (t=0), '
                   'LoRA em runtime no contrato do treino, guidance independente da referência. '
                   'Use CFG 4, shift 3, sampler er_sde/res_multistep + scheduler simple.')

    def apply(self, model, vae, image, lora_name, mode='auto',
              lora_strength=1.0, width=672, height=400, batch_size=1,
              ref_cfg=1.0, expected_cfg=4.0,
              appearance_strength=1.0, cross_attn_strength=1.0,
              llm_adapter_strength=1.0, block_range='all'):
        if expected_cfg <= 0:
            raise ValueError('expected_cfg must be greater than zero')
        if ref_cfg < 0:
            raise ValueError('ref_cfg must be non-negative')
        mode = _resolve_mode(mode, lora_name)
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
        entries_p = _match_entries(dit, pairs, device, torch.bfloat16)
        keep_blocks = _parse_block_range(block_range)

        def _in_range(path):
            idx = _block_index(path)
            return keep_blocks is None or idx is None or idx in keep_blocks

        llm_entries = [e[1:] for e in entries_p if e[0].startswith('llm_adapter')]
        sem_entries = [e[1:] for e in entries_p if 'cross_attn' in e[0] and _in_range(e[0])]
        app_entries = [e[1:] for e in entries_p
                       if not (e[0].startswith('llm_adapter') or 'cross_attn' in e[0])
                       and _in_range(e[0])]
        app_scale = lora_strength * appearance_strength
        sem_scale = lora_strength * cross_attn_strength
        llm_scale = lora_strength * llm_adapter_strength
        print(f'[CtxRushAnima] mode={mode} ref_first={ref_first} masked={masked} '
              f'aparencia={len(app_entries)}x{app_scale:g} cross_attn={len(sem_entries)}x{sem_scale:g} '
              f'llm_adapter={len(llm_entries)}x{llm_scale:g} blocks={block_range} '
              f'expected_cfg={expected_cfg} ref_cfg={ref_cfg}')

        if llm_entries and llm_scale != 0:
            # O llm_adapter roda no preprocess_text_embeds (extra_conds), ANTES
            # do wrapper do DiT — o delta dele precisa de object patch próprio.
            orig_pre = dit.preprocess_text_embeds

            def _pre_with_lora(text_embeds, text_ids, t5xxl_weights=None):
                with _LoraScope(llm_entries, llm_scale, masked=None):
                    return orig_pre(text_embeds, text_ids, t5xxl_weights=t5xxl_weights)

            patched.add_object_patch('diffusion_model.preprocess_text_embeds', _pre_with_lora)

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
            ref_with = ref.clone()
            ref_zero = torch.zeros_like(ref)

            to = kwargs.get('transformer_options') or {}
            c_or_u = to.get('cond_or_uncond')
            guider_ref_branches = to.get(_GUIDER_BRANCH_OPTION)
            if c_or_u:
                if bs % len(c_or_u) != 0:
                    raise RuntimeError(
                        f'Batch {bs} is not divisible by cond_or_uncond branches {len(c_or_u)}'
                    )
            elif guider_ref_branches is not None:
                raise RuntimeError('CtxRush Anima Dual Guider requires cond_or_uncond branch metadata')

            t = timesteps if timesteps.ndim == 1 else timesteps[:, 0]
            t_zero = torch.zeros_like(t)

            def run(ref_frames):
                if ref_first:
                    x_cat = torch.cat([ref_frames, x], dim=2)
                    t_cat = torch.stack([t_zero, t], dim=1)
                else:
                    x_cat = torch.cat([x, ref_frames], dim=2)
                    t_cat = torch.stack([t, t_zero], dim=1)
                with _LoraScope(app_entries, app_scale, masked=masked), \
                     _LoraScope(sem_entries, sem_scale, masked=None):
                    out = executor(x_cat, t_cat, context, fps, padding_mask, **kwargs)
                return out[:, :, -1:, :, :] if ref_first else out[:, :, :1, :, :]

            if guider_ref_branches is not None:
                ref_for_branches = ref.clone()
                chunk = bs // len(c_or_u)
                for i, flag in enumerate(c_or_u):
                    if flag < 0 or flag >= len(guider_ref_branches):
                        raise RuntimeError(f'Unexpected dual-guider branch index: {flag}')
                    if not guider_ref_branches[flag]:
                        ref_for_branches[i * chunk:(i + 1) * chunk] = 0
                out = run(ref_for_branches)
            else:
                cond_mask = torch.ones(bs, dtype=torch.bool, device=x.device)
                if c_or_u:
                    chunk = bs // len(c_or_u)
                    for i, flag in enumerate(c_or_u):
                        if flag not in (0, 1):
                            raise RuntimeError(f'Unexpected cond_or_uncond flag: {flag}')
                        if flag == 1:  # uncond chunk: ref sempre zerada (branch u)
                            ref_with[i * chunk:(i + 1) * chunk] = 0
                            cond_mask[i * chunk:(i + 1) * chunk] = False

                if ref_cfg == 0 or not cond_mask.any():
                    out = run(ref_zero)
                else:
                    out_with_ref = run(ref_with)  # c nos chunks cond, u nos chunks uncond
                    if abs(ref_ratio - 1.0) <= 1e-6:
                        out = out_with_ref
                    else:
                        out_no_ref = run(ref_zero)  # t nos chunks cond, u nos chunks uncond
                        out = _mix_reference_guidance(
                            out_no_ref, out_with_ref, cond_mask, ref_ratio,
                        )

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


class _CtxRushAnimaDualGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive, negative):
        self.inner_set_conds({'positive': positive, 'negative': negative})

    def set_cfg(self, text_cfg, ref_cfg):
        if text_cfg < 0:
            raise ValueError('text_cfg must be non-negative')
        if ref_cfg < 0:
            raise ValueError('ref_cfg must be non-negative')
        self.cfg = text_cfg
        self.ref_cfg = ref_cfg

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        model_options = {} if model_options is None else model_options
        positive = self.conds.get('positive')
        negative = self.conds.get('negative')
        local_options = comfy.model_patcher.create_model_options_clone(model_options)
        local_options.setdefault('transformer_options', {})[_GUIDER_BRANCH_OPTION] = _GUIDER_REF_BRANCHES

        # Branch order: u = negative/no-ref, t = positive/no-ref,
        # c = positive/with-ref. The model wrapper selects the reference using
        # cond_or_uncond indices, even when Comfy evaluates branches separately.
        uncond, text_no_ref, text_with_ref = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [negative, positive, positive],
            x,
            timestep,
            local_options,
        )
        text_prediction = comfy.samplers.cfg_function(
            self.inner_model,
            text_no_ref,
            uncond,
            self.cfg,
            x,
            timestep,
            model_options=local_options,
            cond=positive,
            uncond=negative,
        )
        return _add_reference_guidance(
            text_prediction, text_no_ref, text_with_ref, self.ref_cfg,
        )


class CtxRushAnimaDualGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'positive': ('CONDITIONING',),
                'negative': ('CONDITIONING',),
                'text_cfg': ('FLOAT', {'default': 4.0, 'min': 0.0, 'max': 30.0, 'step': 0.1}),
                'ref_cfg': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
            },
        }

    RETURN_TYPES = ('GUIDER',)
    FUNCTION = 'get_guider'
    CATEGORY = 'CtxRush/Anima'
    DESCRIPTION = ('Guidance de 3 branches: texto negativo sem referência, texto positivo '
                   'sem referência e texto positivo com referência.')

    def get_guider(self, model, positive, negative, text_cfg=4.0, ref_cfg=1.0):
        guider = _CtxRushAnimaDualGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(text_cfg, ref_cfg)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    'CtxRushAnimaNextScene': CtxRushAnimaNextScene,
    'CtxRushAnimaDualGuider': CtxRushAnimaDualGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'CtxRushAnimaNextScene': 'CtxRush - Anima Next-Scene (ic_lora / routed / omini)',
    'CtxRushAnimaDualGuider': 'CtxRush - Anima Dual Guider (Text + Reference)',
}
