#!/usr/bin/env python3
"""Reavaliação dos checkpoints Anima com protocolo correto (audit 2026-07-18).

- preprocessing corrigido (crop-fit + [-1,1]);
- guidance 3 branches: text_cfg=4 fixo, sweep de ref_cfg {0, 0.5, 1.0, 1.5};
- buckets do TREINO por par (720x368 p/ AR 2.39; 672x400 p/ held-out);
- 2 seeds; checkpoints 500 e 1000; 3 braços;
- LoRA aplicado por escopo runtime (nunca merge) -> um único load de modelos;
- grid: ref | target real | vanilla | rc0 | rc0.5 | rc1.0 | rc1.5
"""
import os
import sys
import glob

sys.path.insert(0, '/workspace/diffusion-pipe-easycontrol')
os.chdir('/workspace/diffusion-pipe-easycontrol')

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np

from infer_easycontrol import (
    load_dit, load_vae, load_text_encoder, encode_prompt, encode_control,
    sample_ic_lora_full, sample_ominicontrol, sample_normal,
    load_routed_lora_entries, _RoutedLoraScope,
)

DEVICE = torch.device('cuda')
DTYPE = torch.bfloat16
DIT = '/workspace/models_anima/split_files/diffusion_models/anima-base-v1.0.safetensors'
VAE = '/workspace/models/qwen_image_vae.safetensors'
LLM = '/workspace/models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors'
NEG = 'worst quality, low quality, score_1, score_2, score_3, artist name'
OUT = '/workspace/outputs/anima_reeval_v2'
STEPS, TEXT_CFG, SHIFT = 30, 4.0, 3.0
SEEDS = [76, 200]
REF_CFGS = [0.0, 0.5, 1.0, 1.5]

DS = '/workspace/datasets/contexto_rush'
REFS = {
    'hockey': dict(ref=f'{DS}/input_A/imagem000001.jpg', target=f'{DS}/input_B/imagem000001.jpg',
                   prompt=open(f'{DS}/input_B/imagem000001.txt').read().strip(), w=720, h=368),
    'controlroom': dict(ref=f'{DS}/input_A/imagem000003.jpg', target=f'{DS}/input_B/imagem000003.jpg',
                        prompt=open(f'{DS}/input_B/imagem000003.txt').read().strip(), w=720, h=368),
    'heldout': dict(ref='/workspace/datasets/test_refs/anime_random.jpg', target=None,
                    prompt='the same girl now stands up and smiles brightly, arms spread wide, '
                           'cherry blossom petals swirling around her in the schoolyard', w=672, h=400),
}

def _ck(pattern):
    d = glob.glob(pattern)
    return d[0] + '/adapter_model.safetensors' if d else None

ARMS = {
    'iclora_v2': dict(layout='ref_first', masked=None, ckpts={
        500: _ck('/workspace/checkpoints/anima_iclora_250/*/step500'),
        1000: _ck('/workspace/checkpoints/anima_iclora_250/*/step1000')}),
    'routed': dict(layout='ref_first', masked='first', ckpts={
        500: _ck('/workspace/checkpoints/anima_iclora_routed/*/step500'),
        1000: _ck('/workspace/checkpoints/anima_iclora_routed/*/step1000')}),
    'omini': dict(layout='target_first', masked=None, ckpts={
        500: _ck('/workspace/checkpoints/anima_omini_250/*/step500'),
        1000: _ck('/workspace/checkpoints/anima_omini_250/*/step1000')}),
}


class _FullLoraScope(_RoutedLoraScope):
    """Delta em TODAS as rows (adapters globais), reaproveitando o wrap."""

    def __enter__(self):
        for module, a, b, scale in self.entries:
            orig = module.forward

            def wrapped(x, *args, _orig=orig, _a=a, _b=b, _s=scale * self.strength, **kwargs):
                result = _orig(x, *args, **kwargs)
                if _s == 0:
                    return result
                delta = F.linear(F.linear(x.to(_a.dtype), _a), _b) * _s
                return result + delta.to(result.dtype)

            module.forward = wrapped
            self._originals.append((module, orig))
        return self


def decode(vae, latents):
    with torch.no_grad():
        px = vae.model.decode(latents.to(DTYPE), vae.scale)
    if px.ndim == 5:
        px = px.squeeze(2)
    px = ((px.float().clamp(-1, 1) + 1) / 2 * 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(px)


def main():
    os.makedirs(OUT, exist_ok=True)
    dit = load_dit(DIT, DEVICE, DTYPE)
    vae = load_vae(VAE, DEVICE, DTYPE, None)
    text_encoder, tokenizer, t5_tokenizer = load_text_encoder(LLM, DEVICE, DTYPE)

    def build_context(enc):
        """Embeds do Qwen -> contexto do DiT via LLM adapter (etapa que o main
        do runner faz em 'Running LLM adapter...' — SEM ela sai puro ruído)."""
        emb, mask, t5_ids, t5_mask = enc
        assert dit.use_llm_adapter and hasattr(dit, 'llm_adapter')
        with torch.autocast('cuda', dtype=DTYPE):
            ctx = dit.llm_adapter(
                source_hidden_states=emb.to(DTYPE),
                target_input_ids=t5_ids,
                target_attention_mask=t5_mask,
                source_attention_mask=mask,
            )
        ctx[~t5_mask.bool()] = 0
        return ctx

    ref_data, neg_ctx_cache = {}, {}
    neg_ctx = build_context(encode_prompt(text_encoder, tokenizer, t5_tokenizer, NEG, DEVICE, DTYPE))
    for name, r in REFS.items():
        pos_ctx = build_context(encode_prompt(text_encoder, tokenizer, t5_tokenizer, r['prompt'], DEVICE, DTYPE))
        ctrl = encode_control(vae, r['ref'], r['h'], r['w'], DEVICE, DTYPE)
        ref_data[name] = (pos_ctx, ctrl)
        neg_ctx_cache[(r['w'], r['h'])] = neg_ctx

    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        # UMA vanilla + UMA geração com adapter; inspecionar antes do batch.
        name, r = 'hockey', REFS['hockey']
        pos_ctx, ctrl = ref_data[name]
        lat = sample_normal(dit, pos_ctx, neg_ctx_cache[(r['w'], r['h'])],
                            r['h'], r['w'], STEPS, TEXT_CFG, SHIFT, 76, DEVICE, DTYPE)
        decode(vae, lat).save(f'{OUT}/SMOKE_vanilla.png')
        entries = load_routed_lora_entries(dit, ARMS['iclora_v2']['ckpts'][1000], DEVICE, DTYPE)
        with _FullLoraScope(entries, 1.0):
            lat = sample_ic_lora_full(dit, pos_ctx, neg_ctx_cache[(r['w'], r['h'])], ctrl,
                                      r['h'], r['w'], STEPS, TEXT_CFG, SHIFT, 76, DEVICE, DTYPE, ref_cfg=1.0)
        decode(vae, lat).save(f'{OUT}/SMOKE_iclora_rc1.png')
        print('SMOKE COMPLETO', flush=True)
        return

    # vanilla (sem lora, sem ref) por ref x seed — compartilhado entre braços
    print('== vanilla ==', flush=True)
    for name, r in REFS.items():
        pos_ctx, _ = ref_data[name]
        for seed in SEEDS:
            f = f'{OUT}/vanilla_{name}_seed{seed}.png'
            if os.path.exists(f):
                continue
            lat = sample_normal(dit, pos_ctx, neg_ctx_cache[(r['w'], r['h'])],
                                r['h'], r['w'], STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE)
            decode(vae, lat).save(f)

    for arm, cfg in ARMS.items():
        for step, lora in cfg['ckpts'].items():
            if lora is None:
                print(f'!! sem checkpoint: {arm} {step}', flush=True)
                continue
            entries = load_routed_lora_entries(dit, lora, DEVICE, DTYPE)
            scope_cls = _RoutedLoraScope if cfg['masked'] else _FullLoraScope
            scope_kwargs = dict(masked=cfg['masked']) if cfg['masked'] else {}
            for name, r in REFS.items():
                pos_ctx, ctrl = ref_data[name]
                neg_ctx = neg_ctx_cache[(r['w'], r['h'])]
                for seed in SEEDS:
                    for rc in REF_CFGS:
                        f = f'{OUT}/{arm}_s{step}_{name}_seed{seed}_rc{rc}.png'
                        if os.path.exists(f):
                            continue
                        with scope_cls(entries, 1.0, **scope_kwargs):
                            if cfg['layout'] == 'ref_first':
                                lat = sample_ic_lora_full(
                                    dit, pos_ctx, neg_ctx, ctrl, r['h'], r['w'],
                                    STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE, ref_cfg=rc)
                            else:
                                lat = sample_ominicontrol(
                                    dit, pos_ctx, neg_ctx, ctrl, r['h'], r['w'],
                                    STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE,
                                    position_mode='subject', ref_cfg=rc)
                        decode(vae, lat).save(f)
                        print(f'ok {f}', flush=True)

    # grids: um por braço x step x seed
    W, H, lh = 320, 172, 22
    cols = ['referencia', 'target real', 'vanilla'] + [f'ref_cfg {rc}' for rc in REF_CFGS]
    for arm, cfg in ARMS.items():
        for step in cfg['ckpts']:
            for seed in SEEDS:
                grid = Image.new('RGB', (W * len(cols), (H + lh) * len(REFS) + 26), 'white')
                d = ImageDraw.Draw(grid)
                d.text((8, 5), f'{arm} step{step} seed{seed} — text_cfg 4, shift 3, 30 steps, buckets de treino', fill='black')
                for row, (name, r) in enumerate(REFS.items()):
                    paths = [r['ref'], r['target'], f'{OUT}/vanilla_{name}_seed{seed}.png'] + \
                            [f'{OUT}/{arm}_s{step}_{name}_seed{seed}_rc{rc}.png' for rc in REF_CFGS]
                    for cix, p in enumerate(paths):
                        y = 26 + row * (H + lh)
                        d.text((cix * W + 5, y + 3), f'{name} — {cols[cix]}', fill='black')
                        if p is None or not os.path.exists(p):
                            d.rectangle([cix * W, y + lh, (cix + 1) * W - 1, y + lh + H - 1], outline='red')
                            continue
                        grid.paste(Image.open(p).convert('RGB').resize((W, H)), (cix * W, y + lh))
                grid.save(f'{OUT}/GRID_{arm}_s{step}_seed{seed}.png')
    print('REEVAL COMPLETA', flush=True)


if __name__ == '__main__':
    main()
