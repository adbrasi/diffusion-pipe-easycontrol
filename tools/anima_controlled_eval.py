#!/usr/bin/env python3
"""Avaliação dos 3 braços CONTROLADOS (rank 32, lr 5e-5, sampler padrão):
routed_targetfirst vs routed_reffirst vs global_targetfirst.

Refs NOVAS (pedido do usuário): pares 100/500 do dataset (bucket 672x400,
com target real) + 2 held-outs (anime_random e Screenshot_158).
Protocolo: 3 branches, text_cfg 4, ref_cfg {0, 0.5, 1.0, 1.5}, 2 seeds,
checkpoints {500, 1000, 1500}.
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
OUT = '/workspace/outputs/anima_controlled_eval'
STEPS, TEXT_CFG, SHIFT = 30, 4.0, 3.0
SEEDS = [76, 200]
REF_CFGS = [0.0, 0.5, 1.0, 1.5]
CKPTS = [500, 1000, 1500]

DS = '/workspace/datasets/contexto_rush'
REFS = {
    'par100': dict(ref=f'{DS}/input_A/imagem000100.jpg', target=f'{DS}/input_B/imagem000100.jpg',
                   prompt=open(f'{DS}/input_B/imagem000100.txt').read().strip(), w=672, h=400),
    'par500': dict(ref=f'{DS}/input_A/imagem000500.jpg', target=f'{DS}/input_B/imagem000500.jpg',
                   prompt=open(f'{DS}/input_B/imagem000500.txt').read().strip(), w=672, h=400),
    'heldout_sakura': dict(ref='/workspace/datasets/test_refs/anime_random.jpg', target=None,
                           prompt='the same girl now stands up and smiles brightly, arms spread wide, '
                                  'cherry blossom petals swirling around her in the schoolyard', w=672, h=400),
    'heldout_screenshot': dict(ref='/workspace/Screenshot_158.png', target=None,
                               prompt='the same character riding a motorcycle at high speed through neon-lit '
                                      'city streets at night, dramatic wind in the hair', w=672, h=400),
}

ARMS = {
    'routed_tf': dict(layout='target_first', masked='last',
                      root='/workspace/checkpoints/anima_routed_targetfirst'),
    'routed_rf': dict(layout='ref_first', masked='first',
                      root='/workspace/checkpoints/anima_routed_reffirst'),
    'global_tf': dict(layout='target_first', masked=None,
                      root='/workspace/checkpoints/anima_global_targetfirst'),
}


class _FullLoraScope(_RoutedLoraScope):
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
        emb, mask, t5_ids, t5_mask = enc
        with torch.autocast('cuda', dtype=DTYPE):
            ctx = dit.llm_adapter(source_hidden_states=emb.to(DTYPE), target_input_ids=t5_ids,
                                  target_attention_mask=t5_mask, source_attention_mask=mask)
        ctx[~t5_mask.bool()] = 0
        return ctx

    neg_ctx = build_context(encode_prompt(text_encoder, tokenizer, t5_tokenizer, NEG, DEVICE, DTYPE))
    ref_data = {}
    for name, r in REFS.items():
        pos_ctx = build_context(encode_prompt(text_encoder, tokenizer, t5_tokenizer, r['prompt'], DEVICE, DTYPE))
        ref_data[name] = (pos_ctx, encode_control(vae, r['ref'], r['h'], r['w'], DEVICE, DTYPE))

    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        name, r = 'par100', REFS['par100']
        pos_ctx, ctrl = ref_data[name]
        lat = sample_normal(dit, pos_ctx, neg_ctx, r['h'], r['w'], STEPS, TEXT_CFG, SHIFT, 76, DEVICE, DTYPE)
        decode(vae, lat).save(f'{OUT}/SMOKE_vanilla.png')
        print('SMOKE COMPLETO', flush=True)
        return

    print('== vanilla ==', flush=True)
    for name, r in REFS.items():
        pos_ctx, _ = ref_data[name]
        for seed in SEEDS:
            f = f'{OUT}/vanilla_{name}_seed{seed}.png'
            if not os.path.exists(f):
                lat = sample_normal(dit, pos_ctx, neg_ctx, r['h'], r['w'], STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE)
                decode(vae, lat).save(f)

    for arm, cfg in ARMS.items():
        for step in CKPTS:
            found = glob.glob(f"{cfg['root']}/*/step{step}/adapter_model.safetensors")
            if not found:
                print(f'!! sem checkpoint: {arm} step{step}', flush=True)
                continue
            entries = load_routed_lora_entries(dit, found[0], DEVICE, DTYPE)
            scope_cls = _RoutedLoraScope if cfg['masked'] else _FullLoraScope
            scope_kwargs = dict(masked=cfg['masked']) if cfg['masked'] else {}
            for name, r in REFS.items():
                pos_ctx, ctrl = ref_data[name]
                for seed in SEEDS:
                    for rc in REF_CFGS:
                        f = f'{OUT}/{arm}_s{step}_{name}_seed{seed}_rc{rc}.png'
                        if os.path.exists(f):
                            continue
                        with scope_cls(entries, 1.0, **scope_kwargs):
                            if cfg['layout'] == 'ref_first':
                                lat = sample_ic_lora_full(dit, pos_ctx, neg_ctx, ctrl, r['h'], r['w'],
                                                          STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE, ref_cfg=rc)
                            else:
                                lat = sample_ominicontrol(dit, pos_ctx, neg_ctx, ctrl, r['h'], r['w'],
                                                          STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE,
                                                          position_mode='subject', ref_cfg=rc)
                        decode(vae, lat).save(f)
                        print(f'ok {f}', flush=True)

    W, H, lh = 300, 179, 22
    cols = ['referencia', 'target real', 'vanilla'] + [f'ref_cfg {rc}' for rc in REF_CFGS]
    for arm in ARMS:
        for step in CKPTS:
            for seed in SEEDS:
                grid = Image.new('RGB', (W * len(cols), (H + lh) * len(REFS) + 26), 'white')
                d = ImageDraw.Draw(grid)
                d.text((8, 5), f'{arm} step{step} seed{seed} — controlados: rank32 lr5e-5 sampler padrao (text_cfg4)', fill='black')
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
    print('CONTROLLED EVAL COMPLETA', flush=True)


if __name__ == '__main__':
    main()
