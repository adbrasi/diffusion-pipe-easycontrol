#!/usr/bin/env python3
"""Avaliação dos braços de ESCOPO LARGO: ic_lora_v3, ic_lora_dual, omini_broad.
Mesmo protocolo da bateria controlada (refs novas, 3 branches, target real)."""
import os
import sys
import glob

sys.path.insert(0, '/workspace/diffusion-pipe-easycontrol')
os.chdir('/workspace/diffusion-pipe-easycontrol')

import torch
from PIL import Image, ImageDraw
import numpy as np

from infer_easycontrol import (
    load_dit, load_vae, load_text_encoder, encode_prompt, encode_control,
    sample_ominicontrol, sample_normal,
    load_routed_lora_entries, _RoutedLoraScope, _FullGlobalLoraScope,
)

DEVICE, DTYPE = torch.device('cuda'), torch.bfloat16
DIT = '/workspace/models_anima/split_files/diffusion_models/anima-base-v1.0.safetensors'
VAE = '/workspace/models/qwen_image_vae.safetensors'
LLM = '/workspace/models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors'
NEG = 'worst quality, low quality, score_1, score_2, score_3, artist name'
OUT = '/workspace/outputs/anima_broad_eval'
STEPS, TEXT_CFG, SHIFT = 30, 4.0, 3.0
SEEDS = [76, 200]
REF_CFGS = [0.0, 0.5, 1.0, 1.5]
CKPTS = [250, 500]

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
    'iclora_v3': dict(dual=False, root='/workspace/checkpoints/anima_iclora_v3'),
    'iclora_dual': dict(dual=True, root='/workspace/checkpoints/anima_iclora_dual'),
    'omini_broad': dict(dual=False, root='/workspace/checkpoints/anima_omini_broad'),
}


def decode(vae, lat):
    with torch.no_grad():
        px = vae.model.decode(lat.to(DTYPE), vae.scale)
    if px.ndim == 5:
        px = px.squeeze(2)
    px = ((px.float().clamp(-1, 1) + 1) / 2 * 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(px)


def main():
    os.makedirs(OUT, exist_ok=True)
    dit = load_dit(DIT, DEVICE, DTYPE)
    vae = load_vae(VAE, DEVICE, DTYPE, None)
    text_encoder, tokenizer, t5_tokenizer = load_text_encoder(LLM, DEVICE, DTYPE)

    def ctx(text):
        emb, mask, t5_ids, t5_mask = encode_prompt(text_encoder, tokenizer, t5_tokenizer, text, DEVICE, DTYPE)
        with torch.autocast('cuda', dtype=DTYPE):
            c = dit.llm_adapter(source_hidden_states=emb.to(DTYPE), target_input_ids=t5_ids,
                                target_attention_mask=t5_mask, source_attention_mask=mask)
        c[~t5_mask.bool()] = 0
        return c

    neg_ctx = ctx(NEG)
    ref_data = {}
    for name, r in REFS.items():
        ref_data[name] = (ctx(r['prompt']), encode_control(vae, r['ref'], r['h'], r['w'], DEVICE, DTYPE))

    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        pos_ctx, ctrl = ref_data['par100']
        r = REFS['par100']
        found = glob.glob(f"{ARMS['iclora_v3']['root']}/*/step250/adapter_model.safetensors")
        if not found:
            print('SMOKE SEM CHECKPOINT', flush=True)
            return
        entries = load_routed_lora_entries(dit, found[0], DEVICE, DTYPE)
        with _FullGlobalLoraScope(entries, 1.0):
            lat = sample_ominicontrol(dit, pos_ctx, neg_ctx, ctrl, r['h'], r['w'],
                                      STEPS, TEXT_CFG, SHIFT, 76, DEVICE, DTYPE,
                                      position_mode='subject', ref_cfg=1.0)
        decode(vae, lat).save(f'{OUT}/SMOKE_v3.png')
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

    # NOTE: os frames de ref zerados das branches t/u do 3-branch continuam
    # existindo; o LoRA global (incl. cross_attn/llm_adapter) também age lá,
    # como no treino com condition_dropout.
    for arm, cfg in ARMS.items():
        for step in CKPTS:
            found = glob.glob(f"{cfg['root']}/*/step{step}/adapter_model.safetensors")
            if not found:
                print(f'!! sem checkpoint: {arm} step{step}', flush=True)
                continue
            entries_p = load_routed_lora_entries(dit, found[0], DEVICE, DTYPE, with_paths=True)
            sem = [e[1:] for e in entries_p if 'cross_attn' in e[0] or e[0].startswith('llm_adapter')]
            app = [e[1:] for e in entries_p if not ('cross_attn' in e[0] or e[0].startswith('llm_adapter'))]
            for name, r in REFS.items():
                pos_ctx, ctrl = ref_data[name]
                for seed in SEEDS:
                    for rc in REF_CFGS:
                        f = f'{OUT}/{arm}_s{step}_{name}_seed{seed}_rc{rc}.png'
                        if os.path.exists(f):
                            continue
                        if cfg['dual']:
                            scopes = (_RoutedLoraScope(app, 1.0, masked='last'),
                                      _FullGlobalLoraScope(sem, 1.0))
                        else:
                            scopes = (_FullGlobalLoraScope(app + sem, 1.0),)
                        try:
                            for s in scopes:
                                s.__enter__()
                            lat = sample_ominicontrol(dit, pos_ctx, neg_ctx, ctrl, r['h'], r['w'],
                                                      STEPS, TEXT_CFG, SHIFT, seed, DEVICE, DTYPE,
                                                      position_mode='subject', ref_cfg=rc)
                        finally:
                            for s in reversed(scopes):
                                s.__exit__()
                        decode(vae, lat).save(f)
                        print(f'ok {f}', flush=True)

    W, H, lh = 300, 179, 22
    cols = ['referencia', 'target real', 'vanilla'] + [f'ref_cfg {rc}' for rc in REF_CFGS]
    for arm in ARMS:
        for step in CKPTS:
            for seed in SEEDS:
                grid = Image.new('RGB', (W * len(cols), (H + lh) * len(REFS) + 26), 'white')
                d = ImageDraw.Draw(grid)
                d.text((8, 5), f'{arm} step{step} seed{seed} — ESCOPO LARGO (cross_attn+llm_adapter), batch8 lr1e-4 r32', fill='black')
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
    print('BROAD EVAL COMPLETA', flush=True)


if __name__ == '__main__':
    main()
