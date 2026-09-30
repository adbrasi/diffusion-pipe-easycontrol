#!/usr/bin/env python3
"""Held-out evaluation for anima_nextscene checkpoints — metrics that do NOT reward copying.

Lesson of the whole saga: eyeballing 3 examples and palette metrics misled us
five times, and "similarity to the reference" rewards the copy failure. Here
every score is measured against the GROUND-TRUTH scene 2 of held-out pairs,
with the copy failure measured explicitly.

For each checkpoint and held-out pair (ref A, real B, caption of B):
    out_true  = generate(caption, ref=A)
    out_shuf  = generate(caption, ref=A of another pair)       (reference swap test)
    out_null  = generate(caption, ref=zeros)                   (trained null of ref_dropout)
Metrics (DINOv2 cosine; CCIP anime-identity if dghs-imgutils is installed):
    gt_true      sim(out_true, B)                 higher = closer to the real next scene
    ref_gain     sim(out_true, B) - sim(out_shuf, B)   > 0 = the reference is really used
    null_gain    sim(out_true, B) - sim(out_null, B)
    copy_gap     sim(out_true, A) - sim(B, A)     > 0 = closer to the reference than the real
                                                  next scene is (the "gives me back A" failure)
    copy_rate    share of pairs whose out_true is a near-duplicate of A (dHash)
    ccip_true    CCIP "same character" rate between out_true and B (identity)

Outputs per checkpoint: <out>/<ckpt_name>/grid.png (rows = pairs; cols = A | B | true | shuffled | null)
and metrics.json; plus <out>/summary.csv comparing checkpoints.

Usage:
    python tools/nextscene_eval.py --dit ... --vae ... --llm ... \\
        --pairs /workspace/heldout   (target/ + control/ like the training layout, 10-30 pairs) \\
        --ckpt run/step2500 --ckpt run/step5000 ... \\
        --width 768 --height 768 --steps 30 --cfg 4 --flow_shift 3 --out /workspace/eval_ns
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
import infer_easycontrol as ie  # noqa: E402
from models.anima_nextscene import install_nextscene_rope  # noqa: E402

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


def load_pairs(root, limit):
    root = Path(root)
    t_dir, c_dir = root / 'target', root / 'control'
    controls = {p.stem: p for p in c_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = []
    for tp in sorted(p for p in t_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        if tp.stem in controls and tp.with_suffix('.txt').exists():
            pairs.append((tp.stem, controls[tp.stem], tp, tp.with_suffix('.txt').read_text().strip()))
    return pairs[:limit]


class Feats:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
        try:
            from imgutils.metrics import ccip_batch_same
            self.ccip = ccip_batch_same
        except Exception:  # noqa: BLE001
            self.ccip = None
            print('dghs-imgutils not installed: CCIP identity metric skipped (pip install dghs-imgutils)')

    @torch.no_grad()
    def dino(self, images):
        x = self.proc(images=[i.convert('RGB') for i in images], return_tensors='pt').to(self.device)
        f = self.model(**x).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(f.float(), dim=-1)

    def ccip_same(self, a, b):
        if self.ccip is None:
            return None
        try:
            return bool(self.ccip([a, b])[0, 1])
        except Exception:  # noqa: BLE001  (no character detected, etc.)
            return None


def dhash_ham(a, b, size=8):
    def h(img):
        g = img.convert('L').resize((size + 1, size))
        px = list(g.tobytes())
        return [px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] for r in range(size) for c in range(size)]
    return sum(x != y for x, y in zip(h(a), h(b)))


def fit(img, w, h):
    from PIL import ImageOps
    return ImageOps.fit(img.convert('RGB'), (w, h))


class Runner:
    def __init__(self, args, device, dtype):
        self.args, self.device, self.dtype = args, device, dtype
        self.dit = ie.load_dit(args.dit, device, dtype)
        self.base = {n: p.detach().to('cpu', copy=True) for n, p in self.dit.named_parameters()}
        self.vae = ie.load_vae(args.vae, device, dtype)
        te, tok, t5tok = ie.load_text_encoder(args.llm, device, dtype)
        self.te, self.tok, self.t5tok = te, tok, t5tok
        self.neg = self.encode(args.negative_prompt) if args.cfg > 1 else None

    def encode(self, prompt):
        emb, mask, t5, t5mask = ie.encode_prompt(self.te, self.tok, self.t5tok, prompt, self.device, self.dtype)
        with torch.autocast('cuda', dtype=self.dtype):
            ctx = self.dit.llm_adapter(source_hidden_states=emb.to(self.dtype), target_input_ids=t5,
                                       target_attention_mask=t5mask, source_attention_mask=mask)
        ctx[~t5mask.bool()] = 0
        return ctx

    def load_ckpt(self, ckpt):
        with torch.no_grad():
            for n, p in self.dit.named_parameters():
                p.copy_(self.base[n].to(p.device, p.dtype))
        path = ckpt if ckpt.endswith('.safetensors') else os.path.join(ckpt, 'adapter_model.safetensors')
        contract = ie.read_nextscene_contract(path)
        layout = self.args.rope_layout or (contract or {}).get('rope_layout')
        index = self.args.ref_temporal_index if self.args.ref_temporal_index is not None \
            else (contract or {}).get('ref_temporal_index')
        if layout is None or index is None:
            raise SystemExit(f'{path}: no nextscene contract; pass --rope_layout/--ref_temporal_index')
        install_nextscene_rope(self.dit.pos_embedder, layout, int(index))
        ie.load_peft_lora(self.dit, path, self.device, self.dtype, lora_strength=self.args.lora_strength)
        return path

    def ref_latent(self, path):
        return ie.encode_control(self.vae, path, self.args.height, self.args.width, self.device, self.dtype)

    def generate(self, ctx, ref_lat, seed):
        a = self.args
        lat = ie.sample_nextscene(self.dit, ctx, self.neg, ref_lat, a.height, a.width, a.steps, a.cfg,
                                  a.flow_shift, seed, self.device, self.dtype, ref_cfg=a.ref_cfg,
                                  uncond_ref=a.uncond_ref)
        with torch.no_grad():
            px = self.vae.model.decode(lat.to(self.dtype), self.vae.scale)
        px = px.squeeze(2) if px.ndim == 5 else px
        x = ((px[0].float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray(x)


def grid(rows, cell=256):
    cols = max(len(r) for r in rows)
    g = Image.new('RGB', (cols * cell, len(rows) * cell), 'white')
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            im = im.copy()
            im.thumbnail((cell, cell))
            g.paste(im, (j * cell, i * cell))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dit', required=True)
    ap.add_argument('--vae', required=True)
    ap.add_argument('--llm', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--height', type=int, default=768)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--flow_shift', type=float, default=3.0)
    ap.add_argument('--ref_cfg', type=float, default=1.0)
    ap.add_argument('--uncond_ref', default='keep', choices=['keep', 'zero'])
    ap.add_argument('--lora_strength', type=float, default=1.0)
    ap.add_argument('--rope_layout', default=None)
    ap.add_argument('--ref_temporal_index', type=int, default=None)
    ap.add_argument('--negative_prompt', default='worst quality, low quality, blurry, jpeg artifacts')
    ap.add_argument('--seed', type=int, default=76)
    ap.add_argument('--copy_dhash', type=int, default=6, help='out vs ref Hamming <= this = copy')
    args = ap.parse_args()

    device, dtype = torch.device('cuda'), torch.bfloat16
    pairs = load_pairs(args.pairs, args.limit)
    assert len(pairs) >= 2, 'need >= 2 held-out pairs (shuffled-ref test)'
    run = Runner(args, device, dtype)
    feats = Feats(device)
    ctxs = [run.encode(cap) for _, _, _, cap in pairs]
    run.te.to('cpu')
    torch.cuda.empty_cache()
    refs = [run.ref_latent(c) for _, c, _, _ in pairs]
    A = [fit(Image.open(c), args.width, args.height) for _, c, _, _ in pairs]
    B = [fit(Image.open(t), args.width, args.height) for _, _, t, _ in pairs]
    fA, fB = feats.dino(A), feats.dino(B)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for ckpt in args.ckpt:
        path = run.load_ckpt(ckpt)
        name = Path(ckpt).name if not ckpt.endswith('.safetensors') else Path(ckpt).parent.name
        rows, per = [], []
        for i, (stem, _, _, _) in enumerate(pairs):
            j = (i + 1) % len(pairs)
            o_true = run.generate(ctxs[i], refs[i], args.seed)
            o_shuf = run.generate(ctxs[i], refs[j], args.seed)
            o_null = run.generate(ctxs[i], torch.zeros_like(refs[i]), args.seed)
            f = feats.dino([o_true, o_shuf, o_null])
            m = {
                'stem': stem,
                'gt_true': float(f[0] @ fB[i]),
                'ref_gain': float(f[0] @ fB[i] - f[1] @ fB[i]),
                'null_gain': float(f[0] @ fB[i] - f[2] @ fB[i]),
                'copy_gap': float(f[0] @ fA[i] - fB[i] @ fA[i]),
                'copy': int(dhash_ham(o_true, A[i]) <= args.copy_dhash),
                'ccip_true': feats.ccip_same(o_true, B[i]),
            }
            per.append(m)
            rows.append([A[i], B[i], o_true, o_shuf, o_null])
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        grid(rows).save(d / 'grid.png')

        def mean(k):
            v = [p[k] for p in per if p[k] is not None]
            return round(sum(v) / len(v), 4) if v else None
        agg = {'ckpt': name, 'adapter': path, 'n': len(per),
               **{k: mean(k) for k in ('gt_true', 'ref_gain', 'null_gain', 'copy_gap', 'ccip_true')},
               'copy_rate': mean('copy')}
        (d / 'metrics.json').write_text(json.dumps({'summary': agg, 'pairs': per}, indent=2))
        summary.append(agg)
        print(json.dumps(agg))

    with open(Path(args.out) / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f'summary: {Path(args.out) / "summary.csv"}  (grids: <out>/<ckpt>/grid.png — the verdict is still visual)')


if __name__ == '__main__':
    main()
