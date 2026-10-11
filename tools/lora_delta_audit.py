"""Measure how far a LoRA moves the base weights, without torch or a GPU.

For every adapted Linear it reports the spectral (|dW|_2 / |W|_2) and Frobenius
(|dW|_F / |W|_F) ratios of the LoRA delta against the base weight, optionally next
to a reference LoRA (e.g. Comfy-Org krea2_turbo_lora_rank_64_bf16, i.e. the whole
Raw->Turbo distillation delta). Used in docs/DIAGNOSTICO_KREA2_2026-10-11.md.

The base can be a local .safetensors or an HTTP(S) URL: only the requested
tensors are fetched with range requests (a 26 GB checkpoint is never downloaded).

    python tools/lora_delta_audit.py \
        --base https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_bf16.safetensors \
        --ref loras/krea2_turbo_lora_rank_64_bf16.safetensors \
        --blocks 0 14 27 --adapters step500/adapter_model.safetensors step3000/adapter_model.safetensors
"""

import argparse
import json
import struct
import urllib.request

import numpy as np

def _decode(raw, dtype, shape):
    if dtype == 'BF16':
        a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dtype == 'F16':
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == 'F32':
        a = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f'unsupported dtype {dtype} (use a bf16/fp16/fp32 base, not fp8)')
    return a.reshape(shape)


class SafeTensors:
    def __init__(self, path):
        self.path = path
        self.remote = path.startswith(('http://', 'https://'))
        n = struct.unpack('<Q', self._read(0, 8))[0]
        self.header = json.loads(self._read(8, n))
        self.base = 8 + n

    def _read(self, start, length):
        if self.remote:
            req = urllib.request.Request(self.path, headers={'Range': f'bytes={start}-{start + length - 1}'})
            with urllib.request.urlopen(req) as r:
                return r.read()
        with open(self.path, 'rb') as f:
            f.seek(start)
            return f.read(length)

    def keys(self):
        return [k for k in self.header if k != '__metadata__']

    def get(self, key):
        meta = self.header[key]
        s, e = meta['data_offsets']
        return _decode(self._read(self.base + s, e - s), meta['dtype'], meta['shape'])


def lora_delta(st, module_key):
    """module_key like 'blocks.0.attn.wq'; handles PEFT (lora_A/B) and kohya (lora_down/up + alpha)."""
    keys = [k for k in st.keys() if f'{module_key}.' in k]
    down = next((k for k in keys if 'lora_A.' in k or 'lora_down.' in k), None)
    if down is None:
        return None
    up = down.replace('lora_A.', 'lora_B.').replace('lora_down.', 'lora_up.')
    A, B = st.get(down), st.get(up)
    alpha = next((k for k in keys if k.endswith('.alpha')), None)
    scale = float(st.get(alpha).reshape(-1)[0]) / A.shape[0] if alpha else 1.0
    return B * scale, A


def spectral_norm(M, tol=1e-7, max_iters=2000):
    """Largest singular value. Low-rank deltas are exact (QR); dense weights use
    power iteration on the Gram matrix until converged (30 fixed iterations are NOT
    enough for these weights: the top singular values are close together)."""
    M = np.asarray(M, dtype=np.float64)
    v = np.random.default_rng(0).standard_normal(M.shape[1])
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(max_iters):
        w = M.T @ (M @ v)
        s_new = np.sqrt(np.linalg.norm(w))
        v = w / np.linalg.norm(w)
        if abs(s_new - s) <= tol * s_new:
            return s_new
        s = s_new
    return s


def lowrank_spectral_norm(B, A):
    """Exact largest singular value of B @ A for B (out, r), A (r, in)."""
    rb = np.linalg.qr(np.asarray(B, dtype=np.float64))[1]
    ra = np.linalg.qr(np.asarray(A, dtype=np.float64).T)[1]
    return np.linalg.svd(rb @ ra.T, compute_uv=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='bf16/fp16/fp32 base checkpoint (path or URL)')
    ap.add_argument('--ref', help='reference LoRA to compare against (e.g. the Turbo distillation LoRA)')
    ap.add_argument('--blocks', type=int, nargs='+', default=[0, 14, 27])
    ap.add_argument('--modules', nargs='+', default=['attn.wq', 'attn.wk', 'attn.wo', 'mlp.gate', 'mlp.down'])
    ap.add_argument('--adapters', nargs='+', required=True)
    args = ap.parse_args()

    base = SafeTensors(args.base)
    ref = SafeTensors(args.ref) if args.ref else None
    adapters = [SafeTensors(p) for p in args.adapters]
    print('ratios are |dW|_2/|W|_2 (spectral) and |dW|_F/|W|_F (Frobenius)')
    for b in args.blocks:
        for m in args.modules:
            key = f'blocks.{b}.{m}'
            W = base.get(f'{key}.weight')
            sW, fW = spectral_norm(W), np.linalg.norm(W)
            cols = [f'{key:22s} |W|2={sW:7.2f}']
            for name, st in ([('ref', ref)] if ref else []) + [(p, a) for p, a in zip(args.adapters, adapters)]:
                f = lora_delta(st, key)
                if f is None:
                    ratio = '-'
                else:
                    Bs, A = f
                    ratio = f'{lowrank_spectral_norm(Bs, A) / sW:.3f}/{np.linalg.norm(Bs @ A) / fW:.4f}'
                cols.append(f'{name.split("/")[-2] if "/" in name else name}: {ratio}')
            print('  '.join(cols), flush=True)


if __name__ == '__main__':
    main()
