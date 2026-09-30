"""Compare the native training forward with stock on real BF16 GPU weights.

No sampling, optimizer, adapters or text encoder are involved. The same model
and deterministic inputs serve both paths, isolating the timestep/geometry
contract and BF16 rounding from data or learned deltas.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    import comfy.model_management
    import comfy.sd
    sys.path.remove(str(args.comfy))
    sys.path.insert(0, str(ROOT))
    import utils.common
    utils.common.AUTOCAST_DTYPE = torch.bfloat16
    from models.krea2_reference import Krea2ReferenceInitialLayer, Krea2ReferenceFinalLayer

    patcher = comfy.sd.load_diffusion_model(str(args.weights), model_options={'dtype': torch.bfloat16})
    net = patcher.model.diffusion_model.to('cuda').eval()
    generator = torch.Generator(device='cuda').manual_seed(42)
    x = torch.randn(1, 16, 48, 86, device='cuda', dtype=torch.bfloat16, generator=generator)
    reference = torch.randn(x.shape, device='cuda', dtype=x.dtype, generator=generator)
    context = torch.randn(1, 300, 12 * 2560, device='cuda', dtype=x.dtype, generator=generator)
    mask = torch.ones(1, context.shape[1], device='cuda', dtype=torch.bool)
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for method, mode in [('index_timestep_zero', 'zero'), ('index', 'target')]:
            initial = Krea2ReferenceInitialLayer(net, position_mode='subject',
                                               reference_timestep_mode=mode)
            final = Krea2ReferenceFinalLayer(net)
            t = torch.tensor([0.6], device='cuda')
            stock = net._forward(x, t, context, ref_latents=[reference],
                                 ref_latents_method=method, transformer_options={})
            combined, tf, tvec, freqs, amask, sizes = initial(
                (x.unsqueeze(2), t, context, mask, reference.unsqueeze(2)))
            for block in net.blocks:
                combined = block(combined, tvec, freqs, amask)
            fork = final((combined, tf, tvec, freqs, amask, sizes))[:, :, 0]
            relative = float((fork.float() - stock.float()).norm() / stock.float().norm())
            result = dict(method=method, rel_l2=relative,
                          max_abs=float((fork.float() - stock.float()).abs().max()),
                          stock_norm=float(stock.float().norm()),
                          finite=bool(torch.isfinite(fork).all() and torch.isfinite(stock).all()))
            results.append(result)
            print(json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(weights=str(args.weights), dtype='bfloat16',
                                      target_shape=list(x.shape), results=results), indent=2))
    if any(not row['finite'] or row['rel_l2'] > 0.02 for row in results):
        raise SystemExit('Real-weight GPU parity failed (>2% relative L2)')


if __name__ == '__main__':
    main()
