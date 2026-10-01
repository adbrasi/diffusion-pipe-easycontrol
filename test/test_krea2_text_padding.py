"""Regression: batching must not change conditioning or adapter gradients."""
import torch
import pytest
from torch.utils.checkpoint import checkpoint

from test.test_krea2_native import kn


def make_pipeline():
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT
    from models.krea2_reference import Krea2ReferenceInitialLayer, Krea2ReferenceFinalLayer
    torch.manual_seed(76)
    pipe = object.__new__(kn.Krea2NativePipeline)
    pipe.diffusion_model = SingleStreamDiT(features=128, tdim=32, txtdim=32,
        heads=4, kvheads=2, multiplier=2, layers=2, txtlayers=3,
        txtheads=2, txtkvheads=2, operations=comfy.ops.disable_weight_init).float()
    for p in pipe.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=.05)
    pipe.configure_adapter(dict(type='lora', rank=4, alpha=4, dropout=0., dtype=torch.float32))
    # Exercise gradients of A and B rather than only zero-initialized B.
    for name, p in pipe.diffusion_model.named_parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.03)
    net = pipe.diffusion_model
    layers = [Krea2ReferenceInitialLayer(net), *net.blocks, Krea2ReferenceFinalLayer(net)]
    return pipe, layers


def forward(layers, inputs, checkpointed):
    initial, *blocks, final = layers
    h = checkpoint(initial, inputs, use_reentrant=False) if checkpointed else initial(inputs)
    combined, tf, tvec, freqs, mask, sizes = h
    for block in blocks:
        combined = (checkpoint(block, combined, tvec, freqs, mask, use_reentrant=False)
                    if checkpointed else block(combined, tvec, freqs, mask))
    return final((combined, tf, tvec, freqs, mask, sizes))


@pytest.mark.parametrize('checkpointed', [False, True])
def test_variable_text_batch_matches_individual_outputs_and_gradients(checkpointed):
    pipe, layers = make_pipeline()
    target, ref = torch.randn(2,16,1,6,8), torch.randn(2,16,1,6,8)
    text = [torch.randn(n,96) for n in [3,9]]
    context, mask = pipe.get_conds(dict(text_embeds_0=text,
        attention_mask_0=[torch.ones(n) for n in [3,9]]))
    t = torch.tensor([.63,.27])
    params = [p for p in pipe.diffusion_model.parameters() if p.requires_grad]
    batch = forward(layers, (target,t,context,mask,ref), checkpointed)
    batch.square().mean().backward()
    batch_grad = [p.grad.detach().clone() for p in params]
    pipe.diffusion_model.zero_grad(set_to_none=True)
    individual = []
    for i,n in enumerate([3,9]):
        out = forward(layers,(target[i:i+1],t[i:i+1],text[i][None],mask[i:i+1,:n],ref[i:i+1]),checkpointed)
        individual.append(out.detach())
        (out.square().mean()/2).backward()
    torch.testing.assert_close(batch,torch.cat(individual),rtol=2e-5,atol=1e-7)
    for p, g in zip(params,batch_grad):
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(g,p.grad,rtol=2e-4,atol=2e-8)

