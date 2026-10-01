"""CPU tests for the anima_nextscene contract (tiny random Cosmos DiT, no weights).

Run: python -m pytest -q test/test_anima_nextscene.py
"""

import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _import_pipeline_module():
    # ComfyUI (imported by models.base) probes CUDA at import time unless --cpu.
    if not torch.cuda.is_available() and 'comfy' not in sys.modules:
        sys.path.insert(0, os.path.join(ROOT, 'submodules', 'ComfyUI'))
        saved = sys.argv
        sys.argv = ['pytest', '--cpu']
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            import comfy.model_management  # noqa: F401  (parse args now)
        finally:
            sys.argv = saved
            sys.path.pop(0)
    import utils.common  # noqa: F401  (repo utils before ComfyUI's)
    from models import anima_nextscene
    return anima_nextscene


ns = _import_pipeline_module()
from models.cosmos_predict2_modeling import MiniTrainDIT  # noqa: E402


def tiny_dit():
    torch.manual_seed(0)
    dit = MiniTrainDIT(
        max_img_h=1024, max_img_w=1024, max_frames=128,
        in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True, model_channels=256, num_blocks=2, num_heads=2,
        crossattn_emb_channels=64, pos_emb_cls='rope3d', pos_emb_learnable=True,
        pos_emb_interpolation='crop', use_adaln_lora=True, adaln_lora_dim=32,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
    )
    # Blocks are zero-initialised in places; randomise so outputs are informative.
    with torch.no_grad():
        for p in dit.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.02)
    return dit.eval()


def _rope(pos_embedder, T, H, W):
    return pos_embedder.generate_embeddings(torch.Size([1, T, H, W, 1]), fps=None)


def test_frame0_matches_stock_t2i_embedding():
    dit = tiny_dit()
    pe = dit.pos_embedder
    stock_1 = _rope(pe, 1, 6, 8).clone()
    stock_2 = _rope(pe, 2, 6, 8).clone()
    for layout in ns.ROPE_LAYOUTS:
        restore = ns.install_nextscene_rope(pe, layout, 1)
        assert torch.equal(_rope(pe, 1, 6, 8), stock_1), layout  # T2I untouched
        got2 = _rope(pe, 2, 6, 8)
        assert torch.equal(got2[:48], stock_2[:48]), layout      # target frame untouched
        restore()
    # 'aligned' with index 1 is exactly the old IC-LoRA packing.
    ns.install_nextscene_rope(pe, 'aligned', 1)
    assert torch.equal(_rope(pe, 2, 6, 8), stock_2)


def test_disjoint_layout_moves_reference_off_the_target_grid():
    H, W = 6, 8
    pos = ns.frame_positions('disjoint_w', 1, 2, H, W)
    assert pos == [(0, 0, 0), (1, 0, W)]
    assert ns.frame_positions('disjoint_h', -1, 2, H, W) == [(0, 0, 0), (-1, H, 0)]
    assert ns.frame_positions('disjoint_diag', 1, 3, H, W) == [(0, 0, 0), (1, H, W), (2, 2 * H, 2 * W)]
    dit = tiny_dit()
    pe = dit.pos_embedder
    ns.install_nextscene_rope(pe, 'disjoint_w', 1)
    emb = _rope(pe, 2, H, W)[:, 0, 0]
    half = emb.shape[-1] // 2
    dim_t = pe._dim_t // 2
    dim_h = pe._dim_h // 2
    # w-part of the reference token (r,c) must equal the target token at column c+W
    ref_w = emb[H * W:, :half][:, dim_t + dim_h:]
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range)
    expected = torch.outer(torch.arange(W).float() + W, w_freqs)
    assert torch.allclose(ref_w.view(H, W, -1)[0], expected, atol=1e-6)


def test_tiny_dit_forward_two_frames_per_frame_timestep():
    dit = tiny_dit()
    ns.install_nextscene_rope(dit.pos_embedder, 'disjoint_w', 1)
    x = torch.randn(2, 16, 2, 8, 8)
    t = torch.tensor([[0.7, 0.0], [0.3, 0.0]])
    ctx = torch.randn(2, 5, 64)
    with torch.no_grad():
        out = dit(x, t, ctx, padding_mask=torch.zeros(2, 1, 8, 8))
    assert out.shape == (2, 16, 2, 8, 8)
    assert torch.isfinite(out).all()
    # Changing the reference changes the target-frame prediction (cross-frame attention path).
    x2 = x.clone()
    x2[:, :, 1] = torch.randn_like(x2[:, :, 1])
    with torch.no_grad():
        out2 = dit(x2, t, ctx, padding_mask=torch.zeros(2, 1, 8, 8))
    assert not torch.allclose(out[:, :, 0], out2[:, :, 0])


def test_geometry_changes_what_the_target_sees():
    dit = tiny_dit()
    x = torch.randn(1, 16, 2, 8, 8)
    t = torch.tensor([[0.9, 0.0]])
    ctx = torch.randn(1, 5, 64)
    outs = {}
    for layout in ('aligned', 'disjoint_w'):
        ns.install_nextscene_rope(dit.pos_embedder, layout, 1)
        with torch.no_grad():
            outs[layout] = dit(x, t, ctx, padding_mask=torch.zeros(1, 1, 8, 8))[:, :, 0]
    assert not torch.allclose(outs['aligned'], outs['disjoint_w'])


def test_diff_weight_map_redistributes_without_changing_mean():
    torch.manual_seed(0)
    tgt = torch.randn(3, 16, 1, 8, 8)
    ref = tgt.clone()
    ref[:, :, :, :, :4] += 2.0     # left half changed, right half identical
    w = ns.diff_weight_map(tgt, ref)
    assert w.shape == (3, 1, 1, 8, 8)
    assert torch.allclose(w.mean(dim=(2, 3, 4)), torch.ones(3, 1), atol=1e-5)
    assert (w[..., :4].mean() > 2 * w[..., 4:].mean())


class _Stub(ns.AnimaNextScenePipeline):
    def __init__(self, section, model_config=None):
        self.model_config = model_config or {'sigmoid_scale': 1.0}
        self.cache_text_embeddings = True
        self._parse_nextscene_config({'nextscene': section})


def _inputs(bs=4, h=8, w=8):
    return {
        'latents': torch.randn(bs, 16, 1, h, w),
        'control_latents': torch.randn(bs, 16, 1, h, w),
        'mask': None,
        'prompt_embeds': torch.randn(bs, 7, 1024),
        'attn_mask': torch.ones(bs, 7, dtype=torch.long),
        't5_input_ids': torch.ones(bs, 7, dtype=torch.long),
        't5_attn_mask': torch.ones(bs, 7, dtype=torch.long),
    }


def test_prepare_inputs_contract():
    torch.manual_seed(0)
    p = _Stub({'ref_dropout': 0.0, 'high_noise_prob': 0.0})
    inputs = _inputs()
    (x, t, *text), (target, mask, weight) = p.prepare_inputs(inputs)
    assert x.shape == (4, 16, 2, 8, 8)
    assert torch.equal(x[:, :, 1:], inputs['control_latents'])   # clean ref, second frame
    assert t.shape == (4, 2) and torch.all(t[:, 1] == 0) and torch.all(t[:, 0] > 0)
    # target frame is the flow interpolation of the TARGET latents
    sig = t[:, 0].view(-1, 1, 1, 1, 1)
    noise = target + inputs['latents']
    assert torch.allclose(x[:, :, :1], (1 - sig) * inputs['latents'] + sig * noise, atol=1e-5)
    assert weight.shape == (4, 1, 1, 8, 8)


def test_ref_dropout_blanks_reference_and_resets_weight():
    torch.manual_seed(0)
    p = _Stub({'ref_dropout': 1.0})
    (x, t, *_), (_, _, weight) = p.prepare_inputs(_inputs())
    assert torch.all(x[:, :, 1:] == 0)
    assert torch.all(weight == 1)


def test_high_noise_mixture_and_eval_quantile():
    torch.manual_seed(0)
    p = _Stub({'high_noise_prob': 1.0, 'ref_dropout': 0.0})
    (_, t, *_), _ = p.prepare_inputs(_inputs(bs=64))
    assert torch.all(t[:, 0] >= 0.8)
    (_, t_eval, *_), _ = p.prepare_inputs(_inputs(bs=2), timestep_quantile=0.5)
    assert torch.allclose(t_eval[:, 0], torch.full((2,), 0.5), atol=1e-4)   # eval path is deterministic


def test_ref_noise_sets_reference_timestep():
    torch.manual_seed(0)
    p = _Stub({'ref_noise_prob': 1.0, 'ref_dropout': 0.0})
    inputs = _inputs()
    (x, t, *_), _ = p.prepare_inputs(inputs)
    assert torch.all(t[:, 1] > 0) and torch.all(t[:, 1] <= 0.3)
    assert not torch.equal(x[:, :, 1:], inputs['control_latents'])


def test_loss_uses_target_frame_only_and_backprops():
    torch.manual_seed(0)
    p = _Stub({'ref_dropout': 0.0})
    loss_fn = p.get_loss_fn()
    target = torch.randn(2, 16, 1, 4, 4)
    weight = torch.ones(2, 1, 1, 4, 4)
    out = torch.zeros(2, 16, 2, 4, 4, requires_grad=True)
    with torch.no_grad():
        out[:, :, 0] = target[:, :, 0]
        out[:, :, 1] = 100.0   # garbage on the reference frame must not matter
    loss = loss_fn(out, (target, None, weight))
    assert loss.item() < 1e-8
    loss2 = loss_fn(out * 1.0 + 1.0, (target, None, weight))
    loss2.backward()
    assert out.grad[:, :, 1].abs().sum() == 0 and out.grad[:, :, 0].abs().sum() > 0


def test_end_to_end_step_with_lora_gradients():
    import peft
    dit = tiny_dit().train()
    ns.install_nextscene_rope(dit.pos_embedder, 'disjoint_w', 1)
    targets = sorted({n for n, m in dit.named_modules()
                      if isinstance(m, torch.nn.Linear) and ('self_attn' in n or 'mlp' in n)})
    model = peft.get_peft_model(dit, peft.LoraConfig(r=4, lora_alpha=4, target_modules=targets))
    for n, prm in model.named_parameters():   # PEFT inits B=0; perturb so every A gets grad
        if 'lora_B' in n:
            torch.nn.init.normal_(prm, std=0.01)
    p = _Stub({'ref_dropout': 0.0})
    inp = _inputs(bs=2)
    inp['prompt_embeds'] = torch.randn(2, 5, 64)
    (x, t, ctx, *_), label = p.prepare_inputs(inp)
    out = model(x, t, ctx, padding_mask=torch.zeros(2, 1, 8, 8))
    loss = p.get_loss_fn()(out, label)
    loss.backward()
    grads = [prm.grad for n, prm in model.named_parameters() if 'lora_A' in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


def test_contract_metadata_roundtrip():
    import json
    p = _Stub({'rope_layout': 'disjoint_h', 'ref_temporal_index': -1})
    meta = {'nextscene_contract': json.dumps(p.contract())}
    c = ns.contract_from_metadata(meta)
    assert c['rope_layout'] == 'disjoint_h' and c['ref_temporal_index'] == -1
    assert ns.contract_from_metadata({}) is None
    with pytest.raises(ValueError):
        _Stub({'rope_layout': 'nope'})


def test_adapter_excludes_llm_bridge_with_matching_block_suffixes():
    import copy
    dit = tiny_dit()
    # The real bridge has blocks with the same suffixes as the main DiT.
    dit.llm_adapter = torch.nn.Module()
    dit.llm_adapter.blocks = copy.deepcopy(dit.blocks)
    p = _Stub({'lora_cross_attn': True})
    p.transformer = dit
    p.configure_adapter({'type': 'lora', 'rank': 4, 'alpha': 4,
                         'dropout': 0.0, 'dtype': torch.float32})
    trainable = [n for n, prm in p.transformer.named_parameters() if prm.requires_grad]
    assert trainable and any('cross_attn' in n for n in trainable)
    assert not any('llm_adapter' in n or 'adaln_modulation' in n for n in trainable)


def test_pipeline_dataloader_preserves_and_splits_loss_weights():
    from types import SimpleNamespace
    from utils.dataset import PipelineDataLoader
    x = torch.randn(4, 16, 2, 4, 4)
    target = torch.randn(4, 16, 1, 4, 4)
    weight = torch.arange(4.).reshape(4, 1, 1, 1, 1).expand(4, 1, 1, 4, 4)
    loader = PipelineDataLoader.__new__(PipelineDataLoader)
    loader.dataloader = [{}]
    loader.model = SimpleNamespace(
        prepare_inputs=lambda *a, **k: ((x,), (target, None, weight)),
        prepare_inputs_per_microbatch=False,
    )
    loader.model_engine = SimpleNamespace(is_pipe_parallel=False)
    loader.eval_quantile = None
    loader.gradient_accumulation_steps = 2
    loader.num_batches_pulled = 0
    batches = list(loader._pull_batches_from_dataloader())
    assert len(batches) == 2
    for i, (_, label) in enumerate(batches):
        assert len(label) == 3 and label[1].numel() == 0
        assert torch.equal(label[0], target[2*i:2*i+2])
        assert torch.equal(label[2], weight[2*i:2*i+2])


def test_runner_sampler_cfg_algebra_and_smoke():
    import infer_easycontrol as ie
    c, n, z = torch.full((1,), 3.0), torch.full((1,), 1.0), torch.full((1,), 2.0)
    assert torch.allclose(ie.combine_nextscene(c, n, z, 4.0, 1.0), torch.tensor([9.0]))   # text CFG only
    assert torch.allclose(ie.combine_nextscene(c, n, z, 4.0, 2.0), torch.tensor([10.0]))  # + (ref_cfg-1)(c-z)
    assert torch.allclose(ie.combine_nextscene(c, None, None, 1.0, 1.0), c)

    dit = tiny_dit()
    ns.install_nextscene_rope(dit.pos_embedder, 'disjoint_w', 1)
    ref = torch.randn(1, 16, 1, 8, 8)
    ctx, neg = torch.randn(1, 5, 64), torch.randn(1, 5, 64)
    dit = dit.to(torch.bfloat16)
    out = ie.sample_nextscene(dit, ctx.bfloat16(), neg.bfloat16(), ref, 64, 64, 2, 3.0, 3.0, 7,
                              torch.device('cpu'), torch.bfloat16, ref_cfg=1.5, uncond_ref='keep', ref_renoise=0.1)
    assert out.shape == (1, 16, 1, 8, 8) and torch.isfinite(out.float()).all()
