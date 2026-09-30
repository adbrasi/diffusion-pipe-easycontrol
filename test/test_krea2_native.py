"""CPU tests for the krea2_native contract (stock ComfyUI Krea 2 edit path).

The forward-parity test needs a checkout of the ComfyUI you run
(>= c9602625) in KREA2_STOCK_COMFY; it is skipped otherwise.
"""

import math
import os
import subprocess
import sys

import pytest
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _import():
    if not torch.cuda.is_available() and 'comfy' not in sys.modules:
        sys.path.insert(0, os.path.join(ROOT, 'submodules', 'ComfyUI'))
        saved, sys.argv = sys.argv, ['pytest', '--cpu']
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            import comfy.model_management  # noqa: F401
        finally:
            sys.argv = saved
            sys.path.pop(0)
    import utils.common  # noqa: F401
    from models import krea2_native
    return krea2_native


kn = _import()


def _node_sizes(w, h):
    # Verbatim arithmetic of comfy_extras/nodes_qwen.py::TextEncodeQwenImageEditPlus
    s = math.sqrt(384 * 384 / (w * h))
    vl = (round(w * s), round(h * s))
    s = math.sqrt(1024 * 1024 / (w * h))
    ref = (round(w * s / 8.0) * 8, round(h * s / 8.0) * 8)
    return vl, ref


@pytest.mark.parametrize('w,h', [(1920, 1080), (512, 768), (1000, 1000), (333, 1777), (4096, 2160)])
def test_sizes_match_node_arithmetic(w, h):
    vl, ref = _node_sizes(w, h)
    assert kn.native_vl_size(w, h) == vl
    assert kn.native_ref_size(w, h) == ref


def test_text_layout_and_template():
    assert kn.native_text('the same girl now sits', 1) == \
        'Picture 1: <|vision_start|><|image_pad|><|vision_end|>the same girl now sits'
    assert kn.native_text('x', 2).count('<|vision_start|>') == 2 and 'Picture 2: ' in kn.native_text('x', 2)
    assert kn.native_text('x', 1, grounded=False) == 'x'
    assert kn.QWEN_EDIT_PLUS_TEMPLATE.startswith('<|im_start|>system\nDescribe the key features of the input image')
    assert kn.QWEN_EDIT_PLUS_TEMPLATE.endswith('<|im_start|>assistant\n')


def test_reference_preprocess_is_node_resize(tmp_path):
    g = torch.Generator().manual_seed(0)
    arr = (torch.rand(300, 520, 3, generator=g) * 255).to(torch.uint8).numpy()
    path = tmp_path / 'ref.png'
    Image.fromarray(arr).save(path)
    got = kn.PreprocessNativeControlFile()((None, str(path)), None, size_bucket=(512, 512))[0][0]
    w, h = kn.native_ref_size(520, 300)
    assert got.shape == (3, 1, h, w)
    assert got.min() >= -1 and got.max() <= 1
    # identical to comfy.utils.common_upscale(area) on the LoadImage tensor, mapped to [-1,1]
    import comfy.utils
    img = torch.from_numpy(arr).float().div(255).unsqueeze(0).movedim(-1, 1)
    ref = comfy.utils.common_upscale(img, w, h, 'area', 'disabled')[0] * 2 - 1
    assert torch.allclose(got[:, 0], ref, atol=1e-6)
    vl = kn.native_vl_image(str(path))
    assert vl.shape[1:3] == tuple(reversed(kn.native_vl_size(520, 300)))


def test_config_guards_reject_the_broken_setups():
    base = {'model': {'type': 'krea2_native', 'diffusion_model_dtype': 'float8'}}
    with pytest.raises(ValueError, match='re-quantizes'):
        kn.Krea2NativePipeline(dict(base))
    geo = {'model': {'type': 'krea2_native'}, 'krea2_native': {'position_mode': 'width_shift'}}
    with pytest.raises(ValueError, match='stock ComfyUI contract'):
        kn.Krea2NativePipeline(geo)


def test_reference_latents_may_have_their_own_grid():
    p = object.__new__(kn.Krea2NativePipeline)
    ref = torch.randn(1, 16, 1, 128, 96)
    tgt = torch.randn(1, 16, 1, 64, 64)
    assert kn.Krea2NativePipeline.prepare_reference_latents(p, ref, tgt) is ref


def test_multiref_fusion_rank_preserves_condition_routing():
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT
    from models.condition_lora import ConditionOnlyLoRARouter
    from models.krea2_multiref import Krea2MultiRefGroundedPipeline

    pipe = object.__new__(Krea2MultiRefGroundedPipeline)
    pipe.diffusion_model = SingleStreamDiT(features=128, tdim=32, txtdim=32,
        heads=4, kvheads=2, multiplier=2, layers=1, txtlayers=3,
        txtheads=2, txtkvheads=2, operations=comfy.ops.disable_weight_init)
    for parameter in pipe.diffusion_model.parameters():
        torch.nn.init.normal_(parameter, std=.02)
    pipe.txtfusion_rank = 8
    pipe.condition_only_lora = True
    pipe.condition_lora_router = ConditionOnlyLoRARouter(True)
    pipe.configure_adapter(dict(type='lora', rank=4, alpha=4, dropout=0., dtype=torch.float32))
    for name, module in pipe.diffusion_model.named_modules():
        if hasattr(module, 'lora_A'):
            assert module.lora_A['default'].weight.shape[0] == (8 if 'txtfusion' in name else 4)
    module = pipe.condition_lora_router._installed[0]
    pipe.condition_lora_router.set_reference_span(3, 5)
    module.lora_B['default'].weight.data.fill_(.1)
    value = torch.randn(1, 5, module.in_features)
    delta = module(value) - module.base_layer(value)
    assert torch.count_nonzero(delta[:, :3]) == 0
    assert torch.count_nonzero(delta[:, 3:]) > 0


def test_microbatch_preparation_keeps_reference_grids_and_text_lengths():
    from utils.dataset import PipelineDataLoader

    pipe = object.__new__(kn.Krea2NativePipeline)
    pipe.model_config = {'timestep_sample_method': 'uniform'}
    pipe.caption_dropout = 0.
    references = [torch.randn(16, 1, height, 8) for height in (6, 8, 10, 12)]
    batch = dict(latents=torch.randn(4, 16, 1, 8, 8), mask=None,
        control_latents=references,
        text_embeds_0=[torch.randn(length, 96) for length in (3, 5, 7, 9)],
        attention_mask_0=[torch.ones(length) for length in (3, 5, 7, 9)])
    loader = object.__new__(PipelineDataLoader)
    loader.model = pipe
    loader.dataloader = [batch]
    loader.gradient_accumulation_steps = 4
    loader.eval_quantile = .5
    loader.num_batches_pulled = 0
    loader._broadcast_target = lambda value: value
    items = list(loader._pull_batches_from_dataloader())
    assert len(items) == 4
    for index, (features, labels) in enumerate(items):
        assert features[2].shape[1] == (3, 5, 7, 9)[index]
        assert torch.equal(features[-1][0], references[index])
        assert labels[0].shape[0] == 1


@pytest.mark.skipif(not os.environ.get('KREA2_STOCK_COMFY'), reason='set KREA2_STOCK_COMFY=/path/to/ComfyUI')
@pytest.mark.parametrize('method', ['index_timestep_zero', 'index'])
def test_forward_parity_against_stock_comfy(tmp_path, method):
    tool = os.path.join(ROOT, 'tools', 'krea2_native_parity.py')
    dump = str(tmp_path / f'{method}.pt')
    subprocess.run([sys.executable, tool, 'stock', '--comfy', os.environ['KREA2_STOCK_COMFY'],
                    '--out', dump, '--method', method], check=True)
    out = subprocess.run([sys.executable, tool, 'fork', '--dump', dump], capture_output=True, text=True)
    assert out.returncode == 0 and 'PARITY OK' in out.stdout, out.stdout + out.stderr
