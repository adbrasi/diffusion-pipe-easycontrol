"""Testes NUMÉRICOS de contrato treino↔inferência para o caminho Anima.

Motivação (audit 2026-07-18): os testes de string não pegaram o runner
esticando a referência e alimentando o VAE em [0,1] em vez de [-1,1].
Estes testes comparam tensores de verdade.
"""

import numpy as np
import pytest
import torch
from PIL import Image


def _gradient_image(w=97, h=53):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, 'RGB')


def test_runner_preprocessing_matches_training_exactly():
    """preprocess_control_image DEVE reproduzir o pipeline do treino:
    ImageOps.fit (crop central, AR preservado) + ToTensor + Normalize([-1,1])."""
    from infer_easycontrol import preprocess_control_image
    from models.base import convert_crop_and_resize
    from torchvision import transforms

    img = _gradient_image(311, 173)
    width, height = 144, 80

    got = preprocess_control_image(img, width, height)  # (1, 3, 1, H, W)
    assert got.shape == (1, 3, 1, height, width)

    train_img = convert_crop_and_resize(img, (width, height))
    train_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )(train_img)

    assert torch.allclose(got[0, :, 0], train_tensor, atol=1e-6)
    # range de treino: [-1, 1]
    assert got.min() >= -1.0 - 1e-6 and got.max() <= 1.0 + 1e-6
    assert got.min() < -0.05, 'sem valores negativos => provavelmente [0,1] de novo'


def test_three_branch_guidance_math():
    """pred = u + text_cfg*(t-u) + ref_cfg*(c-t); escalas independentes."""
    from infer_easycontrol import combine_three_branch

    g = torch.Generator().manual_seed(3)
    u = torch.randn(2, 4, generator=g)
    t = torch.randn(2, 4, generator=g)
    c = torch.randn(2, 4, generator=g)

    # ref_cfg=0 => referência não guia nada (CFG de texto puro)
    assert torch.allclose(combine_three_branch(u, t, c, 4.0, 0.0), u + 4.0 * (t - u))
    # text_cfg=1, ref_cfg=1 => c puro
    assert torch.allclose(combine_three_branch(u, t, c, 1.0, 1.0), c)
    # forma geral
    expected = u + 4.0 * (t - u) + 1.5 * (c - t)
    assert torch.allclose(combine_three_branch(u, t, c, 4.0, 1.5), expected)
    # o bug antigo: 2 branches com ref só no positivo == ref_cfg acoplado ao texto
    two_branch = u + 4.0 * (c - u)
    assert torch.allclose(combine_three_branch(u, t, c, 4.0, 4.0), two_branch)


def test_node_ref_ratio_composition_recovers_three_branch():
    """A composição do node (chunk cond = t + ratio*(c-t), KSampler faz
    u + cfg*(cond-u)) tem que reconstruir exatamente as 3 branches."""
    from infer_easycontrol import combine_three_branch

    g = torch.Generator().manual_seed(11)
    u = torch.randn(3, 5, generator=g)
    t = torch.randn(3, 5, generator=g)
    c = torch.randn(3, 5, generator=g)
    text_cfg, ref_cfg = 4.0, 0.5

    ratio = ref_cfg / text_cfg
    cond_out = t + ratio * (c - t)
    ksampler = u + text_cfg * (cond_out - u)
    assert torch.allclose(ksampler, combine_three_branch(u, t, c, text_cfg, ref_cfg), atol=1e-6)


def test_router_masks_reference_rows_only():
    """AnimaConditionRouter: delta só nas rows da referência, nos dois layouts
    (3D self_attn flatten 't h w' e 5D mlp), para ref_first e target-first."""
    from models.ic_lora_routed import AnimaConditionRouter

    t_frames, h, w = 2, 3, 4
    hw = h * w
    for ref_first in (True, False):
        router = AnimaConditionRouter(enabled=True, ref_first=ref_first)
        router.set_frame_geometry(t_frames, h, w)

        m3 = router.mask_for(torch.zeros(2, t_frames * hw, 8))
        assert m3.shape == (1, t_frames * hw, 1)
        ref_rows = m3[0, :, 0].bool()
        expected = torch.zeros(t_frames * hw, dtype=torch.bool)
        if ref_first:
            expected[:hw] = True
        else:
            expected[-hw:] = True
        assert torch.equal(ref_rows, expected)

        m5 = router.mask_for(torch.zeros(2, t_frames, h, w, 8))
        assert m5.shape == (1, t_frames, 1, 1, 1)
        assert m5[0, 0 if ref_first else -1].item() == 1
        assert m5[0, -1 if ref_first else 0].item() == 0

    # sem frame de referência (T=1): delta zero em tudo
    router = AnimaConditionRouter(enabled=True, ref_first=True)
    router.set_frame_geometry(1, h, w)
    assert router.mask_for(torch.zeros(2, hw, 8)).sum() == 0
