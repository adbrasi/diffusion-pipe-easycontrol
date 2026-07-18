import ast
from pathlib import Path
import py_compile

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "comfyui_nodes" / "ctxrush_anima"
NODES = PACKAGE / "nodes.py"


def _source():
    return NODES.read_text(encoding="utf-8")


def _load_function(name):
    tree = ast.parse(_source())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NODES), 'exec'), namespace)
    return namespace[name]


def test_sources_compile(tmp_path):
    py_compile.compile(str(PACKAGE / "__init__.py"), cfile=str(tmp_path / "i.pyc"), doraise=True)
    py_compile.compile(str(NODES), cfile=str(tmp_path / "n.pyc"), doraise=True)


def test_exports_next_scene_node():
    source = _source()
    assert "'CtxRushAnimaNextScene'" in source
    assert "'CtxRushAnimaDualGuider'" in source
    assert "RETURN_TYPES = ('GUIDER',)" in source
    assert "add_wrapper_with_key" in source


def test_modes_match_training_contracts():
    source = _source()
    # ref_first para os ic_lora, target-first para o omini; routed mascarado.
    assert "'ic_lora_v2': (True, None)" in source
    assert "'ic_lora_routed': (True, 'first')" in source
    assert "'omini_subject': (False, None)" in source
    assert "'routed_targetfirst': (False, 'last')" in source


def test_three_branch_guidance_present():
    source = _source()
    assert "ref_ratio = float(ref_cfg) / float(expected_cfg)" in source
    assert "_mix_reference_guidance" in source


def test_node_mix_recovers_independent_reference_guidance():
    mix = _load_function('_mix_reference_guidance')
    # Branches [positive, negative]: t=10, u=1, c=12. Antes do KSampler o
    # positivo vira 10.5; depois: 1 + 4*(10.5-1) = 39 = u+4*(t-u)+(c-t).
    out_no_ref = torch.tensor([10.0, 1.0])
    out_with_ref = torch.tensor([12.0, 99.0])
    cond_mask = torch.tensor([True, False])
    prepared = mix(out_no_ref, out_with_ref, cond_mask, 1.0 / 4.0)
    final = prepared[1] + 4.0 * (prepared[0] - prepared[1])
    assert torch.allclose(prepared, torch.tensor([10.5, 1.0]))
    assert torch.allclose(final, torch.tensor(39.0))


def test_dual_guider_recovers_independent_reference_guidance():
    add_reference = _load_function('_add_reference_guidance')
    # u=1, t=10, c=12, text_cfg=4, ref_cfg=1:
    # u + 4*(t-u) + 1*(c-t) = 39.
    uncond = torch.tensor(1.0)
    text_no_ref = torch.tensor(10.0)
    text_with_ref = torch.tensor(12.0)
    text_prediction = uncond + 4.0 * (text_no_ref - uncond)
    final = add_reference(text_prediction, text_no_ref, text_with_ref, 1.0)
    assert torch.allclose(final, torch.tensor(39.0))


def test_dual_guider_uses_three_explicit_branches():
    source = _source()
    assert "_GUIDER_REF_BRANCHES = (False, False, True)" in source
    assert "[negative, positive, positive]" in source
    assert "uncond, text_no_ref, text_with_ref" in source


def test_cfg_uncond_zeroes_reference():
    source = _source()
    assert "cond_or_uncond" in source
    assert "ref_with[i * chunk:(i + 1) * chunk] = 0" in source


def test_reference_is_pixel_cropfit_never_latent_resized():
    source = _source()
    assert "_crop_fit" in source
    assert "interpolate" not in source  # nunca resize de latente
    assert "ref_weight" not in source   # nunca escalar a ref


def test_no_mutable_default_arguments():
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for default in (*node.args.defaults, *node.args.kw_defaults):
            assert not isinstance(default, (ast.Dict, ast.List, ast.Set)), node.name


def test_readme_documents_wiring_and_settings():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "anima-base-v1.0" in readme
    assert "er_sde" in readme
    assert "ref_cfg" in readme
    assert "expected_cfg" in readme
    assert "SamplerCustomAdvanced" in readme
    assert "u + text_cfg * (t - u) + ref_cfg * (c - t)" in readme


def test_block_range_parser_and_channel_dials():
    parse = _load_function('_parse_block_range')
    bidx = _load_function('_block_index')
    assert parse('all') is None and parse('') is None
    assert parse('4-6') == {4, 5, 6}
    assert parse('0-1,27') == {0, 1, 27}
    try:
        parse('28')
        assert False, 'devia rejeitar block 28'
    except ValueError:
        pass
    assert bidx('blocks.13.self_attn.q_proj') == 13
    assert bidx('llm_adapter.blocks.0.mlp') is None
    source = _source()
    for dial in ('appearance_strength', 'cross_attn_strength', 'llm_adapter_strength', 'block_range'):
        assert dial in source
