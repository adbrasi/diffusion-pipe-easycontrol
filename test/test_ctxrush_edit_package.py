import ast
from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "comfyui_nodes" / "ctxrush_edit"
NODES = PACKAGE / "nodes.py"


def _source():
    return NODES.read_text(encoding="utf-8")


def test_custom_node_sources_compile(tmp_path):
    py_compile.compile(
        str(PACKAGE / "__init__.py"),
        cfile=str(tmp_path / "init.pyc"),
        doraise=True,
    )
    py_compile.compile(
        str(NODES),
        cfile=str(tmp_path / "nodes.pyc"),
        doraise=True,
    )


def test_custom_node_exports_recommended_and_modular_nodes():
    source = _source()
    for node_id in (
        "CtxRushKrea2EditSetup",
        "CtxRushKrea2ReferenceEncode",
        "CtxRushKrea2EditCFGEncode",
        "CtxRushKrea2EditModelPatch",
    ):
        assert f'"{node_id}"' in source


def test_reference_travels_with_conditioning_and_object_patch():
    source = _source()
    assert '{"reference_latents": [reference.latent]}' in source
    assert 'patched.add_object_patch("extra_conds", extra_conds)' in source
    assert 'patched.add_object_patch("diffusion_model.forward", forward)' in source
    assert "add_wrapper_with_key" not in source


def test_contract_has_fixed_clean_timestep_and_reference_frame():
    source = _source()
    assert "timestep_embedding(torch.zeros_like(timesteps)" in source
    assert "positions[..., 0] = 1.0" in source
    assert "position_offset" not in source


def test_no_mutable_default_arguments():
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for default in (*node.args.defaults, *node.args.kw_defaults):
            assert not isinstance(default, (ast.Dict, ast.List, ast.Set)), node.name


def test_readme_documents_daily_workflow_and_raw_profile():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "Krea 2 Edit Setup" in readme
    assert "Krea 2 Raw | 28 | 5.5" in readme
    assert "training_crop" in readme
    assert "LoRA strength to zero is not a vanilla" in readme
