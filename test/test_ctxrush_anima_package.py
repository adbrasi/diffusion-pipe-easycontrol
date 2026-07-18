import ast
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "comfyui_nodes" / "ctxrush_anima"
NODES = PACKAGE / "nodes.py"


def _source():
    return NODES.read_text(encoding="utf-8")


def test_sources_compile(tmp_path):
    py_compile.compile(str(PACKAGE / "__init__.py"), cfile=str(tmp_path / "i.pyc"), doraise=True)
    py_compile.compile(str(NODES), cfile=str(tmp_path / "n.pyc"), doraise=True)


def test_exports_next_scene_node():
    source = _source()
    assert "'CtxRushAnimaNextScene'" in source
    assert "add_wrapper_with_key" in source


def test_modes_match_training_contracts():
    source = _source()
    # ref_first para os ic_lora, target-first para o omini; routed mascarado.
    assert "'ic_lora_v2': (True, None)" in source
    assert "'ic_lora_routed': (True, 'first')" in source
    assert "'omini_subject': (False, None)" in source


def test_cfg_uncond_zeroes_reference():
    source = _source()
    assert "cond_or_uncond" in source
    assert "ref[i * chunk:(i + 1) * chunk] = 0" in source


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
    assert "zero_ref_in_uncond" in readme
