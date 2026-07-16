import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'preflight_krea2_edit',
    ROOT / 'tools' / 'preflight_krea2_edit.py',
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def write_safetensors(path, tensor_names):
    """Write a minimal safetensors file with the given tensor names."""
    offset = 0
    header = {}
    for name in tensor_names:
        header[name] = {'dtype': 'F32', 'shape': [1], 'data_offsets': [offset, offset + 4]}
        offset += 4
    payload = json.dumps(header).encode('utf-8')
    with open(path, 'wb') as handle:
        handle.write(struct.pack('<Q', len(payload)))
        handle.write(payload)
        handle.write(b'\x00' * offset)


class Krea2EditPreflightTest(unittest.TestCase):
    def _make_project(self, root, te_tensor_names=None):
        models = root / 'models'
        targets = root / 'targets'
        controls = root / 'controls'
        for directory in (models, targets, controls):
            directory.mkdir()

        diffusion = models / 'krea2_raw.safetensors'
        vae = models / 'qwen_image_vae.safetensors'
        text_encoder = models / 'qwen3vl_4b.safetensors'
        diffusion.touch()
        vae.touch()
        if te_tensor_names is None:
            te_tensor_names = [
                'model.layers.0.self_attn.q_proj.weight',
                'visual.patch_embed.proj.weight',
                'visual.blocks.0.attn.qkv.weight',
            ]
        write_safetensors(text_encoder, te_tensor_names)

        (targets / 'shot_01.png').touch()
        (targets / 'shot_01.txt').write_text('same character entering the next room')
        (controls / 'shot_01.jpg').touch()

        dataset = root / 'dataset.toml'
        dataset.write_text(
            f"""resolutions = [512]
frame_buckets = [1]
[[directory]]
path = '{targets}'
control_path = '{controls}'
"""
        )

        config = root / 'config.toml'
        config.write_text(
            f"""dataset = '{dataset}'
[model]
type = 'krea2_edit'
diffusion_model = '{diffusion}'
vae = '{vae}'
text_encoders = [{{path = '{text_encoder}', type = 'krea2'}}]
[krea2_edit]
condition_dropout = 0.0
position_mode = 'subject'
reference_position_offset = 1.0
vl_image_max_pixels = 147456
"""
        )
        return config, controls, text_encoder

    def test_valid_dual_conditioning_project_passes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _, _ = self._make_project(Path(temporary_directory))
            errors, warnings, summaries = PREFLIGHT.validate(config)

            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])
            self.assertTrue(any('total pairs: 1' in item for item in summaries))
            self.assertTrue(any('vision-tower tensors found' in item for item in summaries))
            self.assertTrue(any('VL grounding' in item for item in summaries))

    def test_text_only_te_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _, _ = self._make_project(
                Path(temporary_directory),
                te_tensor_names=['model.layers.0.self_attn.q_proj.weight'],
            )
            errors, _, _ = PREFLIGHT.validate(config)

            self.assertTrue(any('vision tower' in error for error in errors))

    def test_missing_control_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, controls, _ = self._make_project(Path(temporary_directory))
            (controls / 'shot_01.jpg').unlink()
            errors, _, _ = PREFLIGHT.validate(config)

            self.assertTrue(any('no reference' in error for error in errors))
            self.assertTrue(any('No valid control/target pairs' in error for error in errors))

    def test_nonzero_dropout_warns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config, _, _ = self._make_project(root)
            text = config.read_text().replace('condition_dropout = 0.0', 'condition_dropout = 0.1')
            config.write_text(text)
            errors, warnings, _ = PREFLIGHT.validate(config)

            self.assertEqual(errors, [])
            self.assertTrue(any('local extension' in warning for warning in warnings))


if __name__ == '__main__':
    unittest.main()
