import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'preflight_krea2_ic_lora',
    ROOT / 'tools' / 'preflight_krea2_ic_lora.py',
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class Krea2PreflightTest(unittest.TestCase):
    def _make_project(self, root):
        models = root / 'models'
        targets = root / 'targets'
        controls = root / 'controls'
        for directory in (models, targets, controls):
            directory.mkdir()

        model_files = [
            models / 'krea2_raw.safetensors',
            models / 'qwen_image_vae.safetensors',
            models / 'qwen3vl_4b.safetensors',
        ]
        for path in model_files:
            path.touch()

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
type = 'krea2_ic_lora'
diffusion_model = '{model_files[0]}'
vae = '{model_files[1]}'
text_encoders = [{{path = '{model_files[2]}', type = 'krea2'}}]
[krea2_ic_lora]
condition_dropout = 0.1
position_mode = 'subject'
reference_position_offset = 1.0
"""
        )
        return config, controls

    def test_valid_one_reference_project_passes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, _ = self._make_project(Path(temporary_directory))
            errors, warnings, summaries = PREFLIGHT.validate(config)

            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])
            self.assertTrue(any('total pairs: 1' in item for item in summaries))
            self.assertTrue(any('1024 target + 1024 reference' in item for item in summaries))

    def test_missing_control_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config, controls = self._make_project(Path(temporary_directory))
            (controls / 'shot_01.jpg').unlink()
            errors, _, _ = PREFLIGHT.validate(config)

            self.assertTrue(any('no reference' in error for error in errors))
            self.assertTrue(any('No valid control/target pairs' in error for error in errors))


if __name__ == '__main__':
    unittest.main()
