import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class _RecordingTMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().clone())
        return self.projection(value)


class _FakeTextFusion(nn.Module):
    def forward(self, value, mask=None):
        return value.squeeze(2)


class _RecordingPositions(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_positions = None

    def forward(self, positions):
        self.last_positions = positions.detach().clone()
        return positions


class _FakeLast(nn.Module):
    def forward(self, hidden_states, timestep_features):
        return hidden_states[..., :4]


class _FakeKreaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = 2
        self.channels = 1
        self.tdim = 4
        self.first = nn.Linear(4, 8)
        self.tmlp = _RecordingTMLP(4, 8)
        self.tproj = nn.Linear(8, 48)
        self.txtfusion = _FakeTextFusion()
        self.txtmlp = nn.Linear(3, 8)
        self.pe_embedder = _RecordingPositions()
        self.last = _FakeLast()

    @staticmethod
    def _unpack_context(context):
        return context.unsqueeze(2)


def _load_pipeline_module():
    safetensors_package = types.ModuleType('safetensors')
    safetensors_torch = types.ModuleType('safetensors.torch')
    safetensors_torch.save_file = mock.Mock()
    safetensors_package.torch = safetensors_torch

    peft_module = types.ModuleType('peft')
    krea_module = types.ModuleType('models.krea2')
    krea_module.Krea2Pipeline = object
    krea_module.TransformerLayer = object

    base_module = types.ModuleType('models.base')
    base_module.make_contiguous = lambda *values: tuple(
        value.contiguous() for value in values
    )

    common_module = types.ModuleType('utils.common')
    common_module.AUTOCAST_DTYPE = torch.bfloat16
    common_module.get_git_commit = lambda: 'test-commit'
    common_module.is_main_process = lambda: False

    comfy_package = types.ModuleType('comfy')
    comfy_ldm = types.ModuleType('comfy.ldm')
    comfy_common_dit = types.ModuleType('comfy.ldm.common_dit')
    comfy_common_dit.pad_to_patch_size = lambda value, patch: value
    comfy_flux = types.ModuleType('comfy.ldm.flux')
    comfy_flux_layers = types.ModuleType('comfy.ldm.flux.layers')

    def timestep_embedding(value, dim):
        return value.unsqueeze(-1).expand(*value.shape, dim)

    comfy_flux_layers.timestep_embedding = timestep_embedding
    comfy_ldm.common_dit = comfy_common_dit
    comfy_ldm.flux = comfy_flux
    comfy_flux.layers = comfy_flux_layers
    comfy_package.ldm = comfy_ldm

    replacements = {
        'safetensors': safetensors_package,
        'safetensors.torch': safetensors_torch,
        'peft': peft_module,
        'models.krea2': krea_module,
        'models.base': base_module,
        'utils.common': common_module,
        'comfy': comfy_package,
        'comfy.ldm': comfy_ldm,
        'comfy.ldm.common_dit': comfy_common_dit,
        'comfy.ldm.flux': comfy_flux,
        'comfy.ldm.flux.layers': comfy_flux_layers,
    }
    module_path = ROOT / 'models' / 'krea2_reference.py'
    spec = importlib.util.spec_from_file_location('_krea2_reference_test_module', module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class Krea2ReferencePackingTest(unittest.TestCase):
    def test_target_reference_packing_timestep_position_and_output_slice(self):
        module = _load_pipeline_module()
        model = _FakeKreaModel()
        initial = module.Krea2ReferenceInitialLayer(
            model,
            position_mode='subject',
            reference_position_offset=1.0,
        )

        target = torch.randn(1, 1, 1, 4, 4)
        reference = torch.randn_like(target)
        timestep = torch.tensor([0.25])
        context = torch.randn(1, 2, 3)
        text_mask = torch.tensor([[True, False]])
        packed = initial((target, timestep, context, text_mask, reference))

        combined, target_timestep, tvec, _, attention_mask, sizes = packed
        self.assertEqual(tuple(combined.shape), (1, 10, 8))
        self.assertEqual(tuple(target_timestep.shape), (1, 1, 8))
        self.assertEqual(tuple(tvec.shape), (1, 10, 48))
        torch.testing.assert_close(sizes, torch.tensor([2, 4, 2, 2, 4, 4]))

        per_token_embedding = model.tmlp.inputs[1]
        torch.testing.assert_close(per_token_embedding[0, :6, 0], torch.full((6,), 0.25))
        torch.testing.assert_close(per_token_embedding[0, 6:, 0], torch.zeros(4))

        positions = model.pe_embedder.last_positions
        torch.testing.assert_close(positions[0, 2:6, 0], torch.zeros(4))
        torch.testing.assert_close(positions[0, 6:, 0], torch.ones(4))
        self.assertEqual(tuple(attention_mask.shape), (1, 1, 1, 10))
        self.assertFalse(attention_mask[0, 0, 0, 1])
        self.assertTrue(torch.all(attention_mask[0, 0, 0, 2:]))

        final = module.Krea2ReferenceFinalLayer(model)
        output = final(packed)
        self.assertEqual(tuple(output.shape), (1, 1, 1, 4, 4))

    def test_dropout_is_training_only_and_drops_whole_reference(self):
        module = _load_pipeline_module()
        contract = types.SimpleNamespace(condition_token_stride=1, condition_dropout=1.0)
        reference = torch.ones(2, 1, 1, 4, 4)
        target = torch.zeros_like(reference)

        dropped = module.Krea2ReferencePipeline.prepare_reference_latents(
            contract, reference, target, timestep_quantile=None
        )
        preserved = module.Krea2ReferencePipeline.prepare_reference_latents(
            contract, reference, target, timestep_quantile=0.5
        )

        self.assertEqual(torch.count_nonzero(dropped).item(), 0)
        torch.testing.assert_close(preserved, reference)


if __name__ == '__main__':
    unittest.main()
