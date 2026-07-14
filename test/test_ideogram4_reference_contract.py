import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn

from models.ideogram4_reference_contract import (
    apply_reference_dropout,
    build_model_timesteps,
    offset_reference_positions,
)


ROOT = Path(__file__).resolve().parents[1]


class ReferenceContractHelpersTest(unittest.TestCase):
    def test_position_offset_changes_only_temporal_axis(self):
        positions = torch.tensor([[65536, 65536, 65536], [65536, 65537, 65538]])
        reference = offset_reference_positions(positions, 1)

        torch.testing.assert_close(reference[:, 0], positions[:, 0] + 1)
        torch.testing.assert_close(reference[:, 1:], positions[:, 1:])
        torch.testing.assert_close(positions[:, 0], torch.tensor([65536, 65536]))

    def test_reference_timestep_is_clean_and_target_is_inverted(self):
        timesteps = torch.tensor([0.25, 0.75])
        model_timesteps = build_model_timesteps(timesteps, 8, 5, 1.0)

        torch.testing.assert_close(model_timesteps[0, :5], torch.full((5,), 0.75))
        torch.testing.assert_close(model_timesteps[1, :5], torch.full((5,), 0.25))
        torch.testing.assert_close(model_timesteps[:, 5:], torch.ones(2, 3))

    def test_reference_dropout_drops_whole_samples(self):
        reference = torch.ones(4, 2, 3, 3)
        dropped = apply_reference_dropout(reference, 1.0)
        preserved = apply_reference_dropout(reference, 1.0, enabled=False)

        self.assertEqual(torch.count_nonzero(dropped).item(), 0)
        torch.testing.assert_close(preserved, reference)


class _FakeTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.last_input = None

    def forward(self, value, dtype):
        self.last_input = value.detach().clone()
        return value.to(dtype).unsqueeze(-1).expand(*value.shape, self.dim)


class _RecordingEmbedding(nn.Embedding):
    def __init__(self, count, dim):
        super().__init__(count, dim)
        self.last_indices = None

    def forward(self, indices):
        self.last_indices = indices.detach().clone()
        return super().forward(indices)


class _FakeIdeogramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4, 8)
        self.t_embedding = _FakeTimestepEmbedding(8)
        self.adaln_proj = nn.Linear(8, 6)
        self.llm_cond_norm = nn.Identity()
        self.llm_cond_proj = nn.Linear(3, 8)
        self.embed_image_indicator = _RecordingEmbedding(2, 8)
        self.final_layer = _FakeFinalLayer()
        self.head_dim = 2
        self.rope_theta = 10000
        self.mrope_section = (1, 1, 0)

    @staticmethod
    def _img_to_tokens(image):
        return image.permute(0, 2, 3, 1).reshape(image.shape[0], -1, image.shape[1])

    @staticmethod
    def _tokens_to_img(tokens, grid_h, grid_w):
        return tokens.reshape(tokens.shape[0], grid_h, grid_w, tokens.shape[-1]).permute(0, 3, 1, 2)

    @staticmethod
    def _image_position_ids(grid_h, grid_w, device):
        h = torch.arange(grid_h, device=device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w = torch.arange(grid_w, device=device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        return torch.stack([torch.zeros_like(h), h, w], dim=1) + 65536


class _FakeFinalLayer(nn.Module):
    def forward(self, hidden_states, adaln_input):
        return hidden_states[..., :4]


def _load_pipeline_module(position_capture):
    safetensors_package = types.ModuleType('safetensors')
    safetensors_torch = types.ModuleType('safetensors.torch')
    safetensors_torch.save_file = mock.Mock()
    safetensors_package.torch = safetensors_torch

    ideogram_module = types.ModuleType('models.ideogram4')
    ideogram_module.Ideogram4Pipeline = object
    ideogram_module.LLM_TOKEN_INDICATOR = 3
    ideogram_module.OUTPUT_IMAGE_INDICATOR = 2
    ideogram_module.SEQUENCE_PADDING_INDICATOR = -1
    ideogram_module.TransformerLayer = object

    base_module = types.ModuleType('models.base')
    base_module.make_contiguous = lambda *values: tuple(value.contiguous() for value in values)

    common_module = types.ModuleType('utils.common')
    common_module.get_git_commit = lambda: 'test-commit'

    comfy_package = types.ModuleType('comfy')
    comfy_text_encoders = types.ModuleType('comfy.text_encoders')
    comfy_llama = types.ModuleType('comfy.text_encoders.llama')

    def precompute_freqs_cis(head_dim, position_ids, *args, **kwargs):
        position_capture.append(position_ids.detach().clone())
        length = position_ids.shape[1]
        return (torch.zeros(length, head_dim), torch.zeros(length, head_dim))

    comfy_llama.precompute_freqs_cis = precompute_freqs_cis

    replacements = {
        'safetensors': safetensors_package,
        'safetensors.torch': safetensors_torch,
        'models.ideogram4': ideogram_module,
        'models.base': base_module,
        'utils.common': common_module,
        'comfy': comfy_package,
        'comfy.text_encoders': comfy_text_encoders,
        'comfy.text_encoders.llama': comfy_llama,
    }
    module_path = ROOT / 'models' / 'ideogram4_ic_lora.py'
    spec = importlib.util.spec_from_file_location('_ideogram4_ic_lora_test_module', module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class ReferencePackingTest(unittest.TestCase):
    def test_initial_and_final_layers_follow_reference_contract(self):
        captured_positions = []
        module = _load_pipeline_module(captured_positions)
        model = _FakeIdeogramModel()
        initial = module.Ideogram4ReferenceInitialLayer(model)

        target = torch.randn(1, 4, 2, 2)
        reference = torch.randn_like(target)
        timestep = torch.tensor([0.25])
        context = torch.randn(1, 3, 3)
        text_mask = torch.tensor([[1, 1, 0]])
        packed = initial((target, timestep, context, text_mask, reference))

        hidden_states, _, adaln_input, sizes, *_ = packed
        self.assertEqual(tuple(hidden_states.shape), (1, 11, 8))
        self.assertEqual(tuple(adaln_input.shape), (1, 11, 6))
        torch.testing.assert_close(sizes, torch.tensor([3, 2, 2]))

        model_t = model.t_embedding.last_input
        torch.testing.assert_close(model_t[:, :7], torch.full((1, 7), 0.75))
        torch.testing.assert_close(model_t[:, 7:], torch.ones(1, 4))

        image_roles = model.embed_image_indicator.last_indices
        torch.testing.assert_close(image_roles[:, :3], torch.zeros(1, 3, dtype=torch.long))
        torch.testing.assert_close(image_roles[:, 3:], torch.ones(1, 8, dtype=torch.long))

        positions = captured_positions[0].transpose(0, 1)
        torch.testing.assert_close(positions[3:7, 0], torch.full((4,), 65536))
        torch.testing.assert_close(positions[7:, 0], torch.full((4,), 65537))

        final = module.Ideogram4ReferenceFinalLayer(model)
        output = final(packed)
        self.assertEqual(tuple(output.shape), (1, 4, 2, 2))
        expected = -model._tokens_to_img(hidden_states[:, 3:7, :4], 2, 2)
        torch.testing.assert_close(output, expected)


if __name__ == '__main__':
    unittest.main()
