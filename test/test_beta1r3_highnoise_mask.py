"""BETA1-R3: tokens VAE da referência mascarados quando t > ref_vae_mask_above_t (só em treino)."""
import unittest
from unittest import mock

import torch

from test.test_beta1r2_anticopy import _batch, _pipe
from models.krea2_multiref import Krea2MultiRefGroundedPipeline, Krea2OminiGroundedPipeline


def _parent(ts):
    def prepare_inputs(self, inputs, timestep_quantile=None):
        ref = inputs['control_latents']
        return (inputs['latents'], ts.clone(), ref), (torch.ones_like(inputs['latents']), None)
    return prepare_inputs


class HighNoiseMask(unittest.TestCase):
    FILES = [f'/r/{i}_1.jpg' for i in range(4)]
    TS = torch.tensor([0.99, 0.93, 0.92, 0.3])

    def _run(self, thr, keep=None, quantile=None):
        pipe = _pipe()
        pipe.ref_vae_mask_above_t = thr
        batch = _batch(self.FILES)
        if keep is not None:
            batch['reference_keep'] = keep
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _parent(self.TS), create=True):
            feats, _ = Krea2MultiRefGroundedPipeline.prepare_inputs(pipe, batch, timestep_quantile=quantile)
        return feats

    def test_masks_only_above_threshold(self):
        feats = self._run(0.92)
        self.assertEqual(len(feats), 4)                        # 6º input = keep
        self.assertEqual(feats[-1].tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertTrue(bool((feats[-2][:2] == 0).all()))     # latente da ref zerado (cosmético)
        self.assertTrue(bool((feats[-2][2:] != 0).any()))

    def test_combines_with_reference_dropout(self):
        feats = self._run(0.92, keep=torch.tensor([1.0, 1.0, 1.0, 0.0]))
        self.assertEqual(feats[-1].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_off_and_eval_untouched(self):
        self.assertEqual(len(self._run(1.0)), 3)               # desligado: caminho do BETA1-R
        self.assertEqual(len(self._run(0.92, quantile=0.5)), 3)  # eval nunca mascara


if __name__ == '__main__':
    unittest.main()
