"""BETA1-R4: máscara espacial de loss nas posições do alvo que são cópia alinhada da referência."""
import unittest
from unittest import mock

import torch

from test.test_beta1r2_anticopy import _batch, _pipe
from models.krea2_multiref import (Krea2MultiRefGroundedPipeline, Krea2OminiGroundedPipeline,
                                   copy_region_mask)


def _parent(seen):
    def prepare_inputs(self, inputs, timestep_quantile=None):
        ref = inputs['control_latents']
        return (inputs['latents'], torch.full((ref.shape[0],), 0.5), ref), (torch.ones_like(inputs['latents']), None)
    return prepare_inputs


class CopyMask(unittest.TestCase):
    def test_mask_marks_only_aligned_identical_positions(self):
        lat = torch.randn(2, 16, 1, 8, 12)
        ref = torch.randn(2, 16, 1, 8, 12) * 3
        ref[0, :, 0, :4] = lat[0, :, 0, :4]            # metade de cima da amostra 0 é cópia exata
        m = copy_region_mask(lat, ref, 0.234, 0.15)
        self.assertEqual(tuple(m.shape), (2, 1, 1, 8, 12))
        self.assertTrue(bool((m[0, 0, 0, :4] == 0.15).all()))
        self.assertTrue(bool((m[0, 0, 0, 4:] == 1).all()))
        self.assertTrue(bool((m[1] == 1).all()))

    def _run(self, keep=None, quantile=None, thr=0.234, w=0.15):
        pipe = _pipe()
        pipe.copy_mask_threshold, pipe.copy_mask_weight = thr, w
        batch = _batch([f'/r/{i}_1.jpg' for i in range(4)])
        batch['control_latents'][0] = batch['latents'][0:1].clone()      # amostra 0: ref == alvo
        if keep is not None:
            batch['reference_keep'] = keep
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _parent(None), create=True):
            _, (target, mask) = Krea2MultiRefGroundedPipeline.prepare_inputs(pipe, batch, timestep_quantile=quantile)
        return mask

    def test_weight_reaches_labels(self):
        mask = self._run()
        self.assertTrue(bool((mask[0] == 0.15).all()))
        self.assertTrue(bool((mask[1:] == 1).all()))

    def test_dropped_reference_not_masked_and_eval_untouched(self):
        mask = self._run(keep=torch.tensor([0.0, 1.0, 1.0, 1.0]))
        self.assertTrue(bool((mask[0] == 1).all()))
        self.assertIsNone(self._run(quantile=0.5))
        self.assertIsNone(self._run(thr=0.0))


if __name__ == '__main__':
    unittest.main()
