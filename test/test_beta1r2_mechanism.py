"""BETA1-R2: referência com ruído (Captain Cinema/LCT) e descarte de tokens (DreamO).

Contrato: com t_ref=0 e todos os tokens mantidos, o caminho de 7 inputs é idêntico ao
de inferência (5 inputs); com todos os tokens descartados, idêntico ao reference_dropout.
"""
import random
import unittest

import torch

from test.test_krea2_reference_dropout import _layers, _pad, _run, _tiny_model, TXTLAYERS, TXTDIM
from models.krea2_multiref import sample_reference_corruption


class MechanismContract(unittest.TestCase):
    def setUp(self):
        self.model, self.lora, self.router = _tiny_model()
        self.layers = _layers(self.model, self.router)
        g = torch.Generator().manual_seed(1)
        self.target = torch.randn(2, 4, 1, 8, 8, generator=g)
        self.ref = torch.randn(2, 4, 1, 8, 8, generator=g)
        self.t = torch.tensor([0.7, 0.3])
        self.emb, self.mask = _pad([torch.randn(5, TXTLAYERS * TXTDIM, generator=g),
                                    torch.randn(9, TXTLAYERS * TXTDIM, generator=g)])
        self.L = 16   # (8/2)*(8/2) tokens de referência

    @torch.no_grad()
    def test_clean_full_equals_inference_path(self):
        base = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref))
        out = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref,
                                 torch.ones(2, self.L), torch.zeros(2)))
        torch.testing.assert_close(out, base, atol=1e-6, rtol=1e-6)

    @torch.no_grad()
    def test_all_tokens_dropped_equals_reference_dropout(self):
        drop = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref,
                                  torch.tensor([0.0, 1.0])))
        tok = torch.ones(2, self.L)
        tok[0] = 0
        out = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref, tok, torch.zeros(2)))
        torch.testing.assert_close(out, drop, atol=1e-6, rtol=1e-6)

    @torch.no_grad()
    def test_ref_timestep_and_partial_tokens_are_used(self):
        base = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref))
        t_out = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref,
                                   torch.ones(2, self.L), torch.tensor([0.3, 0.0])))
        self.assertGreater((t_out[0] - base[0]).abs().max().item(), 1e-4)   # t_ref entra no embedding
        torch.testing.assert_close(t_out[1], base[1], atol=1e-6, rtol=1e-6)  # amostra com t=0 intacta
        tok = torch.ones(2, self.L)
        tok[1, ::2] = 0
        p_out = _run(self.layers, (self.target, self.t, self.emb, self.mask, self.ref, tok, torch.zeros(2)))
        self.assertGreater((p_out[1] - base[1]).abs().max().item(), 1e-4)
        torch.testing.assert_close(p_out[0], base[0], atol=1e-6, rtol=1e-6)

    def test_sampler_off_is_identity_and_on_respects_ranges(self):
        ref = torch.randn(64, 4, 1, 8, 8)
        r, t, k = sample_reference_corruption(ref, 2, 0.0, 0.05, 0.35, 0.0, 0.05, 0.5)
        self.assertTrue(torch.equal(r, ref))
        self.assertTrue(bool((t == 0).all()) and bool((k == 1).all()))
        random.seed(0)
        torch.manual_seed(0)
        r, t, k = sample_reference_corruption(ref, 2, 0.5, 0.05, 0.35, 0.3, 0.05, 0.5)
        noised = t > 0
        self.assertTrue(10 < int(noised.sum()) < 54)
        self.assertTrue(bool((t[noised] >= 0.05).all()) and bool((t[noised] <= 0.35).all()))
        self.assertTrue(torch.equal(r[~noised], ref[~noised]))       # amostras limpas intactas
        dropped = k.mean(1) < 1
        self.assertTrue(5 < int(dropped.sum()) < 35)
        self.assertTrue(bool((k[dropped].mean(1) <= 0.75).all()))
        self.assertTrue(bool((k.sum(1) >= 1).all()))


if __name__ == '__main__':
    unittest.main()
