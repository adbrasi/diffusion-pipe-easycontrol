"""BETA1-R2 anti-cópia (docs/BETA1R2.md) — CPU.

1. Peso de loss por par: o peso chega ao loss_fn real (models.base) pela
   máscara de labels, sobrevive ao split_batch do dataloader e peso 0 dá
   gradiente exatamente zero; peso 1 = BETA1-R bit a bit.
2. Zoom-crop da referência: shape/dtype preservados, só a referência muda,
   p=0 é identidade, o conteúdo é de fato um recorte ampliado, eval nunca aumenta.
3. LR hold_linear: valores numéricos dentro do SequentialLR do train.py.
"""
import json
import random
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import utils.common  # noqa: F401  (antes do ComfyUI entrar no sys.path)
import models.base  # noqa: F401
from models.base import ComfyPipeline
from models.krea2_multiref import (Krea2MultiRefGroundedPipeline, Krea2OminiGroundedPipeline,
                                   apply_pair_weights, load_pair_weights, pair_weights_for,
                                   zoom_crop_reference)
from utils.dataset import split_batch
from utils.lr_schedule import hold_linear_lambda

ROOT = Path(__file__).resolve().parents[1]
B, C, H, W = 4, 16, 8, 12


def _pipe(weights=None, prob=0.0, zoom=1.0):
    pipe = object.__new__(Krea2MultiRefGroundedPipeline)   # sem __init__ (não carrega modelos)
    pipe.pair_weights, pipe.ref_aug_prob, pipe.ref_aug_max_zoom = weights, prob, zoom
    return pipe


def _batch(files):
    torch.manual_seed(0)
    return {
        'latents': torch.randn(len(files), C, 1, H, W),
        'control_latents': [torch.randn(1, C, 1, H, W) for _ in files],
        'control_file': files,
        'mask': None,
    }


def _fake_parent(seen):
    """Substitui o prepare_inputs do pai: devolve (inputs do modelo, (target, mask)) e
    guarda a referência que chegou."""
    def prepare_inputs(self, inputs, timestep_quantile=None):
        ref = inputs['control_latents']
        seen.append(ref)
        target = torch.ones_like(inputs['latents'])
        return (inputs['latents'], torch.full((ref.shape[0],), 0.5), ref), (target, None)
    return prepare_inputs


class PairWeights(unittest.TestCase):
    FILES = ['/workspace/data/beta1r/poxima_v2/refs/poxima_v2_ds2_005577_1.jpg',   # idêntico
             ['/dev/shm/beta1r_data/beta1r/parents/refs/parents_x_1.jpg'],          # lista, quase igual
             '/workspace/data/beta1r/recortados/refs/recortados_y_1.jpg',            # ausente -> 1
             None]
    W = {'poxima_v2_ds2_005577_1.jpg': 0.0, 'parents_x_1.jpg': 0.3}

    def test_lookup_by_basename(self):
        self.assertEqual(pair_weights_for(self.FILES, self.W).tolist(), [0.0, 0.30000001192092896, 1.0, 1.0])

    def test_repo_weights_file_matches_proposal(self):
        w = load_pair_weights(ROOT / 'data/beta1r2/pair_weights.json')
        counts = {v: sum(1 for x in w.values() if x == v) for v in (0.0, 0.3, 0.7)}
        self.assertEqual(counts, {0.0: 783, 0.3: 607, 0.7: 1685})
        self.assertEqual(w['poxima_v2_ds2_005577_1.jpg'], 0.0)   # o "Zubrowka" pixel-idêntico
        cfg = (ROOT / 'examples/beta1r2/train.toml').read_text()
        self.assertIn("pair_weights_path = '/workspace/dpe-beta1r/data/beta1r2/pair_weights.json'", cfg)

    def test_weight_reaches_loss_through_split_batch(self):
        seen = []
        pipe = _pipe(self.W)
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _fake_parent(seen), create=True):
            batch = _batch(self.FILES)
            batch['latents'] = batch['latents'][:1].repeat(B, 1, 1, 1, 1)   # mesmo resíduo em todas
            features, labels = Krea2MultiRefGroundedPipeline.prepare_inputs(pipe, batch)
        self.assertEqual(labels[1].shape, (B, 1, 1, 1, 1))
        loss_fn = ComfyPipeline.get_loss_fn(types.SimpleNamespace(config={}))
        micro = split_batch((features, labels), B)          # micro 1 x accum 4, como no treino
        grads = []
        for feats, (target, mask) in micro:
            out = feats[0].clone().requires_grad_(True)
            loss_fn(out, (target, mask)).backward()
            grads.append(out.grad.abs().sum().item())
        self.assertEqual(grads[0], 0.0)                     # idêntico: zero gradiente
        self.assertGreater(grads[1], 0.0)
        self.assertAlmostEqual(grads[1] / grads[2], 0.3, places=5)
        self.assertGreater(grads[3], 0.0)

    def test_no_weights_is_beta1r(self):
        seen = []
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _fake_parent(seen), create=True):
            _, labels = Krea2MultiRefGroundedPipeline.prepare_inputs(_pipe(None), _batch(self.FILES))
        self.assertIsNone(labels[1])

    def test_eval_not_weighted(self):
        seen = []
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _fake_parent(seen), create=True):
            _, labels = Krea2MultiRefGroundedPipeline.prepare_inputs(_pipe(self.W), _batch(self.FILES),
                                                                     timestep_quantile=0.5)
        self.assertIsNone(labels[1])

    def test_existing_mask_is_multiplied(self):
        target = torch.zeros(2, C, 1, H, W)
        mask = torch.full((2, 1, 1, H, W), 0.5)
        _, m = apply_pair_weights((target, mask), torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.equal(m[0], torch.zeros(1, 1, H, W)))
        self.assertTrue(torch.equal(m[1], torch.full((1, 1, H, W), 0.5)))


class ZoomCrop(unittest.TestCase):
    def test_identity_when_off(self):
        ref = torch.randn(B, C, 1, H, W, dtype=torch.bfloat16)
        self.assertIs(zoom_crop_reference(ref, 0.0, 1.25), ref)
        self.assertIs(zoom_crop_reference(ref, 0.5, 1.0), ref)

    def test_shape_dtype_and_fraction(self):
        ref = torch.randn(200, C, 1, H * 4, W * 4, dtype=torch.bfloat16)
        out = zoom_crop_reference(ref, 0.5, 1.25, rng=random.Random(0))
        self.assertEqual((out.shape, out.dtype), (ref.shape, ref.dtype))
        changed = sum(not torch.equal(out[b], ref[b]) for b in range(200))
        self.assertTrue(80 <= changed <= 120, changed)      # ~p=0.5 (alguns s~1 não mudam)
        self.assertFalse(torch.equal(ref, out))

    def test_content_is_zoomed_crop(self):
        # rampa linear: o recorte ampliado continua uma rampa, com faixa menor que a original
        h, w = 32, 48
        ramp = torch.arange(w, dtype=torch.float32).view(1, 1, 1, 1, w).expand(1, 1, 1, h, w).clone()
        out = zoom_crop_reference(ramp, 1.0, 1.25, rng=random.Random(3))[0, 0, 0]
        span = out[0, -1] - out[0, 0]
        self.assertLess(span.item(), w - 1)
        self.assertGreaterEqual(span.item(), (w - 1) / 1.25 - 1.5)
        self.assertTrue(torch.allclose(out[0], out[-1]))   # sem flip/rotação: linhas iguais
        self.assertTrue(bool((out[0, 1:] >= out[0, :-1]).all()))   # sem flip horizontal

    def test_only_reference_changes_in_prepare_inputs(self):
        seen = []
        batch = _batch(['a_1.jpg'] * B)
        orig = torch.stack([c.squeeze(0) for c in batch['control_latents']])
        random.seed(1)
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _fake_parent(seen), create=True):
            feats, _ = Krea2MultiRefGroundedPipeline.prepare_inputs(_pipe(None, 1.0, 1.25), batch)
        self.assertEqual(seen[0].shape, orig.shape)
        self.assertFalse(torch.equal(seen[0], orig))
        self.assertTrue(torch.equal(feats[0], batch['latents']))  # target intacto
        seen.clear()
        with mock.patch.object(Krea2OminiGroundedPipeline, 'prepare_inputs', _fake_parent(seen), create=True):
            Krea2MultiRefGroundedPipeline.prepare_inputs(_pipe(None, 1.0, 1.25), batch, timestep_quantile=0.5)
        self.assertTrue(torch.equal(seen[0], orig))                # eval nunca aumenta


class LRSchedule(unittest.TestCase):
    def test_hold_linear_inside_sequential(self):
        # mesma composição do train.py: warmup LinearLR(1/50) -> LambdaLR(hold_linear)
        cfg = json.loads('{"warmup": 50, "start": 500, "end": 2000, "factor": 0.2}')
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1 / cfg['warmup'], total_iters=cfg['warmup'])
        main = torch.optim.lr_scheduler.LambdaLR(
            opt, hold_linear_lambda(cfg['start'], cfg['end'], cfg['factor'], cfg['warmup']))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, main], milestones=[cfg['warmup']])
        lrs = {}
        for step in range(0, 3001):
            lrs[step] = opt.param_groups[0]['lr']
            opt.step()
            sched.step()
        self.assertAlmostEqual(lrs[0], 2e-6, delta=1e-12)
        for s in (50, 250, 500):
            self.assertAlmostEqual(lrs[s], 1e-4, delta=1e-10)
        self.assertAlmostEqual(lrs[1000], 1e-4 - 8e-5 * 500 / 1500, delta=1e-10)   # 7.33e-5
        self.assertAlmostEqual(lrs[1250], 6e-5, delta=1e-10)
        for s in (2000, 2500, 3000):
            self.assertAlmostEqual(lrs[s], 2e-5, delta=1e-10)
        self.assertTrue(all(lrs[s + 1] <= lrs[s] + 1e-15 for s in range(50, 3000)))


if __name__ == '__main__':
    unittest.main()
