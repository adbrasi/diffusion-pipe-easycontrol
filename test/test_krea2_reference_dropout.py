"""reference_dropout (krea2_multiref_grounded) — contrato de equivalência.

Usa um SingleStreamDiT REAL do ComfyUI (pesos aleatórios, minúsculo, fp32 em
CPU) com LoRA routada condition-only, e verifica:

1. Uma amostra com a referência descartada, dentro de um batch misto com
   padding de texto, produz EXATAMENTE o text-to-image stock do ComfyUI de B
   (SingleStreamDiT._forward sem ref_latents) — a ref não vaza nem pelos
   tokens VAE nem pela LoRA routada.
2. A amostra mantida no mesmo batch é idêntica a rodá-la sozinha (batch 1,
   sem padding): o txtfusion mascara o padding.
3. encode_text_live: amostra descartada vai ao Qwen3-VL sem arquivo de
   referência (sem bloco de visão) e com a legenda intacta; reference_keep=0.
"""
import random
import types
import unittest

import torch

import utils.common  # noqa: F401  (antes do ComfyUI entrar no sys.path: ele tem um pacote utils)
import models.base  # noqa: F401  (coloca submodules/ComfyUI no sys.path)
import comfy.ops
import peft
from comfy.ldm.krea2.model import SingleStreamDiT

from models.condition_lora import ConditionOnlyLoRARouter
from models.krea2_multiref import Krea2MultiRefGroundedPipeline, Krea2MultiRefInitialLayer
from models.krea2_ominicontrol import Krea2OminiTransformerLayer
from models.krea2_reference import Krea2ReferenceFinalLayer

TXTLAYERS, TXTDIM = 3, 32


class _NoOffload:
    def wait_for_block(self, idx):
        pass

    def submit_move_blocks_forward(self, idx):
        pass


def _tiny_model():
    torch.manual_seed(0)
    model = SingleStreamDiT(
        features=64, tdim=16, txtdim=TXTDIM, heads=4, kvheads=2, multiplier=2, layers=3,
        patch=2, channels=4, txtlayers=TXTLAYERS, txtheads=4, txtkvheads=4,
        dtype=torch.float32, operations=comfy.ops.manual_cast,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    targets = sorted({n for n, m in model.named_modules()
                      if n.startswith('blocks.') and isinstance(m, torch.nn.Linear)})
    lora = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=4, target_modules=targets))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.normal_(0, 0.5)   # LoRA forte: se vazar para o target, aparece
    router = ConditionOnlyLoRARouter(True)
    router.install(model.blocks)
    model.eval()
    return model, lora, router


def _layers(model, router):
    layers = [Krea2MultiRefInitialLayer(
        model, position_mode='width_shift', reference_position_offset=1.0,
        reference_position_scale=1.0, independent_condition=False,
        reference_timestep_mode='zero', slot_axis='width')]
    layers += [Krea2OminiTransformerLayer(b, i, _NoOffload(), router) for i, b in enumerate(model.blocks)]
    layers.append(Krea2ReferenceFinalLayer(model))
    return layers


def _run(layers, inputs):
    x = inputs
    for layer in layers:
        x = layer(x)
    return x


def _pad(ctxs):
    L = max(c.shape[0] for c in ctxs)
    emb = torch.stack([torch.cat([c, c.new_zeros(L - c.shape[0], c.shape[1])]) for c in ctxs])
    mask = torch.stack([torch.arange(L) < c.shape[0] for c in ctxs])
    return emb, mask


class ReferenceDropoutContract(unittest.TestCase):
    def setUp(self):
        self.model, self.lora, self.router = _tiny_model()
        self.layers = _layers(self.model, self.router)
        g = torch.Generator().manual_seed(1)
        self.H = self.W = 8
        self.target = torch.randn(2, 4, 1, self.H, self.W, generator=g)
        self.ref = torch.randn(2, 4, 1, self.H, self.W, generator=g)   # ref NÃO zerada de propósito
        self.t = torch.tensor([0.7, 0.3])
        # amostra 0 (descartada) tem texto curto -> padding; amostra 1 longo
        self.ctx = [torch.randn(5, TXTLAYERS * TXTDIM, generator=g),
                    torch.randn(9, TXTLAYERS * TXTDIM, generator=g)]

    @torch.no_grad()
    def test_dropped_sample_equals_stock_text_to_image(self):
        emb, mask = _pad(self.ctx)
        keep = torch.tensor([0.0, 1.0])
        out = _run(self.layers, (self.target, self.t, emb, mask, self.ref, keep))
        with self.lora.disable_adapter():
            stock = self.model._forward(self.target[0:1, :, 0], self.t[0:1], self.ctx[0][None])
        torch.testing.assert_close(out[0:1, :, 0], stock, atol=2e-5, rtol=1e-4)
        # controle negativo: sem o keep a ref vaza (saída diferente do t2i)
        leak = _run(self.layers, (self.target, self.t, emb, mask, self.ref))
        self.assertGreater((leak[0:1, :, 0] - stock).abs().max().item(), 1e-3)

    @torch.no_grad()
    def test_kept_sample_unaffected_by_batchmate_padding(self):
        emb, mask = _pad(self.ctx)
        keep = torch.tensor([0.0, 1.0])
        out = _run(self.layers, (self.target, self.t, emb, mask, self.ref, keep))
        alone = _run(self.layers, (self.target[1:2], self.t[1:2], self.ctx[1][None],
                                   torch.ones(1, 9, dtype=torch.bool), self.ref[1:2]))
        torch.testing.assert_close(out[1:2], alone, atol=2e-5, rtol=1e-4)
        # e o padded curto (amostra 0) mantida == ela sozinha também
        out_k = _run(self.layers, (self.target, self.t, emb, mask, self.ref, torch.ones(2)))
        alone0 = _run(self.layers, (self.target[0:1], self.t[0:1], self.ctx[0][None],
                                    torch.ones(1, 5, dtype=torch.bool), self.ref[0:1]))
        torch.testing.assert_close(out_k[0:1], alone0, atol=2e-5, rtol=1e-4)

    def test_encode_text_live_drops_vision_block_and_keeps_caption(self):
        calls = {}

        def fake_fn(captions, is_video, files):
            calls['captions'], calls['files'] = list(captions), list(files)
            return {'text_embeds_0': [torch.zeros(3, 2) for _ in captions],
                    'attention_mask_0': [torch.ones(3) for _ in captions]}

        pipe = types.SimpleNamespace(
            reference_dropout=0.5, caption_dropout=0.0, text_encoders=[torch.nn.Identity()],
            get_call_text_encoder_fn=lambda te: fake_fn)
        random.seed(3)
        batch = {'caption': [f'cap{i}' for i in range(64)],
                 'control_file': [[f'/r/{i}_1.jpg'] for i in range(64)]}
        orig_to = torch.nn.Module.to
        torch.nn.Module.to = lambda self, *a, **k: self   # sem CUDA no teste
        try:
            out = Krea2MultiRefGroundedPipeline.encode_text_live(pipe, batch)
        finally:
            torch.nn.Module.to = orig_to
        keep = out['reference_keep']
        self.assertEqual(calls['captions'], [f'cap{i}' for i in range(64)])
        for i, k in enumerate(keep.tolist()):
            self.assertEqual(calls['files'][i] is None, k == 0.0)
        self.assertTrue(0 < keep.sum().item() < 64)

        pipe.reference_dropout = 0.0
        out = Krea2MultiRefGroundedPipeline.encode_text_live(
            pipe, {'caption': ['a'], 'control_file': [['/r/x_1.jpg']]})
        self.assertNotIn('reference_keep', out)
        self.assertEqual(calls['files'], [['/r/x_1.jpg']])


if __name__ == '__main__':
    unittest.main()
