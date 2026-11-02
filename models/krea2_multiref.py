"""Krea 2 com N referências separadas (Macro-Dataset).

Estende o `krea2_omini_grounded` — o contrato validado do projeto — para
aceitar N referências em vez de uma. Mantém tudo que funcionava:
width-shift, referências moduladas a t=0, LoRA condition-only routada nos
blocks, txtfusion global e treinável, grounding Qwen3-VL.

CONTRATO DA SEQUÊNCIA
---------------------
    [texto (caption + N blocos de visão) | target ruidoso | ref_1 | … | ref_N]

O tensor de referência chega como ``(B, C, N, H, W)`` — as N referências
empilhadas no eixo de frame do latente. Elas NÃO são tratadas como frames de
vídeo: o eixo de frame é só transporte, e cada uma vira seu próprio span de
tokens com sua própria posição.

O PONTO CRÍTICO — por que o offset é cumulativo
-----------------------------------------------
O `width_shift` original (`krea2_reference.py:359`) soma uma CONSTANTE::

    reference_pos[..., 2] += float(target_grid_w)

Reusar isso num laço sobre N daria a todos os spans posições IDÊNTICAS. E aí
a saída do DiT sobre o target fica *exatamente* invariante à troca de duas
referências, porque:

  1. atenção é soma sobre keys ponderada por softmax — permutation-invariant
     sobre o conjunto de keys, dadas as rotações que eles carregam;
  2. `tvec` é o mesmo para toda linha de referência (zeros);
  3. a máscara é a mesma para todas;
  4. o router LoRA aplica a mesma máscara a todas (span contíguo, sem
     granularidade por slot);
  5. MLP e norms são pointwise.

Trocar dois sub-spans vira uma permutação de tokens com metadados idênticos
=> saída idêntica. Não é "difícil de aprender": é IMPOSSÍVEL em princípio,
para qualquer quantidade de dados.

Por isso o slot ``i`` recebe ``w += (i + 1) * ref_grid_w``. Com offsets
crescentes os spans deixam de ser permutáveis e o modelo PODE distingui-los.
Se ele vai distinguir vira questão de dados — que é o enquadramento certo:
``<image 1>`` é só texto no caption, e o binding é aprendido por correlação
(precedente: o dataset de abril do Anima ensinou três verbos-operação com
~21k pares).

O eixo de frame (`position_mode='subject'`) é a alternativa: índice discreto,
invariante a tamanho, e é o contrato público do Krea Edit ("RoPE frame 1").
Está implementado aqui como `slot_axis = 'frame'` para A/B, mas o default é
`width` porque é o que o único treino bem-sucedido de Krea 2 do projeto usou.
"""

import json
import math
import os
import random

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding

from models.base import make_contiguous
from models.krea2_edit import Krea2EditPipeline
from models.krea2_omini_grounded import Krea2OminiGroundedPipeline
from models.krea2_ominicontrol import Krea2OminiTransformerLayer
from models.krea2_reference import Krea2ReferenceFinalLayer, Krea2ReferenceInitialLayer
from utils.common import AUTOCAST_DTYPE, is_main_process


# ---------------------------------------------------------------------------
# BETA1-R2 anti-cópia (docs/BETA1R2.md). Os dois são no-op com a config padrão.
# ---------------------------------------------------------------------------

def load_pair_weights(path):
    """{nome do arquivo de referência: peso}; ausente = 1.0."""
    with open(path) as f:
        weights = {str(k): float(v) for k, v in json.load(f).items()}
    if any(not 0.0 <= w <= 1.0 for w in weights.values()):
        raise ValueError(f'pesos de par fora de [0, 1] em {path}')
    return weights


def pair_weights_for(control_files, weights):
    """Peso por amostra a partir do control_file do batch (str ou lista de str;
    casa pelo nome do arquivo, imune a /workspace vs /dev/shm)."""
    out = []
    for control_file in control_files:
        if isinstance(control_file, (list, tuple)):
            control_file = control_file[0] if control_file else None
        out.append(weights.get(os.path.basename(control_file), 1.0) if control_file else 1.0)
    return torch.tensor(out, dtype=torch.float32)


def apply_pair_weights(labels, pair_weight):
    """Multiplica a máscara de loss por amostra. `base.loss_fn` faz loss *= mask;
    máscara vazia/None = sem máscara. (B,1,1,1,1) segue o split_batch no dim 0."""
    target, mask = labels
    w = pair_weight.to(target.device, torch.float32).view(-1, *([1] * (target.ndim - 1)))
    if mask is None or mask.numel() == 0:
        mask = w
    else:
        mask = mask.to(torch.float32) * w
    return (target, mask)


def zoom_crop_reference(reference, prob, max_zoom, rng=random):
    """Zoom-crop aleatório no latente da referência (B, C, N, h, w): com prob.
    `prob` por amostra, recorta uma janela de lado h/s x w/s (s ~ U[1, max_zoom],
    posição uniforme) e volta ao MESMO grid por bilinear. Sem flip, sem cor.
    Cada referência da amostra recebe o seu próprio recorte."""
    if prob <= 0 or max_zoom <= 1.0:
        return reference
    B, C, N, h, w = reference.shape
    out = None
    for b in range(B):
        if rng.random() >= prob:
            continue
        if out is None:
            out = reference.clone()
        for n in range(N):
            s = rng.uniform(1.0, max_zoom)
            ch, cw = max(2, round(h / s)), max(2, round(w / s))
            if ch == h and cw == w:
                continue
            y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
            crop = reference[b, :, n, y0:y0 + ch, x0:x0 + cw].unsqueeze(0).float()
            out[b, :, n] = F.interpolate(crop, size=(h, w), mode='bilinear',
                                         align_corners=False)[0].to(reference.dtype)
    return reference if out is None else out


def copy_region_mask(latents, reference, threshold, weight):
    """BETA1-R4 (revisão independente): máscara espacial de loss contra a cópia alinhada.

    Posição latente do alvo é "cópia alinhada" da referência quando média_c |z_alvo - z_ref| < threshold
    (latentes normalizados do cache; a ref está no mesmo grid do alvo). Calibrado contra o mapa de pixels
    (|A-B|<0.08 em blocos 8x8): tools/beta1r4_calibrate_mask.py, F1 0.81, correlação por par 0.95.
    Essas posições recebem peso `weight` na loss (o resto 1): o modelo continua aprendendo a cena, mas
    deixa de ser recompensado por colar os pixels da referência.

    latents (B, C, 1, h, w), reference (B, C, N, h, w) -> (B, 1, 1, h, w) float32.
    """
    ref = reference[:, :, :1].float()
    if ref.shape[-2:] != latents.shape[-2:]:
        return torch.ones(latents.shape[0], 1, 1, *latents.shape[-2:], device=latents.device)
    d = (latents.float() - ref).abs().mean(dim=1, keepdim=True)
    return torch.where(d < threshold, torch.full_like(d, weight), torch.ones_like(d))


def sample_reference_corruption(reference, patch, noise_prob, noise_min, noise_max,
                                token_drop_prob, keep_min, keep_max):
    """Anti-cópia no mecanismo (BETA1-R2), só em treino e só numa fração das amostras;
    o resto fica com a referência limpa/inteira = exatamente o contrato de inferência.

    - Referência com ruído (Captain Cinema / Long Context Tuning): com prob. noise_prob,
      t_ref ~ U[noise_min, noise_max] e ref = (1-t)·ref + t·ε (mesma fórmula do flow);
      os tokens da referência recebem t_ref no timestep embedding (não 0).
    - Descarte de tokens (DreamO): com prob. token_drop_prob, mantém só uma fração
      ~U[keep_min, keep_max] dos tokens da referência (os outros viram keys inválidas).

    reference: (B, C, N, h, w). Devolve (reference, reference_t (B,), token_keep (B, L)).
    Sorteado aqui (prepare_inputs) e não na camada: o activation checkpointing recomputa
    a camada no backward e sortearia de novo.
    """
    B, C, N, h, w = reference.shape
    device = reference.device
    reference_t = torch.zeros(B, device=device, dtype=torch.float32)
    if noise_prob > 0:
        sel = torch.rand(B, device=device) < noise_prob
        if bool(sel.any()):
            t = torch.empty(B, device=device).uniform_(noise_min, noise_max)
            reference_t = torch.where(sel, t, reference_t)
            tt = reference_t.view(B, 1, 1, 1, 1).to(reference.dtype)
            reference = (1 - tt) * reference + tt * torch.randn_like(reference)
    grid = N * (-(-h // patch)) * (-(-w // patch))
    token_keep = torch.ones(B, grid, device=device, dtype=torch.float32)
    if token_drop_prob > 0:
        for b in range(B):
            if random.random() >= token_drop_prob:
                continue
            frac = random.uniform(keep_min, keep_max)
            keep = torch.rand(grid, device=device) < frac
            if not bool(keep.any()):
                keep[torch.randint(grid, (1,), device=device)] = True
            token_keep[b] = keep.float()
    return reference, reference_t, token_keep


class Krea2MultiRefInitialLayer(Krea2ReferenceInitialLayer):
    """Igual ao pai, mas com N spans de referência e offset por slot."""

    def __init__(self, *args, slot_axis='width', ref_bias=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_axis = slot_axis
        # Só inferência (anti-cópia): viés aditivo log(b) nos logits
        # alvo→referência, contrato do CtxRushKrea2MultiRefControlledApply
        # (ctxrush-edit nodes_multiref.py). 1.0 = desligado, máscara intacta.
        self.ref_bias = float(ref_bias)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # 6th input (optional): reference_keep (B,) — 0 where reference_dropout
        # removed the reference from this sample (see encode_text_live).
        # 7th input (optional): reference_t (B,) — timestep da referência com ruído
        # (sample_reference_corruption); reference_keep pode então ser (B, L_ref) por token.
        reference_t_in = None
        if len(inputs) == 7:
            target, timesteps, context, text_attention_mask, reference, reference_keep, reference_t_in = inputs
        elif len(inputs) == 6:
            target, timesteps, context, text_attention_mask, reference, reference_keep = inputs
        else:
            target, timesteps, context, text_attention_mask, reference = inputs
            reference_keep = None
        if target.shape[2] != 1:
            raise ValueError('Krea2 multi-ref suporta exatamente um frame de target')
        target = target[:, :, 0]
        batch, channels, target_h_orig, target_w_orig = target.shape
        num_refs = reference.shape[2]

        patch = self.patch
        target = comfy.ldm.common_dit.pad_to_patch_size(target, (patch, patch))
        target_h, target_w = target.shape[-2:]
        target_grid_h, target_grid_w = target_h // patch, target_w // patch

        context = self._unpack_context(context)
        target_tokens = self.first(
            rearrange(target, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch)
        )

        # --- os N spans de referência -------------------------------------
        ref_token_list = []
        ref_pos_list = []
        scale_bias = (self.reference_position_scale - 1.0) / 2.0
        width_cursor = float(target_grid_w)
        for slot in range(num_refs):
            ref = comfy.ldm.common_dit.pad_to_patch_size(
                reference[:, :, slot], (patch, patch)
            )
            grid_h, grid_w = ref.shape[-2] // patch, ref.shape[-1] // patch
            ref_token_list.append(self.first(
                rearrange(ref, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch)
            ))

            pos = self._grid_positions(batch, grid_h, grid_w, target.device).clone()
            pos[..., 1:] = pos[..., 1:] * self.reference_position_scale + scale_bias
            if self.slot_axis == 'frame':
                # índice discreto por slot no eixo de frame (contrato Krea Edit)
                pos[..., 0] = self.reference_position_offset + slot
            elif self.position_mode == 'subject':
                pos[..., 0] = self.reference_position_offset + slot
            elif self.position_mode == 'width_shift':
                # O OFFSET CUMULATIVO. Ver o docstring do módulo: com uma
                # constante aqui os spans viram permutation-invariant e o
                # modelo fica incapaz de distingui-los, por construção.
                pos[..., 2] = pos[..., 2] + width_cursor
            width_cursor += float(grid_w)
            ref_pos_list.append(pos)

        reference_tokens = torch.cat(ref_token_list, dim=1)
        reference_pos = torch.cat(ref_pos_list, dim=1)

        # Com micro batch > 1 o texto chega com padding até o max do batch. O
        # stock ComfyUI roda batch 1 (mask=None, sem padding); para o batch>1
        # ser equivalente a N execuções batch-1, os refiners do txtfusion não
        # podem atender ao padding (port do fix da branch elegant-ptolemy).
        # Com batch 1 / sem padding a máscara é toda True = idêntico ao stock.
        fusion_mask = text_attention_mask.to(device=context.device, dtype=torch.bool)
        fusion_mask = None if bool(fusion_mask.all()) else fusion_mask[:, None, None, :]
        context = self.txtfusion(context, mask=fusion_mask)
        context = self.txtmlp(context)
        text_length = context.shape[1]
        target_length = target_tokens.shape[1]
        reference_length = reference_tokens.shape[1]
        combined = torch.cat([context, target_tokens, reference_tokens], dim=1)

        target_timestep_features = self.tmlp(
            timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(combined.dtype)
        )
        target_t = timesteps[:, None].expand(batch, text_length + target_length)
        if self.reference_timestep_mode == 'target':
            reference_t = timesteps[:, None].expand(batch, reference_length)
        elif reference_t_in is not None:
            reference_t = reference_t_in.to(timesteps)[:, None].expand(batch, reference_length)
        else:
            reference_t = timesteps.new_zeros(batch, reference_length)
        per_token_timestep = torch.cat([target_t, reference_t], dim=1)
        embedded_timesteps = timestep_embedding(
            per_token_timestep.reshape(-1), self.tdim
        ).reshape(batch, combined.shape[1], self.tdim)
        tvec = self.tproj(self.tmlp(embedded_timesteps.to(combined.dtype)))

        target_pos = self._grid_positions(batch, target_grid_h, target_grid_w, combined.device)
        text_pos = combined.new_zeros(batch, text_length, 3)
        positions = torch.cat([text_pos, target_pos, reference_pos], dim=1)
        freqs = self.pe_embedder(positions)

        image_mask = torch.ones(
            batch, target_length + reference_length,
            dtype=torch.bool, device=text_attention_mask.device,
        )
        if reference_keep is not None:
            # reference_dropout: as keys da referência ficam inválidas para TODAS
            # as queries da amostra descartada. As linhas de texto+target então
            # computam exatamente o text-to-image de B (a ref não entra em
            # nenhuma soma de atenção; o LoRA routado só toca as linhas da ref,
            # que não alimentam nada e ficam fora da loss).
            if self.independent_condition:
                raise NotImplementedError('reference_dropout com independent_condition não suportado')
            keep_mask = reference_keep.to(device=image_mask.device).bool()
            image_mask[:, target_length:] = keep_mask[:, None] if keep_mask.ndim == 1 else keep_mask
        valid_keys = torch.cat([text_attention_mask.bool(), image_mask], dim=1)
        if self.independent_condition:
            attention_mask = valid_keys[:, None, None, :].expand(
                batch, 1, combined.shape[1], combined.shape[1]
            ).clone()
            reference_start = text_length + target_length
            attention_mask[:, :, reference_start:, :] = False
            attention_mask[:, :, reference_start:, reference_start:] = True
            attention_mask = torch.zeros_like(attention_mask, dtype=combined.dtype).masked_fill_(
                ~attention_mask, -torch.finfo(combined.dtype).max
            )
        elif math.isclose(self.ref_bias, 1.0, rel_tol=0.0, abs_tol=1e-9):
            attention_mask = valid_keys[:, None, None, :]
        else:
            seq_len = combined.shape[1]
            attention_mask = torch.zeros(batch, 1, seq_len, seq_len,
                                         device=combined.device, dtype=combined.dtype)
            attention_mask[:, :, text_length:text_length + target_length,
                           text_length + target_length:] = math.log(max(self.ref_bias, 1e-4))
            attention_mask.masked_fill_(~valid_keys[:, None, None, :].to(combined.device),
                                        torch.finfo(combined.dtype).min)

        sizes = torch.tensor(
            [text_length, target_length, target_grid_h, target_grid_w, target_h_orig, target_w_orig],
            device=combined.device,
        )
        outputs = make_contiguous(combined, target_timestep_features, tvec, freqs, attention_mask, sizes)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs


class Krea2MultiRefGroundedPipeline(Krea2OminiGroundedPipeline):
    name = 'krea2_multiref_grounded'
    config_section = 'krea2_multiref_grounded'
    checkpointable_layers = [
        'Krea2MultiRefInitialLayer', 'TransformerLayer', 'Krea2OminiTransformerLayer'
    ]

    def __init__(self, config):
        super().__init__(config)
        section = config.get(self.config_section, {})
        self.max_refs = int(section.get('max_refs', 3))
        self.slot_axis = section.get('slot_axis', 'width')
        if self.slot_axis not in ('width', 'frame'):
            raise ValueError("slot_axis must be 'width' or 'frame'")
        self.txtfusion_rank = int(section.get('txtfusion_rank', 0))
        # reference_dropout: fração de amostras em que a referência é REMOVIDA
        # por inteiro e de forma coerente — tokens VAE (keys mascaradas) E a
        # imagem do grounding Qwen3-VL (prompt sem bloco "image 1:"). A legenda
        # fica. A amostra vira text-to-image puro de B. Exige encoding de texto
        # ao vivo (o prompt muda por step).
        self.reference_dropout = float(section.get('reference_dropout', 0.0))
        if not 0.0 <= self.reference_dropout <= 0.5:
            raise ValueError('reference_dropout must be between 0.0 and 0.5')
        self.live_text_encoding = bool(config['model'].get('live_text_encoding', False))
        if self.reference_dropout > 0 and not self.live_text_encoding:
            raise ValueError('reference_dropout > 0 exige [model] live_text_encoding = true')
        if self.reference_dropout > 0 and self.independent_condition:
            raise ValueError('reference_dropout não suporta independent_condition')
        # BETA1-R2: peso de loss por par (similaridade A-B) e zoom-crop da referência.
        pair_weights_path = section.get('pair_weights_path')
        self.pair_weights = load_pair_weights(pair_weights_path) if pair_weights_path else None
        self.ref_aug_prob = float(section.get('ref_aug_prob', 0.0))
        self.ref_aug_max_zoom = float(section.get('ref_aug_max_zoom', 1.0))
        if not 0.0 <= self.ref_aug_prob <= 1.0 or not 1.0 <= self.ref_aug_max_zoom <= 2.0:
            raise ValueError('ref_aug_prob em [0, 1] e ref_aug_max_zoom em [1, 2]')
        self.ref_noise_prob = float(section.get('ref_noise_prob', 0.0))
        self.ref_noise_min = float(section.get('ref_noise_min', 0.05))
        self.ref_noise_max = float(section.get('ref_noise_max', 0.35))
        self.ref_token_drop_prob = float(section.get('ref_token_drop_prob', 0.0))
        self.ref_token_keep_min = float(section.get('ref_token_keep_min', 0.05))
        self.ref_token_keep_max = float(section.get('ref_token_keep_max', 0.5))
        if not (0.0 <= self.ref_noise_prob <= 1.0 and 0.0 <= self.ref_noise_min <= self.ref_noise_max <= 1.0):
            raise ValueError('ref_noise_prob em [0, 1] e 0 <= ref_noise_min <= ref_noise_max <= 1')
        if not (0.0 <= self.ref_token_drop_prob <= 1.0 and 0.0 < self.ref_token_keep_min <= self.ref_token_keep_max <= 1.0):
            raise ValueError('ref_token_drop_prob em [0, 1] e 0 < ref_token_keep_min <= ref_token_keep_max <= 1')
        # BETA1-R3: tokens VAE da referência mascarados quando t > limiar (só ruído alto; o Qwen3-VL
        # continua vendo a imagem). A cópia nasce no 1º passo de amostragem (σ≈1): o layout passa a vir
        # do texto + semântica, a identidade fina dos tokens VAE nos passos seguintes. Inferência
        # consistente: pular os tokens VAE enquanto σ > o mesmo limiar (REF_SKIP_SIGMA / node ref_skip_sigma).
        # BETA1-R4: máscara espacial de loss nas posições do alvo que são cópia alinhada da referência
        self.copy_mask_threshold = float(section.get('copy_mask_threshold', 0.0))
        self.copy_mask_weight = float(section.get('copy_mask_weight', 1.0))
        if self.copy_mask_threshold < 0 or not 0.0 <= self.copy_mask_weight <= 1.0:
            raise ValueError('copy_mask_threshold >= 0 e copy_mask_weight em [0, 1]')
        self.ref_vae_mask_above_t = float(section.get('ref_vae_mask_above_t', 1.0))
        if not 0.0 < self.ref_vae_mask_above_t <= 1.0:
            raise ValueError('ref_vae_mask_above_t em (0, 1]; 1.0 = desligado')
        if self.ref_vae_mask_above_t < 1.0 and self.independent_condition:
            raise ValueError('ref_vae_mask_above_t não suporta independent_condition')
        if (self.ref_noise_prob > 0 or self.ref_token_drop_prob > 0) and self.independent_condition:
            raise ValueError('ref_noise/ref_token_drop não suportam independent_condition')
        if self.ref_noise_prob > 0 and self.reference_timestep_mode == 'target':
            raise ValueError("ref_noise exige reference_timestep = 'zero'")
        if is_main_process():
            print(f'[{self.name}] ref_noise p={self.ref_noise_prob} t~U[{self.ref_noise_min},{self.ref_noise_max}] '
                  f'ref_token_drop p={self.ref_token_drop_prob} keep~U[{self.ref_token_keep_min},{self.ref_token_keep_max}]')
        if is_main_process():
            print(f'[{self.name}] max_refs={self.max_refs} slot_axis={self.slot_axis} '
                  f'position_mode={self.position_mode} reference_dropout={self.reference_dropout} '
                  f'live_text_encoding={self.live_text_encoding}')
            if self.pair_weights is not None or self.ref_aug_prob > 0:
                print(f'[{self.name}] pair_weights={pair_weights_path} '
                      f'({0 if self.pair_weights is None else len(self.pair_weights)} pares com peso < 1) '
                      f'ref_aug_prob={self.ref_aug_prob} ref_aug_max_zoom={self.ref_aug_max_zoom}')

    def model_specific_dataset_config_validation(self, dataset_config):
        # o loader multi-ref usa manifest_path, não control_path
        for index, directory in enumerate(dataset_config.get('directory', [])):
            if not directory.get('manifest_path') and not directory.get('control_path'):
                raise ValueError(
                    f'{self.name} exige manifest_path (ou control_path) em cada diretório; '
                    f'faltando na entrada {index}'
                )

    def configure_adapter(self, adapter_config):
        """Igual ao krea2_edit, mas com rank maior no txtfusion.

        Motivo: o canal semântico carrega ~1,13% da energia do adapter e o
        rank efetivo colapsou a ~1/64 em 250 steps
        (docs/OMINI_GROUNDED_CANAL_SEMANTICO.md). Com N referências, LER
        seletivamente é a tarefa — e o modo de falha documentado do routing
        condition-only é exatamente "não ensina ninguém a ler" (armC do
        Anima). Aqui isso deixa de ser refinamento e vira pré-requisito.

        `txtfusion_rank = 0` reproduz o comportamento antigo (rank uniforme).
        """
        if not self.txtfusion_rank or adapter_config['type'] != 'lora':
            super().configure_adapter(adapter_config)
            return
        # Os nomes COMPLETOS têm que sair do modelo: o PEFT normaliza
        # `target_modules` para um conjunto de sufixos, então ler de volta do
        # peft_config depois do get_peft_model não devolve nomes casáveis.
        full_names = [
            name for name, module in self.diffusion_model.named_modules()
            if 'txtfusion' in name and isinstance(module, torch.nn.Linear)
            and not name.endswith('projector')
        ]
        if not full_names:
            raise RuntimeError('rank do txtfusion pedido mas nenhum Linear de txtfusion existe')
        pattern = {name: self.txtfusion_rank for name in full_names}
        adapter_config = dict(adapter_config, rank_pattern=pattern, alpha_pattern=dict(pattern))
        super().configure_adapter(adapter_config)
        if is_main_process():
            print(f'[{self.name}] rank {self.txtfusion_rank} em {len(pattern)} linears do txtfusion')
            self.lora_model.print_trainable_parameters()

    def get_call_vae_fn(self, vae):
        """As N referências chegam como batch do VAE; remonta no eixo de frame.

        O loader empilha as N refs na dimensão de batch da chamada
        (`utils/dataset.py`, ramo multi_ref) porque o target ocupa a batch
        real. Aqui `(N, C, 1, h, w)` vira `(1, C, N, h, w)`.
        """
        parent_vae_fn = Krea2EditPipeline.get_call_vae_fn(self, vae)

        def fn(*args):
            if len(args) == 1:
                return parent_vae_fn(args[0])
            if len(args) != 2:
                raise RuntimeError(f'Número inesperado de entradas do VAE: {len(args)}')
            target, references = args
            result = parent_vae_fn(target)
            ref_latents = parent_vae_fn(self.prepare_reference_media(references))['latents']
            # (N, C, 1, h, w) -> (1, C, N, h, w)
            if ref_latents.ndim == 5:
                ref_latents = ref_latents.squeeze(2)
            num_refs = ref_latents.shape[0]
            if num_refs > self.max_refs:
                raise RuntimeError(f'{num_refs} referências excedem max_refs={self.max_refs}')
            result['control_latents'] = ref_latents.permute(1, 0, 2, 3).unsqueeze(0)
            return result

        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        """O collate entrega `control_latents` como LISTA (uma entrada por
        amostra, com N variável). Empilha em `(B, C, N_max, h, w)` com padding
        de zeros — que é exatamente a representação de "referência ausente"
        que o `condition_dropout` do Anima usa (`ic_lora_full.py:143`): o slot
        continua na sequência, só fica em branco. Logo o padding é
        in-distribution por construção, não um caso especial."""
        control = inputs.get('control_latents')
        if isinstance(control, (list, tuple)):
            per_sample = [c if c.ndim == 4 else c.squeeze(0) for c in control]
            num_refs = max(c.shape[1] for c in per_sample)
            padded = []
            for c in per_sample:
                if c.shape[1] < num_refs:
                    pad = c.new_zeros(c.shape[0], num_refs - c.shape[1], *c.shape[2:])
                    c = torch.cat([c, pad], dim=1)
                padded.append(c)
            inputs = dict(inputs)
            inputs['control_latents'] = torch.stack(padded)
        training = timestep_quantile is None   # eval (quantis fixos) nunca aumenta nem pondera
        copy_mask = None
        if training and getattr(self, 'copy_mask_threshold', 0.0) > 0 and self.copy_mask_weight < 1.0:
            copy_mask = copy_region_mask(inputs['latents'], inputs['control_latents'],
                                         self.copy_mask_threshold, self.copy_mask_weight)
        if training and self.ref_aug_prob > 0:
            inputs = dict(inputs)
            inputs['control_latents'] = zoom_crop_reference(
                inputs['control_latents'], self.ref_aug_prob, self.ref_aug_max_zoom)
        model_inputs, labels = super().prepare_inputs(inputs, timestep_quantile=timestep_quantile)
        if training and self.pair_weights is not None:
            control_files = inputs.get('control_file')
            if control_files is None:
                raise RuntimeError('pair_weights_path exige control_file no batch')
            labels = apply_pair_weights(labels, pair_weights_for(control_files, self.pair_weights))
        keep = inputs.get('reference_keep', None)
        if copy_mask is not None:
            if keep is not None:
                # referência descartada: o modelo não a vê, então não há cópia a desencorajar
                dropped = (keep.to(copy_mask.device) == 0).view(-1, 1, 1, 1, 1)
                copy_mask = torch.where(dropped, torch.ones_like(copy_mask), copy_mask)
            target, mask = labels
            mask = copy_mask if mask is None or mask.numel() == 0 else mask.to(torch.float32) * copy_mask.to(mask.device)
            labels = (target, mask)
        mask_t = getattr(self, 'ref_vae_mask_above_t', 1.0)
        if training and mask_t < 1.0:
            high = model_inputs[1].detach().float().cpu() > mask_t
            if bool(high.any()):
                base = torch.ones(high.shape[0]) if keep is None else keep.float().cpu()
                keep = base * (~high).float()
        noise_p = getattr(self, 'ref_noise_prob', 0.0)
        drop_p = getattr(self, 'ref_token_drop_prob', 0.0)
        corrupt = training and (noise_p > 0 or drop_p > 0)
        if keep is None and not corrupt:
            return model_inputs, labels
        reference = model_inputs[-1]
        if corrupt:
            reference, reference_t, token_keep = sample_reference_corruption(
                reference, self.diffusion_model.patch, self.ref_noise_prob, self.ref_noise_min,
                self.ref_noise_max, self.ref_token_drop_prob, self.ref_token_keep_min,
                self.ref_token_keep_max)
        if keep is not None:
            # Referência descartada: zera também o latente (cosmético — as keys já
            # são mascaradas na camada inicial) e passa o keep como 6º input.
            keep = keep.to(device=reference.device, dtype=reference.dtype)
            reference = reference * keep.view(-1, *([1] * (reference.ndim - 1)))
        if not corrupt:
            return (*model_inputs[:-1], reference, keep), labels
        if keep is not None:
            token_keep = token_keep * keep.to(token_keep)[:, None]
            reference_t = reference_t * keep.to(reference_t)   # ref ausente: t irrelevante, 0
        return (*model_inputs[:-1], reference, token_keep, reference_t), labels

    def encode_text_live(self, batch):
        """Encoding ao vivo do Qwen3-VL (mesmo fn do cache) + reference_dropout.

        Por amostra: com prob. reference_dropout o arquivo de referência some
        do prompt (sem bloco de visão; a legenda fica) e reference_keep=0
        manda a camada inicial mascarar as keys da ref. Senão, com prob.
        caption_dropout, legenda vazia mantendo a ref (uncond grounded).
        """
        captions = batch.get('caption', None)
        if captions is None:
            raise RuntimeError('live_text_encoding needs raw captions in the batch')
        if isinstance(captions, str):
            captions = [captions]
        control_files = batch.get('control_file', None)
        if control_files is None:
            raise RuntimeError('live_text_encoding needs control_file in the batch')
        n = len(captions)
        keep, caps, files = [], [], []
        for caption, control_file in zip(captions, control_files):
            if random.random() < self.reference_dropout:
                keep.append(0.0)
                caps.append(caption)
                files.append(None)
            else:
                keep.append(1.0)
                caps.append('' if random.random() < self.caption_dropout else caption)
                files.append(control_file)
        text_encoder = self.text_encoders[0]
        if isinstance(text_encoder, nn.Module):
            text_encoder = text_encoder.to('cuda')
        else:
            text_encoder.load_model_if_needed()
        outputs = self.get_call_text_encoder_fn(text_encoder)(caps, [False] * n, files)
        # get_call_text_encoder_fn é inference_mode: clonar FORA dele para os
        # tensores poderem entrar no grafo (LoRA do txtfusion).
        for key, value in outputs.items():
            batch[key] = [tensor.clone() for tensor in value]
        if self.reference_dropout > 0:
            batch['reference_keep'] = torch.tensor(keep, dtype=torch.float32)
        return batch

    def prepare_reference_latents(self, reference, noisy_target, timestep_quantile=None):
        """Aceita (B, C, N, h, w). Só o eixo de frame pode diferir do target."""
        if reference.shape[:2] != noisy_target.shape[:2]:
            raise ValueError(
                f'batch/canais precisam bater: {tuple(reference.shape[:2])} != '
                f'{tuple(noisy_target.shape[:2])}'
            )
        if reference.shape[-2:] != noisy_target.shape[-2:]:
            raise ValueError(
                f'shape espacial da referência {tuple(reference.shape[-2:])} != target '
                f'{tuple(noisy_target.shape[-2:])}; regenere o cache do VAE.'
            )
        # condition_dropout continua proibido no caminho grounded (herdado do
        # krea2_edit): zerar o VAE sem tirar a imagem do Qwen3-VL é dropout
        # parcial incoerente. O substituto é caption_dropout.
        return reference

    def to_layers(self):
        model = self.diffusion_model
        layers = [
            Krea2MultiRefInitialLayer(
                model,
                position_mode=self.position_mode,
                reference_position_offset=self.reference_position_offset,
                reference_position_scale=self.reference_position_scale,
                independent_condition=self.independent_condition,
                reference_timestep_mode=self.reference_timestep_mode,
                slot_axis=self.slot_axis,
            )
        ]
        layers.extend(
            Krea2OminiTransformerLayer(block, index, self.offloader, self.condition_lora_router)
            for index, block in enumerate(model.blocks)
        )
        layers.append(Krea2ReferenceFinalLayer(model))
        return layers

    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        metadata['control_family'] = 'krea2_multiref_grounded'
        metadata['max_refs'] = str(self.max_refs)
        metadata['slot_axis'] = self.slot_axis
        metadata['reference_dropout'] = str(self.reference_dropout)
        metadata['reference_dropout_mode'] = 'full_vae_keys_masked_and_no_vl_image'
        metadata['live_text_encoding'] = str(self.live_text_encoding).lower()
        if self.pair_weights is not None:
            metadata['pair_weights'] = str(len(self.pair_weights))
        if self.copy_mask_threshold > 0 and self.copy_mask_weight < 1.0:
            metadata['copy_mask'] = f'latent_absdiff<{self.copy_mask_threshold} weight={self.copy_mask_weight}'
        if self.ref_vae_mask_above_t < 1.0:
            metadata['ref_vae_mask_above_t'] = str(self.ref_vae_mask_above_t)
            metadata['inference_ref_skip_sigma'] = str(self.ref_vae_mask_above_t)
        if self.ref_noise_prob > 0:
            metadata['ref_noise'] = f'p={self.ref_noise_prob} t=U[{self.ref_noise_min},{self.ref_noise_max}] timestep=t_ref'
        if self.ref_token_drop_prob > 0:
            metadata['ref_token_drop'] = f'p={self.ref_token_drop_prob} keep=U[{self.ref_token_keep_min},{self.ref_token_keep_max}]'
        if self.ref_aug_prob > 0:
            metadata['ref_aug'] = f'zoom_crop p={self.ref_aug_prob} max={self.ref_aug_max_zoom}'
        return metadata
