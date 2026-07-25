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

import torch
from einops import rearrange

import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding

from models.base import make_contiguous
from models.krea2_edit import Krea2EditPipeline
from models.krea2_omini_grounded import Krea2OminiGroundedPipeline
from models.krea2_ominicontrol import Krea2OminiTransformerLayer
from models.krea2_reference import Krea2ReferenceFinalLayer, Krea2ReferenceInitialLayer
from utils.common import AUTOCAST_DTYPE, is_main_process


class Krea2MultiRefInitialLayer(Krea2ReferenceInitialLayer):
    """Igual ao pai, mas com N spans de referência e offset por slot."""

    def __init__(self, *args, slot_axis='width', **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_axis = slot_axis

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        target, timesteps, context, text_attention_mask, reference = inputs
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

        context = self.txtfusion(context, mask=None)
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
        valid_keys = torch.cat([text_attention_mask, image_mask], dim=1)
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
        else:
            attention_mask = valid_keys[:, None, None, :]

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
        if is_main_process():
            print(f'[{self.name}] max_refs={self.max_refs} slot_axis={self.slot_axis} '
                  f'position_mode={self.position_mode}')

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
        super().configure_adapter(adapter_config)
        if not self.txtfusion_rank or adapter_config['type'] != 'lora':
            return
        pattern = {
            name: self.txtfusion_rank
            for name in self.peft_config.target_modules
            if 'txtfusion' in name
        }
        if not pattern:
            raise RuntimeError('rank do txtfusion pedido mas nenhum alvo txtfusion encontrado')
        self.peft_config.rank_pattern = pattern
        self.peft_config.alpha_pattern = dict(pattern)
        # reconstrói com o padrão aplicado (o get_peft_model do pai já rodou
        # com rank uniforme; PEFT não permite mutar depois)
        import peft
        self.lora_model = peft.get_peft_model(self.diffusion_model, self.peft_config)
        if is_main_process():
            print(f'[{self.name}] rank {self.txtfusion_rank} em {len(pattern)} linears do txtfusion')
            self.lora_model.print_trainable_parameters()
        for name, parameter in self.diffusion_model.named_parameters():
            parameter.original_name = name

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
        return metadata
