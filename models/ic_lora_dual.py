"""Braços de ESCOPO LARGO para Anima (a arquitetura de abril + fixes de julho).

Arqueologia 2026-07-18: os adapters de abril que o usuário lembra como muito
melhores treinavam LoRA também em cross_attn e llm_adapter (o canal semântico).
O fix de maio (97fc94c/f85e808) cortou esses alvos junto com o adaln — o adaln
tinha mecanismo de defeito comprovado (double-LoRA sobre o adaln interno), mas
cross_attn/llm_adapter foram excluídos por precaução, não por evidência.

Três pipelines:
- ic_lora_v3        : abril fiel — escopo largo GLOBAL, target-first
                      [alvo T=0 | ref T=1] (contrato do V1 que funcionou).
- ominicontrol_broad: mesmo contrato com a cara do omini (use condition_dropout
                      0.1 na config para o A/B contra o v3 sem dropout).
- ic_lora_dual      : escopo largo + lição do Krea 2 — canal de APARÊNCIA
                      routado (self_attn+mlp, delta só nas rows da ref, zero
                      drift do target) + canal SEMÂNTICO global (cross_attn +
                      llm_adapter), o análogo exato do blocks-routado+txtfusion
                      do omini-grounded.

adaln_modulation continua FORA por padrão (include_adaln=true na seção do
modelo para o abril literal — não recomendado; em abril ele era descartado na
inferência via --skip_adaln de qualquer forma).
"""

import peft
import torch.nn as nn

from models.ic_lora_v2 import ICLoraV2Pipeline
from models.ic_lora_routed import ICLoraRoutedPipeline, AnimaConditionRouter
from utils.common import is_main_process


def configure_anima_broad_adapter(pipeline, adapter_config, log_tag,
                                  include_adaln=False, include_cross_attn=True):
    """LoRA em self_attn + mlp (+cross_attn, +llm_adapter, +adaln opcionais).

    include_cross_attn=False + include_adaln=True = o contrato do v2 de abril
    ('raiz_iclora'): adaln treina como absorvedor de erro e é SKIPADO na
    inferência (workflow do usuário usava skip_adaln ativo).
    """
    target_linear_modules = set()
    for name, module in pipeline.transformer.named_modules():
        if module.__class__.__name__ not in pipeline.adapter_target_modules:
            continue
        for full_submodule_name, submodule in module.named_modules(prefix=name):
            if not isinstance(submodule, nn.Linear):
                continue
            parts = full_submodule_name.split('.')
            if not include_adaln and any(p.startswith('adaln_modulation') for p in parts):
                continue
            if not include_cross_attn and 'cross_attn' in parts:
                continue
            target_linear_modules.add(full_submodule_name)
    target_linear_modules = list(target_linear_modules)

    n_cross = sum('cross_attn' in m for m in target_linear_modules)
    n_llm = sum(m.startswith('llm_adapter') for m in target_linear_modules)
    n_adaln = sum('adaln_modulation' in m for m in target_linear_modules)
    if is_main_process():
        print(f'[{log_tag}] BROAD LoRA targets: {len(target_linear_modules)} linears '
              f'({n_cross} cross_attn, {n_llm} llm_adapter, {n_adaln} adaln)')
    assert n_llm > 0, f'[{log_tag}] escopo largo esperava llm_adapter nos alvos'
    if include_cross_attn:
        assert n_cross > 0, f'[{log_tag}] esperava cross_attn nos alvos'
    if include_adaln:
        assert n_adaln > 0, f'[{log_tag}] esperava adaln nos alvos'

    peft_config = peft.LoraConfig(
        r=adapter_config['rank'],
        lora_alpha=adapter_config['alpha'],
        lora_dropout=adapter_config['dropout'],
        bias='none',
        target_modules=target_linear_modules,
    )
    pipeline.peft_config = peft_config
    pipeline.lora_model = peft.get_peft_model(pipeline.transformer, peft_config)
    if is_main_process():
        pipeline.lora_model.print_trainable_parameters()
    for name, p in pipeline.transformer.named_parameters():
        p.original_name = name
        if p.requires_grad:
            p.data = p.data.to(adapter_config['dtype'])


class ICLoraV3Pipeline(ICLoraV2Pipeline):
    """Abril reborn: escopo largo global. Use ref_first=false na config para o
    contrato target-first do V1 original."""

    adapter_log_tag = 'IC-LoRA V3 (broad)'
    # adaln continua auditado no save; cross_attn/llm_adapter são legítimos aqui.
    forbidden_adapter_key_patterns = ('adaln_modulation',)

    def __init__(self, config):
        super().__init__(config)
        self.include_adaln = bool(config.get('ic_lora_full', {}).get('include_adaln', False))
        if self.include_adaln:
            self.forbidden_adapter_key_patterns = ()

    def configure_adapter(self, adapter_config):
        configure_anima_broad_adapter(self, adapter_config, self.adapter_log_tag,
                                      include_adaln=self.include_adaln)


class OminiControlBroadPipeline(ICLoraV3Pipeline):
    """Mesmo contrato matemático (target-first no Anima == omini subject);
    existe como tipo separado para o A/B de condition_dropout do abril-omini."""

    adapter_log_tag = 'OminiControl BROAD'


class RaizICLoraPipeline(ICLoraV2Pipeline):
    """RAIZ IC-LORA — réplica do anima_ic_lora_v2_next_scene_v2_s1950_r64
    (14/abr, o adapter que o usuário considera o melhor de todos):
    ref_first [ref T=0 | alvo T=1], LoRA em self_attn+mlp+ADALN+llm_adapter
    (SEM cross_attn), shifted logit-normal, rank 64.
    Na INFERÊNCIA o adaln deve ser SKIPADO (--skip_adaln / node descarta):
    ele treina como absorvedor de erro e é jogado fora na hora de gerar."""

    adapter_log_tag = 'RAIZ IC-LORA'
    forbidden_adapter_key_patterns = ('cross_attn',)  # adaln é intencional aqui

    def configure_adapter(self, adapter_config):
        configure_anima_broad_adapter(self, adapter_config, self.adapter_log_tag,
                                      include_adaln=True, include_cross_attn=False)


class ICLoraDualPipeline(ICLoraRoutedPipeline):
    """Dual-channel (lição do Krea 2): aparência routada + semântica global."""

    adapter_log_tag = 'IC-LoRA DUAL'
    forbidden_adapter_key_patterns = ('adaln_modulation',)

    def __init__(self, config):
        super().__init__(config)
        self.include_adaln = bool(config.get('ic_lora_full', {}).get('include_adaln', False))
        if self.include_adaln:
            self.forbidden_adapter_key_patterns = ()

    def configure_adapter(self, adapter_config):
        configure_anima_broad_adapter(self, adapter_config, self.adapter_log_tag,
                                      include_adaln=self.include_adaln)
        if self.condition_only_lora:
            # Routing SÓ no canal de aparência (self_attn+mlp dos blocks).
            # cross_attn (mesmo dentro dos blocks) e llm_adapter ficam globais.
            installed = self.condition_router.install(
                self.transformer.blocks, exclude_name_parts=('cross_attn',),
            )
            print(f'[{self.adapter_log_tag}] routing em {installed} linears de aparência '
                  f'(ref_first={self.ref_first}); cross_attn/llm_adapter globais')
