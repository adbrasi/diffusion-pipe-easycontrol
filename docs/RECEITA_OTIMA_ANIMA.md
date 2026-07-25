# Receita ótima Anima — estado em 2026-07-25 (fim da Rodada 2 parcial)

**Melhor configuração encontrada: `arm1` (ic_lora_v3, llm_adapter
congelado, SEM dropout), checkpoint ~500.**

> ### ⚠️ Correção do usuário (2026-07-25)
> A primeira versão deste doc recomendava inferir com `ref_cfg 2.5`. O
> usuário corrigiu, com razão: **o critério de avaliação tem que ser
> `lora_strength 1.0` / `ref_cfg 1.0`.** Se um adapter só fica bom com
> `ref_cfg` alto, ele não está bom o suficiente — o `ref_cfg` é um dial de
> ajuste fino, não uma muleta para compensar treino fraco.
>
> O achado do `ref_cfg` alto continua válido como *fato observado* (a
> identidade aparece mais forte em 2-3), mas não deve ser usado para
> julgar braços nem como recomendação padrão. O protocolo de avaliação foi
> ajustado: a coluna de julgamento é `lora 1.0`, com uma coluna extra
> `ref_cfg 1.75` só para ver o headroom do dial.

Confirma o veredito original do usuário ("arm 1 é incrível") e adiciona um
ajuste que a bateria descobriu: o checkpoint certo é o ~500, não o 1000.

## Treino

`examples/battery_2026-07-25/arm1_broad_llm_frozen.toml`

```toml
[model]
type = 'ic_lora_v3'        # escopo largo: self_attn+mlp+cross_attn+llm_adapter
llm_adapter_lr = 0         # CONGELADO (Rodada 1: vence com folga)
sigmoid_scale = 1.0

[ic_lora_full]
ref_first = false          # target-first [alvo T=0 | ref T=1]
condition_dropout = 0.0    # ver ressalva abaixo
condition_timestep = 0.0
shifted_logit_normal = false
include_adaln = false      # adaln FORA (Rodada 2 Arm A perdeu)

[adapter]
rank = 32                  # alpha = rank

[optimizer]
type = 'adamw_optimi'
lr = 1e-4                  # batch efetivo 8 (micro 1 x accum 8)
```

**Parar em ~500 steps.** A métrica de sensibilidade à referência tem pico
em s500 (.778) e cai pela metade em s1000 (.372). Treinar até 1000 neste
dataset degrada o uso da referência.

## Inferência

```bash
python infer_easycontrol.py --mode ominicontrol_subject \
  --lora <checkpoint step500>/adapter_model.safetensors \
  --control_image <referencia> --prompt "<caption>" \
  --steps 30 --cfg 4.0 --flow_shift 3.0 \
  --lora_strength 1.0 --ref_cfg 1.0 --seed <n>
```

**Padrão: `lora_strength 1.0` e `ref_cfg 1.0`.** É assim que o adapter
tem que funcionar bem — esse é o critério.

`ref_cfg` é um dial de ajuste fino disponível se você quiser puxar mais
identidade numa geração específica (em 1.75-2.5 as marcas de figurino e
adornos ficam mais fortes, ver `docs/ACHADO_REF_CFG_ALTO.md`). Mas subir o
`ref_cfg` tem custo: em valores altos a composição da referência começa a
sobrepor o prompt (ver tabela abaixo). Não usar como padrão nem como
critério de avaliação.

Caption: manter o sufixo do schema de abril na última linha —
`Character continuity: same character. Background continuity: new view of
the same background.`

## O trade-off do condition_dropout (por que 0.0 na receita)

O Arm B (`condition_dropout = 0.1`) tem uma propriedade genuinamente
melhor: **não colapsa em treino longo**. A sensibilidade à referência do
arm1 despenca de .778 (s500) para .372 (s1000), enquanto o armB fica
estável (.641 em s1000).

**Mas** em `ref_cfg` alto o dropout vira sobre-dependência. No exemplo
held-out ex3 (prompt pede "upper body shot" de duas garotas, referência é
uma paisagem de floresta ampla e vazia):

| config | resultado |
|---|---|
| arm1 s500 @ ref_cfg 1.0 | garotas grandes, fundo genérico (sem a árvore) |
| **arm1 s500 @ ref_cfg 2.5** | **garotas grandes E a floresta/lua/árvores da referência** ✅ |
| armB s1000 @ ref_cfg 1.0 | garotas grandes, fundo com árvore |
| armB s1000 @ ref_cfg 2.5 | **colapso: garotas viram pontinhos, cena ≈ cópia da referência** ❌ |

Comparativo: `/workspace/outputs/_comparativos/COMPARE_refcfg_sobrecopia_ex3.png`.

Ou seja: dropout + ref_cfg alto = copia a composição da referência e
ignora o prompt. Sem dropout + ref_cfg alto = pega o cenário mantendo a
obediência ao prompt.

**Quando o dropout ainda pode valer:** se o plano for treinar bem mais que
500-1000 steps (dataset maior, mais épocas), a proteção contra o colapso
pode compensar — usando `ref_cfg` mais baixo (1.0-1.5) na inferência. O
Arm D (dropout, 2000 steps) está testando exatamente isso.

## Hipóteses testadas e REFUTADAS (para não repetir)

- **Treinar o llm_adapter ajuda** (a "eureka" de julho): refutado na
  Rodada 1 — o Arm 3 perdeu com clareza. `llm_adapter_lr = 0` é o certo,
  batendo com a recomendação oficial do criador do Anima.
- **adaln dentro do LoRA ajuda** (o "para-raios" de abril): refutado no
  Arm A — sem ganho e com artefato de instabilidade transitório.
- **O branch OOD do CFG explica a perda de sensibilidade**: refutado por
  álgebra — o termo cancela. Ver `docs/CFG_REFERENCIA_ANIMA.md`.
- **O dial `ref_cfg` seria errático sem dropout**: refutado pelo sweep —
  monotônico nos dois braços.

## Pendências

- Arm C (routing condition-only) — treinando.
- Arm D (dropout 2000 steps) — treinando; decide se o dropout vale para
  treinos longos.
- Reavaliar Arm C/D com a coluna `ref_cfg 2.5` (protocolo já corrigido).
- Nunca testado: `ref_first = true` vs target-first; rank 64 vs 32.
