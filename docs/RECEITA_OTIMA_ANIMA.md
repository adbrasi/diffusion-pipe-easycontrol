# Receita ótima Anima — estado em 2026-07-25 (fim da Rodada 2 parcial)

**Melhor configuração encontrada: `arm1` (ic_lora_v3, llm_adapter
congelado, SEM dropout) no checkpoint 500, inferido com `ref_cfg 2.5`.**

Confirma o veredito original do usuário ("arm 1 é incrível") e adiciona
dois ajustes que a bateria descobriu: o checkpoint certo é o 500 (não o
1000) e o dial de inferência certo é `ref_cfg` 2-3 (não 1.0).

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
  --lora_strength 1.0 --ref_cfg 2.5 --seed <n>
```

**`ref_cfg 2.5` é o achado que mais mudou o resultado prático.** Com
`ref_cfg 1.0` (o que a bateria usava até então) a geração pega clima e
paleta da referência, mas NÃO a identidade. Com 2-3 aparecem as marcas
específicas: adorno de cabelo, detalhes de figurino, elementos do cenário.
Ver `docs/ACHADO_REF_CFG_ALTO.md`.

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
