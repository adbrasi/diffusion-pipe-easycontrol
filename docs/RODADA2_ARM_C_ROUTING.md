# Rodada 2 — Arm C: routing condition-only (2026-07-25)

**Veredito: Claude (usuário acompanhando por cima; precisa revisão).
ARM C PERDE para o arm1, de forma consistente nos 4 checkpoints.**

## O que era

`models/ic_lora_dual.py::ICLoraDualPipeline` (`type = 'ic_lora_dual'`):
o LoRA do canal de APARÊNCIA (self_attn + mlp dos blocks) é roteado para
agir **só nas rows da referência** — o alvo passa pelo caminho base
congelado (zero drift). cross_attn e llm_adapter continuam globais. É o
análogo Anima do omini-grounded que funcionou muito bem no Krea 2.

Config: `examples/round2_2026-07-25/armC_routed.toml`.
Herdou `condition_dropout = 0.1` do armB e `llm_adapter_lr = 0` do arm1.

**Inferência:** exige `MODE=ic_lora_dual` (LoRA aplicado mascarado em
runtime). Ver seção de armadilha abaixo.

## Resultado

| step | arm1 (campeão) | armC (routing) |
|---|---|---|
| s250 | **.711 / .217** | .264 / .043 |
| s500 | **.778 / .246** | .438 / .099 |
| s750 | **.771 / .204** | .405 / .138 |
| s1000 | .372 / .078 | .342 / .041 |

*(sensibilidade / fidelidade — ver `tools/battery_metrics.py`)*

Sensibilidade à referência ~2× menor que o arm1 nos checkpoints úteis
(250-750). Bate com a inspeção visual: nos grids, a coluna "ref
EMBARALHADA" fica quase idêntica à "lora 1.0", ou seja, trocar a
referência por outra muda pouco a saída.

Qualidade de imagem em si é boa — não há degradação. O problema é
especificamente **usar a referência**.

## Por que (leitura mecânica)

No routing condition-only, o delta do LoRA de aparência só é somado nas
rows da referência. Consequência: o modelo aprende a **apresentar** a
referência de forma diferente (mudar seu K/V), mas os pesos que o **alvo**
usa para LER essa informação nunca são modificados — o alvo roda sempre no
caminho base congelado. A capacidade de leitura não é treinada.

No Krea 2 o método funcionou porque havia um canal semântico forte em
paralelo: o grounding Qwen3-VL adaptado por LoRA global no TextFusion
(~1% da energia, mas fazendo o trabalho de resolver entidades — ver
`docs/OMINI_GROUNDED_SEGREDO.md`). No Anima o análogo desse canal seria o
`llm_adapter`, que nesta receita está **congelado** (`llm_adapter_lr = 0`,
a escolha vencedora da Rodada 1). Então o armC roda com meio canal
semântico: só cross_attn.

**Hipótese não testada:** routing + llm_adapter treinável poderia ser o
análogo completo do omini-grounded. Mas o arm3 mostrou que destravar o
llm_adapter sozinho já piora, então a combinação é especulativa e não
óbvia. Não testar sem uma razão melhor.

## Armadilha encontrada (custou 2 avaliações perdidas)

A primeira avaliação do armC saiu **ruído puro** nos 3 exemplos. Não era o
método: eu usei `--mode ominicontrol_subject`, que **funde** o LoRA em
todos os pesos. Um adapter com routing condition-only precisa do delta
aplicado **mascarado em runtime** — fundir aplica o delta em todas as rows
e destrói o adapter.

Isso já estava documentado em `docs/OMINI_CONTROL_KREA2.md` ("LoRA
condition-only não pode ser fundido nos pesos") e eu li esse doc no início
da sessão e repeti o erro. Mitigação: `tools/round2_orchestrate.sh` agora
tem no cabeçalho a tabela contrato-de-treino → modo-de-inferência, e a
regra de **sempre fazer smoke de 1 imagem antes da bateria inteira**.

Segundo bug na sequência: PNGs órfãos em `/tmp/battery_tmp_*` (de
avaliações que matei no meio) faziam o glob casar múltiplos arquivos e o
`mv` falhar silenciosamente após a 1ª imagem. Corrigido com tmp dir
isolado por PID.

## Placar da Rodada 2

| braço | eixo testado | veredito |
|---|---|---|
| **arm1** | (baseline) | 🏆 campeão |
| arm3 | llm_adapter treinável | ❌ perdeu (usuário: "praticamente não pega a referência") |
| armA | adaln dentro do LoRA | ❌ perdeu (sem ganho + artefato transitório) |
| armB | condition_dropout 0.1 | ⚖️ misto (só ganha em treino longo; sobre-copia em ref_cfg alto) |
| armC | routing condition-only | ❌ perdeu (sensibilidade ~2× menor) |
| armD | dropout + 2000 steps | ⏳ treinando |
| armE | ref_first = true | 📋 config pronta, não iniciada (decisão do usuário) |

Convergência interessante: o método vencedor continua sendo o mais
**simples** — LoRA global de escopo largo, target-first, adaln fora,
llm_adapter congelado, sem dropout, parado em ~500 steps. Cada tentativa
de sofisticar (routing, adaln, treinar o canal semântico) piorou.
