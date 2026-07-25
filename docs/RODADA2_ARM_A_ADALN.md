# Rodada 2 — Arm A: adaln dentro do LoRA (2026-07-25, madrugada)

**Veredito: Claude, não o usuário** (usuário foi dormir, pediu pra eu
continuar as rodadas sozinho — este veredito precisa de revisão humana
quando ele acordar). **Arm 1 (adaln fora) continua campeão.**

## O teste

`examples/round2_2026-07-25/armA_adaln_in.toml` — idêntico ao Arm 1
vencedor da Rodada 1, exceto `include_adaln = true` (LoRA também em
`adaln_modulation`). Inferência sempre com `--skip_adaln` (adaln treina
como absorvedor de erro, descartado ao gerar — ver
`docs/BATERIA_2026-07-25_LLM_ADAPTER.md`). Mesmo protocolo: grid de 3
exemplos (1 do dataset + 2 held-out) × 6 colunas, checkpoints 250-1000.

## Observações por checkpoint

- **step250**: ex2 ("ref força 1.0" e "1.5") mostrou artefato de painel
  duplicado/dividido — mais cedo que qualquer braço da Rodada 1 (Arm1 só
  mostrou isso no step750; Arm3 no step500). Bate com a previsão
  arquitetural: LoRA em `adaln_modulation` é um SEGUNDO ajuste de baixo
  rank empilhado sobre a correção `adaln_lora` já embutida no
  Cosmos-Predict2 — mecanicamente menos estável (ver prova na memória do
  projeto / commits anteriores da bateria).
- **step500**: artefato sumiu, geração normal. Boa sensibilidade a
  referência em ex2/ex3.
- **step750-1000**: qualidade e comportamento de referência comparáveis
  ao Arm 1 no mesmo checkpoint — mesmo padrão (ex2 mantém diferença
  correto-vs-embaralhado, ex3 quase fecha o gap). Artefato não voltou.

## Por que Arm 1 continua vencendo

Arm A não mostrou ganho claro sobre o Arm 1 em nenhum checkpoint — a
qualidade final é comparável, mas o Arm A teve um episódio de instabilidade
que o Arm 1 não teve. Sem benefício visível que compense o risco, o
princípio da navalha de Occam favorece manter adaln fora (mais simples,
sem o artefato, mesmo resultado final). Também bate com a prova
arquitetural: adaln é conteúdo-cego por construção (só função do
timestep), então treiná-lo só pode ajudar como válvula de otimização — não
deveria mudar fidelidade de referência de forma significativa, e os dados
confirmam isso (nenhuma diferença de fidelidade perceptível, só o
artefato).

**Isto precisa de confirmação humana.** Diferente da Rodada 1 (onde o
usuário viu os grids e decidiu "Arm 1 é muito melhor" com clareza), aqui a
diferença é mais sutil e a decisão foi tomada por mim sozinho. Os grids
completos estão em `/workspace/outputs/round2_2026-07-25/armA_adaln_in/` —
vale revisar quando o usuário acordar, especialmente se ele discordar do
padrão que venho seguindo até aqui.

## Próximo

Arm B (`condition_dropout` 0.0→0.1) sobe agora com `include_adaln = false`
(config do Arm 1), único eixo variando.
