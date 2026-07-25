# Rodada 2 — Arm B: condition_dropout 0.0 → 0.1 (2026-07-25)

**Veredito: Claude (usuário dormindo — precisa revisão humana).
ARM B VENCE o Arm 1. `condition_dropout = 0.1` entra na receita campeã.**

Este é o achado mais forte da Rodada 2 até agora, e resolve exatamente o
problema que eu vinha suspeitando desde a Rodada 1.

## A hipótese testada

Na Rodada 1, os dois braços mostravam o mesmo padrão: a diferença entre
"referência correta" e "referência embaralhada" era forte em 250-500 steps
e ia enfraquecendo até o step 1000. Minha hipótese era que
`condition_dropout = 0.0` (a receita "abril fiel") nunca treina o modelo
para lidar com ausência/imprecisão de referência — então o caminho fácil
(satisfazer a loss só com o caption) vai vencendo o caminho difícil (ler a
referência via self-attention) conforme o treino avança.

`condition_dropout = 0.1` zera os latents da referência em 10% dos passos,
forçando o modelo a nunca poder assumir que a referência está lá.

## O resultado (comparativo direto, step 1000)

Grid comparativo gerado em `/workspace/outputs/COMPARE_arm1_vs_armB_s1000.png`
(colunas: referência correta vs referência EMBARALHADA, mesmo caption/seed):

| Braço | ex2 (elf/dungeon, held-out) | ex3 (floresta, held-out) |
|---|---|---|
| **Arm 1** (dropout 0.0) | correta ≈ embaralhada — quase idênticas: mesma composição, mesma paleta quente, personagens na mesma posição | correta ≈ embaralhada — praticamente idênticas: mesma pose, fundo, iluminação |
| **Arm B** (dropout 0.1) | correta (marrom/quente, garoto à esquerda) vs embaralhada (**azul frio, composição trocada**) — divergência forte | correta (árvore grande, lua brilhante à esquerda) vs embaralhada (**pedras, lua menor à direita, cores outras**) — divergência forte |

Em outras palavras: **no step 1000 o Arm 1 basicamente ignora qual
referência recebe; o Arm B claramente responde a ela.**

Bônus não previsto: o Arm B também é mais **fiel à iluminação** da
referência. A referência do ex2 é uma cena escura de calabouço — o Arm B
mantém a cena escura, enquanto o Arm 1 clareia demais. Ou seja, o dropout
não só preserva a dependência da referência, como melhora a fidelidade
tonal a ela.

## Trade-off honesto

As gerações do Arm B tendem a ser mais escuras/contrastadas que as do
Arm 1. Isso *parece* ser fidelidade à referência (que é escura), não um
defeito — mas é uma diferença estética perceptível, e o usuário pode ter
preferência própria aqui. **Vale ele olhar o comparativo antes de bater o
martelo**, especialmente porque na Rodada 1 ele julgou pela estética geral
("Arm 1 é incrível"), não só pelo teste de referência embaralhada.

## Receita campeã atualizada

```toml
[model]
type = 'ic_lora_v3'
llm_adapter_lr = 0          # Rodada 1: congelado vence

[ic_lora_full]
ref_first = false           # target-first
condition_dropout = 0.1     # <-- NOVO (Arm B): era 0.0
condition_timestep = 0.0
shifted_logit_normal = false
include_adaln = false       # Rodada 2 Arm A: adaln fica fora

[adapter]
rank = 32
```

Config: `examples/round2_2026-07-25/armB_dropout.toml`.

## Próximo

Arm C (routing condition-only, `ic_lora_dual`) sobe com esta base
(dropout 0.1 herdado, adaln fora, llm_adapter congelado). Depois disso, as
hipóteses da lista original acabam — hipóteses novas a considerar:
- **sweep de dropout** (0.05 / 0.2 / 0.3): se 0.1 já ajuda tanto, onde é o
  ótimo? Provável próximo teste de maior valor.
- ref_first vs target-first (nunca testado nesta bateria).
- rank 64 vs 32 no escopo largo.
- treino mais longo (2000+ steps) com dropout, pra ver se a dependência da
  referência se mantém além do que testamos.
