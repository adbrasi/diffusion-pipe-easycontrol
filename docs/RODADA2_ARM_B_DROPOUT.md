# Rodada 2 — Arm B: condition_dropout 0.0 → 0.1 (2026-07-25)

**Veredito: Claude (usuário dormindo — precisa revisão humana).**

> ## ⚠️ CORREÇÃO IMPORTANTE (adicionada depois, ver §"Correção" no fim)
> A primeira versão deste doc dizia "ARM B VENCE o Arm 1", cravado só na
> comparação visual do step 1000. Depois criei métricas objetivas
> (`tools/battery_metrics.py`) e o quadro é mais nuançado:
> **o Arm 1 tem o MELHOR PICO (steps 500-750), mas colapsa no step 1000;
> o Arm B é mais ESTÁVEL (não colapsa), mas seu pico é mais baixo.**
> Ou seja: qual vence depende de quanto tempo você treina. Ver §Correção.

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

Grid comparativo gerado em `/workspace/outputs/_comparativos/COMPARE_arm1_vs_armB_s1000.png`
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

## Correção — o que as métricas objetivas mostraram

Depois de escrever o veredito acima só com base visual, implementei
`tools/battery_metrics.py` com duas métricas complementares:

- **sensibilidade** = distância(geração com ref correta, geração com ref
  embaralhada). Maior = trocar a ref muda mais a saída = usa a ref.
- **fidelidade** = quanto a ref correta APROXIMA a geração da própria
  imagem de referência, comparado à geração sem ref. Positivo = a ref puxa
  na direção dela (controle contra "sensibilidade por instabilidade").

Resultado (média dos 3 exemplos, seed única):

| braço | s250 | s500 | s750 | s1000 |
|---|---|---|---|---|
| **arm1** (dropout 0.0) | .711 / .217 | **.778 / .246** | .771 / .204 | **.372 / .078** ⟵ colapso |
| arm3 (llm_adapter treinável) | .743 / .176 | .588 / .147 | .544 / .112 | .597 / .107 |
| armA (adaln dentro) | .477 / .088 | .418 / .155 | .495 / .192 | .606 / .129 |
| **armB** (dropout 0.1) | .574 / .157 | .581 / .113 | .390 / .097 | **.641 / .148** ⟵ estável |

*(formato: sensibilidade / fidelidade)*

**Validação da métrica:** ela reproduz os julgamentos humanos já
conhecidos — o usuário disse "Arm 1 é muito melhor que o Arm 3" e a
fidelidade média confirma (arm1 .186 vs arm3 .135); o usuário elogiou
especificamente o arm1 s750, que a métrica coloca entre os melhores
(.771/.204). Isso dá confiança de que ela mede o que se propõe.

**Leitura corrigida:**
1. O colapso do Arm 1 no s1000 é REAL e forte (queda de ~2x na
   sensibilidade, com os 3 exemplos caindo juntos — sinal consistente).
2. O dropout 0.1 de fato PREVINE esse colapso (armB s1000 = .641 vs arm1
   s1000 = .372).
3. **Mas** o Arm 1 nos checkpoints 500-750 é melhor que o Arm B em
   qualquer checkpoint. O dropout não "melhora o teto" — ele **protege
   contra a degradação em treino longo**.

**Recomendação prática revista:** para 1000 steps neste dataset, a melhor
configuração testada é **Arm 1 com checkpoint 500-750** (não o 1000!). O
`condition_dropout=0.1` é a escolha certa se o plano for treinar mais
longo — o que leva direto ao próximo teste.

**Ressalva de rigor:** cada ponto é a média de apenas 3 exemplos com 1
seed, e a variância entre exemplos é alta (ex.: arm1 s500 → ex1=.541,
ex2=.811, ex3=.982). Diferenças pequenas na tabela NÃO são conclusivas;
só os efeitos grandes (o colapso do arm1 s1000; armB s1000 > arm1 s1000)
são robustos o bastante para sustentar conclusão.

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
