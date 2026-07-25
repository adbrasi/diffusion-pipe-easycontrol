# Achado: `ref_cfg` 2-3 é onde a IDENTIDADE aparece (2026-07-25)

**Impacto: metodológico.** Toda a bateria até aqui foi avaliada com
`ref_cfg = 1.0`, e isso **subestima todos os braços**.

## Como apareceu

Eu tinha uma previsão falsificável (doc `CFG_REFERENCIA_ANIMA.md`): como o
branch `t` do CFG usa referência zerada, adapters treinados sem
`condition_dropout` teriam esse branch fora da distribuição, e portanto o
dial `ref_cfg` seria errático neles e monotônico nos treinados com dropout.

Rodei o sweep (`tools/refcfg_sweep.sh`, ref_cfg ∈ {0, 0.5, 1, 2, 3}) no
arm1 s1000 (sem dropout) e no armB s1000 (com dropout).

**A previsão FALHOU.** Os dois respondem de forma monotônica e quase
idêntica:

| ref_cfg | arm1 (dist. do ref_cfg=0) | armB (dist. do ref_cfg=0) |
|---|---|---|
| 0.5 | 0.356 | 0.259 |
| 1.0 | 0.444 | 0.434 |
| 2.0 | 0.692 | 0.722 |
| 3.0 | 0.785 | 0.891 |

O branch OOD não quebra o dial na prática. Registrar como hipótese
refutada.

## O que apareceu no lugar (o achado que importa)

Olhando as imagens do sweep (`/workspace/outputs/_comparativos/SWEEP_refcfg_arm1_vs_armB.png`),
no exemplo held-out da elfa (SAO/Leafa):

- **`ref_cfg` 0.0–1.0**: elfa loira genérica, sem adorno de cabelo, roupa
  verde simples. **Em AMBOS os braços.**
- **`ref_cfg` 2.0–3.0**: aparece o **laço branco de cabelo** e os detalhes
  brancos/dourados do figurino — as marcas de identidade específicas da
  imagem de referência. **Em AMBOS os braços.**

Ou seja: a fidelidade de identidade estava lá o tempo todo, mas só se
manifesta em `ref_cfg` alto. Avaliar com `ref_cfg=1.0` mede outra coisa
(clima/paleta), não identidade.

## Consequências

1. **Protocolo corrigido**: `tools/battery_eval.sh` agora gera também uma
   coluna `ref_cfg 2.5`, e o grid a exibe. Todos os braços devem ser
   reavaliados com essa coluna antes de comparação final.
2. **Recomendação de uso**: para transferir identidade de personagem
   (adorno, figurino, marcas específicas), usar `ref_cfg` 2–3, não 1.0.
   Isso é um dial de inferência, não precisa retreinar nada.
3. **Limitação da minha métrica**: a "fidelidade" de
   `tools/battery_metrics.py` mede paleta + estrutura global. Como a tarefa
   é *next scene* (composição muda de propósito), aumentar a fidelidade de
   identidade pode AUMENTAR essa distância. De fato a métrica deu sinal
   oposto ao visual no armB (dist. p/ referência subiu de 1.476 → 1.576
   com ref_cfg, enquanto visualmente a identidade melhorou). **Não usar a
   métrica de fidelidade como critério de identidade.** A de sensibilidade
   continua válida (é uma diferença entre duas gerações, não contra a
   referência).

## Lição de método (segunda vez na mesma sessão)

Fiz uma previsão elegante, ela falhou, e o dado que apareceu no lugar foi
mais útil que a previsão. Vale manter o hábito de rodar o teste mesmo
quando a hipótese parece óbvia — e de olhar as imagens, não só as
métricas: aqui a métrica sozinha teria me levado à conclusão errada.
