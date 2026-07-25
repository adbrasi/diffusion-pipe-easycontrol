# Rodada 2 — Arm D: dropout 0.1 + treino longo (2026-07-25)

> ## ⚠️ VEREDITO REBAIXADO (teste multi-seed, mesmo dia)
> O título original era "Arm D VENCE". Rodei depois um teste multi-seed e a
> diferença **não é estatisticamente conclusiva**. Ver a seção "Teste
> multi-seed" no fim. O armD segue como o candidato mais promissor, mas
> "vence" era forte demais para o dado que eu tinha.

**Isto inverte a conclusão anterior da bateria.** Até aqui o arm1 (sem
dropout, ~500 steps) era o campeão e o dropout parecia apenas "proteger
contra degradação". Com o treino longo completo, o quadro mudou.

## O resultado

Métricas no checkpoint final de cada braço (`tools/battery_metrics.py`):

| braço | step | sensibilidade | fidelidade |
|---|---|---|---|
| **armD** (dropout 0.1, 2000 steps) | s2000 | **0.898** | **0.156** |
| armB (dropout 0.1, 1000 steps) | s1000 | 0.641 | 0.148 |
| armA (adaln dentro) | s1000 | 0.606 | 0.129 |
| arm3 (llm_adapter treinável) | s1000 | 0.597 | 0.107 |
| arm1 (campeão anterior) | s1000 | 0.372 | 0.078 |
| armC (routing) | s1000 | 0.342 | 0.041 |

Para comparação, o melhor checkpoint que o arm1 já produziu foi o s500, com
**0.778 / 0.246**. O armD em s2000 tem sensibilidade **maior que o pico
histórico do arm1** (0.898 vs 0.778), com fidelidade sólida.

Sensibilidade alta com fidelidade alta é o par que importa — sensibilidade
alta sozinha poderia ser só instabilidade, mas a fidelidade de 0.156 é a
maior entre todos os checkpoints finais, então não é ruído.

## Validação visual (o que a métrica não mostra)

Grid: `/workspace/outputs/armD_dropout_2000/GRID_s2000.png`

No exemplo held-out ex2 (elfa/calabouço, imagem que NÃO está no dataset):
- `lora 1.0` — elfa e garoto, composição correta
- `lora 1.0 + ref_cfg 1.75` — **a elfa aparece com o laço branco de cabelo**,
  o marcador de identidade específico da referência, e o figurino verde bate
- `ref EMBARALHADA` — cena completamente diferente (azulada, escura, outra
  composição)

Ou seja: fidelidade real quando a referência é a certa, divergência clara
quando é a errada. É exatamente o comportamento que a bateria inteira estava
tentando obter.

## Por que isso faz sentido

`condition_dropout = 0.1` zera os latentes da referência em 10% dos passos.
Consequência: o modelo vê a referência em apenas 90% do treino, então
**aprende mais devagar** — mas aprende uma solução em que a referência é
genuinamente necessária, porque ele nunca pôde assumir que ela estaria lá.

Sem dropout, o modelo converge rápido para a solução fácil (satisfazer a loss
pelo caption) e depois degrada, porque nada o impede de ir abandonando a
referência. Com dropout, essa rota está bloqueada por construção.

Daí a leitura anterior estar incompleta: a 1000 steps o armB ainda não tinha
convergido, e por isso parecia inferior ao arm1 no pico. A 2000 steps ele
passa.

## Receita vencedora atualizada

`examples/round2_2026-07-25/armD_dropout_2000.toml`

```toml
[model]
type = 'ic_lora_v3'          # escopo largo: self_attn+mlp+cross_attn+llm_adapter
llm_adapter_lr = 0           # congelado (Rodada 1)
sigmoid_scale = 1.0

[ic_lora_full]
ref_first = false            # target-first
condition_dropout = 0.1      # <-- a mudança que decide
condition_timestep = 0.0
shifted_logit_normal = false
include_adaln = false        # adaln fora (Arm A perdeu)

[adapter]
rank = 32

[optimizer]
type = 'adamw_optimi'
lr = 1e-4                    # batch efetivo 8
```

**Treinar 2000 steps, não 500.** Essa é a segunda metade do achado — a
receita com dropout precisa de mais steps para convergir, e parar cedo dá a
impressão errada de que ela é pior.

Inferência: `--mode ominicontrol_subject --lora_strength 1.0 --ref_cfg 1.0`
como padrão; `ref_cfg` 1.75 quando quiser puxar mais identidade.

## A curva completa — e por que ela desqualifica a métrica de fidelidade

Levantei os checkpoints intermediários (só inferência, sem treinar de novo):

| step | sensibilidade | fidelidade | inspeção visual |
|---|---|---|---|
| s500 | 0.685 | **0.226** | **RUIM** — artefato de painel duplicado no ex2 em todas as colunas com referência, imagens escuras a ponto de ficarem ilegíveis, personagens deformadas no ex3 |
| s1000 | 0.525 | 0.172 | — |
| s1500 | 0.446 | 0.123 | — |
| s2000 | **0.898** | 0.156 | **BOM** — limpo, laço da elfa presente, divergência clara na ref embaralhada |

A curva é não-monotônica e a fidelidade tem pico justamente no checkpoint
visualmente pior. Isso não é ruído: é um **defeito da métrica**.

`tools/battery_metrics.py` calcula fidelidade como distância de paleta +
estrutura entre a geração e a imagem de referência. A referência do ex2 é uma
cena escura de calabouço — então **gerações escuras e degradadas pontuam alto
por coincidência**. A métrica recompensou o artefato.

**Consequência prática: a métrica de fidelidade não deve ser usada como
critério.** A de sensibilidade continua útil (é uma diferença entre duas
gerações, não uma comparação contra a referência), mas mesmo ela precisa de
inspeção visual para separar "usa a referência" de "está instável".

Esta é a terceira vez nesta sessão em que a métrica aponta numa direção e a
inspeção visual corrige — as outras duas estão em
`docs/ACHADO_REF_CFG_ALTO.md` e `docs/CFG_REFERENCIA_ANIMA.md`. O padrão é
consistente o bastante para virar regra: **métrica serve para triagem e para
detectar colapso grosseiro; a decisão é visual.**

## Ressalvas

1. **n=1.** Um único treino de 2000 steps, uma seed, três exemplos de
   avaliação. Não replicado.
2. **O teto não foi medido.** 2000 steps foi o limite testado; 3000 pode ser
   melhor. E a curva não-monotônica sugere que o comportamento entre 500 e
   2000 não é estável nem previsível — vale medir mais pontos antes de
   confiar na receita.
3. **O veredito de que o s2000 é o melhor é VISUAL, não métrico.** As
   métricas, tomadas ao pé da letra, escolheriam o s500 — que tem artefatos.
4. **Este veredito é meu, não do usuário.** Ele julgou o arm1 como "incrível"
   olhando os grids; ainda não viu o armD s2000.


---

## Teste multi-seed — o que derruba o veredito

Toda a bateria foi julgada com **uma seed só** (76). Rodei os dois
finalistas em mais duas seeds (1234, 777), medindo sensibilidade:

| braço | média entre seeds | desvio | min / max |
|---|---|---|---|
| arm1 s500 | 0.650 | **0.050** | 0.615 / 0.686 |
| armD s2000 | 0.852 | **0.146** | 0.688 / 0.969 |

diferença armD−arm1 = **+0.201** · desvio combinado = **0.153**

Critério fixado ANTES de rodar: conclusivo se |diferença| > 2×desvio
(0.306). **0.201 < 0.306 → NÃO CONCLUSIVO.**

### O que isso muda

1. **Os números que eu vinha citando eram pontos altos da variação.** O
   0.898 do armD vira 0.969 e 0.688 em outras seeds. O 0.778 do arm1 s500,
   que eu chamava de "pico histórico", vira 0.615 e 0.686.
2. **O armD é ~3× menos estável entre seeds** (desvio 0.146 vs 0.050). Isso
   é informação nova e independente: um adapter cuja qualidade oscila com a
   seed é menos confiável em produção, mesmo com média maior.
3. **A direção continua favorecendo o armD** (0.852 vs 0.650 de média), mas
   agora como tendência, não como fato estabelecido.

### O que faltaria para concluir

Mais seeds (5-10) e/ou mais exemplos de avaliação. Com 3 exemplos × 3 seeds
o erro padrão ainda é grande demais para separar 0.65 de 0.85.

## Nota de método (4ª correção da sessão)

Esta é a quarta vez nesta sessão em que eu afirmo algo e depois o dado
derruba:
1. atribuí a perda de sensibilidade ao branch OOD do CFG — a álgebra mostrou
   que o termo cancela (`docs/CFG_REFERENCIA_ANIMA.md`);
2. previ que o dial `ref_cfg` seria errático sem dropout — o sweep mostrou
   monotônico nos dois (`docs/ACHADO_REF_CFG_ALTO.md`);
3. usei a métrica de fidelidade como critério — ela premiava o checkpoint com
   artefatos (seção acima);
4. declarei o armD vencedor com n=1 — o multi-seed diz não conclusivo.

O padrão é o mesmo: **concluo cedo demais com amostra pequena.** A regra que
sai disso, e que vale para as próximas rodadas: *nenhum veredito de braço sem
pelo menos 3 seeds, e a decisão final continua sendo visual e do usuário.*

---

## Fechamento: 6 seeds, ainda NÃO CONCLUSIVO

Expandi para 6 seeds por braço e troquei a regra improvisada de "2× desvio"
por um **teste de Welch** (apropriado aqui porque as variâncias podem
diferir).

| braço | média | desvio | min / max |
|---|---|---|---|
| arm1 s500 | 0.625 | 0.155 | 0.338 / 0.778 |
| armD s2000 | 0.786 | 0.165 | 0.572 / 0.969 |

diferença = **+0.161** · Welch **t = 1.74** (df ≈ 10) · crítico 5% = **2.23**
→ **NÃO CONCLUSIVO.**

### 5ª correção: o "armD é 3× menos estável" também era artefato

Com 3 seeds eu tinha medido desvio 0.146 (armD) vs 0.050 (arm1) e concluí que
o armD era muito menos estável. Com 6 seeds os desvios ficaram
**praticamente iguais**: 0.165 vs 0.155. Aquele "3×" era ruído de amostra
pequena — exatamente o erro que o teste multi-seed existia para pegar, e que
eu cometi de novo ao interpretar o próprio teste com n=3.

### Quanto faltaria para concluir

- Cohen d = 1.00 (diferença 0.161 / desvio agrupado 0.160)
- Para poder 80% a 5%: **~16 seeds por braço**, faltam ~10
- Custo: ~237 gerações ≈ 47 min de GPU

**Não rodei.** Duas razões: (a) o retorno é baixo — mesmo confirmando, a
diferença é de magnitude modesta e a decisão de produção é visual; (b) há um
caminho mais eficiente, abaixo.

### O caminho mais eficiente é outro

A variância **entre exemplos** (ex1/ex2/ex3 dentro da mesma seed) é maior que
a variância entre seeds. Ex.: arm1 seed 2024 → 1.159 / 0.390 / 0.624 no mesmo
run. Isso significa que **adicionar exemplos held-out reduz o erro padrão
mais rápido do que adicionar seeds**, por geração gasta.

Recomendação para a próxima rodada: subir de 3 para ~10 exemplos de avaliação
held-out antes de aumentar seeds. Isso também torna o julgamento visual mais
confiável, que é o critério que de fato decide.

## Conclusão honesta da Rodada 2

**Com a metodologia atual eu não consigo distinguir arm1 de armD.** A direção
favorece o armD em todas as medições, mas nunca com separação estatística.

O que **sobrevive** ao escrutínio, por ter efeito grande ou por ser
julgamento visual do usuário:
- `llm_adapter` congelado vence treinável — veredito visual do usuário, claro
- adaln fora — artefato de instabilidade observado diretamente
- routing condition-only perde — efeito grande e consistente
- captions exaustivas causam atalho de caption — medido por vários ângulos

O que **não sobrevive**: qualquer ranking fino entre arm1, armB e armD.
