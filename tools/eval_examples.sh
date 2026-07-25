#!/bin/bash
# Conjunto de exemplos de avaliação — v3, expandido para 10.
#
# POR QUE 10 E NÃO 3: o teste multi-seed de 2026-07-25 mostrou que a
# variância ENTRE EXEMPLOS é maior que entre seeds (ex.: arm1 na seed 2024
# deu 1.159 / 0.390 / 0.624 no mesmo run). Logo, adicionar exemplos reduz o
# erro padrão mais rápido, por geração gasta, do que adicionar seeds — e
# melhora o julgamento visual, que é o critério que de fato decide.
# Com 3 exemplos nenhuma diferença fina entre braços era distinguível do
# ruído (Welch t=1.74 com 6 seeds).
#
# HELD-OUT: ex2 e ex3 são imagens do usuário, nunca vistas no treino. Os
# demais usam referências do dataset MAS com prompts NOVOS, escritos aqui —
# então o par (referência, prompt) é inédito, que é o que importa para
# medir uso de referência em vez de memorização.
#
# ESTILO DOS PROMPTS: delta-caption ("the same X, doing Y"), que é o
# contrato que queremos que o adapter aprenda. Nada de descrever aparência
# do que já está na referência.

DS=/workspace/dataset_raw/extracted
OUTS=/workspace/outputs
NEG_DEFAULT="worst quality, low quality, score_1, score_2, score_3, artist name"

declare -A REF=(
  [ex1]="$DS/input_A/imagem000180.jpg"
  [ex2]="$OUTS/image1.webp"
  [ex3]="$OUTS/image2.png"
  [ex4]="$DS/input_A/imagem000297.jpg"
  [ex5]="$DS/input_A/imagem001395.jpg"
  [ex6]="$DS/input_A/imagem001549.jpg"
  [ex7]="$DS/input_A/imagem000409.jpg"
  [ex8]="$DS/input_A/imagem000878.jpg"
  [ex9]="$DS/input_A/imagem000105.jpg"
  [ex10]="$DS/input_A/imagem001063.jpg"
)

declare -A PROMPT=(
  [ex1]="$(cat "$DS/input_B/imagem000180.txt")"
  [ex2]="$(cat "$OUTS/image1.txt")"
  [ex3]="$(cat "$OUTS/image2.txt")"
  # prompts NOVOS, estilo delta — nunca vistos no treino
  [ex4]="the same man sits alone at a small counter, both hands around a cup of coffee, looking down with a tired expression.
Character continuity: same character. Background continuity: new view of the same background."
  [ex5]="the same woman crouches down to pick up a dropped folder, papers scattered on the floor around her.
Character continuity: same character. Background continuity: new view of the same background."
  [ex6]="the same three men stand in a row and bow deeply at the same time, hands at their sides.
Character continuity: same character. Background continuity: new view of the same background."
  [ex7]="the same small creature curls up asleep on a cushion, tail wrapped around itself.
Character continuity: same character. Background continuity: new background."
  [ex8]="the same lion walks slowly across an open plain at dusk, head lowered, seen from the side.
Character continuity: same character. Background continuity: new background."
  [ex9]="the same two characters stand up and face each other, one handing something small to the other.
Character continuity: same character. Background continuity: new view of the same background."
  [ex10]="wide shot of the same street completely empty at dawn, soft light, no characters.
Character continuity: no character. Background continuity: new view of the same background."
)

# resolução ~0.5MP preservando o AR de cada referência
declare -A WH=(
  [ex1]="912 512"  [ex2]="784 592"  [ex3]="912 512"  [ex4]="912 512"
  [ex5]="912 512"  [ex6]="912 512"  [ex7]="912 512"  [ex8]="912 512"
  [ex9]="912 512"  [ex10]="912 512"
)

# ref embaralhada: rotação circular — cada exemplo recebe a referência do seguinte
declare -A SHUFFLED_REF=(
  [ex1]="${REF[ex2]}"  [ex2]="${REF[ex3]}"  [ex3]="${REF[ex4]}"
  [ex4]="${REF[ex5]}"  [ex5]="${REF[ex6]}"  [ex6]="${REF[ex7]}"
  [ex7]="${REF[ex8]}"  [ex8]="${REF[ex9]}"  [ex9]="${REF[ex10]}"
  [ex10]="${REF[ex1]}"
)

# alvo real só existe para os exemplos que vêm de pares do dataset
declare -A TARGET=(
  [ex1]="$DS/input_B/imagem000180.jpg"
)

EXAMPLES="ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10"
