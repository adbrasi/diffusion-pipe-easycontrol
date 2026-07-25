#!/bin/bash
# TESTE DE ORDEM — o único que separa endereçamento de mistura.
#
# Para cada exemplo com 2 referências de papéis CLARAMENTE distintos
# (pessoa vs objeto, objeto vs cenário), gera duas vezes com a mesma seed e
# o mesmo prompt, mudando SÓ a ordem em que as referências são passadas:
#
#   A) --reference ref_1 --reference ref_2   (a ordem do treino)
#   B) --reference ref_2 --reference ref_1   (trocada)
#
# Se A e B saírem iguais, o modelo NÃO está endereçando — está misturando as
# duas referências num caldo só, e '<image 1>' não significa nada para ele.
# Se saírem diferentes, e A fizer sentido com o prompt, o binding pegou.
#
# É o teste de referência embaralhada da bateria do Anima, adaptado: lá se
# trocava a referência por outra, aqui se troca a ORDEM entre duas válidas.
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol

CONFIG="${CONFIG:-examples/macro_multiref/quick50.toml}"
ADAPTER="$1"
OUT="${2:-/workspace/outputs/macro_multiref}"
SEED="${SEED:-76}"
STEPS="${STEPS:-28}"
DS=/workspace/datasets/macro50
PY=/workspace/.venv-diffusion-pipe/bin/python

mkdir -p "$OUT"

# exemplos escolhidos por terem 2 refs de papéis inconfundíveis
for ex in 0000012 0000004 0000019 0000000; do
  prompt="$(cat $DS/target/${ex}.txt)"
  r1="$DS/refs/${ex}_1.jpg"
  r2="$DS/refs/${ex}_2.jpg"
  [ -f "$r1" ] && [ -f "$r2" ] || { echo "pulando $ex (refs faltando)"; continue; }

  echo "=== $ex — ordem CORRETA ==="
  $PY tools/infer_reference_adapter.py --config "$CONFIG" --adapter "$ADAPTER" \
    --reference "$r1" --reference "$r2" \
    --prompt "$prompt" --seed "$SEED" --steps "$STEPS" \
    --width 512 --height 512 --output "$OUT/${ex}_A_ordem_correta.png" 2>&1 | tail -3

  echo "=== $ex — ordem TROCADA ==="
  $PY tools/infer_reference_adapter.py --config "$CONFIG" --adapter "$ADAPTER" \
    --reference "$r2" --reference "$r1" \
    --prompt "$prompt" --seed "$SEED" --steps "$STEPS" \
    --width 512 --height 512 --output "$OUT/${ex}_B_ordem_trocada.png" 2>&1 | tail -3
done

$PY tools/multiref_grid.py "$OUT" "$DS"
echo "GRID: $OUT/GRID_ordem.png"
