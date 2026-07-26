#!/bin/bash
# Avaliação HELD-OUT do adapter multi-referência do Krea 2.
#
# POR QUE ANIME: o Macro é quase todo fotográfico, e o alvo real deste
# projeto é anime. Estas referências vêm do dataset do Anima — domínio
# completamente FORA da distribuição de treino do adapter, e nenhuma delas
# foi vista em nenhum step. É o teste mais duro disponível: se o
# endereçamento "image 1"/"image 2" só funcionasse por memorização do Macro,
# ele quebraria aqui.
#
# Os prompts seguem o registro das captions de treino
# ("Generate an image of the X from image 1 ... the Y from image 2 ...")
# para não introduzir uma segunda variável.
#

# Uso: macro_eval.sh <dir_do_checkpoint> [dir_de_saida]
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol

ADAPTER="$1"
STEP="$(basename "$ADAPTER")"
OUT="${2:-/workspace/outputs/macro_multiref/eval_${STEP}}"
CONFIG="${CONFIG:-examples/macro_multiref/run2_multiref.toml}"
SEED="${SEED:-76}"
STEPS="${STEPS:-28}"
PY=/workspace/.venv-diffusion-pipe/bin/python
A=/workspace/dataset_raw/extracted/input_A

mkdir -p "$OUT"

# nome | ref1 | ref2 | prompt (só sobre image 1 e image 2)
EXEMPLOS=(
"an1|$A/imagem000297.jpg|$A/imagem000409.jpg|Generate an image of the man with glasses from image 1 holding the small blue creature from image 2 in his arms, smiling at it."
"an2|$A/imagem001395.jpg|$A/imagem000878.jpg|Generate an image of the woman from image 1 standing outdoors next to the lion from image 2, looking at it calmly."
"an3|$A/imagem000409.jpg|$A/imagem000105.jpg|Generate an image of the small blue creature from image 1 sitting on a desk inside the room from image 2."
"an4|$A/imagem001549.jpg|$A/imagem001063.jpg|Generate an image of the young man from image 1 walking alone through the street scene from image 2 at night."
"an5|$A/imagem000180.jpg|$A/imagem001395.jpg|Generate an image of the character from image 1 standing in the hallway from image 2, seen from behind."
"an6|$A/imagem000105.jpg|$A/imagem000297.jpg|Generate an image featuring the two characters from image 1 on the left side and the man with glasses from image 2 on the right side, standing together outdoors."
)

for entry in "${EXEMPLOS[@]}"; do
  IFS='|' read -r nome r1 r2 prompt <<< "$entry"
  [ -f "$r1" ] && [ -f "$r2" ] || { echo "pulando $nome (refs faltando)"; continue; }
  # Só a ordem correta. A ordem trocada já cumpriu o papel no step 50: provou
  # que os spans não são permutation-invariant (nos exemplos com posição
  # explícita, trocar a ordem trocava quem estava de cada lado). Manter isso
  # em toda avaliação dobra o tempo de geração sem informação nova — melhor
  # gastar o dobro em mais exemplos.
  destino="$OUT/${nome}_A_correta.png"
  [ -f "$destino" ] && continue
  $PY tools/infer_reference_adapter.py --config "$CONFIG" --adapter "$ADAPTER" \
    --reference "$r1" --reference "$r2" --prompt "$prompt" \
    --seed "$SEED" --steps "$STEPS" --width 512 --height 512 \
    --output "$destino" 2>&1 | grep -E "^Saved|Error|Traceback"
done

$PY tools/macro_eval_grid.py "$OUT" "$STEP"
