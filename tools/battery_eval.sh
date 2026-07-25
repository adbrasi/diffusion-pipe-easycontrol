#!/bin/bash
# Avaliação rápida de um checkpoint da bateria 2026-07-25.
# Uso: tools/battery_eval.sh <adapter_model.safetensors_or_dir> <label> <out_dir>
#
# Gera, para os 3 pares fixos de referência (mesma seed em todos os
# checkpoints/braços para comparação justa): sem referência, com referência
# (força 1.0), com referência (força 1.5) e com referência EMBARALHADA
# (mesmo caption, ref de outro exemplo — separa memorização de uso real).
# Depois monta um grid. Resolução mais alta (v2, pedido do usuário) pra dar
# pra julgar rosto/detalhe de verdade, não só no thumbnail do grid.
set -euo pipefail

ADAPTER="$1"
LABEL="$2"
OUT="$3"
mkdir -p "$OUT"

PY=/workspace/.venv-diffusion-pipe/bin/python
DIT=/workspace/models_anima/split_files/diffusion_models/anima-base-v1.0.safetensors
VAE=/workspace/models/qwen_image_vae.safetensors
LLM=/workspace/models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors
DS=/workspace/dataset_raw/extracted
OUTS=/workspace/outputs
NEG="worst quality, low quality, score_1, score_2, score_3, artist name"
SEED=76
STEPS=30
CFG=4.0
SHIFT=3.0

# 3 exemplos fixos v2: mantém o Demon Slayer (dataset) + os 2 exemplos
# curados pelo usuário (elf/dungeon, floresta noturna). Resolução ~0.5-0.6MP
# preservando o AR original de cada um (antes era 688x384, pequeno demais).
declare -A REF=(
  [ex1]="$DS/input_A/imagem000180.jpg"
  [ex2]="$OUTS/image1.webp"
  [ex3]="$OUTS/image2.png"
)
declare -A PROMPT=(
  [ex1]="$(cat "$DS/input_B/imagem000180.txt")"
  [ex2]="$(cat "$OUTS/image1.txt")"
  [ex3]="$(cat "$OUTS/image2.txt")"
)
declare -A WH=(
  [ex1]="912 512"
  [ex2]="784 592"
  [ex3]="912 512"
)
# ref embaralhada: mesmo caption, referência de OUTRO exemplo (rotação).
# Se a saída não mudar em relação a ref1.0, é atalho de caption; se mudar
# refletindo a ref errada, é uso genuíno da referência.
declare -A SHUFFLED_REF=(
  [ex1]="${REF[ex2]}"
  [ex2]="${REF[ex3]}"
  [ex3]="${REF[ex1]}"
)

for ex in ex1 ex2 ex3; do
  read -r W H <<< "${WH[$ex]}"
  ref_img="${REF[$ex]}"
  prompt="${PROMPT[$ex]}"

  # sem referência (LoRA mergeado, mas sem control_image -> cai em sample_normal)
  f="$OUT/${LABEL}_${ex}_noref.png"
  if [ ! -f "$f" ]; then
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode ominicontrol_subject \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --seed "$SEED" --save_path /tmp/battery_tmp_noref
    mv /tmp/battery_tmp_noref/*.png "$f"
  fi

  # com referência, força 1.0
  f="$OUT/${LABEL}_${ex}_ref1.0.png"
  if [ ! -f "$f" ]; then
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode ominicontrol_subject --control_image "$ref_img" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --ref_cfg 1.0 --seed "$SEED" --save_path /tmp/battery_tmp_ref1
    mv /tmp/battery_tmp_ref1/*.png "$f"
  fi

  # com referência, força 1.5
  f="$OUT/${LABEL}_${ex}_ref1.5.png"
  if [ ! -f "$f" ]; then
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode ominicontrol_subject --control_image "$ref_img" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.5 --ref_cfg 1.0 --seed "$SEED" --save_path /tmp/battery_tmp_ref15
    mv /tmp/battery_tmp_ref15/*.png "$f"
  fi

  # ref embaralhada (mesmo caption, referência de outro exemplo), força 1.0
  f="$OUT/${LABEL}_${ex}_refshuffle.png"
  if [ ! -f "$f" ]; then
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode ominicontrol_subject --control_image "${SHUFFLED_REF[$ex]}" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --ref_cfg 1.0 --seed "$SEED" --save_path /tmp/battery_tmp_refshuffle
    mv /tmp/battery_tmp_refshuffle/*.png "$f"
  fi
done

$PY tools/battery_grid_assemble.py "$LABEL" "$OUT"
echo "GRID pronto: $OUT/GRID_${LABEL}.png"
