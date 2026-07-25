#!/bin/bash
# Avaliação rápida de um checkpoint da bateria 2026-07-25.
# Uso: tools/battery_eval.sh <adapter_model.safetensors_or_dir> <label> <out_dir>
#
# Gera, para os 3 pares fixos de referência (mesma seed em todos os
# checkpoints/braços para comparação justa): sem referência, com referência
# (força 1.0) e com referência (força 1.5). Depois monta um grid.
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
NEG="worst quality, low quality, score_1, score_2, score_3, artist name"
SEED=76
STEPS=30
CFG=4.0
SHIFT=3.0

# 3 exemplos fixos (mesmos em toda a bateria), com bucket ~512px area
# preservando o AR original de cada par.
declare -A REF=(
  [ex1]="$DS/input_A/imagem000180.jpg"
  [ex2]="$DS/input_A/imagem001129.jpg"
  [ex3]="$DS/input_A/imagem001549.jpg"
)
declare -A PROMPT=(
  [ex1]="$(cat "$DS/input_B/imagem000180.txt")"
  [ex2]="$(cat "$DS/input_B/imagem001129.txt")"
  [ex3]="$(cat "$DS/input_B/imagem001549.txt")"
)
declare -A WH=(
  [ex1]="688 384"
  [ex2]="688 384"
  [ex3]="688 384"
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
done

$PY tools/battery_grid_assemble.py "$LABEL" "$OUT"
echo "GRID pronto: $OUT/GRID_${LABEL}.png"
