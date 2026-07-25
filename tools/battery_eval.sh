#!/bin/bash
# Avaliação rápida de um checkpoint da bateria 2026-07-25.
# Uso: tools/battery_eval.sh <adapter_model.safetensors_or_dir> <label> <out_dir> [skip_adaln]
# Passe "skip_adaln" como 4o argumento para braços que treinaram LoRA em
# adaln_modulation (ex.: Rodada 2 Arm A) — o adaln é descartado na
# inferência por design (absorvedor de erro, ver docs da bateria).
#
# Colunas (padrão v3, definido pelo usuário em 2026-07-25):
#   1. sem ref          — LoRA aplicado, sem imagem de referência (baseline)
#   2. lora 1.0         — CRITÉRIO DE AVALIAÇÃO. É aqui que o braço é
#                         julgado. Se o adapter só fica bom com ref_cfg
#                         alto, ele NÃO está bom o suficiente.
#   3. lora 1.0 + ref_cfg 1.75 — headroom do dial (substituiu a antiga
#                         coluna lora_strength 1.5)
#   4. ref EMBARALHADA  — mesmo caption, referência de outro exemplo.
#                         Separa memorização de uso real da referência.
# Mesma seed em todos os checkpoints/braços para comparação justa.
set -euo pipefail

ADAPTER="$1"
LABEL="$2"
OUT="$3"
SKIP_ADALN_FLAG=""
if [ "${4:-}" = "skip_adaln" ]; then
  SKIP_ADALN_FLAG="--skip_adaln"
fi
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
# MODO de inferência: precisa CASAR com a ordem do concat usada no treino.
#   ref_first=false (target-first) -> ominicontrol_subject  [alvo | ref]
#   ref_first=true  (ref-first)    -> ic_lora_full          [ref | alvo]
# Passar via env: MODE=ic_lora_full tools/battery_eval.sh ...
MODE="${MODE:-ominicontrol_subject}"

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
    rm -rf /tmp/battery_tmp_$$; mkdir -p /tmp/battery_tmp_$$
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode "$MODE" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --seed "$SEED" $SKIP_ADALN_FLAG --save_path /tmp/battery_tmp_$$
    mv /tmp/battery_tmp_$$/*.png "$f"
  fi

  # com referência, força 1.0
  f="$OUT/${LABEL}_${ex}_ref1.0.png"
  if [ ! -f "$f" ]; then
    rm -rf /tmp/battery_tmp_$$; mkdir -p /tmp/battery_tmp_$$
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode "$MODE" --control_image "$ref_img" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --ref_cfg 1.0 --seed "$SEED" $SKIP_ADALN_FLAG --save_path /tmp/battery_tmp_$$
    mv /tmp/battery_tmp_$$/*.png "$f"
  fi

  # lora 1.0 + ref_cfg 1.75 (padrão definido pelo usuário 2026-07-25):
  # substitui a antiga coluna lora_strength 1.5. O CRITÉRIO de avaliação
  # continua sendo a coluna lora 1.0 / ref_cfg 1.0 — se o adapter só fica
  # bom com ref_cfg alto, ele não está bom o suficiente. Esta coluna é
  # para ver o headroom do dial, não para julgar o braço.
  f="$OUT/${LABEL}_${ex}_refcfg1.75.png"
  if [ ! -f "$f" ]; then
    rm -rf /tmp/battery_tmp_$$; mkdir -p /tmp/battery_tmp_$$
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode "$MODE" --control_image "$ref_img" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --ref_cfg 1.75 --seed "$SEED" $SKIP_ADALN_FLAG --save_path /tmp/battery_tmp_$$
    mv /tmp/battery_tmp_$$/*.png "$f"
  fi

  # ref embaralhada (mesmo caption, referência de outro exemplo), força 1.0
  f="$OUT/${LABEL}_${ex}_refshuffle.png"
  if [ ! -f "$f" ]; then
    rm -rf /tmp/battery_tmp_$$; mkdir -p /tmp/battery_tmp_$$
    $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
      --lora "$ADAPTER" --mode "$MODE" --control_image "${SHUFFLED_REF[$ex]}" \
      --prompt "$prompt" --negative_prompt "$NEG" \
      --width "$W" --height "$H" --steps "$STEPS" --cfg "$CFG" --flow_shift "$SHIFT" \
      --lora_strength 1.0 --ref_cfg 1.0 --seed "$SEED" $SKIP_ADALN_FLAG --save_path /tmp/battery_tmp_$$
    mv /tmp/battery_tmp_$$/*.png "$f"
  fi
done

$PY tools/battery_grid_assemble.py "$LABEL" "$OUT"
echo "GRID pronto: $OUT/GRID_${LABEL}.png"
