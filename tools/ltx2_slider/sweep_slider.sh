#!/bin/bash
# Sweep do slider de timing: mesma seed, mesmo prompt, mesma imagem inicial,
# variando SÓ o multiplicador da LoRA. É o teste que isola o efeito do eixo.
#
# Roda em I2V (--reference_image), que é o uso real — e não em t2v, que é o
# regime em que o slider foi TREINADO. Essa diferença é justamente o que
# precisa ser verificado: o modo texto treina com latentes de ruído puro, sem
# frame condicionante, então a transferência para i2v é hipótese, não fato.
#
# Uso: sweep_slider.sh <lora.safetensors> <imagem_inicial.png> [saida]
set -uo pipefail
cd /workspace/musubi

LORA="$1"
IMG="$2"
OUT="${3:-/workspace/outputs/slider_anime/sweep_$(basename "$LORA" .safetensors)}"
PROMPT="${PROMPT:-A young anime woman with a purple-to-crimson gradient bob haircut and a floral off-shoulder sundress walks slowly toward the camera down a narrow suburban Japanese alley at sunset. Her hair sways with each step and she tilts her head slightly as her smile widens. Warm golden-pink light rakes across the concrete walls and utility poles behind her. Hand-drawn 2D anime animation, cel shading, static camera.}"
SEED="${SEED:-76}"
PY=/workspace/.venv-musubi/bin/python

mkdir -p "$OUT"

for M in ${MULTS:--2.0 2.0}; do
  nome="mult${M}"
  [ -f "$OUT/${nome}.mp4" ] && continue
  echo "=== multiplicador $M ==="
  $PY ltx2_generate_video.py \
    --ltx2_checkpoint /workspace/models_ltx2/ltx-2.3-22b-dev.safetensors \
    --gemma_root /workspace/models_ltx2/gemma --gemma_load_in_8bit \
    --fp8_base --fp8_scaled \
    --blocks_to_swap "${SWAP:-32}" \
    --sample_with_offloading \
    --sample_vae_tile_size 512 --sample_vae_temporal_tile_size 16 \
    --ltx2_mode video \
    --reference_image "$IMG" \
    --lora_weight "$LORA" --lora_multiplier "$M" \
    --prompt "$PROMPT" \
    --frame_rate 25 \
    --seed "$SEED" \
    --output_dir "$OUT" --output_name "$nome" \
    2>&1 | grep -E "Saved|saved|rror|OOM" | head -3
done

echo "PRONTO: $OUT"
