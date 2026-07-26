#!/bin/bash
# SLIDER de timing — modo REFERENCE (Fase 2). Otimizado para VELOCIDADE.
#
# POR QUE ISTO É MUITO MAIS BARATO QUE A FASE 1 (texto):
#   texto:     1 target x 2 multiplicadores x 3 linhas (2 âncoras entram em
#              CADA passe de gradiente) + passe de referência congelado com 5
#              linhas  = 16 s/step medido
#   reference: 2 passes com backward (positivo em +1, negativo em -1), 1 linha
#              cada, SEM âncoras (o trainer as ignora neste modo) e SEM passe
#              de referência congelado — os alvos vêm dos latentes cacheados
#
# ALAVANCAS DE VELOCIDADE, em ordem de impacto:
#   1. blocks_to_swap BAIXO — swap é tráfego CPU<->GPU e foi o que dominou na
#      Fase 1. Aqui o custo de ativação é menor (1 linha por passe), então cabe
#      mais modelo na GPU.
#   2. SEM --gradient_checkpointing — recompute custa ~30% de tempo. Reference
#      mode tem ativações pequenas, então provavelmente cabe. Se der OOM,
#      ligar de volta (GC=1 abaixo).
#   3. --sdpa — único backend disponível (flash/sage/xformers não instalados;
#      compilar levaria mais tempo que economizaria). No torch 2.13 o SDPA já
#      escolhe o kernel flash quando a atenção não tem máscara.
#   4. data loader persistente com workers.
#
# --ltx2_first_frame_conditioning_p "${FFC:-0.9}": ANCORA O FRAME 0 como condicionamento
# e o exclui da loss. O doc descreve exatamente este caso — "pares que
# compartilham o mesmo frame inicial e diferem principalmente em movimento" —
# e é o regime de i2v, que é o uso real (~100% das vezes).
set -uo pipefail
cd /workspace/musubi

GC="${GC:-1}"          # SEMPRE 1: libera VRAM sem custar trafego CPU<->GPU
SWAP="${SWAP:-0}"      # MEDIDO: 0 -> 2.21 s/step | 20 -> 8.4 s/step. NUNCA subir sem medir OOM.
STEPS="${STEPS:-600}"
DIM="${DIM:-16}"
OUT=/workspace/outputs/slider_ref
mkdir -p "$OUT"

EXTRA=""
[ "$GC" = "1" ] && EXTRA="--gradient_checkpointing"

/workspace/.venv-musubi/bin/accelerate launch \
  --num_cpu_threads_per_process 8 --mixed_precision bf16 \
  ltx2_train_slider.py \
  --mixed_precision bf16 \
  --ltx2_checkpoint /workspace/models_ltx2/ltx-2.3-22b-dev.safetensors \
  --fp8_base --fp8_scaled \
  --sdpa \
  --blocks_to_swap "$SWAP" \
  $EXTRA \
  --max_data_loader_n_workers 4 --persistent_data_loader_workers \
  --network_module networks.lora_ltx2 \
  --network_dim "$DIM" --network_alpha "$DIM" \
  --lora_target_preset video_sa_ca_ff \
  --learning_rate 1e-4 \
  --optimizer_type AdamW8bit \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 \
  --max_train_steps "$STEPS" \
  --ltx2_first_frame_conditioning_p "${FFC:-0.9}" \
  --output_dir "$OUT" --output_name anime_timing_ref \
  --slider_config anime_timing_slider_ref.toml \
  --save_every_n_steps 150 \
  --gemma_root /workspace/models_ltx2/gemma --gemma_load_in_4bit \
  --use_precached_sample_prompts --use_precached_sample_latents \
  --sample_prompts_cache /workspace/datasets/animateka/cache_pos/ltx2_sample_prompts_cache.pt \
  --sample_latents_cache /workspace/datasets/animateka/cache_pos/ltx2_sample_latents_cache.pt \
  --sample_prompts slider_sample_prompts.txt \
  --sample_every_n_steps "${SAMPLE_EVERY:-200}" \
  --logging_dir "$OUT/logs" \
  "$@"
