#!/bin/bash
# SLIDER de densidade temporal de animação — LTX-2.3, modo text-only.
#
# DUAS DECISÕES QUE NÃO SÃO DEFAULT E IMPORTAM:
#
# 1. --latent_frames 7 (o default é 1 = IMAGEM).
#    Um slider sobre TIMING treinado em frame único seria vazio — não existe
#    eixo temporal para ele afetar. O VAE codifica o primeiro frame sozinho e
#    depois em grupos de 8 (pixels = 8k+1 -> latentes = k+1), então 7 latentes
#    = 49 frames = ~2s a 25fps (o target_fps padrão). A 25fps, "on twos"
#    (frame segurado 2x) e "on threes" (3x) são ambos representáveis.
#
# 2. --lora_target_preset video_sa_ca_ff, NÃO t2v.
#    O doc avisa (linha 1190): o preset t2v cria pesos de LoRA para as
#    camadas de áudio e cross-modais. Sem dado de áudio no treino — e um
#    slider de texto não tem áudio nenhum — esses pesos são inicializados e
#    nunca recebem gradiente com sentido, e aplicar a LoRA sobrescreve as
#    camadas de áudio com deltas quase-zero, degradando o áudio do modelo
#    base. Os presets video_* restringem a LoRA ao ramo de vídeo.
set -uo pipefail
cd /workspace/musubi

LTX=/workspace/models_ltx2/ltx-2.3-22b-dev.safetensors
GEMMA=/workspace/models_ltx2/gemma
OUT=/workspace/outputs/slider_anime
mkdir -p "$OUT"

/workspace/.venv-musubi/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  ltx2_train_slider.py \
  --mixed_precision bf16 \
  --ltx2_checkpoint "$LTX" \
  --gemma_root "$GEMMA" \
  --gemma_load_in_8bit \
  --fp8_base --fp8_scaled \
  --gradient_checkpointing \
  --blocks_to_swap "${SWAP:-6}" \
  --network_module networks.lora_ltx2 \
  --network_dim "${DIM:-16}" --network_alpha "${DIM:-16}" \
  --lora_target_preset video_sa_ca_ff \
  --learning_rate 1e-4 \
  --optimizer_type AdamW8bit \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 \
  --max_train_steps "${STEPS:-400}" \
  --output_dir "$OUT" --output_name anime_timing_slider \
  --slider_config anime_animation_slider.toml \
  --latent_frames 7 \
  --latent_height 512 --latent_width 768 \
  --save_every_n_steps 100 \
  --logging_dir "$OUT/logs" \
  "$@"
