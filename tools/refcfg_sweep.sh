#!/bin/bash
# Testa a hipótese do CFG de referência: se condition_dropout é o que
# torna o branch "sem ref" (ctrl_zero) parte da distribuição treinada,
# então variar --ref_cfg deve ter efeito FORTE e monotônico num adapter
# treinado com dropout, e FRACO/errático num treinado sem dropout.
# Uso: tools/refcfg_sweep.sh <adapter> <label> <out_dir>
set -euo pipefail
ADAPTER="$1"; LABEL="$2"; OUT="$3"; mkdir -p "$OUT"
PY=/workspace/.venv-diffusion-pipe/bin/python
DIT=/workspace/models_anima/split_files/diffusion_models/anima-base-v1.0.safetensors
VAE=/workspace/models/qwen_image_vae.safetensors
LLM=/workspace/models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors
OUTS=/workspace/outputs
NEG="worst quality, low quality, score_1, score_2, score_3, artist name"
REF="$OUTS/image1.webp"
PROMPT="$(cat "$OUTS/image1.txt")"
for rc in 0.0 0.5 1.0 2.0 3.0; do
  f="$OUT/${LABEL}_refcfg${rc}.png"
  [ -f "$f" ] && continue
  $PY infer_easycontrol.py --dit "$DIT" --vae "$VAE" --llm "$LLM" \
    --lora "$ADAPTER" --mode ominicontrol_subject --control_image "$REF" \
    --prompt "$PROMPT" --negative_prompt "$NEG" \
    --width 784 --height 592 --steps 30 --cfg 4.0 --flow_shift 3.0 \
    --lora_strength 1.0 --ref_cfg "$rc" --seed 76 --save_path /tmp/rcs_tmp
  mv /tmp/rcs_tmp/*.png "$f"
done
echo "SWEEP OK: $LABEL"
