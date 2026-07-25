#!/bin/bash
# Gera o conjunto COMPLETO de 10 exemplos para TODOS os braços, no checkpoint
# mais treinado de cada um. Até 2026-07-25 só o armD tinha os 10 — os demais
# tinham 3, o que impede a comparação lado a lado que decide o método.
#
# Saída: /workspace/outputs/_eval10_<arm>/GRID_s<step>_eval10.png
# (pasta separada dos runs de 3 exemplos, para não misturar protocolos:
#  a referência EMBARALHADA do ex3 muda quando o conjunto vai de 3 para 10)
#
# TABELA CRÍTICA — o modo de inferência tem que casar com o contrato do treino:
#   arm1 / arm3 / armB / armD : ic_lora_v3, target-first -> ominicontrol_subject
#   armA                      : idem + LoRA no adaln     -> + skip_adaln
#   armC                      : ic_lora_dual, routing    -> ic_lora_dual
# Errar isso produz ruído puro que PARECE "o método falhou" (já custou caro).
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol
mkdir -p /workspace/.tmp

B=/workspace/checkpoints/battery_2026-07-25
R=/workspace/checkpoints/round2_2026-07-25

# arm | raiz dos checkpoints | modo de inferência | skip_adaln | step forçado
# Step vazio = o mais treinado. O arm1 entra DUAS vezes: s1000 (mais treinado,
# comparação justa com os outros braços de 1000 steps) e s500, que foi o pico
# histórico dele e o checkpoint que o usuário julgou "incrível" na Rodada 1.
ARMS=(
  "arm1_broad_llm_frozen|$B/arm1_broad_llm_frozen|ominicontrol_subject||"
  "arm1_broad_llm_frozen_s500|$B/arm1_broad_llm_frozen|ominicontrol_subject||500"
  "arm3_broad_llm_full|$B/arm3_broad_llm_full|ominicontrol_subject||"
  "armA_adaln_in|$R/armA_adaln_in|ominicontrol_subject|skip_adaln|"
  "armB_dropout|$R/armB_dropout|ominicontrol_subject||"
  "armC_routed|$R/armC_routed|ic_lora_dual||"
  "armD_dropout_2000|$R/armD_dropout_2000|ominicontrol_subject||"
)

for entry in "${ARMS[@]}"; do
  IFS='|' read -r arm root mode skip forced <<< "$entry"
  if [ -n "$forced" ]; then
    step="$forced"
  else
    step=$(find "$root" -name "step*" -type d 2>/dev/null | grep -oP 'step\K[0-9]+$' | sort -n | tail -1)
  fi
  [ -z "$step" ] && { echo "SKIP ${arm} (sem checkpoint)"; continue; }
  ck="$(find "$root" -path "*/step${step}/adapter_model.safetensors" | head -1)"
  outdir="/workspace/outputs/_eval10_${arm}"
  mkdir -p "$outdir"
  echo "=== ${arm} step${step} (mode=${mode} ${skip}) ==="
  MODE="$mode" bash tools/battery_eval.sh "$ck" "s${step}_eval10" "$outdir" "$skip" \
    >> /workspace/.tmp/eval10_${arm}.log 2>&1
  echo "PRONTO ${arm} -> ${outdir}/GRID_s${step}_eval10.png"
done

echo "=== EVAL10 COMPLETO PARA TODOS OS BRAÇOS ==="
