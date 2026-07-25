#!/bin/bash
# Gera UM grid por método, usando o checkpoint MAIS TREINADO de cada um.
# Protocolo v3: sem ref | lora 1.0 (CRITERIO) | lora 1.0 + ref_cfg 1.75 | ref EMBARALHADA
#
# TABELA CRÍTICA — o modo de inferência tem que casar com o contrato do treino:
#   arm1 / arm3 / armB / armD : ic_lora_v3, target-first -> ominicontrol_subject
#   armA                      : idem + LoRA no adaln     -> + skip_adaln
#   armC                      : ic_lora_dual, routing    -> ic_lora_dual
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol

B=/workspace/checkpoints/battery_2026-07-25
R=/workspace/checkpoints/round2_2026-07-25

# arm | raiz dos checkpoints | modo de inferência | skip_adaln
ARMS=(
  "arm1_broad_llm_frozen|$B/arm1_broad_llm_frozen|ominicontrol_subject|"
  "arm3_broad_llm_full|$B/arm3_broad_llm_full|ominicontrol_subject|"
  "armA_adaln_in|$R/armA_adaln_in|ominicontrol_subject|skip_adaln"
  "armB_dropout|$R/armB_dropout|ominicontrol_subject|"
  "armC_routed|$R/armC_routed|ic_lora_dual|"
  "armD_dropout_2000|$R/armD_dropout_2000|ominicontrol_subject|"
)

for entry in "${ARMS[@]}"; do
  IFS='|' read -r arm root mode skip <<< "$entry"
  # checkpoint mais treinado disponível
  step=$(find "$root" -name "step*" -type d 2>/dev/null | grep -oP 'step\K[0-9]+$' | sort -n | tail -1)
  [ -z "$step" ] && { echo "SKIP ${arm} (sem checkpoint)"; continue; }
  ck="$(find "$root" -path "*/step${step}/adapter_model.safetensors" | head -1)"
  outdir="/workspace/outputs/${arm}"
  mkdir -p "$outdir"
  MODE="$mode" bash tools/battery_eval.sh "$ck" "s${step}" "$outdir" "$skip" \
    >> /workspace/.tmp/regen_${arm}.log 2>&1
  echo "PRONTO ${arm} (step${step}) -> ${outdir}/GRID_s${step}.png"
done

echo "=== TODOS OS GRIDS PRONTOS ==="
