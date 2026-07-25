#!/bin/bash
# Orquestra a Rodada 2 (torneio: cada arm compara contra o campeão atual).
# Uso: tools/round2_orchestrate.sh <logfile_treino> <config_dir/arm_name> <output_subdir> [skip_adaln]
# Espera o treino (já rodando em processo separado) terminar, avalia os 4
# checkpoints com o mesmo protocolo da Rodada 1.
#
# ⚠️ O MODO DE INFERÊNCIA PRECISA CASAR COM O CONTRATO DO TREINO.
# Passar via env MODE=... (default: ominicontrol_subject). Errar isso não
# gera "resultado ruim" — gera RUÍDO PURO, e é fácil confundir com o
# método tendo falhado. Aconteceu com o armC em 2026-07-25.
#
#   type = ic_lora_v3   (LoRA global, target-first)  -> ominicontrol_subject
#   type = ic_lora_dual (routing condition-only)     -> ic_lora_dual
#   ref_first = true    (ordem [ref|alvo])           -> ic_lora_full
#
# Motivo: adapters com routing condition-only NÃO podem ter o LoRA
# fundido nos pesos — o delta tem que ser aplicado mascarado em runtime,
# só nas rows da referência. O loader padrão mergeia em todas as rows e
# destrói o adapter. Isso já estava documentado em
# docs/OMINI_CONTROL_KREA2.md (lição do Krea 2) e eu repeti o erro.
#
# SEMPRE fazer um smoke de 1 imagem antes de rodar a bateria inteira.
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol
CKPT_STEPS="${CKPT_STEPS:-250 500 750 1000}"

LOGFILE="$1"
CKPT_ROOT="$2"
ARM_NAME="$3"
SKIP_ADALN="${4:-}"

wait_for_training() {
  local logfile="$1"
  local target_step="${TARGET_STEP:-1000}"
  # espera o processo de treino APARECER antes de vigiar (evita corrida em
  # que o orquestrador sobe antes do train.py existir e conclui na hora)
  local waited=0
  while ! pgrep -f "train.py --local_rank=0" > /dev/null; do
    sleep 5; waited=$((waited+5))
    if [ "$waited" -ge 180 ]; then
      echo "MILESTONE: FALHA — treino nunca apareceu em 180s"; return 1
    fi
  done
  while true; do
    if grep -q "steps: ${target_step} " "$logfile" 2>/dev/null; then
      sleep 20
      return 0
    fi
    if ! pgrep -f "train.py --local_rank=0" > /dev/null; then
      if grep -q "steps: ${target_step} " "$logfile" 2>/dev/null; then
        return 0
      fi
      echo "MILESTONE: FALHA — treino morreu antes de completar (ver $logfile)"
      return 1
    fi
    sleep 15
  done
}

eval_checkpoint() {
  local root="$1" arm="$2" step="$3" skip="$4"
  local found outdir
  found=$(find "$root" -path "*/step${step}/adapter_model.safetensors" 2>/dev/null | head -1)
  if [ -z "$found" ]; then
    echo "MILESTONE: ${arm} step${step} SEM CHECKPOINT (pulei)"
    return
  fi
  outdir="/workspace/outputs/${arm}"
  mkdir -p "$outdir"
  bash tools/battery_eval.sh "$found" "s${step}" "$outdir" "$skip" \
    >> /workspace/.tmp/eval_round2_${arm}_s${step}.log 2>&1
  echo "MILESTONE: GRID PRONTO ${arm} step${step} -> ${outdir}/GRID_s${step}.png"
}

echo "MILESTONE: aguardando treino de ${ARM_NAME} terminar (step 1000)"
wait_for_training "$LOGFILE"
echo "MILESTONE: ${ARM_NAME} treino completo, avaliando checkpoints"
for s in $CKPT_STEPS; do
  eval_checkpoint "$CKPT_ROOT" "$ARM_NAME" "$s" "$SKIP_ADALN"
done
echo "MILESTONE: ${ARM_NAME} RODADA 2 AVALIACAO COMPLETA -> /workspace/outputs/${ARM_NAME}/"
