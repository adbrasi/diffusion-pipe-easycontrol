#!/bin/bash
# Orquestra a bateria 2026-07-25 v2: re-roda a avaliação do arm1 (já
# treinado) com os exemplos/resolução novos, em pasta própria, e faz o
# mesmo para arm3 assim que cada checkpoint dele sair (treino do arm3 já
# está rodando em processo separado, este script só espera e avalia).
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol
CKPT_STEPS="250 500 750 1000"

wait_for_training() {
  local logfile="$1"
  local target_step="$2"
  while true; do
    if grep -q "steps: ${target_step} " "$logfile" 2>/dev/null; then
      sleep 20  # buffer para o save do checkpoint final terminar de gravar
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
  local root="$1" arm="$2" step="$3"
  local found outdir
  found=$(find "$root" -path "*/step${step}/adapter_model.safetensors" 2>/dev/null | head -1)
  if [ -z "$found" ]; then
    echo "MILESTONE: ${arm} step${step} SEM CHECKPOINT (pulei)"
    return
  fi
  outdir="/workspace/outputs/battery_2026-07-25/${arm}"
  mkdir -p "$outdir"
  bash tools/battery_eval.sh "$found" "s${step}" "$outdir" \
    >> /workspace/.tmp/eval_${arm}_s${step}.log 2>&1
  echo "MILESTONE: GRID PRONTO ${arm} step${step} -> ${outdir}/GRID_s${step}.png"
}

echo "MILESTONE: re-rodando avaliacao do arm1 (ja treinado) com exemplos/resolucao v2"
for s in $CKPT_STEPS; do
  eval_checkpoint /workspace/checkpoints/battery_2026-07-25/arm1_broad_llm_frozen arm1_broad_llm_frozen "$s"
done

echo "MILESTONE: aguardando arm3 (treino ja rodando em paralelo) terminar (step 1000)"
wait_for_training /workspace/.tmp/train_arm3.log 1000
echo "MILESTONE: arm3 treino completo, avaliando checkpoints"
for s in $CKPT_STEPS; do
  eval_checkpoint /workspace/checkpoints/battery_2026-07-25/arm3_broad_llm_full arm3_broad_llm_full "$s"
done

echo "MILESTONE: RODADA 1 COMPLETA (arm1 vs arm3, 2 braços x 4 checkpoints, v2) — grids em /workspace/outputs/battery_2026-07-25/<arm>/"
