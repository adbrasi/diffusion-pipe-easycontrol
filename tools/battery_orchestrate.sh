#!/bin/bash
# Orquestra a bateria 2026-07-25 (Rodada 1, cortada para 2 braços por pedido
# do usuário: só congelado vs lr cheio, sem o meio-termo). Espera arm1 (já
# rodando) terminar, avalia seus 4 checkpoints, sobe arm3 direto, avalia.
set -uo pipefail
cd /home/claude/diffusion-pipe-easycontrol
DS=/workspace/.venv-diffusion-pipe/bin/deepspeed
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
  local found
  found=$(find "$root" -path "*/step${step}/adapter_model.safetensors" 2>/dev/null | head -1)
  if [ -z "$found" ]; then
    echo "MILESTONE: ${arm} step${step} SEM CHECKPOINT (pulei)"
    return
  fi
  bash tools/battery_eval.sh "$found" "${arm}_s${step}" /workspace/outputs/battery_2026-07-25 \
    >> /workspace/.tmp/eval_${arm}_s${step}.log 2>&1
  echo "MILESTONE: GRID PRONTO ${arm} step${step} -> /workspace/outputs/battery_2026-07-25/GRID_${arm}_s${step}.png"
}

echo "MILESTONE: aguardando arm1 (ja rodando) terminar (step 1000)"
wait_for_training /workspace/.tmp/train_arm1.log 1000
echo "MILESTONE: arm1 treino completo, avaliando checkpoints"
for s in $CKPT_STEPS; do
  eval_checkpoint /workspace/checkpoints/battery_2026-07-25/arm1_broad_llm_frozen arm1 "$s"
done

echo "MILESTONE: arm2 (lr baixo) CORTADO por pedido do usuario -- indo direto pro arm3"

echo "MILESTONE: subindo arm3 (llm_adapter_lr=lr base, o fix da eureka)"
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $DS --num_gpus=1 train.py --deepspeed \
  --config examples/battery_2026-07-25/arm3_broad_llm_full.toml \
  > /workspace/.tmp/train_arm3.log 2>&1
wait_for_training /workspace/.tmp/train_arm3.log 1000
echo "MILESTONE: arm3 treino completo, avaliando checkpoints"
for s in $CKPT_STEPS; do
  eval_checkpoint /workspace/checkpoints/battery_2026-07-25/arm3_broad_llm_full arm3 "$s"
done

echo "MILESTONE: RODADA 1 COMPLETA (arm1 vs arm3, 2 braços x 4 checkpoints) — grids em /workspace/outputs/battery_2026-07-25/"
