#!/bin/bash
# Treinos completos 5k: ic_lora_v3 -> ominicontrol_broad, com uploader
# incremental em paralelo (cada step* novo sobe pro HF assim que estabiliza).
set -u
cd /workspace/diffusion-pipe-easycontrol
source .venv/bin/activate
LOG=/workspace/logs/anima_probes

python /workspace/scripts/anima_full5k_uploader.py > $LOG/full5k_uploader.log 2>&1 &
UPLOADER_PID=$!
echo "[full5k] uploader pid=$UPLOADER_PID"

for arm in v3 omini_broad; do
  echo "[full5k] treinando $arm (5000 steps)..."
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed \
    --config /workspace/configs/anima_${arm}_full5k.toml > $LOG/full5k_${arm}.log 2>&1
  ec=$?
  echo "${arm} EXIT: $ec" >> $LOG/full5k_${arm}.log
  [ $ec -eq 0 ] || { echo "[full5k] ABORT: $arm falhou (exit $ec)"; kill $UPLOADER_PID 2>/dev/null; exit 1; }
done

echo "[full5k] treinos concluidos; aguardando uploader drenar (5 min)..."
sleep 300
kill $UPLOADER_PID 2>/dev/null
echo "[full5k] COMPLETO"
