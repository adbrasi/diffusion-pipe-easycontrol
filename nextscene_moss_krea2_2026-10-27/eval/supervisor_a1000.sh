#!/bin/bash
# every 250 steps: save_quit -> 6 Turbo test generations -> resume. Then full 45-case eval at 1000.
OUT=/workspace/checkpoints/krea2_A1000recipe_cap1000
TRAIN_CMD="deepspeed --num_gpus=1 train.py --deepspeed --config /workspace/dpk/configs/krea2_A1000_cap1000.toml --resume_from_checkpoint"
train_alive() { pgrep -f "^/workspace/dpk/venv/bin/python -u train.py" >/dev/null; }
gen6() {  # $1 = step
  source /workspace/dpk/venv/bin/activate; cd /workspace/dpk/repo
  python tools/krea2_v3_eval.py --cases /workspace/evals/krea_a1000/cases6.json --out /workspace/evals/krea_a1000/steps --seeds 42 \
    --comfy /workspace/comfy/ComfyUI --python /workspace/dpk/venv/bin/python --models /workspace/models/krea2 \
    --col "label=s$1,adapter=$RUN/step$1/adapter_model.safetensors,strength=1.0" 2>&1 | grep -E "Traceback|Error" | tail -2
  /workspace/venv/bin/python /workspace/evals/summary_krea.py /workspace/evals/krea_a1000/cases6.json /workspace/evals/krea_a1000/steps rawv3,s250,s500,s750,s1000 \
    "Krea 2 receita A1000 — 6 testes a cada 250 passos (Turbo 8 passos, seed 42)" /workspace/outputs_for_user/KREA2_A1000_testes.jpg >/dev/null
  echo "GRID_OK step $1"
}
for N in 250 500 750 1000; do
  until RUN=$(ls -d $OUT/*/ 2>/dev/null | tail -1) && [ -f "$RUN/step$N/adapter_model.safetensors" ]; do sleep 15; done
  RUN=${RUN%/}; echo "[$(date +%H:%M)] step$N salvo"
  if [ $N -lt 1000 ]; then
    touch $RUN/save_quit; while train_alive; do sleep 5; done; echo "[$(date +%H:%M)] pausado em $N"
  else
    while train_alive; do sleep 5; done
  fi
  gen6 $N
  if [ $N -lt 1000 ]; then
    cd /workspace/k2a1000; source /workspace/dpk/venv/bin/activate
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 nohup $TRAIN_CMD >> /workspace/train_krea_A1000.log 2>&1 &
    echo "[$(date +%H:%M)] retomado"
    sleep 60
  fi
done
/workspace/evals/pipeline_a1000.sh
