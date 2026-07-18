#!/bin/bash
# Corrente autônoma dos braços controlados (usuário ausente):
# espera o routed_targetfirst -> treina routed_reffirst -> treina global_targetfirst
# -> smoke da avaliação -> avaliação completa -> upload HF.
set -u
cd /workspace/diffusion-pipe-easycontrol
source .venv/bin/activate
LOG=/workspace/logs/anima_probes

echo "[chain] esperando routed_targetfirst terminar..."
until grep -q "ROUTEDTF EXIT" $LOG/routed_tf1500.log 2>/dev/null; do sleep 30; done
grep -q "ROUTEDTF EXIT: 0" $LOG/routed_tf1500.log || { echo "[chain] ABORT: targetfirst falhou"; exit 1; }

echo "[chain] treinando routed_reffirst..."
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed \
  --config /workspace/configs/anima_routed_reffirst_1500.toml > $LOG/routed_rf1500.log 2>&1
echo "RF EXIT: $?" >> $LOG/routed_rf1500.log
grep -q "RF EXIT: 0" $LOG/routed_rf1500.log || { echo "[chain] ABORT: reffirst falhou"; exit 1; }

echo "[chain] treinando global_targetfirst..."
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed \
  --config /workspace/configs/anima_global_targetfirst_1500.toml > $LOG/global_tf1500.log 2>&1
echo "GLOBAL EXIT: $?" >> $LOG/global_tf1500.log
grep -q "GLOBAL EXIT: 0" $LOG/global_tf1500.log || { echo "[chain] ABORT: global falhou"; exit 1; }

echo "[chain] smoke da avaliacao..."
python /workspace/scripts/anima_controlled_eval.py smoke > $LOG/controlled_eval.log 2>&1
grep -q "SMOKE COMPLETO" $LOG/controlled_eval.log || { echo "[chain] ABORT: smoke da eval falhou"; exit 1; }

echo "[chain] avaliacao completa..."
python /workspace/scripts/anima_controlled_eval.py >> $LOG/controlled_eval.log 2>&1
echo "EVAL EXIT: $?" >> $LOG/controlled_eval.log

echo "[chain] upload HF..."
python - <<'PY' >> $LOG/controlled_eval.log 2>&1
from huggingface_hub import HfApi
import glob
api = HfApi()
for arm, root in [('routed_tf', 'anima_routed_targetfirst'), ('routed_rf', 'anima_routed_reffirst'),
                  ('global_tf', 'anima_global_targetfirst')]:
    for step in (500, 1000, 1500):
        d = glob.glob(f'/workspace/checkpoints/{root}/*/step{step}')
        if d:
            api.upload_folder(repo_id='AdwolfCzar/groundedsecret', repo_type='model', folder_path=d[0],
                              path_in_repo=f'anima_probes/controlled_{arm}_step{step}',
                              commit_message=f'Controlado {arm} step{step}')
api.upload_folder(repo_id='AdwolfCzar/groundedsecret', repo_type='model',
                  folder_path='/workspace/outputs/anima_controlled_eval',
                  path_in_repo='anima_probes/controlled_eval', allow_patterns=['GRID_*.png'],
                  commit_message='Grids da comparacao controlada')
print('UPLOADS OK')
PY
echo "[chain] COMPLETO"
