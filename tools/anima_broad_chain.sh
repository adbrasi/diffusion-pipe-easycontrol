#!/bin/bash
# Corrente dos braços de escopo largo: v3 -> dual -> omini_broad -> smoke -> eval -> HF.
set -u
cd /workspace/diffusion-pipe-easycontrol
source .venv/bin/activate
LOG=/workspace/logs/anima_probes

for arm in v3 dual omini_broad; do
  echo "[broad-chain] treinando $arm..."
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed \
    --config /workspace/configs/anima_${arm}_500.toml > $LOG/broad_${arm}.log 2>&1
  ec=$?
  echo "${arm} EXIT: $ec" >> $LOG/broad_${arm}.log
  [ $ec -eq 0 ] || { echo "[broad-chain] ABORT: $arm falhou (exit $ec)"; exit 1; }
done

echo "[broad-chain] smoke da avaliacao..."
python /workspace/scripts/anima_broad_eval.py smoke > $LOG/broad_eval.log 2>&1
grep -q "SMOKE COMPLETO" $LOG/broad_eval.log || { echo "[broad-chain] ABORT: smoke falhou"; exit 1; }

echo "[broad-chain] avaliacao completa..."
python /workspace/scripts/anima_broad_eval.py >> $LOG/broad_eval.log 2>&1

echo "[broad-chain] upload HF..."
python - <<'PY' >> $LOG/broad_eval.log 2>&1
from huggingface_hub import HfApi
import glob
api = HfApi()
for arm, root in [('iclora_v3', 'anima_iclora_v3'), ('iclora_dual', 'anima_iclora_dual'),
                  ('omini_broad', 'anima_omini_broad')]:
    for step in (250, 500):
        d = glob.glob(f'/workspace/checkpoints/{root}/*/step{step}')
        if d:
            api.upload_folder(repo_id='AdwolfCzar/groundedsecret', repo_type='model', folder_path=d[0],
                              path_in_repo=f'anima_probes/broad_{arm}_step{step}',
                              commit_message=f'Broad {arm} step{step}')
api.upload_folder(repo_id='AdwolfCzar/groundedsecret', repo_type='model',
                  folder_path='/workspace/outputs/anima_broad_eval',
                  path_in_repo='anima_probes/broad_eval', allow_patterns=['GRID_*.png'],
                  commit_message='Grids escopo largo')
print('UPLOADS OK')
PY
echo "[broad-chain] COMPLETO"
