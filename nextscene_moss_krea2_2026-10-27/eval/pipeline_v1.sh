#!/bin/bash
set -e
source /workspace/venv/bin/activate
while pgrep -f "run.py /workspace/scripts/moss_nextscene_v1.yaml" >/dev/null; do sleep 15; done
python /workspace/scripts/grids.py moss_nextscene_v1
LORA=/workspace/output/moss_nextscene_v1/moss_nextscene_v1.safetensors
ls -la $LORA
rm -f /workspace/evals/engine/engine.json
cd /workspace/ai-toolkit
python run.py /workspace/evals/engine.yaml > /workspace/evals/engine.log 2>&1 &
EPID=$!
until [ -f /workspace/evals/engine/engine.json ]; do sleep 5; kill -0 $EPID || { echo ENGINE_DIED; exit 1; }; done
sleep 5
python /workspace/evals/run_tests.py /workspace/evals/v1_tests.json $LORA /workspace/evals/v1_final
python /workspace/evals/summary.py /workspace/evals/v1_final "MOSS-Image Base + LoRA next-scene v1 (768px, 1500 passos) — seed 42, 30 passos, cfg 3" /workspace/outputs_for_user/RESUMO_TESTES_v1.jpg
kill $EPID; sleep 10; pkill -f "run.py /workspace/evals/engine.yaml" || true
echo PIPELINE_DONE
