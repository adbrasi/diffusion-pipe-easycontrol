# Usage: eval_raw512.sh <run_dir> "<steps>" <out>
source /workspace/venv/bin/activate; cd /workspace/dpk
RUN=$1; STEPS=$2; OUT=$3
AA=/workspace/adapters/a1000/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000/adapter_model.safetensors
COLS="--col label=native --col label=A1000,adapter=$AA"
for s in $STEPS; do COLS="$COLS --col label=s$s,adapter=$RUN/step$s/adapter_model.safetensors"; done
python tools/krea2_v3_eval.py --cases /workspace/k2eval/cases_v1.json --out $OUT $COLS
