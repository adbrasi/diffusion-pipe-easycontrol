# Runs after the 1024 finish: eval its checkpoints, then launch the 3-epoch run.
source /workspace/venv/bin/activate; cd /workspace/dpk
until grep -aq "TRAINING COMPLETE\|Traceback" /workspace/logs/finish1024.log; do sleep 30; done
F=$(ls -d /workspace/checkpoints/krea2_raw_v3/finish1024/2*)
R=$(ls -d /workspace/checkpoints/krea2_raw_v3/raw512_r32/2*)
AA=/workspace/adapters/a1000/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000/adapter_model.safetensors
python tools/krea2_v3_eval.py --cases /workspace/k2eval/cases_v2.json --out /workspace/k2eval/finish1024_eval \
  --col label=native_vl --col label=A1000,adapter=$AA --col label=r512_s1220,adapter=$R/step1220/adapter_model.safetensors \
  --col label=f120,adapter=$F/step120/adapter_model.safetensors --col label=f240,adapter=$F/step240/adapter_model.safetensors \
  --col label=f240_s07,adapter=$F/step240/adapter_model.safetensors,strength=0.7 \
  --col label=r512_s1220_s07,adapter=$R/step1220/adapter_model.safetensors,strength=0.7 \
  --col label=f240_shuf,adapter=$F/step240/adapter_model.safetensors,shuffle=1 > /workspace/logs/eval_finish1024.log 2>&1
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 --master_port=29606 train.py --deepspeed --config examples/krea2_raw_v3/raw512_long3ep.toml > /workspace/logs/raw512_long3ep.log 2>&1
