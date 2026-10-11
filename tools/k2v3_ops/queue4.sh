source /workspace/venv/bin/activate; cd /workspace/dpk
until grep -aq "TRAINING COMPLETE\|Traceback" /workspace/logs/raw1024_long_resume.log; do sleep 30; done
sleep 30
L=$(ls -d /workspace/checkpoints/krea2_raw_v3/raw1024_long/2*)
T=/workspace/adapters/t1024/checkpoints/20261009_02-44-34/step3000/adapter_model.safetensors
python tools/krea2_v3_eval.py --cases /workspace/k2eval/cases_small.json --out /workspace/k2eval/raw1024_long_eval \
  --col label=native_vl --col label=t3000_s06,adapter=$T,strength=0.6 --col label=L1250,adapter=$L/step1250/adapter_model.safetensors \
  --col label=L2000,adapter=$L/step2000/adapter_model.safetensors --col label=L3000,adapter=$L/step3000/adapter_model.safetensors > /workspace/logs/eval_raw1024_long2.log 2>&1
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 --master_port=29606 train.py --deepspeed --config examples/krea2_raw_v3/raw512_shorttier.toml > /workspace/logs/raw512_shorttier.log 2>&1
