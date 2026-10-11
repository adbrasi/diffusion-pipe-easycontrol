source /workspace/venv/bin/activate; cd /workspace/dpk
R=$(ls -d /workspace/checkpoints/krea2_raw_v3/raw512_r32/2*)/step1220/adapter_model.safetensors
T=/workspace/adapters/t1024/checkpoints/20261009_02-44-34/step3000/adapter_model.safetensors
python tools/krea2_v3_eval.py --cases /workspace/k2eval/cases_small.json --ids web_gura,web_aya --out /workspace/k2eval/styletest --variant Raw \
  --col label=raw_native_vl --col label=raw_r512,adapter=$R --col label=turbo_t3000_s06,adapter=$T,strength=0.6,variant=Turbo > /workspace/logs/styletest.log 2>&1
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 --master_port=29606 train.py --deepspeed --config examples/krea2_raw_v3/raw1024_long.toml > /workspace/logs/raw1024_long.log 2>&1
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 --master_port=29606 train.py --deepspeed --config examples/krea2_raw_v3/raw512_shorttier.toml > /workspace/logs/raw512_shorttier.log 2>&1
