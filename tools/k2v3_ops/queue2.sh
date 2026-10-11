source /workspace/venv/bin/activate; cd /workspace/dpk
R=$(ls -d /workspace/checkpoints/krea2_raw_v3/raw512_r32/2*)/step1220/adapter_model.safetensors
python tools/krea2_v3_eval.py --cases /workspace/k2eval/cases_small.json --ids web_gura,web_aya,web_cirno2,web_hood --out /workspace/k2eval/strength_web \
  --col label=r512_s14,adapter=$R,strength=1.4 --col label=r512_s18,adapter=$R,strength=1.8 --col label=r512_shuf,adapter=$R,shuffle=1 > /workspace/logs/strength_web.log 2>&1
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 --master_port=29606 train.py --deepspeed --config examples/krea2_raw_v3/raw1024_long.toml > /workspace/logs/raw1024_long.log 2>&1
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 --master_port=29606 train.py --deepspeed --config examples/krea2_raw_v3/raw512_shorttier.toml > /workspace/logs/raw512_shorttier.log 2>&1
