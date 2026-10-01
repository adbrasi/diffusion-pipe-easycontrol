import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_XET_LOG_LEVEL']='warn'
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
names=['diffusion_models/krea2_raw_bf16.safetensors','text_encoders/qwen3vl_4b_bf16.safetensors','loras/krea2_turbo_lora_rank_64_bf16.safetensors']
def get(name):
 print('Downloading',name,flush=True)
 path=hf_hub_download('Comfy-Org/Krea-2',name,local_dir='/workspace/models/krea2')
 print('Ready',name,Path(path).stat().st_size,flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(get,names))
