set -e
source /workspace/env-dl/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("circlestone-labs/Anima-Base-v1.0-Diffusers", local_dir="/workspace/models/circlestone-labs/Anima-Base-v1.0-Diffusers")
for f in ["split_files/diffusion_models/anima-base-v1.0.safetensors","split_files/text_encoders/qwen_3_06b_base.safetensors","split_files/vae/qwen_image_vae.safetensors"]:
    hf_hub_download("circlestone-labs/Anima", f, local_dir="/workspace/models/circlestone-labs/Anima")
print("HF done")
PY
mkdir -p /workspace/models/esrgan
curl -sL -o /workspace/models/esrgan/realesr-animevideov3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
ls -la /workspace/models/esrgan
echo ALL DONE
