set -x
cd /workspace/dpk
uv venv /workspace/venv --python 3.12 --seed
source /workspace/venv/bin/activate
uv pip install torch==2.11.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
grep -v -E "^(torch|torchvision|torchaudio)$" requirements.txt > /tmp/req_dp.txt
grep -v -E "^(torch|torchvision|torchaudio)([=<> ].*)?$" /workspace/comfy/ComfyUI/requirements.txt > /tmp/req_comfy.txt
echo "torch==$TV" > /tmp/constraints.txt
uv pip install -r /tmp/req_dp.txt -c /tmp/constraints.txt --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install -r /tmp/req_comfy.txt -c /tmp/constraints.txt --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install requests numpy huggingface_hub hf_transfer
python -c "import torch;print('TORCH',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name())"
echo DONE_INSTALL2
