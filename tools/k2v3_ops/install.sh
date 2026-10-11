source /venv/main/bin/activate
cd /workspace/dpk
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -r submodules/ComfyUI/requirements.txt
python -c "import torch;print('TORCH',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name())"
echo DONE_INSTALL
