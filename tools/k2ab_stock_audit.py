"""Verify the trained adapter with the real stock LoraLoaderModelOnly node."""
import argparse
import logging
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--adapter', required=True)
args = parser.parse_args()
sys.path.insert(0, '/workspace/ComfyUI_stock')
import comfy.sd
import folder_paths
import nodes
import torch

messages = []


class Capture(logging.Handler):
    def emit(self, record):
        if 'lora key not loaded' in record.getMessage():
            messages.append(record.getMessage())


logging.getLogger().addHandler(Capture())
adapter = Path(args.adapter)
folder_paths.add_model_folder_path('loras', str(adapter.parent))
model = comfy.sd.load_diffusion_model(
    '/workspace/models/krea2/diffusion_models/krea2_raw_bf16.safetensors',
    model_options={'dtype': torch.bfloat16})
patched, = nodes.LoraLoaderModelOnly().load_lora_model_only(model, adapter.name, 1.)
if messages:
    raise RuntimeError('\n'.join(messages))
print('STOCK LORA AUDIT OK:', len(patched.patches), 'patches, 0 unloaded keys')
patched.detach()
model.detach()
del patched, model
