"""Dedicated Ideogram 4 unconditional transformer for deployment-parity CFG.

Loaded through ComfyUI's scaled-fp8 path (comfy.sd.load_diffusion_model), NOT
the trainer's destructive fp8 recast. The unconditional pass is image-only
(context=None), exactly the official DualModelGuider negative branch.
"""

import torch

UNCOND_PATH = '/workspace/models/ideogram4_unconditional_fp8_scaled.safetensors'


class Ideogram4Uncond:
    def __init__(self, path=UNCOND_PATH, device='cuda'):
        import comfy.sd
        self.patcher = comfy.sd.load_diffusion_model(path)
        self.model = self.patcher.model.diffusion_model
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def velocity(self, latent, sigma):
        timestep = latent.new_full((latent.shape[0],), float(sigma))
        return self.model(latent.to(self.device), timestep, context=None).float()
