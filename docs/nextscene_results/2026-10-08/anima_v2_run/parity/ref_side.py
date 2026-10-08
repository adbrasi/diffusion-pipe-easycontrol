"""Reference side (training repo inference code). Saves noise, ref latent, step-0 denoised, final latent, image."""
import argparse, sys
sys.path.insert(0, '/workspace/diffusion-pipe-easycontrol')
import utils.common  # noqa: F401  (repo utils before anything else)
import torch, numpy as np
from PIL import Image
sys.argv_backup = sys.argv
import infer_easycontrol as ie
from tools.nextscene_eval import Runner
from diffusers.utils.torch_utils import randn_tensor

ap = argparse.ArgumentParser()
for a in ('--ckpt', '--image', '--prompt', '--negative', '--out'): ap.add_argument(a, required=True)
ap.add_argument('--steps', type=int, default=30); ap.add_argument('--cfg', type=float, default=4.0)
ap.add_argument('--shift', type=float, default=3.0); ap.add_argument('--seed', type=int, default=76)
a = ap.parse_args()
M = '/workspace/models_anima/split_files/'
args = argparse.Namespace(dit=M+'diffusion_models/anima-base-v1.0.safetensors', vae=M+'vae/qwen_image_vae.safetensors',
                          llm=M+'text_encoders/qwen_3_06b_base.safetensors', cfg=a.cfg, negative_prompt=a.negative,
                          rope_layout=None, ref_temporal_index=None, lora_strength=1.0, width=None, height=None,
                          steps=a.steps, flow_shift=a.shift, ref_cfg=1.0, uncond_ref='keep')
dev, dt = torch.device('cuda'), torch.bfloat16
img = Image.open(a.image); W, H = img.size
run = Runner(args, dev, dt)
ctx = run.encode(a.prompt)
run.load_ckpt(a.ckpt)
ref = run.ref_latent(a.image, W, H)
gen = torch.Generator(device='cpu').manual_seed(a.seed)
noise = randn_tensor((1, 16, 1, H // 8, W // 8), generator=gen, device=dev, dtype=torch.bfloat16)
# step-0 combined prediction (same formula as sample_nextscene)
_, sigmas = ie.get_timesteps_sigmas(a.steps, a.shift, dev)
t0 = sigmas[0].to(dev, torch.bfloat16).view(1)
def fwd(ctx_):
    x = torch.cat([noise, ref], dim=2); tf = torch.stack([t0, torch.zeros_like(t0)], dim=1)
    pm = torch.zeros(1, 1, H // 8, W // 8, dtype=torch.bfloat16, device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return run.dit(x, tf, ctx_, padding_mask=pm)[:, :, :1].float()
c, n = fwd(ctx), fwd(run.neg)
v0 = n + a.cfg * (c - n)
den0 = noise.float() - sigmas[0].item() * v0
lat = ie.sample_nextscene(run.dit, ctx, run.neg, ref, H, W, a.steps, a.cfg, a.shift, a.seed, dev, dt, ref_cfg=1.0, uncond_ref='keep')
with torch.no_grad():
    px = run.vae.model.decode(lat.to(dt), run.vae.scale)
px = px.squeeze(2) if px.ndim == 5 else px
Image.fromarray(((px[0].float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)).save(a.out + '_ref.png')
np.savez(a.out + '_ref.npz', noise=noise.float().cpu().numpy(), ref_latent=ref.float().cpu().numpy(),
         den0=den0.cpu().numpy(), final=lat.float().cpu().numpy(), sigmas=sigmas.cpu().numpy())
print('ref side done', W, H)
