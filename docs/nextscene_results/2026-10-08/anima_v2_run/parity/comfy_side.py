"""ComfyUI side: native Anima + comfyui_nextscene nodes, same noise/sigmas/sampler as the reference side."""
import argparse, sys, os
ap = argparse.ArgumentParser()
for x in ('--lora_dir', '--lora_name', '--image', '--prompt', '--negative', '--out', '--nodes_parent'): ap.add_argument(x, required=True)
ap.add_argument('--steps', type=int, default=30); ap.add_argument('--cfg', type=float, default=4.0)
a = ap.parse_args()
sys.argv = [sys.argv[0]]
os.chdir('/workspace/comfy/ComfyUI'); sys.path.insert(0, '/workspace/comfy/ComfyUI'); sys.path.append(a.nodes_parent)
import torch, numpy as np
from PIL import Image
import folder_paths, comfy.sd, comfy.utils, comfy.sample, comfy.model_management
from comfyui_nextscene import AnimaNextSceneLoader, AnimaNextSceneConditioning
M = '/workspace/models_anima/split_files/'
folder_paths.add_model_folder_path('loras', a.lora_dir)
d = np.load(a.out + '_ref.npz')
model = comfy.sd.load_diffusion_model(M + 'diffusion_models/anima-base-v1.0.safetensors')
clip = comfy.sd.load_clip(ckpt_paths=[M + 'text_encoders/qwen_3_06b_base.safetensors'], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(M + 'vae/qwen_image_vae.safetensors'))
with torch.inference_mode():
    pos = clip.encode_from_tokens_scheduled(clip.tokenize(a.prompt))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(a.negative))
    img = torch.from_numpy(np.asarray(Image.open(a.image).convert('RGB')).astype(np.float32) / 255.0)[None]
    lat = vae.encode(img)
    patched = AnimaNextSceneLoader().load(model, a.lora_name, 1.0, 1.0)[0]
    pos, neg = AnimaNextSceneConditioning().condition(pos, neg, {'samples': lat}, 'image', 'keep')
    noise = torch.from_numpy(d['noise'])
    sigmas = torch.from_numpy(d['sigmas']).float()
    den = {}
    def cb(step, x0, x, total):
        if step == 0: den[0] = x0.float().cpu()
    out = comfy.sample.sample(patched, noise, a.steps, a.cfg, 'euler', 'simple', pos, neg, torch.zeros_like(noise),
                              sigmas=sigmas, callback=cb, disable_pbar=True, seed=0)
    image = vae.decode(out)
    arr = (image[0].reshape(-1, *image.shape[-3:])[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(a.out + '_comfy.png')
    ref_proc = patched.get_model_object('latent_format').process_in(lat if lat.ndim == 5 else lat.unsqueeze(2))
    np.savez(a.out + '_comfy.npz', ref_latent=ref_proc.float().cpu().numpy(), den0=den[0].numpy(), final=out.float().cpu().numpy())
    print('comfy side done', tuple(out.shape))
