"""NextScene geometry and reference packing around ComfyUI's native Anima."""

import torch

import comfy.conds
import comfy.samplers
from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb


class NextSceneRope(VideoRopePosition3DEmb):
    def __init__(self, original, layout, reference_index):
        head_dim = 2 * (2 * original.dim_spatial_range.numel() + original.dim_temporal_range.numel())
        super().__init__(head_dim=head_dim, len_h=original.max_h, len_w=original.max_w,
                         len_t=2, base_fps=original.base_fps, enable_fps_modulation=False)
        self.h_ntk_factor = original.h_ntk_factor
        self.w_ntk_factor = original.w_ntk_factor
        self.t_ntk_factor = original.t_ntk_factor
        self.layout = layout
        self.reference_index = reference_index

    def generate_embeddings(self, shape, fps=None, h_ntk_factor=None,
                            w_ntk_factor=None, t_ntk_factor=None, device=None, dtype=None):
        _, frames, height, width, _ = shape
        spatial_range = self.dim_spatial_range.to(device=device)
        temporal_range = self.dim_temporal_range.to(device=device)
        hf = self.h_ntk_factor if h_ntk_factor is None else h_ntk_factor
        wf = self.w_ntk_factor if w_ntk_factor is None else w_ntk_factor
        tf = self.t_ntk_factor if t_ntk_factor is None else t_ntk_factor
        freq_h = 1.0 / ((10000.0 * hf) ** spatial_range)
        freq_w = 1.0 / ((10000.0 * wf) ** spatial_range)
        freq_t = 1.0 / ((10000.0 * tf) ** temporal_range)
        angles = []
        for i in range(frames):
            t = 0 if i == 0 else self.reference_index + i - 1
            dh = height * i if self.layout in ("disjoint_h", "disjoint_diag") else 0
            dw = width * i if self.layout in ("disjoint_w", "disjoint_diag") else 0
            h = torch.arange(height, device=device, dtype=torch.float32) + dh
            w = torch.arange(width, device=device, dtype=torch.float32) + dw
            et = (t * freq_t).view(1, 1, -1).expand(height, width, -1)
            eh = torch.outer(h, freq_h).view(height, 1, -1).expand(height, width, -1)
            ew = torch.outer(w, freq_w).view(1, width, -1).expand(height, width, -1)
            angles.append(torch.cat((et, eh, ew), dim=-1))
        angle = torch.stack(angles).flatten(0, 2)
        cos, sin = angle.cos(), angle.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).unflatten(-1, (2, 2))


class NextSceneExtraConds:
    def __init__(self, original, latent_format):
        self.original = original
        self.latent_format = latent_format

    def __call__(self, **kwargs):
        out = self.original(**kwargs)
        latent = kwargs.get("nextscene_reference_latent")
        if latent is None:
            raise ValueError("Connect Anima NextScene Conditioning to the sampler's positive and negative inputs.")
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        if latent.shape[1] != 16 or latent.shape[2] != 1:
            raise ValueError("NextScene needs one image encoded with qwen_image_vae.safetensors.")
        reference = self.latent_format.process_in(latent)
        if kwargs.get("nextscene_null_reference", False):
            reference = torch.zeros_like(reference)
        out["nextscene_reference"] = comfy.conds.CONDRegular(reference)
        return out


def nextscene_forward(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
    reference = kwargs.pop("nextscene_reference").to(x)
    if x.shape[2] != 1 or reference.shape[2:] != x.shape[2:]:
        raise ValueError("NextScene requires one target frame and the same reference/target size. "
                         "Use the workflow's shared width and height controls.")
    if kwargs.get("transformer_options", {}).get("nextscene_null_reference", False):
        reference = torch.zeros_like(reference)
    packed = torch.cat((x, reference), dim=2)
    times = torch.stack((timesteps.reshape(-1), torch.zeros_like(timesteps.reshape(-1))), dim=1)
    return executor(packed, times, context, fps, padding_mask, **kwargs)[:, :, :1]


class NextSceneReferenceGuidance:
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, args):
        options = args["model_options"].copy()
        options["transformer_options"] = options.get("transformer_options", {}).copy()
        options["transformer_options"]["nextscene_null_reference"] = True
        null = comfy.samplers.calc_cond_batch(args["model"], [args["cond"]],
                                            args["input"], args["sigma"], options)[0]
        return args["denoised"] + (self.strength - 1.0) * (args["cond_denoised"] - null)
