import os, sys, yaml
root = "/workspace/data/ns_full"
name, steps, res = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
samples = []
for f in sorted(os.listdir(f"{root}/val/target")):
    if not f.endswith(".txt"): continue
    stem = f[:-4]
    samples.append({"prompt": open(f"{root}/val/target/{f}").read().strip(),
                    "ctrl_img_1": f"{root}/val/control/{stem}.jpg",
                    "width": 1024, "height": 576, "seed": 42})
cfg = {"job": "extension", "config": {"name": name, "process": [{
  "type": "diffusion_trainer", "training_folder": "/workspace/output", "device": "cuda:0",
  "network": {"type": "lora", "linear": 32, "linear_alpha": 32,
              "network_kwargs": {"ignore_if_contains": ["layers.22."]}},
  "save": {"dtype": "bf16", "save_every": 250, "max_step_saves_to_keep": 10},
  "datasets": [{"folder_path": f"{root}/train/target", "control_path": [f"{root}/train/control"],
                "caption_ext": "txt", "caption_dropout_rate": 0.05, "resolution": [res],
                "cache_latents_to_disk": False}],
  "train": {"batch_size": 1, "steps": steps, "gradient_accumulation": 1, "train_unet": True,
            "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
            "timestep_type": "shift", "optimizer": "adamw8bit", "lr": 1e-4, "dtype": "bf16",
            "unload_text_encoder": False},
  "model": {"name_or_path": "OpenMOSS-Team/MOSS-Image-Base", "arch": "moss_image",
            "quantize": True, "qtype": "convrot8", "quantize_te": True, "qtype_te": "convrot8",
            "low_vram": False,
            "model_kwargs": {"edit": True, "kv_cache": True, "match_target_res": True, "uncond_skip_layers": [22]}},
  "sample": {"sampler": "flowmatch", "sample_every": 250, "sample_start_step": 0, "width": 1024, "height": 576,
             "neg": "", "seed": 42, "walk_seed": False, "guidance_scale": 3, "sample_steps": 30,
             "samples": samples}}]},
  "meta": {"name": "[name]", "version": "1.0"}}
yaml.safe_dump(cfg, open(f"/workspace/scripts/{name}.yaml", "w"), sort_keys=False, allow_unicode=True)
print(len(samples), "samples")
