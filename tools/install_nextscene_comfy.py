#!/usr/bin/env python3
"""Link existing Anima assets and original adapter checkpoints into ComfyUI."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--comfy", type=Path, default=Path("/workspace/comfy/ComfyUI"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    links = {args.comfy / "custom_nodes/ComfyUI-Anima-NextScene": repo / "comfyui_nextscene"}
    assets = args.workspace / "models_anima/split_files"
    for folder, name in (("diffusion_models", "anima-base-v1.0.safetensors"),
                         ("vae", "qwen_image_vae.safetensors"),
                         ("text_encoders", "qwen_3_06b_base.safetensors")):
        links[args.comfy / "models" / folder / name] = assets / folder / name
    for experiment in ("E1", "E2"):
        for arm, layout in (("A", "aligned"), ("B", "disjoint_w")):
            checkpoints = args.workspace / "checkpoints/anima_nextscene" / f"{experiment}_{arm}"
            for checkpoint in sorted(checkpoints.glob("*/*/adapter_model.safetensors")):
                step = "step5685_epoch1" if checkpoint.parent.name == "epoch1" else checkpoint.parent.name
                name = f"{experiment}_{arm}_{layout}_{step}.safetensors"
                links[args.comfy / "models/loras/anima_nextscene" / name] = checkpoint
    for destination, source in links.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Preserving existing file: {destination}")
        destination.symlink_to(source, target_is_directory=source.is_dir())
        print(f"{destination} -> {source}")


if __name__ == "__main__":
    main()
