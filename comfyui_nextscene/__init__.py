"""Two custom nodes; loaders, text encoding, VAE and sampling stay native."""

import json
import logging
from pathlib import Path

from safetensors import safe_open
import comfy.lora
import comfy.utils
from comfy.ldm.anima.model import Anima
from comfy.patcher_extension import WrappersMP
import folder_paths
import node_helpers

from .model import NextSceneExtraConds, NextSceneReferenceGuidance, NextSceneRope, nextscene_forward


class AnimaNextSceneLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05}),
            "reference_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                "tooltip": "1 = avaliação E2. Outros valores adicionam um forward sem referência por step."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Anima/NextScene"
    DESCRIPTION = "Aplica o LoRA nativamente e lê a geometria aligned/disjoint do checkpoint. Strength 0 testa o modelo base com a mesma geometria."

    def load(self, model, lora_name, strength, reference_guidance):
        if not isinstance(model.get_model_object("diffusion_model"), Anima):
            raise ValueError("Load anima-base-v1.0.safetensors with the native Load Diffusion Model node.")
        if model.get_attachment("anima_nextscene_contract") is not None:
            raise ValueError("Apply only one NextScene adapter to the base model.")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
        if "nextscene_contract" not in metadata:
            raise ValueError("Select a NextScene E1/E2 adapter with nextscene_contract metadata.")
        contract = json.loads(metadata["nextscene_contract"])
        if (contract.get("version") != "1" or contract.get("layout") != "target_first"
                or contract.get("rope_layout") not in ("aligned", "disjoint_w", "disjoint_h", "disjoint_diag")
                or contract.get("ref_timestep") != 0.0 or contract.get("null_ref") != "zeros"):
            raise ValueError(f"Unsupported NextScene contract: {contract}")
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        # Symlinks resolve to the original checkpoint, including its PEFT alpha/rank config.
        config = json.loads((Path(path).resolve().parent / "adapter_config.json").read_text())
        modules = set()
        for key in list(lora):
            if any(forbidden in key for forbidden in ("llm_adapter", "adaln_modulation")):
                raise ValueError(f"Forbidden NextScene LoRA parameter: {key}")
            if key.endswith(".lora_A.weight"):
                prefix = key.removesuffix(".lora_A.weight")
                if prefix + ".lora_B.weight" not in lora:
                    raise ValueError(f"Incomplete LoRA pair: {prefix}")
                modules.add(prefix)
                # Native LoRA loader supports these exact keys; preserve PEFT alpha/r math.
                alpha = config.get("alpha_pattern", {}).get(prefix.removeprefix("diffusion_model."), config["lora_alpha"])
                lora[prefix + ".alpha"] = lora[key].new_tensor(alpha)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(lora, key_map)
        if len(patches) != len(modules) or not modules:
            raise ValueError(f"NextScene LoRA mismatch: loaded {len(patches)} of {len(modules)} linears.")
        patched = model.clone()
        loaded = patched.add_patches(patches, strength)
        if len(loaded) != len(patches):
            raise ValueError("The base model did not accept every NextScene LoRA patch.")
        patched.set_attachments("anima_nextscene_contract", contract)
        patched.add_object_patch("diffusion_model.pos_embedder", NextSceneRope(
            model.get_model_object("diffusion_model.pos_embedder"), contract["rope_layout"], int(contract["ref_temporal_index"])))
        patched.add_object_patch("extra_conds", NextSceneExtraConds(
            model.get_model_object("extra_conds"), model.get_model_object("latent_format")))
        patched.add_object_patch("memory_usage_factor", model.get_model_object("memory_usage_factor") * 2.0)
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "anima_nextscene", nextscene_forward)
        if reference_guidance != 1.0:
            patched.set_model_sampler_post_cfg_function(NextSceneReferenceGuidance(reference_guidance))
        logging.info("NextScene: %s, %s, %d LoRA linears, strength %.2f, ref guidance %.2f",
                     lora_name, contract["rope_layout"], len(loaded), strength, reference_guidance)
        return (patched,)


class AnimaNextSceneConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "reference": ("LATENT",),
            "reference_mode": (["image", "null"], {"default": "image"}),
            "negative_reference": (["keep", "zero"], {"default": "keep"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "condition"
    CATEGORY = "Anima/NextScene"
    DESCRIPTION = "Use VAE Encode da imagem A. image/keep reproduz os testes E2; null remove a referência com zeros no espaço normalizado."

    def condition(self, positive, negative, reference, reference_mode, negative_reference):
        values = {"nextscene_reference_latent": reference["samples"],
                  "nextscene_null_reference": reference_mode == "null"}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative_values = {**values, "nextscene_null_reference": reference_mode == "null" or negative_reference == "zero"}
        negative = node_helpers.conditioning_set_values(negative, negative_values)
        return positive, negative


NODE_CLASS_MAPPINGS = {
    "AnimaNextSceneLoader": AnimaNextSceneLoader,
    "AnimaNextSceneConditioning": AnimaNextSceneConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaNextSceneLoader": "Anima NextScene — Adapter",
    "AnimaNextSceneConditioning": "Anima NextScene — Reference",
}
