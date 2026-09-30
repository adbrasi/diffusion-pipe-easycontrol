#!/usr/bin/env python3
"""Build UI/API workflows against the running ComfyUI's actual node schemas."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:18818")
    parser.add_argument("--comfy", type=Path, default=Path("/workspace/comfy/ComfyUI"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/nextscene_artifacts/ComfyUI"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image = "nextscene_classroom_reference.jpg"
    shutil.copy2("/workspace/heldout_short/control/05_ds2_004723.jpg", args.comfy / "input" / image)
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "AnimaNextSceneLoader", "inputs": {"model": ["1", 0], "lora_name": "anima_nextscene/E2_B_disjoint_w_step5685_epoch1.safetensors", "strength": 1.0, "reference_guidance": 1.0}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image}},
        "6": {"class_type": "ImageScale", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "width": ["15", 0], "height": ["16", 0], "crop": "center"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "the same boy in the red and green striped shirt, now turning toward his classmate and speaking, in the same classroom, medium shot"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "worst quality, low quality, blurry, jpeg artifacts"}},
        "10": {"class_type": "AnimaNextSceneConditioning", "inputs": {"positive": ["8", 0], "negative": ["9", 0], "reference": ["7", 0], "reference_mode": "image", "negative_reference": "keep"}},
        "11": {"class_type": "EmptyLatentImage", "inputs": {"width": ["15", 0], "height": ["16", 0], "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["11", 0], "seed": 76, "steps": 20, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "NextScene/E2_B_disjoint_w_5685"}},
        "15": {"class_type": "PrimitiveInt", "inputs": {"value": 512}},
        "16": {"class_type": "PrimitiveInt", "inputs": {"value": 512}},
        "17": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
    }
    positions = {1:(0,0),2:(0,160),3:(0,315),4:(410,0),5:(0,550),6:(410,550),7:(795,555),
                 8:(410,220),9:(410,405),10:(1110,360),11:(1110,610),12:(1500,0),13:(1500,430),
                 14:(1880,0),15:(0,435),16:(205,435),17:(795,705)}
    titles = {4:"ADAPTER — escolha A/aligned ou B/disjoint_w",5:"IMAGEM A — envie sua referência",8:"PROMPT — descreva a próxima cena",9:"PROMPT NEGATIVO",10:"REFERÊNCIA — image / null",15:"LARGURA (múltiplo de 16)",16:"ALTURA (múltiplo de 16)",17:"IMAGEM A redimensionada",14:"RESULTADO — salvo em /workspace/"}
    schema = json.load(urllib.request.urlopen(args.server + "/object_info"))
    nodes = []
    for node_id, entry in prompt.items():
        info = schema[entry["class_type"]]
        node = {"id": int(node_id), "type": entry["class_type"], "pos": positions[int(node_id)],
                "size": [340, 155], "flags": {}, "order": int(node_id)-1, "mode": 0,
                "inputs": [], "outputs": [], "properties": {"Node name for S&R": entry["class_type"]}, "widgets_values": []}
        if int(node_id) in titles:
            node["title"] = titles[int(node_id)]
        if entry["class_type"] in ("SaveImage", "PreviewImage", "LoadImage"):
            node["size"] = [360, 420]
        elif entry["class_type"] == "KSampler":
            node["size"] = [340, 330]
        elif entry["class_type"] == "PrimitiveInt":
            node["size"] = [190, 90]
        elif entry["class_type"] == "CLIPTextEncode":
            node["size"] = [340, 160]
        for name, spec in {**info["input"].get("required", {}), **info["input"].get("optional", {})}.items():
            if name not in entry["inputs"]:
                continue
            value = entry["inputs"][name]
            kind = spec[0]
            widget = isinstance(kind, list) or kind in ("INT", "FLOAT", "STRING", "BOOLEAN")
            linked = isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt
            if linked:
                port = {"name": name, "type": "COMBO" if isinstance(kind, list) else kind, "link": None}
                if widget:
                    port["widget"] = {"name": name}
                node["inputs"].append(port)
            if widget:
                default = spec[1].get("default", 512) if len(spec) > 1 and isinstance(spec[1], dict) else 512
                node["widgets_values"].append(default if linked else value)
                if len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("control_after_generate"):
                    node["widgets_values"].append("fixed")
        for index, output_type in enumerate(info["output"]):
            name = info.get("output_name", info["output"])[index]
            node["outputs"].append({"name": name, "type": output_type, "links": [], "slot_index": index})
        nodes.append(node)
    by_id = {str(n["id"]): n for n in nodes}
    links = []
    for node_id, entry in prompt.items():
        for name, value in entry["inputs"].items():
            if not isinstance(value, list):
                continue
            source, output = value
            node = by_id[node_id]
            slot = next(i for i, port in enumerate(node["inputs"]) if port["name"] == name)
            link_id = len(links) + 1
            kind = by_id[source]["outputs"][output]["type"]
            links.append([link_id, int(source), output, int(node_id), slot, kind])
            node["inputs"][slot]["link"] = link_id
            by_id[source]["outputs"][output]["links"].append(link_id)
    workflow = {"last_node_id": 17, "last_link_id": len(links), "nodes": nodes, "links": links,
                "groups": [], "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [90, 60]}}, "version": 0.4}
    workflow["nodes"].append({"id": 18, "type": "Note", "pos": [1880, 460], "size": [365, 250],
        "flags": {}, "order": 18, "mode": 0, "properties": {}, "widgets_values": [
        "NextScene E2 — 1 época = 5685 steps.\nA = aligned; B = disjoint_w. O node lê a geometria automaticamente.\n\nTroque IMAGEM A e PROMPT. LARGURA/ALTURA controlam referência e resultado juntos.\n\nComece com strength 1, reference_guidance 1, image/keep, CFG 4, Euler, 20 steps.\nStrength 0 = base com referência; null = sem referência. Fixe a seed para comparar.\n\nOutputs: ComfyUI/output/NextScene/. O prefixo é só um nome: atualize-o ao trocar o adapter."]})
    workflow["last_node_id"] = 18
    for arm, layout in (("B", "disjoint_w"), ("A", "aligned")):
        prompt["4"]["inputs"]["lora_name"] = f"anima_nextscene/E2_{arm}_{layout}_step5685_epoch1.safetensors"
        prompt["14"]["inputs"]["filename_prefix"] = f"NextScene/E2_{arm}_{layout}_5685"
        by_id["4"]["widgets_values"][0] = prompt["4"]["inputs"]["lora_name"]
        by_id["14"]["widgets_values"][0] = prompt["14"]["inputs"]["filename_prefix"]
        name = f"Anima_NextScene_E2_{arm}_{layout}"
        (args.out / (name + ".json")).write_text(json.dumps(workflow, ensure_ascii=False, indent=2))
        (args.out / (name + "_API.json")).write_text(json.dumps(prompt, ensure_ascii=False, indent=2))
        user_workflows = args.comfy / "user/default/workflows"
        user_workflows.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out / (name + ".json"), user_workflows / (name + ".json"))
        print(args.out / (name + ".json"))


if __name__ == "__main__":
    main()
