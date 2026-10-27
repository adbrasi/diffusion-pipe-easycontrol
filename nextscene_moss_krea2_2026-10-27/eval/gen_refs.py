# generates out-of-dataset reference images with MOSS-Image Base (text-to-image, no LoRA)
import json, os, shutil, urllib.request
eng = json.load(open("/workspace/evals/engine/engine.json"))
URL, TOK = f"http://127.0.0.1:{eng['port']}", eng["token"]
REFS = {
 "foto_ruiva": "A photograph of a young woman with short curly red hair and freckles, wearing a yellow raincoat, standing between shelves in a small cozy bookstore, natural window light, 35mm photo.",
 "raposa_3d": "A 3D animated movie still of a small orange fox character wearing a blue scarf and a green backpack, standing in a sunny forest clearing, soft lighting.",
 "cavaleiro_aquarela": "A watercolor illustration of an old knight with a long white beard in dented silver armor and a red cape, sitting by a campfire at dusk in a meadow.",
 "manga_garoto": "A black and white manga panel, screentone shading: a teenage boy with spiky white hair and an eyepatch, wearing a school uniform, leaning on a rooftop fence.",
}
model = {"arch": "moss_image", "name_or_path": "OpenMOSS-Team/MOSS-Image-Base", "quantize": True, "qtype": "convrot8",
         "quantize_te": True, "qtype_te": "convrot8", "model_kwargs": {"uncond_skip_layers": [22]}}
for name, p in REFS.items():
    dst = f"/workspace/test_refs/gen/{name}.jpg"
    if os.path.exists(dst): continue
    body = {"model": model, "wait": True, "sample": {"prompt": p, "width": 1024, "height": 576, "num_inference_steps": 30, "guidance_scale": 3, "seed": 7, "negative_prompt": ""}}
    req = urllib.request.Request(URL + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {TOK}"})
    info = json.load(urllib.request.urlopen(req, timeout=3600))
    shutil.copy(info["results"][0]["path"], dst); print("ref", name, flush=True)
