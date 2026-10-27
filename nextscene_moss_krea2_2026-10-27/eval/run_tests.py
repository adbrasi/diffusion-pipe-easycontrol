# usage: run_tests.py <spec.json> <lora.safetensors> <out_dir>
import json, os, sys, time, urllib.request
spec, lora, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
DATA = spec.get("data_dir", "/workspace/data/ns_full")
VAL, TRAIN = f"{DATA}/val", f"{DATA}/train"
eng = json.load(open("/workspace/evals/engine/engine.json"))
URL, TOK = f"http://127.0.0.1:{eng['port']}", eng["token"]
MODEL = {"arch": "moss_image", "name_or_path": "OpenMOSS-Team/MOSS-Image-Base",
         "quantize": True, "qtype": "convrot8", "quantize_te": True, "qtype_te": "convrot8",
         "model_kwargs": {"edit": True, "kv_cache": True, "match_target_res": True, "uncond_skip_layers": [22]}}

MODEL.update(spec.get("model", {}))
STEPS, CFG = spec.get("steps", 30), spec.get("guidance", 3)

def gen(prompt, ctrl, w, h, strength):
    model = dict(MODEL, loras=[{"path": lora, "strength": strength}] if strength else [])
    body = {"model": model, "wait": True, "sample": {"prompt": prompt, "ctrl_img": ctrl, "width": w, "height": h,
            "num_inference_steps": STEPS, "guidance_scale": CFG, "seed": 42, "negative_prompt": ""}}
    req = urllib.request.Request(URL + "/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {TOK}"})
    info = json.load(urllib.request.urlopen(req, timeout=3600))
    paths = [r["path"] for r in info.get("results", [])]
    if not paths: raise RuntimeError(info)
    return paths[0]

manifest = []
for r in spec["refs"]:
    base = VAL if r.get("split") == "val" else TRAIN
    ctrl = r.get("ctrl") or f"{base}/control/{r['stem']}.jpg"
    target = f"{base}/target/{r['stem']}.jpg" if "stem" in r else None
    w, h = r.get("w", 1024), r.get("h", 576)
    prompts = []
    if "stem" in r: prompts.append(("identico", open(f"{base}/target/{r['stem']}.txt").read().strip()))
    prompts += [("parecido", r["similar"])] + ([("parecido_delta", r["delta"])] if "delta" in r else []) + [("diferente", r["different"])]
    for kind, p in prompts:
        row = {"ref": r["id"], "split": r.get("split", "fora"), "kind": kind, "prompt": p, "ctrl": ctrl, "target": target, "imgs": {}}
        for s in spec["strengths"]:
            dst = f"{out}/{r['id']}_{kind}_s{s}.jpg"
            if not os.path.exists(dst):
                t = time.time(); src = gen(p, ctrl, w, h, s)
                os.system(f"cp '{src}' '{dst}'"); print(f"{r['id']} {kind} s={s} {time.time()-t:.0f}s", flush=True)
            row["imgs"][str(s)] = dst
        manifest.append(row)
        json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1, ensure_ascii=False)
print("DONE", len(manifest), "rows")
