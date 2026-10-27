import json, os, shutil
src, dst = "/workspace/data/ns_full", "/workspace/data/ns_cap1000"
for split, jl in [("train", "/workspace/recap/cap1000.jsonl"), ("val", "/workspace/recap/val6.jsonl")]:
    for k in ("target", "control"): os.makedirs(f"{dst}/{split}/{k}", exist_ok=True)
    n = 0
    for l in open(jl):
        r = json.loads(l)
        if not r.get("ok"): continue
        s = r["stem"]
        shutil.copy(f"{src}/{split}/control/{s}.jpg", f"{dst}/{split}/control/{s}.jpg")
        shutil.copy(f"{src}/{split}/target/{s}.jpg", f"{dst}/{split}/target/{s}.jpg")
        open(f"{dst}/{split}/target/{s}.txt", "w").write(r["caption"]); n += 1
    print(split, n)
