# usage: prep_big.py <src full_git> <dst>   (ai-toolkit edit format, shared by MOSS v3 and Krea2)
import os, re, sys, json
from multiprocessing import Pool
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
VAL = ["ds2_004705", "ds4_imagem000073", "ds4_imagem000305", "ds4_imagem000381", "ds4_imagem000511", "ds4_imagem001191"]
def num(stem):
    m = re.match(r"(ds[24])_(?:imagem)?(\d+)$", stem); return (m.group(1), int(m.group(2))) if m else None
held = {num(v) for v in VAL}
def excluded(stem):
    n = num(stem)
    return n is not None and any(n[0] == h[0] and abs(n[1] - h[1]) <= 8 for h in held)
jobs, report = [], {}
for sub in sorted(d for d in os.listdir(src) if d.startswith("ds")):
    A, B = f"{src}/{sub}/images_A", f"{src}/{sub}/images_B"
    a = {os.path.splitext(f)[0]: f for f in os.listdir(A) if f.lower().endswith((".jpg", ".png", ".webp"))}
    b = {os.path.splitext(f)[0]: f for f in os.listdir(B) if f.lower().endswith((".jpg", ".png", ".webp"))}
    ok = [s for s in a if s in b and os.path.exists(f"{B}/{s}.txt")]
    n_tr = n_val = n_ex = 0
    for s in ok:
        cap = open(f"{B}/{s}.txt").read().strip()
        if len(cap) < 40: n_ex += 1; continue
        if s in VAL: split = "val"; n_val += 1
        elif excluded(s): n_ex += 1; continue
        else: split = "train"; n_tr += 1
        jobs.append((f"{A}/{a[s]}", f"{B}/{b[s]}", cap, split, s))
    report[sub] = {"A": len(a), "B": len(b), "pares": len(ok), "train": n_tr, "val": n_val, "excluidos": n_ex}
for sp in ["train", "val"]:
    for k in ["target", "control"]: os.makedirs(f"{dst}/{sp}/{k}", exist_ok=True)
def save(pin, pout, maxpx=1024 * 1024):
    im = Image.open(pin).convert("RGB"); w, h = im.size
    if w * h > maxpx:
        s = (maxpx / (w * h)) ** 0.5; im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    im.save(pout, quality=93)
def work(j):
    a, b, cap, sp, s = j
    try:
        save(a, f"{dst}/{sp}/control/{s}.jpg"); save(b, f"{dst}/{sp}/target/{s}.jpg")
        open(f"{dst}/{sp}/target/{s}.txt", "w").write("Next scene of Picture 1. " + cap)
        return 1
    except Exception as e:
        print("skip", s, e); return 0
with Pool(16) as p: done = sum(p.map(work, jobs, chunksize=32))
report["total_escritos"] = done
json.dump(report, open(f"{dst}/report.json", "w"), indent=1); print(json.dumps(report, indent=1))
