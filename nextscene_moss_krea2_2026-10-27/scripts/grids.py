# usage: grids.py <run_name> -> /workspace/outputs_for_user/<run_name>/
import os, re, sys, shutil
from PIL import Image, ImageDraw, ImageFont
run = sys.argv[1]
val = "/workspace/data/ns_full/val"
sdir = f"/workspace/output/{run}/samples"
out = f"/workspace/outputs_for_user/{run}"
os.makedirs(out, exist_ok=True)
import yaml
_cp = f"/workspace/output/{run}/config.yaml"
cfg = yaml.safe_load(open(_cp if os.path.exists(_cp) else f"/workspace/scripts/{run}.yaml"))["config"]["process"][0]["sample"]["samples"]
pairs = []  # (name, control, target or None, prompt)
for x in cfg:
    c = x["ctrl_img_1"]; stem = os.path.splitext(os.path.basename(c))[0]
    t = c.replace("/control/", "/target/")
    pairs.append((stem, c, t if "/control/" in c and os.path.exists(t) else None, x["prompt"]))
by = {}
for f in os.listdir(sdir):
    m = re.match(r".*__(\d+)_(\d+)\.\w+$", f)
    if m: by.setdefault(int(m.group(2)), {})[int(m.group(1))] = f"{sdir}/{f}"
steps = sorted({s for d in by.values() for s in d})
W, H, T = 512, 288, 28
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception: font = ImageFont.load_default()
cols = ["A (referencia)", "B (real)"] + [f"passo {s}" for s in steps]
rows = []
for i, (stem, ctrl, tgt, prompt) in enumerate(pairs):
    ims = [ctrl, tgt] + [by.get(i, {}).get(s) for s in steps]
    row = Image.new("RGB", (W * len(ims), H + T), (20, 20, 20)); d = ImageDraw.Draw(row)
    for j, p in enumerate(ims):
        if p:
            im = Image.open(p).convert("RGB"); im.thumbnail((W, H)); row.paste(im, (j * W + (W - im.width) // 2, T))
        d.text((j * W + 6, 4), cols[j], fill=(255, 220, 80), font=font)
    row.save(f"{out}/par{i+1}_{stem}.jpg", quality=90); rows.append(row)
    open(f"{out}/par{i+1}_{stem}_prompt.txt", "w").write(prompt)
# overview: latest step only
if steps:
    last = steps[-1]; g = Image.new("RGB", (W * 3, (H + T) * len(pairs)), (20, 20, 20))
    for i, r in enumerate(rows):
        g.paste(r.crop((0, 0, 2 * W, H + T)), (0, i * (H + T)))
        g.paste(r.crop((r.width - W, 0, r.width, H + T)), (2 * W, i * (H + T)))
    g.save(f"{out}/resumo_passo{last}.jpg", quality=88)
print("steps", steps, "->", out)
