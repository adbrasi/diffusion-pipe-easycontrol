# usage: summary.py <test_out_dir> <title> <dst.jpg>
import json, sys, textwrap
from PIL import Image, ImageDraw, ImageFont
out, title, dst = sys.argv[1], sys.argv[2], sys.argv[3]
rows = json.load(open(f"{out}/manifest.json"))
strengths = list(rows[0]["imgs"].keys())
W, H, LBL, HDR, PAD = 480, 270, 420, 34, 6
F = lambda s, b=False: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf", s)
cols = ["A (referência)", "B (real)"] + [("sem LoRA" if float(s) == 0 else f"LoRA força {s}") for s in strengths]
KIND = {"identico": "PROMPT IDÊNTICO (legenda do dataset)", "parecido": "PROMPT PARECIDO (outra ação/câmera)", "diferente": "PROMPT DIFERENTE (curto / outro lugar)"}
SPLIT = {"val": "validação (não visto no treino)", "train": "TREINO (visto)", "fora": "FORA do dataset"}
total_w = LBL + len(cols) * (W + PAD)
total_h = 90 + HDR + len(rows) * (H + PAD)
g = Image.new("RGB", (total_w, total_h), (16, 16, 18)); d = ImageDraw.Draw(g)
d.text((16, 14), title, fill=(255, 255, 255), font=F(40, True))
for j, c in enumerate(cols):
    d.text((LBL + j * (W + PAD) + 8, 90), c, fill=(255, 210, 80), font=F(24, True))
y = 90 + HDR
prev = None
for r in rows:
    if r["ref"] != prev and prev is not None:
        d.line([(0, y - PAD // 2 - 1), (total_w, y - PAD // 2 - 1)], fill=(110, 110, 120), width=3)
    prev = r["ref"]
    d.text((10, y + 6), f"{r['ref']}  ·  {SPLIT[r['split']]}", fill=(140, 200, 255), font=F(20, True))
    d.text((10, y + 34), KIND[r["kind"]], fill=(255, 210, 80), font=F(17, True))
    p = r["prompt"].replace("Next scene of Picture 1. ", "")
    lines = textwrap.wrap(p, 46)
    if len(lines) > 10: lines = lines[:10]; lines[-1] += " …"
    d.multiline_text((10, y + 60), "\n".join(lines), fill=(205, 205, 210), font=F(15), spacing=3)
    for j, path in enumerate([r["ctrl"], r["target"]] + [r["imgs"][s] for s in strengths]):
        x = LBL + j * (W + PAD)
        if path is None:
            d.rectangle([x, y, x + W, y + H], fill=(34, 34, 38)); d.text((x + W // 2 - 50, y + H // 2 - 10), "(não existe)", fill=(120, 120, 120), font=F(18))
            continue
        im = Image.open(path).convert("RGB"); im.thumbnail((W, H), Image.LANCZOS)
        g.paste(im, (x + (W - im.width) // 2, y + (H - im.height) // 2))
    y += H + PAD
g.save(dst, quality=90)
print(dst, g.size)
