# rows = Krea eval cases; columns: A | B | MOSS v4 (1.0) | Krea columns
import json, os, sys, textwrap
from PIL import Image, ImageDraw, ImageFont
cases = json.load(open(sys.argv[1])); kdir = sys.argv[2]; cols = sys.argv[3].split(","); title = sys.argv[4]; dst = sys.argv[5]
W, H, LBL, HDR, PAD = 480, 270, 420, 34, 6
F = lambda s, b=False: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf", s)
heads = ["A (referência)", "B (real)", "MOSS v4 · LoRA 1.0"] + [{"native": "Krea2 SEM LoRA (Turbo)", "lora_turbo": "Krea2 + LoRA (Turbo 8 passos)", "lora_raw": "Krea2 + LoRA (Raw 28 passos)", "s250": "A1000 · passo 250", "s500": "A1000 · passo 500", "s750": "A1000 · passo 750", "s1000": "A1000 · passo 1000", "rawv3": "Krea2 raw-v3 (anterior) · Turbo", "a1000": "Krea2 receita A1000 · Turbo"}.get(c, c) for c in cols]
g = Image.new("RGB", (LBL + len(heads) * (W + PAD), 90 + HDR + len(cases) * (H + PAD)), (16, 16, 18)); d = ImageDraw.Draw(g)
d.text((16, 14), title, fill=(255, 255, 255), font=F(38, True))
for j, c in enumerate(heads): d.text((LBL + j * (W + PAD) + 8, 90), c, fill=(255, 210, 80) if j < 3 else (120, 200, 255), font=F(21, True))
y, prev = 90 + HDR, None
for c in cases:
    ref, kind = c["id"].split("__")
    if prev is not None and ref != prev: d.line([(0, y - 4), (g.width, y - 4)], fill=(110, 110, 120), width=3)
    prev = ref
    d.text((10, y + 6), ref, fill=(140, 200, 255), font=F(20, True)); d.text((10, y + 34), kind.upper(), fill=(255, 210, 80), font=F(17, True))
    lines = textwrap.wrap(c["prompt"].replace("Next scene of Picture 1. ", ""), 46)
    if len(lines) > 10: lines = lines[:10]; lines[-1] += " …"
    d.multiline_text((10, y + 60), "\n".join(lines), fill=(205, 205, 210), font=F(15), spacing=3)
    paths = [c["ref"], c.get("target"), f"/workspace/evals/v4_final/{ref}_{kind}_s1.0.jpg"] + [f"{kdir}/{c['id']}_s42_{col}.png" for col in cols]
    for j, p in enumerate(paths):
        x = LBL + j * (W + PAD)
        if not p or not os.path.exists(p):
            d.rectangle([x, y, x + W, y + H], fill=(34, 34, 38)); d.text((x + W // 2 - 60, y + H // 2 - 10), "(não existe)", fill=(120, 120, 120), font=F(18)); continue
        im = Image.open(p).convert("RGB"); im.thumbnail((W, H), Image.LANCZOS); g.paste(im, (x + (W - im.width) // 2, y + (H - im.height) // 2))
    y += H + PAD
g.save(dst, quality=90); print(dst, g.size)
