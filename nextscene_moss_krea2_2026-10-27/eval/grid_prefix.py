import json, textwrap
from PIL import Image, ImageDraw, ImageFont
rows = json.load(open("/workspace/evals/ostris_prefix/manifest.json"))
by = {}
for r in rows: by.setdefault(r["ref"], {})[r["kind"]] = r
W, H, L = 560, 315, 380
F = lambda s, b=False: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf", s)
g = Image.new("RGB", (L + 3 * (W + 6), 80 + len(by) * (H + 6)), (16, 16, 18)); d = ImageDraw.Draw(g)
d.text((14, 12), "Krea 2 · método Ostris (passo 1000, Raw cfg 4): COM vs SEM o prefixo 'Next scene of Picture 1.'", fill=(255, 255, 255), font=F(26, True))
for j, t in enumerate(["A (referência)", "COM prefixo", "SEM prefixo"]): d.text((L + j * (W + 6) + 8, 50), t, fill=(255, 210, 80), font=F(20, True))
y = 80
for ref, k in by.items():
    d.text((10, y + 6), ref, fill=(140, 200, 255), font=F(18, True))
    d.multiline_text((10, y + 34), "\n".join(textwrap.wrap(k["diferente"]["prompt"], 40)[:9]), fill=(205, 205, 210), font=F(14))
    for j, p in enumerate([k["parecido"]["ctrl"], k["parecido"]["imgs"]["1.0"], k["diferente"]["imgs"]["1.0"]]):
        im = Image.open(p).convert("RGB"); im.thumbnail((W, H)); g.paste(im, (L + j * (W + 6) + (W - im.width) // 2, y + (H - im.height) // 2))
    y += H + 6
g.save("/workspace/outputs_for_user/KREA2_OSTRIS_com_vs_sem_prefixo.jpg", quality=90); print("ok")
