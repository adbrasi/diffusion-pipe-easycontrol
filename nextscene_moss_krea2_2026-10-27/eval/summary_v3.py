# summary.py variant: adds the v1 image (same ref + same prompt, strength 1.0) as a comparison column
import json, os, sys, textwrap
from PIL import Image, ImageDraw, ImageFont
out, title, dst = sys.argv[1], sys.argv[2], sys.argv[3]
rows = json.load(open(f"{out}/manifest.json"))
CMP = os.environ.get("CMP_DIR", "/workspace/evals/v1_final"); CMP_LABEL = os.environ.get("CMP_LABEL", "v1 · legendas longas (1.0)"); NEW_LABEL = os.environ.get("NEW_LABEL", "v3 · legendas delta")
v1rows = {(r["ref"], r["kind"]): r for r in json.load(open(f"{CMP}/manifest.json"))}
ALIAS = {"naovisto_tranca": "treino_tranca"}
strengths = list(rows[0]["imgs"].keys())
W, H, LBL, HDR, PAD = 480, 270, 420, 34, 6
F = lambda s, b=False: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf", s)
cols = ["A (referência)", "B (real)", CMP_LABEL] + [f"{NEW_LABEL} ({s})" for s in strengths]
KIND = {"identico": "PROMPT IDÊNTICO (legenda nova do dataset)", "parecido": "PROMPT PARECIDO (outra ação/câmera)", "parecido_delta": "PARECIDO no FORMATO DELTA (sem aparência)", "diferente": "PROMPT DIFERENTE (curto / outro lugar)"}
SPLIT = {"val": "validação (não visto)", "train": "TREINO (visto)", "fora": "FORA do dataset / não visto"}
g = Image.new("RGB", (LBL + len(cols) * (W + PAD), 90 + HDR + len(rows) * (H + PAD)), (16, 16, 18)); d = ImageDraw.Draw(g)
d.text((16, 14), title, fill=(255, 255, 255), font=F(40, True))
for j, c in enumerate(cols): d.text((LBL + j * (W + PAD) + 8, 90), c, fill=(255, 210, 80) if j < 3 else (120, 230, 140), font=F(22, True))
y, prev = 90 + HDR, None
for r in rows:
    if prev is not None and r["ref"] != prev: d.line([(0, y - 4), (g.width, y - 4)], fill=(110, 110, 120), width=3)
    prev = r["ref"]
    d.text((10, y + 6), f"{r['ref']}  ·  {SPLIT[r['split']]}", fill=(140, 200, 255), font=F(20, True))
    d.text((10, y + 34), KIND[r["kind"]], fill=(255, 210, 80), font=F(17, True))
    lines = textwrap.wrap(r["prompt"].replace("Next scene of Picture 1. ", ""), 46)
    if len(lines) > 10: lines = lines[:10]; lines[-1] += " …"
    d.multiline_text((10, y + 60), "\n".join(lines), fill=(205, 205, 210), font=F(15), spacing=3)
    v1 = v1rows.get((ALIAS.get(r["ref"], r["ref"]), r["kind"]))
    v1img = v1["imgs"].get("1.0") if v1 and v1["prompt"] == r["prompt"] else None
    for j, path in enumerate([r["ctrl"], r["target"], v1img] + [r["imgs"][s] for s in strengths]):
        x = LBL + j * (W + PAD)
        if not path or not os.path.exists(path):
            d.rectangle([x, y, x + W, y + H], fill=(34, 34, 38)); d.text((x + W // 2 - 60, y + H // 2 - 10), "(não existe)", fill=(120, 120, 120), font=F(18)); continue
        im = Image.open(path).convert("RGB"); im.thumbnail((W, H), Image.LANCZOS)
        g.paste(im, (x + (W - im.width) // 2, y + (H - im.height) // 2))
    y += H + PAD
g.save(dst, quality=90); print(dst, g.size)
