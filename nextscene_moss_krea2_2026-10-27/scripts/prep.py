import json, os, random, re, sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
items = json.load(open(f"{src}/captions.json"))["items"]
nsfw = re.compile(r"nude|naked|breast|chest|nipple|sex|bikini|lingerie|underwear|pant(y|ies)|panties|bare|thigh|butt|crotch|genital|pixelat|moan|lewd|bed|sweat|kiss|lick|blush|flushed|towel|bath|shower|strap|stocking|cleavage|bra\b", re.I)
import json as _j
tags={x["filename"]:x["caption"].split(". ")[0] for x in (lambda d: d["items"] if isinstance(d,dict) else d)(_j.load(open(sys.argv[3])))}
bad=re.compile(r"\b(nude|completely nude|nipples|pussy|penis|sex|vaginal|oral|fellatio|cum|cunnilingus|paizuri|anus|after sex|erection|testicles|masturbation|breasts|cleavage|bikini|underwear|panties|lingerie|ass|swimsuit|bra|topless|censored|mosaic censoring)\b")
items = [x for x in items if os.environ.get("NOFILTER") or (x["filename"].split("_")[0] in ("ds2","ds4") and not bad.search(tags.get(x["filename"],"")))]
items = [x for x in items if len(x["caption"]) >= 40 and os.path.exists(f"{src}/images_B/{x['filename']}")]
random.seed(0)
safe = [x for x in items if not nsfw.search(x["caption"])]
random.shuffle(safe)
val = safe[:24]  # candidates; final pick after visual check
val_names = {x["filename"] for x in val}
for sub in ["train/target", "train/control", "val/target", "val/control"]:
    os.makedirs(f"{dst}/{sub}", exist_ok=True)
def save(path_in, path_out, maxpx=1280*720):
    im = Image.open(path_in).convert("RGB")
    w, h = im.size
    if w * h > maxpx:
        s = (maxpx / (w * h)) ** 0.5
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    im.save(path_out, quality=95)
n = 0
for x in items:
    split = "val" if x["filename"] in val_names else "train"
    stem = os.path.splitext(x["filename"])[0]
    save(f"{src}/images_A/{x['filename']}", f"{dst}/{split}/control/{stem}.jpg")
    save(f"{src}/images_B/{x['filename']}", f"{dst}/{split}/target/{stem}.jpg")
    open(f"{dst}/{split}/target/{stem}.txt", "w").write("Next scene of Picture 1. " + x["caption"].strip())
    n += 1
print("total", n, "val", len(val), "safe", len(safe))
