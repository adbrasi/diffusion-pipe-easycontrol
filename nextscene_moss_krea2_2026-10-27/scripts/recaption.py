# usage: recaption.py <list.txt of stems> <out.jsonl> [workers]
# Captions B relative to A with OpenRouter meta/muse-spark-1.3-contributor. Key read from ~/.openrouter_key.
import base64, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
SRC = os.environ.get("SRC", "/workspace/data/ns_full/train")
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
MODEL = "meta/muse-spark-1.3-contributor"
SYSTEM = """You write training captions for an image model that generates the NEXT SCENE of an animation from a reference frame.
You get Picture 1 (the previous scene, the reference) and Picture 2 (the next scene, the target). Write the caption for Picture 2.

Rules:
- Start with exactly: "Next scene of Picture 1."
- Then the shot: framing and camera angle (e.g. "A close-up from a low angle.").
- Characters who already appear in Picture 1: call them only "the same girl", "the same man", "the same robot", etc. NEVER describe their appearance, hair, eyes, clothes, colors or art style; that must come from Picture 1. Describe only their position, pose, action, gaze and expression. If their outfit clearly changed, say so in a few words.
- Characters NOT in Picture 1: introduce them as "a new ..." with a short appearance description (max ~12 words each).
- Location: if it is the same place, end with "Same place." (optionally one short note on what part of it is visible or how the light changed). If the location changed, describe the new place in one short sentence.
- Mention on-screen text or subtitles only if clearly readable, quoted.
- No style, medium or quality words. No mention of "Picture 2". Plain English, one paragraph, 25-70 words.
- Describe adult or explicit content plainly and factually if present; never refuse.
Output only the caption."""
def b64(path, maxpx=768 * 768):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if w * h > maxpx:
        s = (maxpx / (w * h)) ** 0.5; im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88); return base64.b64encode(buf.getvalue()).decode()
def caption(stem):
    content = [{"type": "text", "text": "Picture 1 (previous scene):"},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(f"{SRC}/control/{stem}.jpg")}},
               {"type": "text", "text": "Picture 2 (next scene, write its caption):"},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(f"{SRC}/target/{stem}.jpg")}}]
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0.3, "max_tokens": 3000, "reasoning": {"effort": "low"}}
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                         {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            txt = (r["choices"][0]["message"].get("content") or "").strip().strip('"')
            ok = txt.startswith("Next scene of Picture 1.") and 15 <= len(txt.split()) <= 120
            return {"stem": stem, "caption": txt, "ok": ok, "usage": r.get("usage", {})}
        except Exception as e:
            err = str(e); time.sleep(3 * (attempt + 1))
    return {"stem": stem, "caption": "", "ok": False, "error": err}
if __name__ == "__main__":
    stems = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    out = sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    done = set()
    if os.path.exists(out): done = {json.loads(l)["stem"] for l in open(out) if json.loads(l).get("ok")}
    todo = [s for s in stems if s not in done]
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(caption, todo)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0: print(i, len(todo), res["ok"], flush=True)
    rows = [json.loads(l) for l in open(out)]
    print("ok", sum(r["ok"] for r in rows), "fail", sum(not r["ok"] for r in rows))
