#!/usr/bin/env python3
"""Legenda delta (Picture 2 relativa à Picture 1) com OpenRouter meta/muse-spark-1.3-contributor.

  python tools/delta_big_caption.py <prompt.txt> <jobs.jsonl> <out.jsonl> [workers]
jobs.jsonl: {"key", "ref", "target"}. Retoma: pula as keys já presentes em out.jsonl com ok=true.
"""
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

MODEL = 'meta/muse-spark-1.3-contributor'
KEY = open(os.environ.get('OPENROUTER_KEY_FILE', '/workspace/clean_forcing/.openrouter_key')).read().strip()


def b64(path, maxpx=768 * 768):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    if w * h > maxpx:
        s = (maxpx / (w * h)) ** 0.5
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def valid(t):
    # o prompt_final atual não pede mais o prefixo "Next scene of Picture 1." (as legendas antigas o tinham)
    return 15 <= len(t.split()) <= 160 and 'Picture 2' not in t


def caption(system, job):
    content = [{'type': 'text', 'text': 'Picture 1 (previous scene, the reference):'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b64(job['ref'])}},
               {'type': 'text', 'text': 'Picture 2 (next scene, write its caption):'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b64(job['target'])}}]
    body = {'model': MODEL, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'temperature': 0.3, 'max_tokens': 3000, 'reasoning': {'effort': 'low'}}
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', json.dumps(body).encode(),
                                         {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'})
            r = json.load(urllib.request.urlopen(req, timeout=240))
            t = (r['choices'][0]['message'].get('content') or '').strip().strip('"')
            cost = (r.get('usage') or {}).get('cost', 0) or 0
            if valid(t):
                return {'key': job['key'], 'caption': t, 'ok': True, 'cost': cost}
            last = 'invalid: ' + t[:200]
        except urllib.error.HTTPError as e:
            last = f'HTTP {e.code}'
            if e.code in (401, 403):
                raise SystemExit(f'auth error {e.code}')
            time.sleep(5 * (attempt + 1))
        except Exception as e:  # noqa
            last = type(e).__name__
            time.sleep(5 * (attempt + 1))
    return {'key': job['key'], 'ok': False, 'error': last}


def main():
    system = open(sys.argv[1]).read().strip()
    jobs = [json.loads(l) for l in open(sys.argv[2])]
    out = sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if r.get('ok'):
                done.add(r['key'])
    todo = [j for j in jobs if j['key'] not in done]
    lock = threading.Lock()
    f = open(out, 'a')
    stats = {'n': 0, 'ok': 0, 'cost': 0.0}

    def run(j):
        r = caption(system, j)
        with lock:
            f.write(json.dumps(r) + '\n')
            f.flush()
            stats['n'] += 1
            stats['ok'] += r['ok']
            stats['cost'] += r.get('cost', 0)
            if stats['n'] % 200 == 0:
                print(f"{stats['n']}/{len(todo)} ok={stats['ok']} custo=${stats['cost']:.3f}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))
    print(f"fim: {stats['n']} novos, ok={stats['ok']}, custo=${stats['cost']:.3f}")


if __name__ == '__main__':
    main()
