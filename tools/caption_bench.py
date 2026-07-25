#!/usr/bin/env python3
"""Benchmarka LLMs de visão do OpenRouter na tarefa de recaptionar pares.

Testa em pares escolhidos a dedo, incluindo FALSOS POSITIVOS conhecidos do
dataset atual — se o modelo não detectar que a "referência" não tem
personagem nenhum, ele não serve.

Uso: caption_bench.py [--models m1,m2] [--mode single|multiturn]
"""
import os
import sys
import json
import time
import base64
import argparse
import io
import urllib.request
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caption_schema import SYSTEM_PROMPT, USER_PROMPT, build_caption

DS = '/workspace/dataset_raw/extracted'
API = 'https://openrouter.ai/api/v1/chat/completions'
MAX_SIDE = 768  # reduz custo de tokens de imagem

# pares de teste: o gabarito é meu julgamento visual, anotado à mão
CASES = {
    'imagem000180': 'ESPERADO same character (Demon Slayer, anime, continuidade real)',
    'imagem000853': 'FALSO POSITIVO: referência é mesa de comida vista de cima, SEM personagem',
    'imagem001700': 'FALSO POSITIVO: 2o personagem muda (homem musculoso -> garota)',
    'imagem001491': 'ESPERADO new character (marcado assim no dataset)',
}


def b64(path):
    im = Image.open(path).convert('RGB')
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def call(model, messages, key, max_tokens=8000, reasoning=None):
    payload = {
        'model': model, 'messages': messages,
        'max_tokens': max_tokens, 'temperature': 0.1,
    }
    if reasoning:
        # OpenRouter: ativa raciocinio em modelos que suportam o parametro.
        # A tarefa (comparar 2 imagens + julgar continuidade com ceticismo)
        # se beneficia de raciocinio; medido em 2026-07-25.
        payload['reasoning'] = {'effort': reasoning}
        payload['max_tokens'] = max(max_tokens, 8000)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API, data=body, headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    gen_id = d.get('id')
    msg = d['choices'][0]['message']
    txt = msg.get('content')
    if not txt:
        # alguns modelos de raciocinio devolvem tudo em 'reasoning' e nada em
        # 'content' quando o budget de tokens acaba durante o pensamento
        rz = msg.get('reasoning') or msg.get('reasoning_content') or ''
        fin = d['choices'][0].get('finish_reason')
        raise RuntimeError(
            f'content vazio (finish_reason={fin}, '
            f'reasoning={len(rz)} chars, completion_tokens='
            f'{(d.get("usage") or {}).get("completion_tokens")})')
    us = d.get('usage', {}) or {}
    us['_gen_id'] = gen_id
    return txt, dt, us


def real_cost(gen_id, key, tries=6):
    """Custo REAL em USD da geracao, do endpoint /generation do OpenRouter.
    Mais confiavel que estimar por preco de tabela: ja inclui tokens de
    raciocinio (cobrados como saida) e tokens de imagem."""
    if not gen_id:
        return None
    url = f'https://openrouter.ai/api/v1/generation?id={gen_id}'
    for i in range(tries):
        try:
            rq = urllib.request.Request(url, headers={'Authorization': f'Bearer {key}'})
            with urllib.request.urlopen(rq, timeout=30) as r:
                g = json.loads(r.read()).get('data') or {}
            if g.get('total_cost') is not None:
                return {
                    'usd': g.get('total_cost'),
                    'tokens_prompt': g.get('tokens_prompt'),
                    'tokens_completion': g.get('tokens_completion'),
                    'reasoning_tokens': g.get('native_tokens_reasoning'),
                    'native_prompt': g.get('native_tokens_prompt'),
                    'native_completion': g.get('native_tokens_completion'),
                }
        except Exception:
            pass
        time.sleep(1.5)
    return None


def parse_json(txt):
    t = txt.strip()
    if t.startswith('```'):
        t = t.split('```')[1]
        if t.startswith('json'):
            t = t[4:]
    t = t.strip()
    i, j = t.find('{'), t.rfind('}')
    if i >= 0 and j > i:
        t = t[i:j+1]
    return json.loads(t)


def run_single(model, stem, key, reasoning=None):
    """1 chamada, as duas imagens juntas."""
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': USER_PROMPT},
            {'type': 'text', 'text': 'IMAGE A (reference):'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64(f"{DS}/input_A/{stem}.jpg")}'}},
            {'type': 'text', 'text': 'IMAGE B (target):'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64(f"{DS}/input_B/{stem}.jpg")}'}},
        ]},
    ]
    return call(model, msgs, key, reasoning=reasoning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='qwen/qwen3.5-flash-02-23,google/gemini-2.5-flash-lite,qwen/qwen3-vl-32b-instruct,google/gemini-3.1-flash-lite')
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--reasoning', default=None, choices=['low','medium','high'],
                    help='ativa raciocinio (OpenRouter reasoning.effort)')
    args = ap.parse_args()

    key = None
    with open('/workspace/.secrets/openrouter.env') as f:
        for line in f:
            if line.startswith('OPENROUTER_API_KEY='):
                key = line.strip().split('=', 1)[1]
    assert key, 'sem chave'

    results = {}
    for model in args.models.split(','):
        print(f'\n{"="*78}\nMODELO: {model}\n{"="*78}')
        results[model] = {}
        for stem in args.cases.split(','):
            print(f'\n--- {stem} | {CASES.get(stem, "")}')
            try:
                txt, dt, us = run_single(model, stem, key, reasoning=args.reasoning)
                js = parse_json(txt)
                cost = {'usd': us.get('cost')}
                results[model][stem] = {'json': js, 'dt': dt, 'usage': us,
                                        'reasoning': args.reasoning, 'cost': cost}
                inh = js.get('inherited') or {}
                flags = ''.join([
                    'C' if inh.get('character_identity') else '-',
                    'P' if inh.get('place_identity') else '-',
                    'S' if inh.get('art_style') else '-',
                    'L' if inh.get('palette_lighting') else '-',
                    'K' if inh.get('camera_language') else '-'])
                print(f'  {js.get("pair_type"):<26} herda[CPSLK]={flags} '
                      f'refhaschar={js.get("reference_has_character")} conf={js.get("confidence")!r}')
                if js.get('reject_reason'):
                    print(f'  REJECT: {js["reject_reason"]}')
                print(f'  action: {(js.get("action") or "")[:150]}')
                if js.get('new_characters'):
                    print(f'  new_chars: {str(js["new_characters"])[:120]}')
                c = cost or {}
                usd = c.get('usd')
                rt = ((us.get('completion_tokens_details') or {}).get('reasoning_tokens'))
                print(f'  [{dt:.1f}s, in={us.get("prompt_tokens","?")} out={us.get("completion_tokens","?")}'
                      + (f', reason={rt}' if rt else '')
                      + (f', USD REAL={usd:.6f} -> 1255 pares = ${usd*1255:.2f}' if usd else '')
                      + ']')
            except Exception as e:
                print(f'  ERRO: {type(e).__name__}: {str(e)[:200]}')
                results[model][stem] = {'error': str(e)[:200]}

    tag = args.models.split(',')[0].replace('/','_').replace('.','')
    if args.reasoning: tag += f'_reason-{args.reasoning}'
    out = f'/workspace/outputs/_caption_bench/results_{tag}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nsalvo: {out}')


if __name__ == '__main__':
    main()
