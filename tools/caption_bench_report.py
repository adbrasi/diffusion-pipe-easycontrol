#!/usr/bin/env python3
"""Página HTML de avaliação do benchmark de captioners (schema v2).

Lê os JSONs de caption_bench.py, embute as imagens em base64 (CSP do
Artifact bloqueia recursos externos) e mostra, por par e por modelo:
  - a taxonomia escolhida (pair_type) e quais eixos são herdados
  - a caption de treino que sairia daquele JSON (build_caption)
  - vazamento de aparência quando o eixo é herdado (o erro que mata o dataset)
  - custo REAL em USD, extrapolado para o dataset inteiro
"""
import os
import json
import base64
import io
import glob
import html
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caption_schema import build_caption, LEAK_WORDS

DS = '/workspace/dataset_raw/extracted'
BENCH = '/workspace/outputs/_caption_bench'
OUT = f'{BENCH}/avaliacao.html'
THUMB = 460
N_PARES_DATASET = 1255

GABARITO = {
    'imagem000180': dict(
        char=True, place=True, refchar=True,
        nota='Continuidade real: mesma garota + mesmo homem de cabelo vermelho, mesmo dojo, câmera girou. '
             'Deve herdar personagem, lugar, estilo e paleta; só a câmera muda.'),
    'imagem000853': dict(
        char=False, place=False, refchar=False,
        nota='FALSO POSITIVO do dataset (marcado "same character"). A referência é uma mesa de comida vista '
             'de cima, sem personagem legível. O correto é reference_has_character=false e herdar só estilo.'),
    'imagem001700': dict(
        char=True, place=False, refchar=True,
        nota='FALSO POSITIVO parcial. O garoto parece ser o mesmo, mas o 2º personagem troca (homem musculoso '
             '-> garota) e o lugar muda (cozinha -> cafeteria). "same_plus_new" é a leitura mais fina; '
             'tratar como personagem totalmente novo também é defensável.'),
    'imagem001491': dict(
        char=False, place=False, refchar=True,
        nota='A referência tem MULHERES, o alvo tem um garoto ruivo: personagens diferentes. Mas HÁ personagens '
             'na referência — então reference_has_character deve ser TRUE. Pega quem confunde '
             '"personagem diferente" com "sem personagem".'),
}


def b64(path, size=THUMB):
    im = Image.open(path).convert('RGB')
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def load():
    merged = {}
    for f in glob.glob(f'{BENCH}/results_*.json'):
        with open(f) as fh:
            try:
                data = json.load(fh)
            except Exception:
                continue
        for model, cases in data.items():
            reasoning = None
            for c in cases.values():
                if isinstance(c, dict) and c.get('reasoning'):
                    reasoning = c['reasoning']; break
            label = f'{model}  ·  thinking={reasoning}' if reasoning else model
            merged.setdefault(label, {}).update(cases)
    return merged


def leak(js):
    """A caption descreveu algo que deveria ser herdado?"""
    inh = js.get('inherited') or {}
    act = (js.get('action') or '').lower()
    bad = []
    if inh.get('character_identity') and any(w in act for w in LEAK_WORDS):
        bad.append('aparência de personagem herdado')
    if inh.get('place_identity') and js.get('new_place'):
        bad.append('descreveu lugar herdado')
    if inh.get('palette_lighting') and js.get('palette_shift'):
        bad.append('descreveu paleta herdada')
    return bad


def score_pair(js, gab):
    """(acertos, total) nos 3 campos que mais importam."""
    inh = js.get('inherited') or {}
    checks = [
        (inh.get('character_identity'), gab['char']),
        (inh.get('place_identity'), gab['place']),
        (js.get('reference_has_character'), gab['refchar']),
    ]
    return sum(1 for a, b in checks if a == b), len(checks)


def main():
    res = load()
    stems = list(GABARITO)
    models = sorted(res)

    agg = {}
    for m in models:
        hit = tot = 0
        leaks = 0
        usd = []
        fails = 0
        for s in stems:
            e = res[m].get(s) or {}
            js = e.get('json')
            if not js:
                fails += 1; tot += 3; continue
            h, t = score_pair(js, GABARITO[s]); hit += h; tot += t
            leaks += bool(leak(js))
            c = (e.get('cost') or {}).get('usd')
            if c:
                usd.append(c)
        agg[m] = dict(hit=hit, tot=tot, leaks=leaks, fails=fails,
                      usd=(sum(usd)/len(usd)) if usd else None)

    P = []; A = P.append
    A('<title>Captioners — qual LLM classifica os pares corretamente</title>')
    A('''<style>
:root{
  --ink:#151b21;--soft:#5b6975;--faint:#8c99a4;--bg:#f5f7f8;--card:#fff;--line:#dee4e9;
  --acc:#1d6fa5;--ok:#137a4d;--okbg:#e4f3ea;--warn:#8a6000;--warnbg:#fdf2da;
  --err:#a32727;--errbg:#fbe8e8;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media(prefers-color-scheme:dark){:root{
  --ink:#e7edf2;--soft:#9dabb7;--faint:#6f7d89;--bg:#0e1317;--card:#161d23;--line:#293238;
  --acc:#5cabdd;--ok:#5fd097;--okbg:#112a1e;--warn:#e2b75c;--warnbg:#2c2411;
  --err:#f18c8c;--errbg:#2d1515;}}
:root[data-theme=dark]{--ink:#e7edf2;--soft:#9dabb7;--faint:#6f7d89;--bg:#0e1317;--card:#161d23;
  --line:#293238;--acc:#5cabdd;--ok:#5fd097;--okbg:#112a1e;--warn:#e2b75c;--warnbg:#2c2411;
  --err:#f18c8c;--errbg:#2d1515;}
:root[data-theme=light]{--ink:#151b21;--soft:#5b6975;--faint:#8c99a4;--bg:#f5f7f8;--card:#fff;
  --line:#dee4e9;--acc:#1d6fa5;--ok:#137a4d;--okbg:#e4f3ea;--warn:#8a6000;--warnbg:#fdf2da;
  --err:#a32727;--errbg:#fbe8e8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans);padding:26px 20px 70px}
.wrap{max-width:1280px;margin:0 auto;display:flex;flex-direction:column;gap:26px}
h1{font-size:21px;margin:0;letter-spacing:-.01em}
h1 span{color:var(--faint);font-weight:400}
.sub{color:var(--soft);font-size:14px;margin:0;max-width:74ch}
.rule{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:5px;padding:12px 15px;font-size:14px}
.rule b{color:var(--acc)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(258px,1fr));gap:11px}
.c{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:13px 15px;
  display:flex;flex-direction:column;gap:8px}
.c .m{font-family:var(--mono);font-size:12px;color:var(--acc);word-break:break-all}
.big{display:flex;align-items:baseline;gap:7px}
.big b{font-size:25px;line-height:1;font-variant-numeric:tabular-nums}
.big .of{color:var(--faint);font-size:13px}
.kv{font-family:var(--mono);font-size:11.5px;color:var(--faint);display:flex;gap:11px;flex-wrap:wrap;
  font-variant-numeric:tabular-nums}
.kv .money{color:var(--ink);font-weight:600}
sec{display:block}
.pair{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pair>header{padding:12px 15px;border-bottom:1px solid var(--line);display:flex;gap:9px;
  align-items:baseline;flex-wrap:wrap}
.pair h2{font-family:var(--mono);font-size:14px;margin:0}
.gab{font-family:var(--mono);font-size:11px;background:var(--bg);border:1px solid var(--line);
  border-radius:4px;padding:2px 7px;color:var(--soft)}
.nota{padding:10px 15px;background:var(--bg);border-bottom:1px solid var(--line);
  font-size:13.5px;color:var(--soft)}
.ims{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
.ims figure{margin:0;padding:9px;background:var(--card)}
.ims img{width:100%;display:block;border-radius:3px}
.ims figcaption{font-family:var(--mono);font-size:10.5px;color:var(--faint);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:6px}
.row{padding:12px 15px;border-top:1px solid var(--line);display:grid;
  grid-template-columns:194px 168px 1fr;gap:14px;align-items:start}
.mdl{font-family:var(--mono);font-size:11.5px;color:var(--acc);word-break:break-all}
.tag{font-family:var(--mono);font-size:11px;padding:2px 6px;border-radius:3px;font-weight:600;
  display:inline-block}
.ok{background:var(--okbg);color:var(--ok)}.warn{background:var(--warnbg);color:var(--warn)}
.err{background:var(--errbg);color:var(--err)}
.axes{font-family:var(--mono);font-size:12px;letter-spacing:.13em;font-weight:700}
.axes i{font-style:normal;color:var(--faint);font-weight:400}
.lbl{font-family:var(--mono);font-size:10px;color:var(--faint);text-transform:uppercase;
  letter-spacing:.06em;display:block;margin-bottom:2px}
.blk{font-size:13.5px}.blk+.blk{margin-top:8px}
.cap{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:9px 11px;
  font-family:var(--mono);font-size:12px;white-space:pre-wrap;line-height:1.45}
.leak{color:var(--err);font-weight:600}
@media(max-width:900px){.row{grid-template-columns:1fr;gap:7px}.ims{grid-template-columns:1fr}}
</style>''')
    A('<div class="wrap">')
    A('<div style="display:flex;flex-direction:column;gap:9px">')
    A('<h1>Captioners <span>— qual LLM entende os pares</span></h1>')
    A('<p class="sub">Cada modelo recebe as duas imagens numa chamada, com <code>thinking</code> ligado, '
      'e devolve JSON. A caption de treino é montada em código a partir desse JSON.</p>')
    A('<div class="rule">A regra que o dataset ensina: <b>o que está escrito é o que muda; o que não está '
      'escrito é herdado da referência.</b> Por isso o erro mais grave não é errar a classificação — é '
      '<b>descrever algo que deveria ser herdado</b>, porque aí o texto basta e o modelo aprende a ignorar '
      'a referência.</div>')
    A('<p class="sub">Os eixos são independentes: <b>C</b>haracter · <b>P</b>lace · <b>S</b>tyle · '
      'palette/<b>L</b>ighting · camera/<b>K</b>. Letra presente = herdado da referência. '
      'Estilo e paleta costumam ser herdados mesmo quando todo o conteúdo muda — é isso que permite usar '
      'a referência só como guia de estilo.</p>')
    A('</div>')

    A('<div class="cards">')
    for m in sorted(models, key=lambda x: (-agg[x]['hit'], agg[x]['leaks'], agg[x]['usd'] or 9)):
        a = agg[m]
        usd = a['usd']
        tot_ds = f'${usd*N_PARES_DATASET:.2f}' if usd else '—'
        A(f'<div class="c"><div class="m">{html.escape(m)}</div>'
          f'<div class="big"><b>{a["hit"]}</b><span class="of">/ {a["tot"]} campos certos</span>'
          + (f'<span class="tag err">{a["leaks"]} vazam</span>' if a['leaks'] else '<span class="tag ok">0 vazam</span>')
          + (f'<span class="tag err">{a["fails"]} falhas</span>' if a['fails'] else '')
          + '</div>'
          f'<div class="kv"><span class="money">{tot_ds} / {N_PARES_DATASET} pares</span>'
          + (f'<span>${usd:.6f}/par</span>' if usd else '') + '</div></div>')
    A('</div>')

    for s in stems:
        g = GABARITO[s]
        A('<sec class="pair">')
        A(f'<header><h2>{s}</h2><span class="gab">gabarito: herda personagem={g["char"]} · '
          f'herda lugar={g["place"]} · ref tem personagem={g["refchar"]}</span></header>')
        A(f'<div class="nota">{html.escape(g["nota"])}</div>')
        A('<div class="ims">')
        for side, lb in (('input_A', 'A — referência'), ('input_B', 'B — alvo')):
            p = f'{DS}/{side}/{s}.jpg'
            if os.path.exists(p):
                A(f'<figure><figcaption>{lb}</figcaption><img src="data:image/jpeg;base64,{b64(p)}" alt="{lb}"></figure>')
        A('</div>')
        for m in models:
            e = res[m].get(s) or {}
            js = e.get('json')
            A('<div class="row">')
            A(f'<div class="mdl">{html.escape(m)}</div>')
            if not js:
                A('<div><span class="tag err">falhou</span></div>')
                A(f'<div class="blk">{html.escape(str(e.get("error","sem resposta"))[:260])}</div></div>')
                continue
            inh = js.get('inherited') or {}
            axes = ''.join(f'{L}' if inh.get(k) else '<i>·</i>' for L, k in
                           (('C', 'character_identity'), ('P', 'place_identity'), ('S', 'art_style'),
                            ('L', 'palette_lighting'), ('K', 'camera_language')))
            h, t = score_pair(js, g)
            cls = 'ok' if h == t else ('warn' if h >= t - 1 else 'err')
            A('<div>')
            A(f'<span class="tag {cls}">{h}/{t} campos</span> <span class="axes">{axes}</span><br>')
            A(f'<span class="lbl" style="margin-top:6px">{html.escape(str(js.get("pair_type")))}</span>')
            A(f'<span class="kv">refchar={js.get("reference_has_character")} · conf={js.get("confidence")}</span>')
            c = (e.get('cost') or {}).get('usd')
            rt = ((e.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
            if c:
                A(f'<span class="kv">${c:.6f}' + (f' · {rt} tok raciocínio' if rt else '') + '</span>')
            A('</div>')
            A('<div>')
            lk = leak(js)
            if lk:
                A(f'<div class="blk"><span class="lbl leak">⚠ vazamento: {html.escape(", ".join(lk))}</span></div>')
            try:
                cap = build_caption(js)
            except Exception as ex:
                cap = f'(erro montando caption: {ex})'
            A(f'<div class="blk"><span class="lbl">caption de treino que sairia</span>'
              f'<div class="cap">{html.escape(cap)}</div></div>')
            if js.get('reject_reason'):
                A(f'<div class="blk"><span class="lbl">reject_reason</span>{html.escape(str(js["reject_reason"]))}</div>')
            A('</div></div>')
        A('</sec>')

    A('</div>')
    os.makedirs(BENCH, exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(P))
    print(f'{OUT} ({os.path.getsize(OUT)/1024:.0f} KB, {len(models)} modelos)')


if __name__ == '__main__':
    main()
