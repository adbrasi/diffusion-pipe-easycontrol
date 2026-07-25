#!/usr/bin/env python3
"""System prompt + schema + builder de captions (v3).

Desenho por Opus 5 com contexto limpo, 2026-07-25. Substitui inteiramente as
versões anteriores — não é incremental.

=============================================================================
O CONTRATO
=============================================================================
    O QUE ESTÁ ESCRITO É O QUE FOI ESPECIFICADO.
    O QUE NÃO ESTÁ ESCRITO É HERDADO DA REFERÊNCIA.

Cada atributo escrito é um atributo que o modelo passa a obter do texto — e
por isso deixa de olhar a referência para obtê-lo. Cada atributo omitido é um
para o qual a referência vira a ÚNICA fonte. Sub-descrever é barato;
super-descrever destrói o dataset silenciosamente (foi o que mediu a bateria
de 2026-07-25: com captions exaustivas o adapter ignora a referência).

=============================================================================
DECISÕES DE DESENHO (e por que a v2 falhou)
=============================================================================
1. PONTEIRO DEÍTICO, não sufixo de metadados.
   "the same girl" (a) força a leitura da referência, (b) é exatamente o que
   o usuário digita na inferência, (c) é uma frase curta repetida milhares de
   vezes que o Qwen3-0.6B aprende como símbolo.
   O sufixo da v2 ("Character continuity: no character continuity.") falhava
   nos três: nunca aparece em inferência real, e é uma NEGAÇÃO — modelos
   pequenos codificam negação mal, então provavelmente ATIVAVA "character".

2. TRÊS TIERS por par (rich / normal / terse), sorteados por época.
   O mesmo alvo aparece com especificação alta e baixa. É isso que ensina
   literalmente "se eu não especificar, herda": a loss só fecha se o que
   sumiu do texto vier da referência.

3. BAN GLOBAL de estilo e paleta em TODA caption.
   Como nenhuma caption jamais contém estilo, estilo é 100% do tempo
   informação exclusiva da referência. É o que faz o caso "referência só de
   estilo" (Goku -> quase-Vegeta) funcionar sem nenhum modo especial.

4. ENUM, não boolean, no eixo de personagem.
   O falso positivo mais grave vem de colapsar "personagem diferente" com
   "sem personagem na referência". São outcomes distintos.

5. CHAIN-OF-THOUGHT DENTRO DO JSON, em ordem.
   Modelos baratos ignoram o canal de thinking com frequência. Campos de
   evidência que PRECEDEM o veredito forçam o condicionamento no próprio
   texto gerado — e ficam auditáveis como colunas (dá para filtrar o dataset
   por confidence, leaks != [], matched_features < 2).

6. Pares sem nada em comum NÃO são lixo — viram style_only, e são os mais
   valiosos do conjunto para ensinar herança de estilo.
"""

SYSTEM_PROMPT = """You annotate PAIRS of anime images for a reference-conditioned generation dataset.
IMAGE 1 = REFERENCE (the model sees it as pixels). IMAGE 2 = TARGET (the model must generate it).

THE CONTRACT THAT GOVERNS EVERYTHING
The final training prompt obeys one rule: WHAT IS WRITTEN IS WHAT IS SPECIFIED;
WHAT IS NOT WRITTEN IS INHERITED FROM THE REFERENCE.
Every attribute you write down is an attribute the model can get from text, so it
stops looking at the reference for it. Every attribute you leave out is one the
reference becomes the ONLY source of. Under-describing is cheap. Over-describing
silently destroys the dataset. When in doubt, write less.

HARD BAN — never appears in any caption slot, in any pair:
  - art style, medium, era, line quality, shading, rendering ("anime style",
    "cel shaded", "manga", "screentone", "90s", "detailed", "high quality")
  - global colour grading, palette or light mood ("warm tones", "soft lighting",
    "vibrant colours", "muted") — EXCEPT when it demonstrably changed (see palette)
  - ANY appearance attribute of a character or place that already exists in the
    REFERENCE: hair colour/length, eye colour, clothing, build, age, skin,
    accessories, furniture, architecture, weather, time of day.
Style and palette are inherited in 100% of pairs by construction. That is the point.

WORK IN THIS ORDER. Each step is a JSON key; fill them in order, do not skip ahead.

STEP 1 — inventory each image separately, before any comparison. For the reference:
how many characters are readable (a character is readable only if face/hair/body is
identifiable; a hand, a silhouette, a blurred crowd, food on a table, an empty room
= 0 readable characters), what they look like, what the setting is, what the shot is.
Then the same for the target. Do not compare yet.

STEP 2 — identity test, evidence first. For each character in the target, ask: is
there a character in the REFERENCE inventory that this is the SAME INDIVIDUAL as?
Cite at least TWO invariant features that match (hair silhouette + hair colour, eye
colour, a specific garment or accessory, a scar/mark, distinctive proportions) and
list any contradiction. Rules:
  - "Both are anime girls", "both wear school uniforms", "both are in the same
    production" is NOT identity. Reject it.
  - If the reference has 0 readable characters, the answer is ABSENT, never
    "different" and never "same". These are distinct outcomes and you must not merge them.
  - If the reference has characters but none of them is the target character, the
    answer is DIFFERENT. This is a perfectly good pair — it becomes a style pair.
  - If motion blur / extreme framing makes it unreadable, answer UNREADABLE and set
    confidence low.
This dataset is known to contain many WRONG "same character" labels. Assume the pair
is NOT continuous until the evidence forces you to say otherwise.

STEP 3 — decide each axis independently: character, place, palette, camera. A pair
can inherit character and replace place, or the reverse, or neither. Style is not a
question; it is always inherited.

STEP 4 — write the caption slots. This is a PROMPT a user would type, not a
description of the target. Voice: plain, concrete, present tense, lowercase, no
flourish. Constraints per slot:
  - framing: 1-3 words for the target's shot ("close-up", "wide shot",
    "over-the-shoulder", "low angle medium shot"). Always fill it.
  - subject: how to refer to the characters WITHOUT describing them, when they are
    inherited. Use "the same girl", "the same two characters", "the same man on the
    left", "the same character". Null when no character is inherited.
  - action: what happens in the target — pose, gesture, expression, gaze, spatial
    relation, motion. The heart of the caption. 1-2 clauses. Zero appearance words
    for inherited subjects.
  - new_subject: appearance of characters that are NEW in the target only. Here
    appearance IS allowed, because the reference cannot supply it. Keep it to what a
    user would bother typing (~10-20 words). Null if nobody is new.
  - place: null if the place is inherited (same location, even from a new angle).
    Otherwise a short description of the new setting — layout and objects only, no
    colour grading, no time of day unless it changed.
  - palette_shift: null in almost every pair. Fill ONLY for an unmistakable global
    change, e.g. "at night", "at sunset", "in the rain", "in black and white".

STEP 5 — audit your own slots. Re-read them and list every phrase that describes
something the reference already shows. Then emit the corrected final slots. If the
draft was clean, copy it verbatim.

STEP 6 — usability. Mark unusable ONLY if the two images come from visibly different
productions/media so nothing at all is inheritable, or if the target is unreadable.
Do NOT mark a pair unusable merely because nothing is in common: content-free pairs
from the same production are the most valuable pairs in the set.

Output ONE JSON object, no markdown fence, no commentary. Use null, not "none"."""

USER_PROMPT = ("IMAGE 1 is the REFERENCE. IMAGE 2 is the TARGET. "
               "Work through the steps in order and output the JSON.")

# ---------------------------------------------------------------- builder ---
import hashlib

INHERIT_PLACE_PHRASES = ('in the same room', 'in the same place', 'in the same background')


def _clean(s):
    if not s:
        return ''
    return str(s).strip().rstrip('.').strip()


def build_caption(js, tier='rich', pair_id=''):
    """Monta a caption de treino.

    tier:
      rich   - especificação alta (framing + tudo)
      normal - sem framing
      terse  - só sujeito + ação principal
    O sorteio de tier por par/época é o que ensina "omitido = herdado".
    """
    cap = js.get('caption') or js.get('caption_draft') or {}
    framing = _clean(cap.get('framing'))
    subject = _clean(cap.get('subject'))
    action = _clean(cap.get('action'))
    new_subject = _clean(cap.get('new_subject'))
    place = _clean(cap.get('place'))
    palette = _clean(cap.get('palette_shift'))

    head = subject or new_subject
    if tier == 'rich' and framing:
        head = f'{framing} of {head}' if head else framing

    parts = [p for p in [head, action] if p]
    if tier != 'terse':
        if new_subject and subject:
            parts.append(f'with {new_subject}')
        if place:
            parts.append(f'in {place}')
        if palette:
            parts.append(palette)

    # afirmação positiva de herança de lugar em parte dos pares: dá ao modelo
    # um sinal explícito ocasional, sem virar sufixo mecânico em todos
    axes = js.get('axes') or {}
    if (axes.get('place', '').startswith('same_place') and not place
            and tier != 'terse' and pair_id):
        h = int(hashlib.md5(f'{pair_id}|{tier}'.encode()).hexdigest()[:8], 16)
        if h % 100 < 40:
            parts.append(INHERIT_PLACE_PHRASES[h % len(INHERIT_PLACE_PHRASES)])

    text = ', '.join(p for p in parts if p)
    return (text[0].lower() + text[1:] + '.') if text else ''


def pick_tier(pair_id, epoch=0):
    """Tier determinístico por (par, época) — o dataloader varia entre épocas."""
    h = int(hashlib.md5(f'{pair_id}|{epoch}'.encode()).hexdigest()[:8], 16) % 100
    return 'rich' if h < 45 else ('normal' if h < 80 else 'terse')


# palavras que NÃO podem aparecer numa caption cujo eixo é herdado
LEAK_WORDS = ('hair', 'wearing', 'jacket', 'dress', 'shirt', 'uniform', 'eyes',
              'haired', 'outfit', 'clothes', 'coat', 'anime', 'style', 'cel',
              'lighting', 'tones', 'palette', 'colours', 'colors', 'detailed')
