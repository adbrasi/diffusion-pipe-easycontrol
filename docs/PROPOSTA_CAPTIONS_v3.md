# Proposta de sistema de captions — v3

**Data:** 2026-07-25
**Autor do desenho:** Opus 5 com contexto limpo (briefing dado por mim, sem
acesso às tentativas anteriores, justamente para não herdar os vícios delas).
**Status: PROPOSTA PARA VOCÊ AVALIAR.** Nada foi implementado no pipeline.

---

## 0. O problema que isso resolve

O dataset atual tem captions **exaustivas** — descrevem a imagem-alvo por
completo:

> "A medium shot of a young person with long, straight black hair with
> teal-colored tips, wearing a dark blue uniform with gold buttons and a white
> belt. They are holding a katana with the blade unsheathed and pointed towards
> the right. Behind them, three additional katanas are mounted horizontally on a
> wooden wall. In the foreground on the right, the back of a person with dark
> reddish hair and a purple cloth draped over their shoulders is partially
> visible, facing the sword-wielder.
>
> Character continuity: same character. Background continuity: new view of the
> same background."

Medimos na bateria de 2026-07-25 (teste de referência embaralhada: gerar a mesma
imagem trocando a referência por outra, mantendo o caption) que o adapter
treinado assim **ignora a referência** — a saída muda pouco ao trocar a
referência. Causa: o texto já contém tudo, a referência é redundante, a loss não
dá motivo para lê-la.

**O contrato que a v3 quer ensinar:**

> O QUE ESTÁ ESCRITO É O QUE FOI ESPECIFICADO.
> O QUE NÃO ESTÁ ESCRITO É HERDADO DA REFERÊNCIA.

---

## 1. Por que a v2 (minha tentativa anterior) falhou

Ela gerava:

```
A close-up. The same character holds a sword.

Character continuity: no character continuity. Background continuity: new background. Style continuity: same style.
```

Três defeitos, e o segundo é o mais sério:

1. **Sufixo mecânico** que nunca aparece num prompt real de inferência. Treinar
   com um formato que você não vai usar cria descasamento treino↔uso.
2. **`"no character continuity"` é uma NEGAÇÃO.** Modelos de linguagem pequenos
   (o text encoder do Anima é um Qwen3-0.6B) codificam negação de forma pouco
   confiável — há chance real de esse texto **ativar** o conceito "character"
   em vez de desativá-lo. Ou seja, podia estar fazendo o oposto do pretendido.
3. **Repetição idêntica em ~1255 exemplos** de um bloco que não carrega
   informação discriminativa vira ruído tokenizado, ocupando espaço de contexto
   sem ensinar nada.

---

## 2. As decisões de desenho da v3

### 2.1 Ponteiro deítico, não sufixo de metadados
`"the same girl"` faz três coisas ao mesmo tempo:
- força a leitura da referência (a identidade não está no texto);
- é **exatamente o que você vai digitar na inferência**;
- é uma frase curta repetida milhares de vezes — o Qwen3-0.6B a aprende como
  símbolo estável.

### 2.2 Três tiers da mesma caption (rich / normal / terse)
O mesmo par aparece, entre épocas, com especificação alta e baixa **para a mesma
imagem-alvo**. Este é o mecanismo de ensino central: a loss só fecha se aquilo
que sumiu do texto vier da referência. Não é variação estética — é o que ensina
literalmente "se eu não especificar, herda".

### 2.3 Ban global de estilo e paleta em TODA caption
Se nenhuma caption jamais menciona estilo, traço, shading ou grading de cor,
então **estilo é 100% do tempo informação exclusiva da referência**. Consequência
elegante: o caso "referência só de estilo" (mandar Goku, pedir cara de camisa
azul com cabelo pra cima, sair quase-Vegeta) funciona **sem nenhum modo
especial** — cai naturalmente do mesmo contrato.

### 2.4 Enum, não boolean, no eixo de personagem
O falso positivo mais grave do dataset vem de colapsar dois casos distintos:
- **personagem diferente** (há personagens na referência, mas são outros)
- **personagem ausente** (a referência não tem personagem legível)

O `gemini-2.5-flash-lite` sem raciocínio errou exatamente assim num par real.
Enum + inventário obrigatório antes da comparação + exigência de 2 invariantes é
o que quebra esse erro.

### 2.5 Chain-of-thought DENTRO do JSON, em ordem
Modelos baratos ignoram o canal de `thinking` com frequência. Campos de evidência
que **precedem** o veredito forçam o condicionamento no próprio texto gerado. E
ficam **auditáveis como colunas**: dá para filtrar o dataset depois por
`confidence`, `leaks != []`, `matched_features < 2`.

### 2.6 Pares "sem nada em comum" NÃO são lixo
Viram `style_only` e são, segundo o desenho, **os mais valiosos do conjunto** para
ensinar herança de estilo. Isso inverte a premissa de que os falsos positivos que
encontrei deveriam ser descartados — eles devem ser **reclassificados**.

---

## 3. SYSTEM PROMPT (íntegra, pronto para uso)

```text
You annotate PAIRS of anime images for a reference-conditioned generation dataset.
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

Output ONE JSON object, no markdown fence, no commentary. Use null, not "none".
```

---

## 4. SCHEMA de saída

```jsonc
{
  "reference": {"readable_characters": 0, "characters": "…", "setting": "…", "shot": "…"},
  "target":    {"readable_characters": 2, "characters": "…", "setting": "…", "shot": "…"},

  "identity_check": {
    "matched_features": ["…"],          // >=2 invariantes, ou []
    "contradictions": ["…"],
    "verdict_reason": "one sentence"
  },

  "axes": {
    "character": "same" | "same_plus_new" | "different" | "absent_in_reference"
               | "absent_in_target" | "unreadable",
    "place":     "same_place_same_view" | "same_place_new_view"
               | "different_place_same_world" | "different_place" | "no_place",
    "palette":   "same" | "shifted",
    "camera":    "same" | "changed"
  },

  "pair_type": "next_shot" | "sequence_page" | "new_pose" | "same_char_new_place"
             | "same_place_new_char" | "place_variation" | "style_only" | "unusable",

  "caption_draft": {"framing":"…","subject":null,"action":"…",
                    "new_subject":null,"place":null,"palette_shift":null},
  "self_audit": {"leaks": ["…"], "clean": true},
  "caption":       {"framing":"…","subject":null,"action":"…",
                    "new_subject":null,"place":null,"palette_shift":null},

  "confidence": "high" | "medium" | "low",
  "usable": true,
  "reject_reason": null
}
```

Note que `caption_draft` → `self_audit` → `caption` é o ciclo de auto-correção:
o modelo escreve, audita o que escreveu procurando vazamento, e reemite.

---

## 5. BUILDER da caption final

Determinístico, semeado pelo id do par. Emite 3 variantes; o dataloader escolhe
uma por época.

```
rich   = "{framing} of {subject|new_subject}, {action}[, with {new_subject}][, in {place}][, {palette_shift}]."
normal = igual, sem framing
terse  = só sujeito + ação principal
affirm = em 40% dos pares com lugar herdado, insere "in the same room|place|background"
```

**Sem sufixo. Sem bloco de continuidade. Sem cauda de metadados. Nunca.**

Distribuição sugerida de tiers: rich 45% · normal 35% · terse 20%.

---

## 6. Exemplos de caption final

### A. Next shot — o exemplo exaustivo do topo, recaptionado
```
medium shot of the same swordsman, he lowers the katana and turns to face the other
character standing at the right edge of the frame, both in profile.
```
*terse:*
```
the same swordsman lowers his sword and turns to the other character.
```

### B. Style-only (Goku → quase-Vegeta)
Nenhum ponteiro de continuidade, nenhuma palavra de estilo — o estilo vem só da
referência:
```
medium shot of a muscular man in a blue shirt, hair swept sharply upward, arms crossed,
facing the viewer with a smirk.
```

### C. Place variation (visual novel: diretoria → sala de estar)
```
wide shot of an empty living room, a low sofa facing a coffee table, a wide window on
the left, no characters.
```

### D. Falso positivo tipo "mesa de comida = mesmo personagem"
`axes.character = absent_in_reference`, `axes.place = same_place_new_view`
```
close-up of a girl with short brown hair leaning over the same table, picking up a bowl
with both hands.
```

### E. Falso positivo tipo "mulheres → garoto ruivo"
`axes.character = different` → `pair_type = style_only` — **não descartado**
```
low angle medium shot of a red-haired boy in a green jacket running down a street,
looking back over his shoulder.
```

---

## 7. Riscos identificados pelo próprio desenho

1. **Tier terse pode ficar ambíguo demais** em pares onde muita coisa mudou.
   Mitigação: limitar terse a pares com ≤1 eixo trocado.
2. **`new_subject` é a única porta por onde aparência entra.** Se o anotador
   errar e marcar `different` num personagem que era o mesmo, vaza aparência.
   Mitigação: auditar amostralmente os pares `character: different` com `place`
   herdado.
3. **Se ~todas as captions ficarem curtas**, o adapter pode perder aderência a
   prompts longos na inferência. O tier `rich` existe para cobrir isso — daí a
   distribuição não ser uniforme.

---

## 8. Contexto de custo (medido, não estimado)

Custo real por par com `thinking` ligado, via `usage.cost` do OpenRouter
(já inclui tokens de raciocínio, cobrados como saída), extrapolado para 1255
pares:

| modelo | $/par | dataset inteiro |
|---|---|---|
| google/gemma-4-31b-it | 0.000783 | **$0.98** |
| xiaomi/mimo-v2.5 | 0.000848 | **$1.06** |
| qwen/qwen3.5-flash | 0.001649 | $2.07 |
| google/gemini-2.5-flash-lite | 0.002120 | $2.66 |
| z-ai/glm-4.6v | 0.002518 | $3.16 |
| minimax/minimax-m3 | 0.003044 | $3.82 |
| qwen/qwen3.7-plus | 0.003887 | $4.88 |
| google/gemini-3.1-flash-lite | 0.005508 | $6.91 |

(kimi-k2.6 removido do pool a seu pedido: $0.013410/par = $16.83, o mais caro,
e ainda falhou num par.)

**Custo não é restrição.** O critério de escolha do modelo deve ser só
qualidade — especificamente: detectar os falsos positivos e não vazar aparência
nos eixos herdados.

---

## 9. O que falta você decidir

1. O formato de caption final está certo? Alguma mudança no estilo de escrita?
2. Manter os 3 tiers ou fixar um só?
3. A afirmação positiva ocasional ("in the same room") ajuda ou polui?
4. O ban total de palavras de estilo é agressivo demais? (ele torna impossível
   pedir mudança de estilo via texto — mas é o que faz herança de estilo
   funcionar)
5. Qual modelo usar para rodar os 1255 pares.
