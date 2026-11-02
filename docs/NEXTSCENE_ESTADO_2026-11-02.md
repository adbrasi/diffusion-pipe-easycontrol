# Próxima cena (Krea 2): estado em 02/11/2026 e receita do próximo treino

## Veredito

- **Problema.** Com mais treino, o adapter devolve a composição da referência em vez da cena nova (cópia, às vezes
  com uma edição pedida pelo prompt). A cópia aparece entre os steps 750 e 1250 em todos os métodos com as legendas
  "longas", que descrevem a cena B inteira.
- **Causa.** Está nos dados.
  - ~34–38 % de cada alvo é cópia alinhada da referência, e ~27–37 % dos pares têm mais da metade da imagem copiada.
  - A legenda longa descreve o que já está na referência, então copiar satisfaz o texto. A cópia se concentra nos
    casos em que o prompt é compatível com a referência: 53 % contra 9 % no BETA1-R 1000.
- **Solução comprovada: legenda delta.** A config do BETA1-R, idêntica, nos 1000 pares do v11 com legendas delta
  (DELTA1000) ficou estável até 1750 steps. A cópia **não cresce** com o treino.

## Resultados (eval64: 56–64 casos fora do treino, 2 seeds, cópia por caso = maioria das seeds)

| run | step | cópia (caso) | em qualquer seed | grupo compatível | DINO contra o alvo |
|---|---|---|---|---|---|
| BETA1-R | 750 | 3 % | 12 % | 6 % | 0,69 |
| BETA1-R | 1000 | **31 %** | 39 % | 53 % | 0,67 |
| R4a (máscara espacial 0,15) | 1000 | 5 % | 19 % | 9 % | 0,70 |
| R4a | 1250 | copiava visualmente (cópia com edição; o dHash não pegava) | | | |
| **DELTA1000** | 1000 | 5 % | 11 % | 10 % | 0,68 |
| **DELTA1000** | **1500** | **2 %** | **2 %** | 3 % | 0,69 |
| DELTA1000 | 1750 | avaliação rápida 0/88 | | | |

O que não funcionou ou foi só paliativo:

| tentativa | resultado |
|---|---|
| pesos por par com dHash global (R2b) | copiou no 750 |
| zoom-crop da referência no latente | piorou |
| ruído na referência, descarte de tokens | sem ganho |
| `ref_bias` | sem efeito no runner |
| pular a referência nos primeiros passos | duplicação e colagem |
| máscara espacial 0,15 | atrasa ~250 steps |

Detalhes: `docs/BETA1R2.md`, `docs/REVISAO_INDEPENDENTE_2026-11-02.md`, `docs/AVALIACAO_EVAL64.md`.

## Receita do próximo treino (definida pelo usuário)

- **Config do BETA1-R sem mudanças** (`examples/beta1r/train.toml`).
  - 512 px; LoRA routed r64 + txtfusion r128.
  - reference_dropout 0.1, caption_dropout 0.
  - base fp8_scaled, lr 1e-4, batch 4.
- **Dataset novo, no caminho delta:**
  1. Juntar tudo o que existe: v11, proxima_cena_grounded, recortados, parents etc.
  2. **Remover cópias:** pares com muita área alinhada (critério: `scratchpad/al.py` / `tools/beta1r4_calibrate_mask.py`;
     a segunda opinião sugeriu corte em 0,30).
  3. **Legendas delta para tudo**, com o system prompt de `nextscene_moss_krea2_2026-10-27/scripts/recaption.py`
     (branch `claude/nextscene-moss-krea2-2026-10-27`).
     - Reaproveitar as 1000 que já existem (`captions/cap1000.jsonl`).
     - Modelos: DeepSeek V4 Flash e Muse Spark 1.3 Contributor (OpenRouter).
  4. **~1000 pares de comics do usuário**, com legenda delta **por painel**: número de painéis e o que acontece em cada um.
  5. Mais pares **realistas**, para corrigir o viés para anime, e mais pares com **mudança de lugar ou câmera**,
     porque 842/1000 das legendas delta terminam em "Same place.".
- **Avaliação:** eval64 com prompts delta (`cases_eval64_d1000.json`, descontaminado contra o dataset de treino),
  visualmente em `/workspace/eval64.html` (`tools/eval64_html.py`).

## Onde está cada coisa

- **Código:** esta branch (`beta1-r3`, que contém R2/R3/R4/DELTA1000 e a avaliação).
  - Config DELTA1000: `examples/delta1000/`.
  - Dataset: `/workspace/data/delta1000` (montado de HF `AdwolfCzar/ofificial_next_scene_v11` + `cap1000.jsonl`).
- **Pesos no HF (públicos, sem README):**
  - `AdwolfCzar/krea2-beta1r` (BETA1-R; o **step750** é o default do usuário)
  - `AdwolfCzar/krea2-delta1000` (steps 250–1750)
  - `AdwolfCzar/krea2-beta1r4a`, `-beta1r2`, `-beta1r2a`, `-beta1r2b`
- **Node ComfyUI:** `adbrasi/ctxrush-edit`.
  - Workflow que funciona com o BETA1-R: `K2 Training Base` com fp8_training_grid=false + turbo, depois `Multi-Ref v2`
    com training_base_quant=false.
  - Opção `ref_skip_sigma` na branch `ref-skip-sigma` (paliativo; não testada de ponta a ponta).
- **Inferência com DELTA1000:** prompt no formato delta, por exemplo
  "Next scene of Picture 1. A wide shot. The same girl … Same place."
