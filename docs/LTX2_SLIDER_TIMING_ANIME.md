# LTX-2.3 — slider de timing de animação anime

**Objetivo:** o LTX anima anime **fluido demais** — movimento contínuo, como
material 60fps interpolado. Anime real é animado "on twos" (12 desenhos/s) ou
"on threes" (8/s), com frames **segurados** e transições em pop entre poses.
O problema não é amplitude de movimento: é excesso de **densidade temporal** —
o modelo preenche os intervalos que a animação deixa vazios de propósito.

Alvo de uso: **i2v, ~100% das vezes.**

Ferramenta: fork [AkaneTendo25/musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner)
branch `ltx-2`, que tem `ltx2_train_slider.py` — Concept Sliders nativos em
três modos (`text`, `reference`, `ic_reference`).

---

## FASE 1 — slider de TEXTO: FALHOU

Treinado 200 steps (de 400 planejados), 5 formulações do conceito
(`animated on twos` / `limited animation` / `stop motion` / `12 drawings per
second` / `traditional hand-drawn timing`) contra os negativos
correspondentes, 2 âncoras de qualidade estática, `latent_frames = 7`.

### Resultado

Sweep i2v com mesma seed, mesmo prompt, mesma imagem inicial, só variando o
multiplicador:

| multiplicador | frames segurados | diff média entre frames |
|---|---|---|
| −2.0 | 33.3% | 0.00718 |
| +2.0 | 35.8% | 0.00674 |

**Diferença dentro do ruído**, e os vídeos são visualmente indistinguíveis.

### Diagnóstico — e por que era previsível

O modo texto trabalha com **latentes de ruído puro** e computa:

```
direction      = pred_pos − pred_neg      (modelo CONGELADO)
target_enhance = norm(pred_neu + gs·direction)
target_erase   = norm(pred_neu − gs·direction)
```

Ou seja: ele só consegue extrair a direção que **o modelo já associa** àqueles
prompts. Se "animated on twos" não tem representação distinta no
condicionamento do LTX-2.3, `direction ≈ 0` e não há o que amplificar.

Sintoma que apareceu no primeiro step e eu devia ter lido na hora: `loss` ≈
0.0003 com `guidance_strength = 1.0`. Como no step 0 a LoRA é ~zero, a loss
inicial **é** essencialmente `‖gs·direction‖²`. Subir `guidance_strength` para
2.0 levou a loss para ~0.0053 (faixa comparável ao exemplo que funciona no
doc), o que confirma que o eixo existia mas era fraco — fraco demais para
sobreviver 200 steps.

### Custo de compute que eu não calculei antes de rodar

Primeira configuração deu **80 s/step** (7-9h para 400 steps). Os
multiplicadores, que eu empilhei sem multiplicar:

| fator | custo |
|---|---|
| 5 targets com `batch_all_targets = true` | ×5 |
| 2 âncoras — cada uma adiciona uma linha ao passe de referência **e a cada passe de gradiente** | ×3 |
| 7 frames latentes (necessário: um slider de timing em frame único é vazio) | ×7 vs imagem |
| 22B, 48 blocos, fp8 + block swap | base pesada |

= **5 targets × 2 multiplicadores × 3 linhas = 30 passes com backward por
step.** Desligar `batch_all_targets` levou a 16 s/step, o ÷5 exato.

**Resolução NÃO era o gargalo:** 512×768 com patch /32 dá 16×24 = 384 tokens
por frame × 7 = 2.688 tokens, sequência modesta.

### Outras armadilhas encontradas

- **`--lora_target_preset t2v` degrada o áudio do modelo base.** O preset cria
  pesos de LoRA nas camadas de áudio e cross-modais; sem dado de áudio no
  treino esses pesos são inicializados e nunca recebem gradiente com sentido,
  e aplicar a LoRA sobrescreve o áudio com deltas quase-zero. Usar
  `video_sa_ca_ff`. O doc só menciona isso numa tabela de troubleshooting.
- **Treino e inferência têm perfis de memória muito diferentes.** Treino cabe
  com `blocks_to_swap 6` (~22 GB); inferência dá OOM e precisa de
  `blocks_to_swap 32` + `--sample_with_offloading` + tiling de VAE, porque o
  decode roda em resolução plena.
- Não dá para treinar e inferir ao mesmo tempo — o 22B ocupa a GPU inteira.
- `setsid nohup` estava sendo morto neste ambiente; usar background rastreável.
- Editar um `.sh` **enquanto ele roda** corrompe a execução (o bash lê o
  arquivo incrementalmente) — parecia erro de sintaxe.

---

## FASE 2 — slider de REFERENCE: a direção vem dos dados

Se o modelo não tem o conceito no texto, a saída é dar o eixo por exemplo.

**Dataset:** [AdwolfCzar/animateka](https://huggingface.co/datasets/AdwolfCzar/animateka)
— 708 clipes de anime de câmera estática que o usuário já validou como
exatamente o alvo estético (e já usou num treino de LoRA para reduzir
movimento).

**Construção do par** — o insight que torna isto barato: o negativo é o
*mesmo vídeo* com a estrutura de frames segurados destruída.

```bash
ffmpeg -i orig.mp4 -vf "mpdecimate,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir" neg.mp4
```

`mpdecimate` remove as duplicatas (sobram só os desenhos distintos);
`minterpolate` reconstrói 24fps com compensação de movimento, então todo frame
passa a ser único e o movimento vira contínuo — **o defeito do LTX aplicado ao
material do usuário**. Não precisa de RIFE; o ffmpeg basta.

### Verificação por medição (antes de escalar)

Métrica: fração de pares de frames consecutivos quase idênticos
(diff média < 0.0002 em luminância 160×90) — a assinatura de "frame segurado".

| | duplicatas | diff mediana |
|---|---|---|
| positivo (original) | **44.3%** | 0.00021 |
| negativo (interpolado) | **0.0%** | 0.00256 (12×) |

E o **alinhamento temporal se preserva**: `orig[i]` vs `neg[i]` difere 0.0003,
*menos* que a diferença entre frames vizinhos do original (0.0022). O par
mostra o mesmo instante em cada índice e difere só na textura temporal — que é
o que faz o slider aprender um EIXO e não uma troca de conteúdo.

### A curadoria é obrigatória — a média esconde pares tóxicos

Medindo 12 pares individualmente, **3 não serviam**:

| clipe | pos | neg | delta | |
|---|---|---|---|---|
| c_038 | 76.6 | 0.0 | +76.6 | ✅ |
| c_001 | 57.4 | 0.0 | +57.4 | ✅ |
| c_024 | 93.6 | 87.2 | +6.4 | ⚠️ clipe quase estático, nada a remover |
| c_037 | 0.0 | 0.0 | 0.0 | ❌ já era fluido: sem eixo |
| c_030 | 2.1 | 29.8 | **−27.7** | ❌ **INVERTIDO** — treinaria ao contrário |

A média agregada (+30 pontos) escondia os dois casos. Um par com delta≈0 dilui
o sinal; um com delta **negativo** o corrompe. `tools/filter_pairs.py` mede
cada par e reprova por `--min-pos` e `--min-delta`.

### O que a Fase 2 ganha de graça

**Âncoras não se aplicam ao modo `reference`** — o trainer as ignora, porque
pares positivo/negativo já se auto-regularizam contra drift. Aqui isso é
literalmente verdade: os dois lados *são o mesmo vídeo*. Some a parte mais
delicada da Fase 1 (escolher âncoras que não briguem com o eixo) e cai o ×3 de
custo por passe.

**E casa com i2v**, que é o uso real: `reference` é o único modo onde
`--ltx2_first_frame_conditioning_p` funciona, e o doc o descreve como feito
para *"pares que compartilham o mesmo frame inicial e diferem principalmente
em movimento"* — a descrição literal deste par.

---

## Arquivos

Em `/workspace/musubi/` (fork do musubi, fora deste repo):

| arquivo | o quê |
|---|---|
| `anime_animation_slider.toml` | config da Fase 1 (texto) — mantido como registro do que falhou |
| `train_anime_slider.sh` | treino Fase 1 |
| `sweep_slider.sh` | sweep i2v de multiplicadores, mesma seed/prompt/imagem |
| `make_negatives.sh` | constrói o lado negativo do par |
| `filter_pairs.py` | curadoria por medição |
| `animateka_pos.toml` / `animateka_neg.toml` | datasets dos dois lados |
| `anime_timing_slider_ref.toml` | config da Fase 2 (reference) |

Os dois datasets usam `target_frames` FIXO e `frame_extraction = "head"`: o
`mpdecimate` encurta o negativo, então deixar o comprimento livre daria
latentes de shapes diferentes e o pareamento por nome falharia.

## FASE 2 — RESULTADO: FUNCIONOU

600 steps, 261 pares, 1h27 (4.3 s/step), loss 0.33 -> 0.18.

Diferença mediana entre frames consecutivos nas amostras do step 600
(mesma seed, mesmo prompt, mesma imagem inicial, só o multiplicador muda):

| multiplicador | diff mediana | |
|---|---|---|
| **-2.0** | 0.01203 | mais movimento por frame (fluido) |
| -1.0 | 0.00489 | |
| 0.0 | 0.00108 | base |
| +1.0 | 0.00081 | |
| **+2.0** | **0.00052** | menos movimento por frame (escalonado) |

**MONOTÔNICO, faixa de 23x.** O eixo existe e responde na direção correta.

O eixo **emergiu entre os steps 400 e 600**: no 400 a curva era irregular
(0.0039 / 0.0023 / 0.0011 / 0.0023 / 0.0019) e no 200 era ruído. Parar em 400
teria dado um falso negativo.

Sinal independente da métrica: o **tamanho dos arquivos** cai monotonicamente
(1.5 MB em -2.0 -> 1.2 MB em +2.0). Menos mudança entre frames = menos dados
para o codec.

Nota sobre métricas: o `dup%` (fração de frames quase idênticos), que serviu
para caracterizar o DATASET, é inútil no material GERADO — um modelo de difusão
praticamente nunca produz frames exatamente iguais, sempre há ruído residual.
Para as saídas, a diff mediana é a métrica que capta o eixo.

## Contraste entre as duas fases

| | Fase 1 (texto) | Fase 2 (reference) |
|---|---|---|
| loss inicial | 0.0003 | **0.33** (1000x) |
| resultado do sweep | indistinguível | monotônico, 23x |
| custo/step | 16 s | **2.2-4.3 s** |
| passes com backward | 6 + ref de 5 linhas | 2 |

A loss inicial é o diagnóstico barato: no modo texto ela É essencialmente
`‖gs·direction‖²` (a LoRA ainda é zero). 0.0003 dizia que o conceito não
estava no modelo — e não estava.

## BUG ENCONTRADO NO TRAINER (corrigido)

`ltx2_train_slider.py` descartava **todos** os pares de vídeo com
"No text cache for ...". Causa: `_LATENT_BASENAME_RE` segue a convenção de
IMAGEM. Todo modo de `frame_extraction` — inclusive `"full"` — anexa o token de
intervalo de frames ao `item_key` (`image_video_dataset.py:3562`), e o dataset
de vídeo compensa resolvendo o cache de texto com `tokens[:-3]` (:3712),
enquanto imagens usam `tokens[:-2]` (:2794). O slider gerava o stem
`c_000_00000-057` e procurava `c_000_00000-057_ltx2_te.safetensors`, que nunca
existe para vídeo. Corrigido com fallback que remove o token de intervalo.

## Configuração final (com os números medidos)

```
blocks_to_swap = 0          # MEDIDO: 0 -> 2.21 s/step | 20 -> 8.4 s/step
gradient_checkpointing = 1  # sempre: libera VRAM sem tráfego CPU<->GPU
--use_precached_sample_prompts --use_precached_sample_latents
  + --sample_prompts_cache / --sample_latents_cache explícitos
  (o modo reference não usa --dataset_config, então não resolve o path sozinho)
--ltx2_first_frame_conditioning_p 0.9   # i2v é o uso real
--lora_target_preset video_sa_ca_ff     # t2v degrada o áudio do base
network_dim 16, lr 1e-4, AdamW8bit, 600 steps
```

Cache do Gemma: precisa de `--gemma_load_in_4bit` + `--batch_size 1` no
caching (o script carrega o LTX 22B **e** o Gemma juntos; bf16 e 8-bit dão OOM
em 32 GB).

## Em aberto
- Se o eixo funcionar, vale reconsiderar `batch_all_targets = true` num refino
  curto — o argumento do doc (a média entre formulações cancela ruído
  específico de contexto) continua válido, só é caro para o run principal.
- Minha métrica de duplicatas é global; em cena de câmera estática grande parte
  do quadro é fundo parado, então ela mede em parte "câmera estática" e não só
  "frame segurado". Para o *veredito* o julgamento é visual do usuário.
