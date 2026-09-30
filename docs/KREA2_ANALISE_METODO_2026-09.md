# Krea 2 — análise da saga e o método de treino recomendado (2026-09-30)

**Queixas do usuário:**
1. "Às vezes funciona com perfeição, mas muitas vezes entrega literalmente a mesma imagem."
2. "Os LoRAs atrapalham muito a qualidade."
3. "Não funciona no ComfyUI normal; precisa do patch `ctxrush-edit`, que mexe no fp8 e altera
   os resultados."

**Base desta análise:**
- todas as notas do repo sobre Krea 2 (KREA2_EDIT_SAGA, AUDITORIA_2026-08-06, OMINI_*,
  KREA2_MULTIREF_*, KREA2_APEX_SPEC, K2_PROXIMACENA_V2_NOTES);
- o config real do `k2-context-rush-ofc-beta1` no HF;
- o código do fork (`models/base.py`, `models/krea2_*.py`);
- o repositório `adbrasi/ctxrush-edit` (NOTES, DESIGN_PACK_V3, `nodes_trainbase.py`);
- uma pesquisa externa, com clone do ComfyUI master `fb2315f1` (30/09), do ai-toolkit, do
  krea2edit-trainer, dos nodes do lbouaraba e do ostris.

---

## 0. Resposta curta

As três queixas têm **uma causa comum**: **o treino foi feito contra um "Krea 2" que não é o
que o ComfyUI roda.** O patch existe para recriar esse Krea 2 alternativo dentro do ComfyUI,
e é por isso que ele "altera resultados".

O ComfyUI oficial, desde 18/07 (commit `c9602625`), **roda edição por referência no Krea 2
nativamente** (`ReferenceLatent` + `index_timestep_zero` + `TextEncodeQwenImageEditPlus`).

**O método recomendado é treinar exatamente nesse contrato nativo:** base com a numérica
correta, encoder de texto idêntico ao do ComfyUI, ref no frame 1 com t=0 e LoRA global.
Somam-se a isso as correções anti-cópia que as duas sagas (Krea 2 e Anima) ensinaram. O
resultado carrega com `LoraLoaderModelOnly` num workflow **sem nenhum node custom**.

---

## 1. Por que os LoRAs degradam a qualidade e exigem patch

### 1.1 A base de treino estava mutilada (causa dominante, verificada no código)

`models/base.py` (~L533–547), com `diffusion_model_dtype = 'float8'` (o config do beta1):

```python
module.register_parameter(p_name, nn.Parameter(p.dequantize()))   # desfaz o fp8_scaled (correto)
...
p.data = p.data.to(diffusion_model_dtype)                          # re-quantiza SEM escala
```

O checkpoint `krea2_raw_fp8_scaled` guarda `fp8 + weight_scale`. O trainer desquantiza e
depois joga os valores no grid fp8 cru, sem escala. Números medidos pela própria saga
(`ctxrush-edit/nodes_trainbase.py`):
- as 224 Linears dos blocos ficam **2,5%–6,7%** diferentes;
- **entre 6,6% e 26,8% dos pesos viram ZERO** (underflow de denormal);
- a velocidade no primeiro forward diverge com **relL2 0,58**.

Consequência: o LoRA aprendeu a corrigir *essa* base.
- No ComfyUI normal, que usa a escala, o delta não encaixa: o "LoRA fraco / diferente".
- O `K2 Training Base` recria a base mutilada, então o LoRA volta a funcionar, mas sobre um
  Krea 2 degradado: o **"LoRA estraga a qualidade"**.

Hipótese (não provada): um prior com até ¼ dos pesos zerados em algumas camadas fica mais
fraco e se apoia mais nos tokens limpos da referência, o que empurra para a cópia.

### 1.2 O encoder de texto do treino não é o do ComfyUI

O fork treinou com o ComfyUI fixado em 23/06, e o Qwen3-VL daquela versão não aplica MRoPE 3D
nem DeepStack na visão. O ComfyUI atual aplica. O contexto do grounding diverge com
**relL2 1,36**. O patch desfaz o `Qwen3VL.forward` para imitar o treino.

### 1.3 O contrato de referência do beta1 não existe no ComfyUI

| elemento | beta1 (fork) | ComfyUI nativo |
|---|---|---|
| posição da ref | `width_shift` (ao lado do alvo) | frame `i`, grid h/w a partir de 0 no tamanho da própria ref |
| t da ref | 0 por token | 0 por token com `index_timestep_zero` ✅ |
| LoRA | **roteado só nas linhas da ref** (não pode ser fundido) | LoRA global comum |
| template VL | `KREA2_TEMPLATE` + "image 1:" | template Qwen-Image-Edit + "Picture 1:" |
| imagem para o VL | bicubic, nunca amplia, 384² | "area", exatamente 384² (amplia) |
| ref no VAE | ajustada ao tamanho do alvo | ~1 MP de área, múltiplo de 8, AR próprio |

Cada linha dessa tabela é um patch necessário. Um LoRA **roteado**, em particular, nunca
funciona num loader padrão: ele exige aplicação mascarada em runtime.

### 1.4 O resto da lista de divergências era paridade de avaliação, não qualidade

Ruído na CPU contra CUDA, decodificação de JPEG, as 7 chaves `.diff_b` da turbo. Essas
afetam comparar node com runner, não a qualidade do LoRA.

---

## 2. Por que ele devolve a mesma imagem

Aqui **a geometria não é a causa**, ao contrário do Anima: o beta1 usava `width_shift`,
que já afasta a ref (núcleo RoPE 0,806 a 512px, contra 0,987 no frame 1 alinhado). As causas
são outras e se somam:

1. **Routing condition-only, a "catraca" (auditoria 08/06).** O único grau de liberdade é
   inflar a saliência da ref; as queries do alvo ficam congeladas. O LoRA do conradlocke mostra
   o oposto do que o routing permite: ~26% da energia em `wq`+`wk`, ou seja, ele aprende
   *para onde o alvo olha*.
2. **Dados com muitas aulas de cópia.** Na fase 1, **26% era pico-banana** (edições locais:
   B ≈ A em quase todos os pixels) e **23% "recortados"** (frames de vídeo; nenhum filtro de
   quase duplicados). Para um prompt fraco, "copiar A" virou o padrão aprendido.
3. **`caption_dropout` 0,1.** Legenda vazia com ref limpa ensina exatamente "copie". O
   AnimaRefLora mediu: o dropout da legenda inteira antecipou a cópia de ~25K para ~10K steps.
4. **13,4k steps sem freio anti-cópia.** A pressão de cópia cresceu com os steps (sauce
   1.500@1024 já "praticamente uma cópia"; groundedsecret parado em 0,8 época não copiava).
5. **O grounding descreve a ref.** Os tokens do Qwen3-VL "contam" a imagem A para o DiT. No
   Ideogram (P3) isso causou colapso de reconstrução com prompt mínimo. Aqui é um agravante,
   não a causa única.
6. **A turbo decide a composição em 2–3 passos de σ alto**, exatamente onde a ref limpa domina.
7. **Fase 2 a 1024 com `flux_shift`**: ~4% de massa em t<0,3, e o LR nunca caiu (resume).
   Isso explica parte do "degrada quando segue".

**O outro polo existe e está registrado:** o `apex` (LoRA global, frame axis, t compartilhado,
55% das legendas completas) caiu em "bonito, sem relação com a ref". A solução não é voltar
ao routing. É **LoRA global + t=0 + legendas que obriguem a ler a ref + dados sem aulas de
cópia + parada guiada por avaliação contra o alvo real**. É a mesma conclusão a que o Anima
chegou.

---

## 3. O mapa externo (o que existe hoje para Krea 2)

- **Nenhum edit oficial da Krea** até 30/09 ("Editing is something we are exploring").
- **ComfyUI nativo:** `c9602625` (18/07) adiciona `ref_latents` ao Krea 2 com dois métodos:
  - `index`: um t para todos os tokens (contrato do conradlocke);
  - `index_timestep_zero`: ref em t=0 (contrato do ostris).
  Sequência `[texto | alvo | refs]`; a saída devolve só o alvo. **Armadilha:** sem o node
  `FluxKontextMultiReferenceLatentMethod`, o `default_ref_method` é `None` e a ref é
  **ignorada em silêncio**.
- **Blueprint oficial "Image Style Reference (Krea-2 Turbo)"** (09/09):
  `TextEncodeQwenImageEditPlus` + `index_timestep_zero` + LoRA do ostris, CFG 1, 8 passos.
  É a prova de que o caminho nativo sem patch funciona com um LoRA treinado no contrato certo.
- **conradlocke Identity Edit:** global r256, shared t, ref centrada, `ref_boost`; precisa
  dos nodes do lbouaraba. Também teve *passthrough* (saída = entrada) em ARs retrato na v1.1,
  corrigido ajustando a resolução para passos de 16 e reformulando a instrução.
- **ai-toolkit (ostris):** matematicamente igual ao `index_timestep_zero` nativo. Há relatos
  de artefatos e de falta de convergência com os defaults (rank 16, 3k steps).
- **Checkpoints (Comfy-Org/Krea-2):** raw/turbo em bf16 (26,3 GB), fp8_scaled (13,1 GB),
  int8_convrot (13,5 GB), mxfp8, nvfp4. Um benchmark da comunidade (150 imagens) coloca o
  **int8_convrot acima do fp8_scaled** em todas as métricas.
- **Memória:** LoRA em bf16 puro não cabe em 32 GB (42 GB medidos para r16 a 512). É preciso
  `blocks_to_swap` ou uma base quantizada **com a escala preservada**.
- **Papers sobre "o editor devolve a entrada":** MotionEdit (os editores preservam a pose),
  Edit-R1/UniWorld-V2 e NP-Edit (RL com recompensa de VLM "a edição foi aplicada?"),
  Kontinuous Kontext (força de edição escalar). A alavanca comum a todos, antes de RL, é
  **filtrar pares quase idênticos e ter pares com mudança real**.

---

## 4. O método recomendado: `krea2_native_edit`

Princípio: **o contrato de treino é o contrato do ComfyUI oficial, byte a byte.** Nenhum patch
na inferência; qualquer adaptação é feita no treino.

### 4.1 Numérica da base (conserta a qualidade)
- **Nunca** `diffusion_model_dtype = 'float8'` com o dequantize atual.
- Opção A (simples e exata): base em bf16 (desquantizada ou `krea2_raw_bf16`) com
  `blocks_to_swap` para caber em 32 GB. Mais lento, numericamente limpo.
- Opção B (rápida): manter o `fp8_scaled` como `QuantizedTensor`, **com escala**, igual à
  inferência. Exige verificar que o gradiente em relação à entrada passa pelas ops quantizadas
  do comfy. Teste obrigatório antes de usar: comparar o gradiente da LoRA contra a opção A num
  bloco.
- Um LoRA treinado sobre a base exata funciona em qualquer quantização fiel (fp8_scaled,
  int8_convrot).

### 4.2 Encoder de texto idêntico ao ComfyUI (conserta a paridade)
- Atualizar `submodules/ComfyUI` para o upstream (≥ `c9602625`; testado em `fb2315f1`).
- Cachear o texto **chamando a mesma lógica do `TextEncodeQwenImageEditPlus`**:
  - template Qwen-Image-Edit;
  - `Picture 1: <|vision_start|><|image_pad|><|vision_end|>` + prompt;
  - imagem VL em área 384² com interpolação "area".
- Teste de paridade: embedding do cache contra a saída do node para 3 imagens (relL2 ≈ 0).

### 4.3 Referência no contrato nativo
- Latente da ref: imagem em **~1 MP de área, múltiplo de 8, AR próprio** (a mesma conta do
  node), no grid do frame 1 com h/w a partir de 0.
- `index_timestep_zero`: ref em t=0 por token.
- Implementar o forward de treino com as mesmas operações do `_forward` upstream, com um
  **teste de paridade em CPU contra o `_forward` do ComfyUI** num Krea 2 minúsculo aleatório
  (a mesma técnica do `test/test_anima_nextscene.py`).

### 4.4 LoRA (conserta a cópia estrutural e a compatibilidade)
- **Global** (sem routing) em todas as Linears dos 28 blocos + `txtfusion` (sem o projector).
  Carregável pelo `LoraLoaderModelOnly`. Rank 64–128, α = rank.
- Salvar no formato de chaves que o `comfy.lora` carrega, com teste: carregar pelo loader
  do ComfyUI e conferir que 100% das chaves casam.

### 4.5 Dados (conserta a maior parte do "devolve a mesma imagem")
- Base: `proxima_cena_grounded_original_dataset` (12.455 pares, ~11,5k casados; faltam
  800 B no ds2).
- **Fora:** pico-banana e edições locais, ou no máximo 5–10% para um modelo de próxima cena.
- `tools/nextscene_pairs.py` para tirar quase duplicados e pares sem relação (os limiares já
  calibrados pelo agente do Anima servem de partida).
- Balancear NSFW (~44% R18 estimado): com 60%, o beta1 puxava conteúdo explícito até com
  prompt neutro.

### 4.6 Legendas (evita o polo "ignora a ref")
- As legendas atuais descrevem B em 60–110 palavras, com aparência. Com grounding, o texto já
  "vê" A, então isso é menos grave que no Anima, mas ainda é o caminho do `apex`.
- Tiers via `captions.json`: `["completa", "curta", "curta"]` (a curta pesa 2×). A curta usa
  o dialeto de inferência ("the same girl … now …"). Idealmente gerada por VLM a partir do
  par A→B, descrevendo **a mudança**.
- **`caption_dropout` = 0.** O uncond vem do negativo com a mesma imagem e prompt vazio
  (a convenção documentada do node conradlocke e do Qwen-Edit). Para a Turbo com CFG 1 isso
  nem entra.

### 4.7 Timesteps e duração
- Lei `LN(m(N), s(N))` do `krea2_apex` (cobre as duas pontas a 512/768/1024) + **15% de σ
  uniforme em [0,8; 1]** (a banda da composição, onde a turbo decide).
- 512 para o grosso e 1024 para 10–20% do final (a fase 2 da saga mostrou que isso ajuda).
  **No resume, conferir o LR no log.**
- Batch 1–2 com o throughput medido; avaliar a cada ~500 steps.

### 4.8 Avaliação (portar do Anima)
- `tools/nextscene_eval.py` adaptado ao Krea 2: `ref_gain`, `copy_rate`, `copy_gap` e CCIP
  contra o **B real** dos held-out, com grid A | B | ref certa | ref trocada | sem ref.
- **Rodar a avaliação no workflow nativo do ComfyUI** (API/headless), não no runner. O teste
  de aceitação é "funciona no ComfyUI normal".
- Avaliar em Raw (28 passos, CFG ~4–5,5) **e** Turbo (8 passos, CFG 1), que é o uso real.

### 4.9 A/B planejados (uma variável por vez, probes curtos)
1. **G0 contra G1:** só VAE (`CLIPTextEncode` + `ReferenceLatent`) contra grounded
   (`TextEncodeQwenImageEditPlus`). Os dois são nativos. O `omini_true` sem grounding teve a
   melhor fidelidade de referência de todos os braços; o grounding traz a resolução de "the
   same X", mas também o risco de reconstrução.
2. **Se a cópia persistir:** ref num frame mais distante ou deslocada em w. O núcleo cai de
   0,987 (frame 1) para 0,911 (frame 5). Isso exige um node **mínimo** (um hook
   `post_input` de ~15 linhas que só soma um offset nos `img_ids` da ref, sem substituir o
   forward). Fica como fallback, porque quebra o "zero custom node".

---

## 5. Resgate sem treinar (para os adapters que já existem)

Barato, vale fazer antes de retreinar:
- beta1 com `block_strength` 0,6–0,8 no node CtxRush (desanda a catraca);
- `reference_guidance`/força do `K2ReferenceGuider` 0,6–0,9 (interpolação convexa já
  implementada);
- prompt reformulado no dialeto do treino (`K2 Prompt Rewriter`), porque o passthrough do
  conradlocke também era sensibilidade à forma da instrução.

---

## 6. Resumo em uma tabela

| queixa | causa | correção no treino |
|---|---|---|
| LoRA degrada a qualidade | base fp8 re-quantizada sem escala (até 26,8% de pesos zerados) | base bf16 exata ou fp8 com escala preservada |
| precisa de patch no ComfyUI | base diferente + TE diferente + routing + geometria custom | contrato nativo: `ReferenceLatent` + `index_timestep_zero` + `TextEncodeQwenImageEditPlus`, LoRA global |
| devolve a mesma imagem | catraca do routing + 49% de dados de cópia + caption dropout + 13k steps sem freio + grounding | LoRA global, sem pico/quase duplicados, sem caption dropout, legendas curtas, avaliação contra o alvo real e parada pelo `copy_rate` |

---

## 7. Revisão após a conversa com o usuário (mesmo dia)

- **"O nativo só dá uma guinada fraca de estilo"**: sem LoRA treinado, sim. O caminho
  nativo é só o encanamento (onde os tokens da ref entram); a fidelidade vem do LoRA.
  Prova: o `krea2_style_reference` do ostris, publicado pelo Comfy-Org, roda nesse caminho
  sem patch.
- **Risco que eu tinha subestimado:** a geometria nativa (frame 1, h/w alinhados) tem núcleo
  RoPE **0,987**, o mesmo atrator de cópia do Anima. O beta1 usava `width_shift` (0,806).
  Ir para o nativo pode piorar o "devolve a mesma imagem". Por isso a decisão é um A/B, não
  uma aposta.
- **Plano aprovado:** A (nativo puro) × B (beta1 corrigido), 500 steps cada; C (nativo +
  deslocamento via hook `post_input`) só se A copiar. As três correções valem para
  qualquer braço: base numérica, encoder de texto e dados sem aulas de cópia.
  Execução: `docs/HANDOFF_KREA2_AB.md`.
- **Implementado e verificado em CPU:** `models/krea2_native.py`. O forward de treino bate
  com o `_forward` do ComfyUI upstream `fb2315f1` (relL2 ~1e-7 em `index_timestep_zero` e
  `index`, ref com grid próprio de tamanho diferente); o controle negativo com `width_shift`
  reprova (3e-3). Tamanhos e resize da ref e do VL são idênticos à conta do node. A paridade
  do encoder de texto precisa dos pesos reais (`tools/krea2_native_parity.py te-*`, na GPU).
- **Duas referências:** suportadas pelo nativo (até 3, frames 1..3). Ficam para depois do A/B.
