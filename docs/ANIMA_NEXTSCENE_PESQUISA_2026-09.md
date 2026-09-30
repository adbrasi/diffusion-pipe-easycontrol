# Anima "próxima cena" — pesquisa consolidada e o método (2026-09-30)

**Para quê:** fechar, com evidência, *por que* as tentativas no Anima chegaram perto
várias vezes sem nunca acertar o ponto, e definir UM método de treino que tenha
sentido para rodar numa RTX 5090 (LoRA sobre o Anima congelado, sem retreinar o modelo).

**Base:** leitura de todas as notas do repo (saga Anima, rodadas 1/2, Krea 2,
Ideogram 4, auditoria), do código do fork, e seis frentes de pesquisa externa
(literatura de edição, arquitetura Cosmos/Anima, anti-cópia e mineração de pares,
consistência/multi-painel, código do krea2edit-trainer/ai-toolkit/LLLite, e
projetos da comunidade Anima). Relatórios brutos ficaram fora do repo; aqui está
só o que sobrevive à checagem.

**Código novo:** `models/anima_nextscene.py` (tipo `anima_nextscene`),
`infer_easycontrol.py --mode nextscene`, `tools/nextscene_pairs.py`,
`tools/nextscene_eval.py`, configs em `examples/anima_nextscene/`, testes CPU em
`test/test_anima_nextscene.py` (13 testes, passando).

---

## 0. A resposta em cinco linhas

1. **O sintoma "devolve a mesma imagem" tem causa geométrica.** A referência entrava
   em `(t=1, h, w)`, no mesmo grid espacial do alvo. O Anima é só T2I (confirmado nos
   pesos), então o eixo temporal nunca foi treinado e quase não separa nada. Para a
   atenção, a ref vira uma cópia sobreposta, pixel a pixel, do próprio alvo.
2. **Quem conseguiu no Anima tirou a referência do grid do alvo** (AnimaRefLora,
   500K steps) e treinou em escala. O paper Stand-In mede o efeito: similaridade de
   identidade 0,724 com a ref fora do grid contra 0,536 com posições compartilhadas.
3. **O Anima nunca recebeu o tratamento que o Krea 2 recebeu.** Os runs no Anima
   usaram 1.255 pares e 500–2.000 steps; o Krea 2 usou ~30k pares e 13k steps; os
   projetos Anima que funcionam usam de 62k pares a 500K steps.
4. **A loss recompensava a cópia.** Em pares parecidos, a maior parte da imagem é
   "copie A". A loss ponderada pela diferença e a mistura de ruído alto atacam isso.
5. **Método:** `anima_nextscene`, com ref limpa em t=0 por frame, RoPE disjunto, LoRA
   global (self_attn + mlp + cross_attn, sem adaln/llm_adapter), dropout de ref, loss
   ponderada pela diferença, 20% dos steps em σ∈[0,8; 1], pares filtrados e avaliação
   contra o alvo real. O primeiro experimento na 5090 é um A/B de uma variável só:
   `aligned` contra `disjoint_w`.

---

## 1. Os pontos que não tínhamos encontrado

### P1 — A geometria criava o atrator de cópia (o principal)

**Fato verificado:** o `x_embedder.proj.1.weight` do Anima tem 68 colunas
(16 canais latentes + 1 padding mask, × patch 2×2). O Cosmos-Predict2-2B-Video2World
tem 72, porque inclui o canal de máscara de condição. O Anima descende do
**Text2Image**: atenção entre frames e posições temporais > 0 **nunca foram treinadas**.

**Conta** (RoPE do Anima: head_dim 128 = 42 h + 42 w + 44 t; θ=10⁴, NTK 4× em h/w):
núcleo posicional médio `K = mean cos(Δ·freq)` entre dois tokens de conteúdo idêntico.

| relação com o token-alvo (0,h,w) | K |
|---|---|
| ele mesmo | 1,000 |
| **ref no IC-LoRA antigo (t=1, mesmo h,w)** | **0,987** |
| vizinho espacial imediato (Δw=1) | 0,988 |
| ref deslocada 32 tokens em w (512px) | 0,909 |
| ref deslocada 64 tokens em w (1024px) | 0,853 |

A ref ficava posicionalmente **indistinguível do pixel vizinho do próprio alvo**.
A atenção pré-treinada, que privilegia vizinhança, lê a ref como "a mesma imagem,
no mesmo lugar". Isso prevê tudo o que vocês viram:
- devolve a referência quase sem mudar (o sintoma principal);
- o artefato de "painel duplicado/dividido" nos grids da rodada 2;
- herança forte de paleta/luz e fraca de identidade (o que está alinhado é o layout,
  não o personagem que se moveu);
- a cópia piora com mais treino (cada step reforça o caminho alinhado).

**Evidência externa independente:**
- **AnimaRefLora** (crazysheep924, Anima Base v1.0): *"colocar todos os frames no
  mesmo grid espacial dá à atenção alinhada por coordenada um caminho direto de
  copiar e colar"*; e *"o prior de vídeo espera frames consecutivos muito parecidos,
  então reproduzir a ref é o comportamento padrão deste backbone"*. A solução deles
  foi dar a cada ref um deslocamento de um frame inteiro em h ou w, além do índice
  temporal.
- **Stand-In** (ablação): ref fora do grid 0,724 contra posições compartilhadas 0,536
  (similaridade de rosto).
- **UNO:** o offset da ref existe para impedir que o modelo *"aprenda a distribuição
  espacial original da referência"*, e isso é o que causa o copy-paste.
- **OminiControl:** deslocar a ref em +32 tokens em tarefa não alinhada "melhora
  significativamente a convergência".
- **ID-LoRA:** posições negativas para os tokens de ref (sem isso o WER vai de
  0,113 para 0,252).
- **Contraponto honesto:** Kontext, Qwen-Image-Edit e Z-Image usam ref **alinhada**
  (eixo de frame, mesmo h,w), e o conradlocke também. Em **edição local** o layout se
  preserva, então o prior alinhado ajuda. Em próxima cena, com câmera e pose novas,
  ele atrapalha. É por isso que o primeiro experimento é o A/B, e não uma afirmação.

### P2 — Escala: o Anima nunca foi treinado de verdade nesta tarefa

| projeto | base | dados | treino | resultado |
|---|---|---|---|---|
| rodadas Anima (jul) | Anima | 1.255 pares | 500–2.000 steps × batch 8, rank 32 | "chegou perto" |
| abril (o melhor Anima) | Anima | ~21k pares | ~1.950 × 8, rank 64 | o melhor até então |
| **Krea 2 beta1 (seu)** | Krea 2 | **30k pares** | **13,4k × 4**, rank 64 | **preserva personagem/estilo/cenário** |
| conradlocke | Krea 2 | pares mesmo-personagem | rank 256, ~2k steps/estágio | identity edit usável |
| darask0 Anima-InContext | Anima | **62k pares** (CCIP) | rank 64 | funciona, "lava o fundo" |
| AnimaRefLora | Anima | 138k imagens | **500K steps**, LoKr 512 | funciona |

A regra "parar em ~500 steps" das rodadas era **overfit de dataset minúsculo**:
500 × 8 = 4.000 amostras ≈ 3 épocas de 1.255 pares. Com um dataset de dezenas de
milhares de pares a dinâmica é outra. O AnimaRefLora registrou que a **fase de cópia
teve pico perto de 95K steps e colapsou depois de 100K**, já com o anti-cópia ligado.
Com a receita certa, cópia pode ser uma fase do treino e não o destino. Isso é n=1,
então vale como alerta, não como lei.

### P3 — A loss recompensava a cópia

Em frames próximos ou páginas seguidas, A e B compartilham quase tudo. A MSE média
é dominada pelas regiões iguais, onde "copie A" é a resposta ótima. A ref chega limpa,
o alvo ruidoso: em σ alto (onde se decide composição) a ref é o único sinal nítido.
**Correções usadas por quem funcionou** (todas em `anima_nextscene`):
- **loss ponderada pela diferença:** regiões iguais pesam 0,2 e regiões que mudaram
  pesam >1, com média mantida em 1 (não altera o LR efetivo);
- **mistura de ruído alto:** 20% dos steps com σ~U[0,8; 1], onde quase nada do alvo
  sobrevive e o modelo precisa usar a ref e o texto;
- **pares quase duplicados removidos** (dHash/thumbnail): cada par desses é uma aula de
  cópia pura;
- **dropout da ref** (ela fica em branco, zeros): nulo treinado, dial de `ref_cfg`, e
  o modelo não pode assumir que a ref sempre está lá. O armD da rodada 2 já apontava
  nessa direção.

### P4 — O canal semântico do Krea 2 não existe no Anima, e a evidência diz que não é o gargalo

O Krea 2 funcionou com grounding (o Qwen3-VL vê a ref). No Anima o texto nunca vê a
imagem. Mas:
- **Kontext** mostra que um caminho só com tokens VAE basta para edições clássicas;
- o **BAGEL** (ablação) mostra que tirar os tokens de visão custa ~16% só em edições
  "de raciocínio";
- no próprio Anima, o **AnimaRefLora testou adapters estilo IP-Adapter e KV de
  referência, e nenhum superou o in-context simples** na avaliação humana. Os tokens
  de identidade CCIP deles ficaram com contribuição quase nula no fim.

Conclusão: não é aqui que o Anima perdia. Se faltar identidade depois do método
principal, a opção barata é pôr os descritores do personagem no prompt (gerados por
VLM na inferência) e não construir um adapter novo.

### P5 — Por que o routing perdeu no Anima (e ganhou no Krea 2)

A forense do LoRA do conradlocke mostra `wq` 16,7% e `wk` 9,2% da energia, contra
`wv` 1,9%. O LoRA aprende principalmente **para onde o alvo olha**. O routing
condition-only congela exatamente as queries do alvo. No Krea 2 o grounding global
compensava isso; no Anima não havia compensação. O método usa **LoRA global**.

### P6 — Captions: os dois atalhos

- **Legenda exaustiva** → o texto resolve tudo, a ref é ignorada (polo "bonito sem
  relação").
- **Pares sem delta, ou legenda vazia com ref limpa** → reconstruir é o ótimo (polo
  "copia"). O AnimaRefLora mediu que **dropout da legenda inteira adiantou o início da
  cópia de ~25K para ~10K steps**. Eles usam dropout por tag (50% dos steps, cada tag
  mantida com p=0,5, mínimo de 3) e **zero** dropout de legenda inteira.
- O `PROPOSTA_CAPTIONS_v3.md` (herança por omissão, tiers rich/normal/terse, ban de
  estilo) está alinhado com isso. No fork, os tiers entram via `captions.json` (cada
  legenda vira uma amostra cacheada), sem precisar de encoder de texto ao vivo.

### P7 — A avaliação premiava o erro

Cinco correções na rodada 2 vieram de métrica fraca com amostra pequena, e a
"fidelidade" media distância de paleta **contra a referência**, ou seja, premiava
cópia. `tools/nextscene_eval.py` mede tudo contra o **alvo real (B) de pares
held-out**:
- `ref_gain` = sim(saída, B) − sim(saída com ref trocada, B)
- `copy_gap` = sim(saída, A) − sim(B, A)
- `copy_rate` = fração de saídas quase duplicadas de A (dHash)
- `ccip_true` = identidade anime (CCIP da deepghs) entre a saída e B, se o
  `dghs-imgutils` estiver instalado

O veredito final continua visual (o grid sai junto), mas agora com números que não
recompensam o erro.

---

## 2. O mapa: como os outros fazem

| sistema | ref entra como | posição da ref | t da ref | escopo | nota |
|---|---|---|---|---|---|
| FLUX.1 Kontext | tokens VAE na sequência | eixo de frame=1, h,w alinhados | t compartilhado | full FT, milhões de pares | channel concat "pior" |
| Qwen-Image-Edit-2511 | VAE + Qwen2.5-VL | eixo de frame | **t=0 (`zero_cond_t`)** | full | a versão "menos drift" adotou t=0 |
| Z-Image-Edit | VAE + SigLIP2 | t+1, h,w alinhados | t diferente | full, T2I:I2I = 4:1 | pares de vídeo |
| ChronoEdit-2B (NVIDIA) | **Cosmos-Predict2.5-2B** | entrada no frame 0, saída mais longe | limpa | full | precedente mais próximo do Anima |
| conradlocke (Krea 2) | VAE + Qwen3-VL (grounding 384–768) | frame 1, centrado | t compartilhado | LoRA global r256 | usa `ref_boost` na inferência |
| ai-toolkit krea2 edit | VAE + VL "Picture N" | grid próprio | t=0 | LoRA global | um usuário achou o pior |
| darask0 (Anima) | frame temporal extra | índice temporal próprio | t=0 | LoRA r64 α32 | 62k pares CCIP; lava o fundo |
| **AnimaRefLora (Anima)** | 2 frames (rosto + ref) | **tiles disjuntos (h ou w)** | t=0 | LoKr 512 | anti-cópia completo |
| LLLite (kohya) | resíduo por posição | alinhado por construção | – | pequeno | **não expressa "mesmo personagem, nova pose"** |

Padrões convergentes:
1. Ref limpa na mesma sequência de atenção, com loss só no alvo.
2. t=0 separado para a ref é a tendência mais recente.
3. Para tarefas não alinhadas (identidade, próxima cena), a posição da ref sai do
   grid do alvo.
4. O controle anti-cópia vem principalmente dos **dados**: pares em que o alvo muda de
   verdade, filtro de quase duplicados, dropout.

---

## 3. Os caminhos avaliados e o veredito

| caminho | veredito | por quê |
|---|---|---|
| IC-LoRA temporal alinhado (o antigo) | ⚠️ controle do A/B | geometria de sobreposição (P1); serve de baseline |
| **In-context com RoPE disjunto** | ✅ **método principal** | evidência mais forte, no próprio Anima |
| Diptych espacial (painel lado a lado) | ↔ equivale ao disjunto | a ref lado a lado *é* o `disjoint_w` com t=0 por frame, sem perder resolução |
| ControlNet-LLLite | ❌ para identidade | resíduo por posição com campo receptivo de ~90px; a ref não tem tokens próprios. Útil só para sinal alinhado (pose/lineart) combinado com o in-context |
| IP-Adapter / tokens de imagem na cross-attn | ❌ como caminho principal | comprime a ref em poucos tokens, perde cenário; no Anima não superou o in-context (AnimaRefLora) |
| EasyControl (máscara causal) | ⏸ | já treinado e funcionando para controle espacial; para próxima cena não traz nada que o disjunto não traga |
| Routing condition-only | ❌ no Anima | congela as queries do alvo (P5); perdeu na rodada 2 |
| Treinar llm_adapter / adaln | ❌ | perdeu nas rodadas 1/2; recomendação oficial também |
| Full fine-tune | ⏸ reserva | 2B cabe numa 5090 com AdamW8bit + checkpointing, mas LoRA r64–128 é o que todos os precedentes de orçamento pequeno usam |

---

## 4. O método: `anima_nextscene`

```
x   = [ alvo ruidoso (T=0) | ref limpa (T=1) ]      (B, 16, 2, H, W)
t   = [ σ                  | 0                 ]    timestep por frame (nativo no Cosmos)
RoPE: alvo (0, h, w) · ref (1, h, w + W)             rope_layout = 'disjoint_w'
loss: MSE de velocidade só no frame do alvo, ponderada pela diferença A↔B
```

| knob (`[nextscene]`) | valor | por quê |
|---|---|---|
| `rope_layout` | `disjoint_w` (A/B contra `aligned`) | P1 |
| `ref_temporal_index` | 1 | aceita negativo; `-1` + `disjoint_w` reproduz a relação do AnimaRefLora com a ref completa |
| `ref_dropout` | 0,1 | nulo treinado, `ref_cfg` confiável, atribuição em vez de acoplamento |
| `lora_cross_attn` | true | escopo vencedor da rodada 1; llm_adapter e adaln sempre fora (auditado no save) |
| `diff_weight` | true (floor 0,2) | P3 |
| `high_noise_prob` | 0,2 em σ∈[0,8; 1] | P3; mesma escolha do AnimaRefLora |
| `ref_hflip_prob` | 0 (A/B futuro) | flip da ref no espaço latente, anti-cópia estilo Paint-by-Example |
| `ref_noise_prob` | 0 (A/B futuro) | ruído pequeno na ref estilo SVD/LTX (σ~LogN(−3; 0,5)); a ref ganha t=σ_ref |
| LoRA | rank 64, α 64 | faixa dos precedentes (darask0 64, ai-toolkit 16–64, conradlocke 256) |
| otimizador | adamw_optimi, lr 5e-5, betas (0,9; 0,99), wd 0,01 | a meio caminho entre o oficial (2e-5) e as rodadas (1e-4) |
| batch | micro 1 × accum 4 | |
| timesteps | logit-normal, `sigmoid_scale` 1,0, sem shift no treino | conselho oficial do Anima; shift 3 só na inferência |
| base | **Anima Base v1.0** | todos os adapters da comunidade que funcionam foram validados nela |

O adapter grava o contrato no header do safetensors (`nextscene_contract`) e num
`nextscene_contract.json` ao lado. O runner lê isso sozinho, então não há como
inferir com a geometria errada. Esse erro, na saga, gerava "ruído puro" que parecia
falha de método.

---

## 5. Plano na 5090

### Fase 0 — dados (CPU, antes de qualquer GPU)
```bash
python tools/nextscene_pairs.py audit --target <B> --control <A> --out report.csv --dino
# olhar o report: % near_dup, % unrelated, mediana de palavras da legenda
python tools/nextscene_pairs.py build --report report.csv --out-root /workspace/ns_filtered --reverse
```
- Separar **16–24 pares held-out** (nunca vistos no treino) em
  `/workspace/heldout/{target,control}`, cobrindo as fontes: vídeo, quadrinho,
  próxima cena com câmera nova, e o mesmo cenário com pose nova ("gato sentado → em pé").
- Medir o balanço de conteúdo por fonte (o viés de 60% NSFW do Krea 2 era previsível
  pela contagem).
- Legendas: ver §6.

### Fase 1 — o A/B de geometria (~2 × 2.500 steps)
```bash
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config examples/anima_nextscene/probe_A_aligned.toml
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config examples/anima_nextscene/probe_B_disjoint_w.toml
python tools/nextscene_eval.py --dit ... --vae ... --llm ... --pairs /workspace/heldout \
  --ckpt <A>/step1000 --ckpt <A>/step2500 --ckpt <B>/step1000 --ckpt <B>/step2500 --out /workspace/eval_probe
```
Os dois configs só diferem em `rope_layout`. **Antes do run longo:** smoke de 10
steps, conferir o log (`RoPE layout=...`, número de alvos do LoRA, audit do adapter)
e medir s/step e VRAM. Não prometo tempo por step sem medir.

**Como ler:**
- `B` com `copy_rate` menor e `ref_gain` > 0 → a hipótese P1 se confirma;
- os dois com `ref_gain` ≈ 0 → ainda não leem a ref: mais steps, ou legendas longas
  demais;
- `B` sem cópia mas com identidade fraca → subir `ref_cfg` na inferência e seguir para
  o run longo (identidade cresce com os steps; o armD já mostrava isso).

### Fase 2 — run longo (`full_run.toml`)
Com a geometria vencedora e todos os pares filtrados. Avaliar a cada 1.000 steps e
escolher o checkpoint pelo `ref_gain`/`copy_rate` e pelo olho, **não pelo último**.

### Fase 3 — acabamento em alta resolução
Retomar a 768/1024 por 10–20% dos steps. Lição do Krea 2: o adapter de 512 já funciona
a 1024, e o detalhe melhora muito. No resume do DeepSpeed, conferir o LR no log.

---

## 6. Dados e legendas (o que mais pesa depois da geometria)

- **Faixa de similaridade:** tirar quase duplicados (cópia pura) e pares sem relação
  (corte de cena, outra obra). Os limiares do `nextscene_pairs.py` são ponto de
  partida; ajuste olhando o report.
- **B→A:** próxima cena não tem direção privilegiada. `--reverse` dobra os pares quando
  a ref tem legenda própria.
- **Legenda descreve B com herança por omissão.** O que está na ref não precisa estar
  no texto. Tiers via `captions.json`: `["descrição completa", "legenda curta só com a
  mudança"]`. O tier curto ensina a ler a ref; o completo mantém a obediência a prompts
  longos.
- **"Mesmo cenário, nova pose"** é o caso que nenhum projeto público cobre (o darask0
  lava o fundo porque os pares dele só compartilham o personagem). Os pares de vídeo e
  quadrinho que você tem são exatamente o que falta ali. Se precisar reforçar: frames
  do mesmo plano com câmera fixa e sujeito em movimento, ou pares gerados por um editor
  (Qwen-Image-Edit-2511) a partir de imagens do próprio Anima, filtrados por CCIP. Se
  for gerar pares com um editor, conferir a licença do modelo professor.
- **Sem dropout de legenda inteira no começo** (P6). Os tiers curtos cumprem esse papel
  sem ensinar "legenda vazia + ref = copie".

---

## 7. Inferência

```bash
python infer_easycontrol.py --dit anima-base-v1.0.safetensors --vae ... --llm ... \
  --lora <run>/stepN/adapter_model.safetensors --mode nextscene \
  --control_image cena1.png --prompt "the same cat now stands on the sofa, stretching" \
  --width 768 --height 768 --steps 30 --cfg 4 --flow_shift 3 --ref_cfg 1.0
```
- **CFG:** por padrão `--uncond_ref keep`, que é CFG só de texto com a ref presente
  nos dois ramos. É a convenção do Cosmos Video2World (a NVIDIA comenta que aplicar
  CFG na condição "piora os resultados") e do Qwen-Image-Edit.
- **`--ref_cfg` > 1:** soma `(ref_cfg−1)·(c − f(texto, ref=zeros))`, usando o nulo
  treinado pelo `ref_dropout`. É o dial de identidade; o achado do `ref_cfg` alto da
  rodada 2 vale aqui também.
- **`--ref_renoise`:** re-ruído da ref estilo LTX (s = k·σ²). É um dial diagnóstico
  anti-cópia. Também dá para testar nos adapters **antigos** (com
  `--rope_layout aligned --ref_temporal_index 1`, target-first) para confirmar o
  mecanismo sem treinar.
- **ComfyUI:** o node atual não conhece o layout disjunto. O AnimaRefLora registrou que
  o sampler nativo de Anima no ComfyUI limita as posições do RoPE (`max_h = 120`). Um
  node próprio precisa instalar o mesmo `generate_embeddings`
  (`models/anima_nextscene.py::install_nextscene_rope`) e ampliar esse limite. Fica
  para depois do A/B, para não construir o node na geometria errada.

---

## 8. Riscos e o que ainda não sabemos

1. **O A/B pode empatar em 2.500 steps.** A separação pode só aparecer com mais treino.
   Nesse caso, seguir com o `disjoint_w`, porque é onde está a evidência externa.
2. **O disjunto pode enfraquecer a preservação do cenário** em planos de câmera fixa,
   onde o alinhamento ajudava. Se o held-out "mesmo cenário, nova pose" piorar, o A/B
   seguinte é `ref_temporal_index` maior com `aligned`. O FramePack-1f mostra que a
   distância temporal funciona como um dial de "quanta mudança".
3. **Treinar por cima do AnimaRefLora** (só ideia): o LoKr deles é formato LyCORIS e
   precisaria ser fundido no base antes. A geometria é compatível via
   `ref_temporal_index=-1` + `disjoint_w`. Não implementado.
4. **Custo por step desconhecido** para T=2 a 512/768 neste fork. Medir no smoke.
5. **n=1 em quase toda a evidência da comunidade.** O AnimaRefLora diz que a receita
   dele foi empilhada sem ablação. Por isso o plano mede uma variável por vez e usa
   métricas contra o GT.

---

## 9. Referências principais

- AnimaRefLora: https://github.com/crazysheep924/AnimaRefLora · https://crazysheep924.github.io/AnimaRefLora/
- Anima-InContext-Character: https://huggingface.co/darask0/Anima-InContext-Character
- Anima (Base v1.0): https://huggingface.co/circlestone-labs/Anima
- Cosmos-Predict2 (Video2World, frame replace, `sigma_conditional`): https://github.com/nvidia-cosmos/cosmos-predict2
- ChronoEdit: https://arxiv.org/abs/2510.04290
- FLUX.1 Kontext: https://arxiv.org/abs/2506.15742
- Qwen-Image-Edit-2511 (`zero_cond_t`): https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Z-Image: https://arxiv.org/abs/2511.22699
- UNO (UnoPE): https://arxiv.org/abs/2504.02160 · OminiControl: https://arxiv.org/abs/2411.15098
- Stand-In: https://arxiv.org/abs/2508.07901
- krea2edit-trainer (conradlocke): https://github.com/lbouaraba/krea2edit-trainer · https://huggingface.co/conradlocke/krea2-identity-edit
- ai-toolkit: https://github.com/ostris/ai-toolkit
- ControlNet-LLLite: https://github.com/kohya-ss/sd-scripts/blob/sdxl/docs/train_lllite_README.md · Anima-LLLite: https://huggingface.co/kohya-ss/Anima-LLLite
- FramePack 1f (distância temporal como dial de mudança): https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack_1f.md
- MotionEdit (editores preservam a pose original): https://arxiv.org/abs/2512.10284
- Cut2Next (próximo plano): https://arxiv.org/abs/2508.08244 · InstructMove: https://arxiv.org/abs/2412.12087
- Paint-by-Example (augmentation anti-cópia): https://arxiv.org/abs/2211.13227
- CCIP (identidade anime): https://huggingface.co/deepghs/ccip
