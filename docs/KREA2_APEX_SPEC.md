# krea2_apex — contrato + spec de implementação

**Ação agora (10 min, 0 GPU):** criar `/workspace/projects/diffusion-pipe-easycontrol/models/krea2_apex.py` a partir do esqueleto do §2.1 e registrar o type. Todo o resto depende dele.

Estado verificado antes de escrever: GPU livre (18 MiB / 32 GB), disco 275 G livres de 428 G, nenhum treino rodando.

---

# 1. O contrato completo

`N` = nº de tokens de imagem do **alvo** = `(h_lat//2)·(w_lat//2)` (512px→1024, 768px→2304, 1024px→4096), a mesma quantidade que `models/krea2.py:147` já usa.

| # | Eixo | Valor apex | Origem | Certeza |
|---|---|---|---|---|
| 1 | Eixo da referência | **frame axis**, `pos[...,0] = 1.0`; alvo e texto em 0 | conradlocke (`custom_nodes/comfyui-krea2edit/__init__.py:47`) + NK2E; provado ótimo pelo derivador de geometria | **Provado.** Δf=1 = viés rígido de −0,144 nats sobre o bloco da ref, não reordena (softmax invariante a constante aditiva por bloco). width_shift = −2,200 nats **e** reordena, com atração de costura de +2,008 nats sobre a correspondência verdadeira |
| 2 | `width_shift` | **descartado** do código de produção | derivação nova | Provado. 13/24 pares de fase descoerentes a 512px, 15/24 a 1024px; piora com resolução |
| 3 | Fit da ref | AR-preservado, fit-inside, **crop-to-grid /16**, ramo CROP_TOL 8 % que preenche o grid; resample **bicubic + antialias** | conradlocke `_fit_prep` (`ai-toolkit/extensions/krea2_edit/krea2_edit.py:519-551`) ≡ node `_fit_encode_image` fit branch (`__init__.py:88-126`) | Provado idêntico ao node. **No-op bit-a-bit em 100 % do `konrad_ds4` e ~95 % do `konrad`** (dims A↔B iguais) → custo zero no treino, compra o contrato de inferência |
| 4 | Offsets h/w da ref | **fracionários centrados**: `off = max(0, (G_tgt − G_ref)/2)` em float, nunca `//2` | conradlocke (`__init__.py:44`, `krea2_edit.py:117-120`) | Provado. Floor custa 0,100 nats (−9,5 % de peso de atenção) em **todo** par de gap ímpar, sistemático |
| 5 | Escala de posição da ref | `1.0` (no-op). Pixels já foram reamostrados à densidade do alvo | derivação | Provado — reescalar posições depois de reamostrar conteúdo fabrica colisão/skip |
| 6 | Ordem da sequência | `[texto \| alvo \| ref]` (a do fork, **inalterada**) | Claim provado no `RELATORIO_ELEMENTO_DISTOANTE.md:95` | Provado idêntica ao `[texto\|ref\|alvo]` do node: sem máscara causal (`src/mmdit.py:451`), softmax é permutação-equivariante; posições e tvec viajam com os tokens; `Krea2ReferenceFinalLayer` fatia por índice |
| 7 | Timestep da ref | **`reference_timestep = 'target'`** (tvec único, ref limpa) | conradlocke — **forçado por compat**, ver §3 | Decidido por compat, não por matemática. O node calcula **um** `tvec` de `timesteps` (`__init__.py:215-216`) e o difunde: t=0 per-token é **inexprimível** ali. A matemática (Teorema 2) favorece t=0; a única evidência dura é sobre *mismatch* |
| 8 | Distribuição de t | **`t = sigmoid(s(N)·z + m(N))`, z~N(0,1)**, com<br>`m(N) = 0.337 + 0.335·ln(N/1024)`<br>`s(N) = 1.538 + 0.130·ln(N/1024)` | **derivação nova** — ajuste aos ótimos medidos no espectro dos nossos latentes (512/768/1024) | Erro do ajuste ≤ 0,002 nas 3 âncoras. Reproduz `timestep_type: weighted` do conradlocke (KL 0,024 contra a densidade de decisão de conteúdo) **e** o estende para 768/1024, onde a tabela fixa dele subcobre |
| 9 | Peso na loss | **`w ≡ 1`** — amostragem por importância, sem tabela | derivação (objetivo §2.1) | Provado que peso e schedule são o mesmo grau de liberdade (Radon–Nikodym). Vantagem operacional: **a loss volta a ser comparável entre braços** |
| 10 | Amostragem de t | **estratificada de baixa discrepância** (golden-ratio na CDF), não iid | derivação nova (objetivo §4.4) | Remove 88–93 % da variância de gradiente induzida pelo schedule a batch 1. Custo: 6 linhas, 0 GPU |
| 11 | Loss | MSE em v, **só no span do alvo**, sem peso por banda/região, sem `fft_loss` | fork (já é o comportamento) + convergência dos 3 | Provado. Piso de Bayes da v-MSE varia só 5,5× em t∈[0,01;0,99] — não há patologia de SNR para consertar |
| 12 | Grounding — imagem | **nativa, pré-fit** (o encoder recebe o *path*) | empate: fork (`models/krea2_edit.py:256-274`) ≡ conradlocke | Provado. Já correto no fork |
| 13 | Grounding — escala | cap **longest-side 768**, jitter uniforme **[384,768] POR STEP** | conradlocke (`krea2_edit.py:743`) | Provado que o nosso "jitter" é `sha256(path)` = constante por arquivo para sempre (`models/krea2_edit.py:229-233`). O default do node é `grounding_px=768` (`__init__.py:427`) |
| 14 | Grounding — cap 0 (nativo) | **não** | derivação (condicionamento §1.3) | Plausível-forte. Tokens de visão entram com RoPE **todo zero** (`krea2_edit.py:126` / fork `krea2_reference.py:342`) → detalhe fino sem endereço espacial; +2,5–2,8× step time; VRAM estoura |
| 15 | Layout do prompt VL | `vl_prompt_style = 'plain'` (blocos de visão antes da caption) + `KREA2_TEMPLATE` | fork já tem; **byte-idêntico ao node** | Provado: `comfy/text_encoders/krea2.py:20` ≡ `__init__.py:407-412` ≡ `ai-toolkit/.../src/text_encoder.py:27-32` |
| 16 | LoRA — escopo | **global, sem routing**: 224 blocks + 32 txtfusion + **2 txtmlp** = **258 módulos / 516 tensores** | conradlocke (256) + adição derivada (`txtmlp`) | Routing descartado: provado que congela a função de query do alvo e quebra a fusibilidade em silêncio. `txtmlp.1/3` é a **única** projeção 2560→6144 por onde os tokens de visão entram na largura do DiT — hoje congelada |
| 17 | LoRA — `txtfusion.projector` | **P2 opcional**, salvo como `.diff` | derivação (condicionamento §2.4②) | 12 parâmetros (`Linear(12,1)`, `comfy/ldm/krea2/model.py:149`) = seletor de profundidade do Qwen3-VL. Alto ganho/param, risco de drift. Exige plumbing de `.diff` (§2.6) |
| 18 | LoRA — fora | `first`, `last`, `tmlp`, `tproj`, `mod.lin`, RMSNorm | derivação | `first` é linear aplicado **identicamente** a ref e alvo (`mmdit.py:423`) → não pode distinguir. `tmlp/tproj` = AdaLN global, briga com o eixo #8 |
| 19 | Rank | **r=64, α=64** (scaling 1) | 3 receitas aprovadas convergem | Rank efetivo medido ≤6 nos blocks, ≤23 no txtfusion ⇒ **`rank_pattern` no txtfusion é injustificado**; A/B legítimo é para **baixo** (r=32) |
| 20 | Otimizador | `AdamW8bitKahan`, lr **1e-4**, betas [0.9, 0.99], wd 0.01, constante, warmup **250** | fork (probe vencedor) + conradlocke + NK2E | lr fechado: nenhum run com 2e-4 foi aprovado. wd irrelevante (1,3 % de encolhimento em 13,5k steps) |
| 21 | Batch | **micro_batch 1, grad_accum 1** | derivação (objetivo §4.3) | Batch 4 = 3.700 optimizer-steps em 8 h vs 14.800. A variância que batch 4 compraria é comprada de graça pelo item #10 |
| 22 | Resoluções | probe **[512]**; run **[512, 768]** AR buckets | fork (probe vencedor usou 512/768/1024) + teto de VRAM | 1024 não cabe com TE residente. O item #8 se auto-ajusta por bucket — vantagem estrutural sobre a tabela fixa |
| 23 | Captions — fase 1 | **mistura de regime por amostra**: 55 % completa · 40 % clause-dropout(ρ=0,60, keep_tokens=1) · 5 % vazia | derivação (objetivo §3.4) | Provado que **domina estritamente** ρ fixo 0,25: P(completa) 0,601 vs 0,098, mesma destruição média (0,240), KL menor (0,361 vs 0,470) |
| 24 | Captions — fase 2 | remoção dirigida de `C_shared` (cláusulas que a ref já carrega), mistura 50/45/5 | derivação (condicionamento §3.3) | Álgebra sólida (`I(A;B\|C_delta) ≥ I(A;B\|C_random)`); exige captionar os 12.238 **sources** (hoje 0 `.txt`) |
| 25 | `caption_dropout` | **0,05** | ambos os derivadores + o node | O node documenta explicitamente o uncond grounded ("*For CFG, ground the NEGATIVE too: second node, empty prompt, same image*", `__init__.py:395-396`). É o `ε(c_I,∅)` do IP2P |
| 26 | Ref-dropout (zerar `z_A`) | **desligado por default**; A/B declarado | **resolvo contra** o derivador de objetivo, §5(c) | O node **não tem** caminho de ref zerada; o dial de fidelidade dele é `ref_boost` (viés aditivo de logit, `__init__.py:139-168`). A coluna "sem ref" do protocolo se faz **removendo o span**, não zerando |
| 27 | Multi-ref | **N=1** | os 3 concordam | Com fit centrado, ref₁ e ref₂ compartilham o suporte h/w e só 0,340 nats de rotação de frame os separam ⇒ mistura de identidades é o modo padrão |
| 28 | Infra de texto | **TE residente + encode ao vivo**, cache de texto desligado | derivação (condicionamento §4) | Provado: 24,4 MB/amostra ⇒ **271 GB** para 12.455 pares. É a causa mecânica documentada do 384² fixo do v2 (`k2_proximacena_v2.toml:5-7`). Sem isso, jitter por step é **impossível por construção** |

**Sobre "a tabela de pesos":** ela some. `timestep_type: weighted` do ai-toolkit é uma logit-normal alargada disfarçada (m=0,479, s=1,578; TV=0,043). Amostrar direto de LN(m(N),s(N)) tem o **mesmo `E[g]`**, cobre 768/1024 corretamente, e devolve a comparabilidade da loss.

Massas por banda (calculadas):

| schedule | t>0,9 | t>0,8 | 0,5–0,8 | t<0,3 | t<0,1 |
|---|---|---|---|---|---|
| **apex @512** LN(0,337; 1,538) | 11,3 % | **24,8 %** | 33,9 % | **22,1 %** | 5,0 % |
| **apex @768** LN(0,608; 1,643) | 16,7 % | 31,8 % | 32,6 % | 18,8 % | 4,4 % |
| `weighted` conradlocke | 13,8 % | 28,3 % | 33,7 % | 20,0 % | 4,5 % |
| probe vencedor (flux@512) | 5,3 % | 21,0 % | 50,9 % | **7,7 %** | 0,3 % |
| proximacena v2 (flux@1024) | 9,8 % | 31,6 % | 50,2 % | **4,0 %** | 0,1 % |

---

# 2. Spec de engenharia no fork

Ordem de implementação. Estimativas para **mim** executando; GPU separada.

## 2.1 `models/krea2_apex.py` (novo, ~200 linhas) — **1 h 30**

Herda de **`Krea2EditPipeline`** (não de `krea2_omini_grounded`: aquele arrasta o `ConditionOnlyLoRARouter`, que estamos descartando).

```
Krea2Pipeline (krea2.py)            → base, timestep sampling, get_conds
 └ Krea2ReferencePipeline           → sequência/posições/tvec per-token, save_adapter+metadata+audit
    └ Krea2EditPipeline             → grounding Qwen3-VL, caption_dropout, LoRA blocks+txtfusion
       └ Krea2ApexPipeline          ← NOVO
```

Conteúdo:

```python
class Krea2ApexPipeline(Krea2EditPipeline):
    name = 'krea2_apex'
    config_section = 'krea2_apex'
    checkpointable_layers = ['Krea2ApexInitialLayer', 'TransformerLayer']
    adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.', '.txtmlp.')
```

1. `__init__`: força `position_mode='frame_fit'`, `reference_position_offset=1.0`,
   `reference_position_scale=1.0`, `reference_timestep_mode='target'` (aceita override
   explícito `'zero'` só com `allow_node_incompatible = true` no TOML, para o A/B).
   Valida `micro_batch_size_per_gpu == 1` (ver 2.3).
2. `get_preprocess_control_file_fn()` → devolve `fit_ref(path, target_h_px, target_w_px)`,
   porte **linha a linha** de `__init__.py:88-126` (ramo `fit`), em torch:
   PIL→RGB (fundo branco)→float[0,1] CHW→CROP_TOL 0,08→ else `/16 floor` + crop-to-grid→
   `F.interpolate(bicubic, antialias=True)`→`Normalize([0.5],[0.5])`→retorna `(3,1,nh,nw)`.
3. `prepare_reference_latents`: **relaxa** o assert de shape de
   `models/krea2_reference.py:126-132` para `ref_h <= tgt_h and ref_w <= tgt_w` (+ múltiplos de 2).
4. `configure_adapter`: copia `krea2_edit.py:169-224` e troca o filtro por
   `class_name in ('SingleStreamBlock','TextFusionTransformer') or full_name.startswith('txtmlp.')`,
   mantendo o `endswith('projector')` excluído. Confere que dá **258** alvos e falha se não der.
5. `to_layers()`: `[Krea2ApexInitialLayer(...)] + [TransformerLayer(...)] + [Krea2ReferenceFinalLayer(...)]`.
6. `Krea2ApexInitialLayer(Krea2ReferenceInitialLayer)`: sobrescreve **só** o bloco de posições
   (hoje `krea2_reference.py:335-353`):

```python
off_h = max(0.0, (target_grid_h - reference_grid_h) / 2.0)   # FLOAT. nunca //2
off_w = max(0.0, (target_grid_w - reference_grid_w) / 2.0)
reference_pos = self._grid_positions(batch, reference_grid_h, reference_grid_w, dev).clone()
reference_pos[..., 0] = 1.0          # frame axis
reference_pos[..., 1] += off_h
reference_pos[..., 2] += off_w
# scale_bias / position_mode 'width_shift' NÃO existem aqui
```

7. `get_reference_metadata()`: ver §3.

## 2.2 Schedule apex em `models/krea2.py` (~20 linhas) — **30 min**

Em `prepare_inputs` (`krea2.py:107-155`), novo ramo antes do `if shift :=`:

```python
if self.model_config.get('timestep_law', None) == 'krea2_apex':
    n_tok = (h // 2) * (w // 2)
    ln = math.log(n_tok / 1024.0)
    m = 0.337 + 0.335 * ln
    s = 1.538 + 0.130 * ln
    if timestep_quantile is None:
        u = self._low_discrepancy_u(bs, device)      # golden-ratio, ver abaixo
    else:
        u = torch.full((bs,), timestep_quantile, device=device)
    z = torch.distributions.normal.Normal(0., 1.).icdf(u)
    t = torch.sigmoid(s * z + m)
```

Estratificação (`_low_discrepancy_u`, 6 linhas): contador por processo,
`u_k = frac(u0 + k·0.6180339887)`, `u0 = seed_uniform()`. Determinístico no resume
(persistir `k` junto do dataloader state) — ou aceitar re-seed, o efeito é o mesmo.
Guarda: `timestep_law` e `flux_shift`/`shift` são mutuamente exclusivos, erro se ambos.

> Nota de equivalência: `sigmoid_scale` (`krea2.py:139-140`) já dá o σ e `shift`
> (`:143`) já dá o μ via `time_shift`, pois `odds' = e^μ·odds`. O ramo novo existe só
> para a **dependência de resolução** e a estratificação — sem ele você teria que
> escolher um μ fixo e perder os buckets.

## 2.3 TE residente + jitter por step, contornando o cache — **3 h** (a peça cara)

Cinco pontos, todos em `utils/dataset.py` salvo o último.

**(a) Não construir o cache de texto.** Flag global `LIVE_TEXT_ENCODING` (mesmo padrão de
`CAPTION_DROPOUT`, `dataset.py:41`, setada em `train.py:450`). Em `_cache_fn`, envolver
o laço `for text_encoder_idx in range(num_text_encoders)` (`dataset.py:1208-1221`) num
`if not LIVE_TEXT_ENCODING:`. Em `DatasetManager.cache()`, envolver
`ds.cache_text_embeddings(None, i)` (`:1311-1312`) no mesmo guard. Isso mata **271 GB de
escrita e a passada completa do Qwen3-VL antes do treino**.

**(b) TE fica em CUDA.** Em `DatasetManager.cache()` (`:1290-1302`), quando live:
não mandar os text encoders para `'meta'` (só o VAE), e trocar `mm.unload_all_models()`
por `mm.unload_all_models(); te.load_model_if_needed(); te.load_model()` —
`comfy/sd.py:290-291` faz `load_models_gpu([self.patcher], force_full_load=True)`.
Nada mais no treino chama `free_memory`, então ele fica.

**(c) `__getitem__` devolve texto cru.** Em `SizeBucketDataset.__getitem__`
(`:335-371`), quando live: pular o laço `for te_idx, (ds, uncond_ds) in zip(...)`
inteiramente e adicionar `ret['control_file']`. **Não** mexer no `iteration_order`
(mudaria o fingerprint do cache de latentes); em vez disso, construir no `__init__`
um dict lateral a partir de `self.metadata_dataset`:

```python
self.control_by_spec = {tuple(s): cf for s, cf in
                        zip(metadata_dataset['image_spec'], metadata_dataset['control_file'])}
```

`Dataset._collate` (`:1080-1108`) já mantém chaves não-tensoriais como listas — `caption`
e `control_file` saem como `list[str]`. Zero trabalho ali.

**(d) O encode roda no processo de treino.** Ponto de enxerto exato:
`PipelineDataLoader._pull_batches_from_dataloader`, `utils/dataset.py:1454-1456`:

```python
for batch in self.dataloader:
    if LIVE_TEXT_ENCODING:
        batch = self.model.encode_text_live(batch)   # injeta text_embeds_0 / attention_mask_0
    features, label = self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)
```

Este generator roda no processo principal (é `next()` do loop de treino), não nos workers
forkados do `DataLoader` — que é o que torna CUDA legal aqui. `models/krea2_edit.py:1345-1348`
documenta exatamente a restrição que estamos respeitando.

`Krea2ApexPipeline.encode_text_live(batch)` chama
`self.get_call_text_encoder_fn(self.text_encoders[0])(batch['caption'], [False]*n, batch['control_file'])`.

> **⚠️ Armadilha que quebra no step 1 se ignorada:** `get_call_text_encoder_fn` está
> decorado com `@torch.inference_mode()` (`models/krea2_edit.py:236`). Tensores de
> inference **não podem entrar num grafo de autograd** — e o `context` entra em
> `txtfusion`/`txtmlp`, que agora **têm LoRA**. `Krea2ReferenceInitialLayer.forward` ainda
> faz `item.requires_grad_(True)` (`krea2_reference.py:361-363`). Correção: no
> `encode_text_live`, `return {k: [t.clone() for t in v] for k, v in out.items()}` **fora**
> do `inference_mode` (ou trocar o decorador por `@torch.no_grad()` no caminho live).

Guard para `pipeline_stages > 1`: só o stage 0 e o último puxam do dataloader e ambos
chamariam o TE. Com `pipeline_stages = 1` (nossa config) é irrelevante — mas deixar um
`assert`.

**(e) Jitter de verdade.** `models/krea2_edit.py:229-233` — `_jittered_grounding_side`
troca `sha256(path)` por `random.randint(lo, hi)` **quando live**. Duas linhas.
Mantém o hash quando cacheado (senão o cache fica inconsistente).

**Bônus grátis:** `_cache_grounded_uncond_embeddings` (`:209-222`) e todo o ramo
`grounded_uncond_datasets` viram código morto no caminho live — o uncond é `caption=''`
em tempo real. −9,6 GB por 1.247 pares e menos superfície de bug.

## 2.4 Fit da referência no preparo de pixels (~25 linhas) — **45 min**

Em `latents_map_fn`, `utils/dataset.py:1155-1169`: quando o modelo expõe
`get_preprocess_control_file_fn()`, usar essa função para os control files em vez de
`preprocess_media_file_fn(...)`. Hoje esse caminho chama
`convert_crop_and_resize` → `ImageOps.fit` (`models/base.py:73`) = **crop central + resize ao
grid cheio** = literalmente a geometria v1 que o conradlocke abandonou após o RCA de
seam-doubling.

`assert caching_batch_size == 1 or todas as refs terem o mesmo shape` — com fit, refs de
AR discrepante produzem latentes de shapes diferentes e `_collate` deixaria uma lista,
que `prepare_inputs` não sabe consumir. Com `micro_batch_size_per_gpu = 1` (nossa config)
o problema não existe; o assert é a rede de segurança.

**Regenerar o cache de latentes** (`--regenerate_cache`) — o conteúdo dos latentes de ref muda.
Custo: ~15 min para 12.238 pares.

## 2.5 Clause dropout por step (~15 linhas) — **30 min**

Só faz sentido depois de (2.3). Em `SizeBucketDataset.__getitem__`, no ramo live,
antes de devolver `ret['caption']`:

```python
r = random.random()
if r < CAPTION_DROPOUT:                    # 0.05  -> vazia, ref mantida nos 2 canais
    caption = ''
elif r < CAPTION_DROPOUT + CLAUSE_DROPOUT_PROB:      # 0.40
    parts = caption.split(', ')
    caption = ', '.join(parts[:KEEP_CLAUSES] +
                        [p for p in parts[KEEP_CLAUSES:]
                         if random.random() >= CLAUSE_DROPOUT_RATE])   # 0.60
# senão: caption completa (55%)
```

`KEEP_CLAUSES = 1` está **certo**: a 1ª cláusula é tipo de plano em 100 % dos casos, e
tipo de plano é o que **muda** entre A e B em "próxima cena" — é `C_delta`, não `C_shared`.

## 2.6 `txtfusion.projector` como `.diff` — **P2, 1 h**

`peft` `modules_to_save=['txtfusion.projector']` → o saver (`utils/saver.py:74`) já tira
`.default` e `.modules_to_save`, produzindo `txtfusion.projector.weight`. **Isso não carrega
no ComfyUI**: `comfy/lora.py:72-76` consome `<key>.diff`, e uma `.weight` avulsa é
silenciosamente ignorada. Em `Krea2ApexPipeline.save_adapter`, antes do
`safetensors.save_file`, converter: `W_treinado − W_base → diffusion_model.txtfusion.projector.diff`.
Widen também o `_audit_adapter_keys` (`krea2_reference.py:246-249`), que hoje rejeita
qualquer chave que não seja `lora_A`/`lora_B`.

## 2.7 TOML — **15 min**

```toml
output_dir = '/workspace/checkpoints/k2_apex_probe'
dataset = '/workspace/configs/apex_ds4_512.toml'
max_steps = 2000
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 1
warmup_steps = 250
save_every_n_steps = 250
activation_checkpointing = true
save_dtype = 'bfloat16'
caching_batch_size = 1
blocks_to_swap = 8

[model]
type = 'krea2_apex'
diffusion_model = '/workspace/models/krea2/diffusion_models/krea2_raw_fp8_scaled.safetensors'
vae = '/workspace/models/krea2/split_files/vae/qwen_image_vae.safetensors'
text_encoders = [{path = '/workspace/models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors', type = 'krea2'}]
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'
timestep_law = 'krea2_apex'        # NÃO usar flux_shift junto
live_text_encoding = true

[krea2_apex]
reference_timestep = 'target'      # obrigatório p/ comfyui-krea2edit
vl_prompt_style = 'plain'
vl_longest_side = 768
vl_grounding_jitter = [384, 768]   # agora POR STEP
caption_dropout = 0.05
clause_dropout_prob = 0.40
clause_dropout_rate = 0.60
keep_clauses = 1

[adapter]
type = 'lora'
rank = 64
alpha = 64
dtype = 'bfloat16'

[optimizer]
type = 'AdamW8bitKahan'
lr = 1e-4
betas = [0.9, 0.99]
weight_decay = 0.01
```

**Esforço total: ~7 h de código + 1 h 30 de smoke.** Ordem obrigatória: 2.1 → 2.2 → 2.4 → 2.3 → 2.5 → (2.6 opcional). O 2.3 é o único com risco real de dia perdido.

**Orçamento de VRAM a 512px (a conferir no smoke):** DiT fp8 12,5 GB + TE bf16 8,5 GB + LoRA/Adam8bit/Kahan ~0,7 GB + ativações c/ checkpointing ~3 GB + ViT do Qwen3-VL @768 ~1 GB ≈ **26 GB**. Cabe. A 768px sobe ~3 GB → use `blocks_to_swap = 8..16`. Contingência se apertar: castar o TE para fp8_e4m3 offline (−4,3 GB).

---

# 3. Compat de inferência — confirmação item a item

Alvo: `LoraLoaderModelOnly` padrão **+** `comfyui-krea2edit` **v1.2.5** (é a versão instalada em `/workspace/comfy/ComfyUI/custom_nodes/comfyui-krea2edit/pyproject.toml:4`).

| Requisito | Garantido? | Prova |
|---|---|---|
| Formato de chave | ✅ | Saver produz `diffusion_model.<módulo>.lora_A/B.weight` (`utils/saver.py:74` + `krea2_reference.py:202`). Confirmei num checkpoint real do conradlocke: 512 chaves, exatamente esse formato, sem `alpha` |
| Fusão no loader padrão | ✅ | `comfy/lora.py:191-196` cria um mapa genérico para **todo** `diffusion_model.*.weight` do state_dict; `comfy/weight_adapter/lora.py:162,179` aceita `lora_A/lora_B` |
| `txtmlp.1/3` fundem? | ✅ | Existem em `comfy/ldm/krea2/model.py:264-269` (`Sequential(RMSNorm, Linear, GELU, Linear)`) ⇒ entram no mapa genérico. E o node chama `m.txtmlp(context)` (`__init__.py:220`) **no modelo patcheado** |
| `txtfusion.projector` | ⚠️ só se salvo como `.diff` | `comfy/ldm/krea2/model.py:149` existe; `comfy/lora.py:72-76` só consome `.diff`. Sem o plumbing do §2.6, **falha em silêncio** |
| Geometria: frame index | ✅ idêntica | node `_imgids_offset(..., frame=i+1)` (`__init__.py:38-50`); apex `pos[...,0]=1.0` |
| Geometria: offsets fracionários | ✅ idêntica | node `off_h,off_w = max(0,(th-gh)/2)` float (`__init__.py:44`); apex idem |
| Geometria: fit da ref | ✅ idêntica em lógica | node `_fit_encode_image` fit branch (`__init__.py:88-126`) — CROP_TOL 0,08, `/16 floor` com cap, crop-to-grid. **Divergência residual mínima:** o node usa `bicubic+antialias`, o ai-toolkit usa `bilinear+antialias`. Especifiquei **bicubic** (casa com o node; o próprio conradlocke carrega essa inconsistência). Irrelevante em 100 % do ds4, onde o fit é no-op |
| Timestep | ✅ **só porque escolhemos `'target'`** | node: `t = m.tmlp(timestep_embedding(timesteps,...)); tvec = m.tproj(t)` — **um** tvec para toda a sequência (`__init__.py:215-216`). Um LoRA treinado com `reference_timestep='zero'` roda ali sem erro e **degrada em silêncio** |
| Ordem da sequência | ✅ irrelevante | node `[texto\|refs\|alvo]`, fork `[texto\|alvo\|ref]`. Provado equivalente |
| Grounding (encode) | ✅ idêntico | `Krea2EditGroundedEncode._template` (`__init__.py:411-416`) ≡ `KREA2_TEMPLATE` (`comfy/text_encoders/krea2.py:20`) + blocos de visão antes do prompt = nosso `vl_prompt_style='plain'`. **E o fork usa literalmente o mesmo código de TE do ComfyUI** (`comfy.sd.load_clip`, `models/base.py:525`) — o strip do prefixo é o mesmo (`comfy/text_encoders/krea2.py:43-62`). É uma vantagem estrutural do fork sobre o ai-toolkit, que usa um slice fixo `START_IDX=34` |
| Padding de texto | ✅ | batch 1 ⇒ nenhum padding; máscara toda 1s, como o node (`mask=None`) |
| `ref_boost` | ✅ funciona | `_ref_attn_bias` (`__init__.py:139-168`) soma `log(b)` aos logits alvo→ref. É exatamente o **gate de frame** exercido em inferência: `log 1,5 = 0,405 nats`, `log 2,5 = 0,916 nats` — dentro do teto treinável de ~1,975 nats |

**Nada diverge, com três condições que a spec já impõe:** `reference_timestep='target'`,
fit bicubic node-matched, e `.diff` para o projector (ou não treinar o projector).

**Metadata obrigatório** no `.safetensors` (`get_reference_metadata`):

```
geometry_contract       = krea2_frame_fit_v1
ref_axis                = frame ; ref_frame_index = 1
ref_offsets             = centered_fractional
ref_fit                 = ar_preserve_crop_to_grid_16_bicubic
px_per_token            = 16
reference_model_timestep= target
node_compat             = comfyui-krea2edit>=1.2.4 ; fit_mode=fit ; grounding_px=768
grounding               = qwen3vl_native_pre_fit_longest768_jitter384_perstep
timestep_law            = krea2_apex m=0.337+0.335*ln(N/1024) s=1.538+0.130*ln(N/1024)
caption_regime          = 0.55 full | 0.40 clause_dropout(0.60,keep1) | 0.05 empty
lora_targets            = blocks+txtfusion+txtmlp
```

---

# 4. Plano de validação

## 4.1 Smoke (10 steps) — **30 min, obrigatório**

Valida em uma passada: modelo carrega, dataset carrega, **VRAM real**, checkpoint escrito e válido, log, **resume**, e as 6 asserções específicas do apex:

1. `configure_adapter` imprime **258** alvos (224 + 32 + 2) e o audit passa (0 chaves fora de `.blocks./.txtfusion./.txtmlp.`).
2. Nenhum arquivo em `cache/*/text_embeddings_*` foi criado; disco não cresce.
3. Log do jitter: 10 steps ⇒ **10 valores distintos** de longest-side em [384,768] para a **mesma** imagem (repetir a mesma amostra). Se sair constante, o §2.3(e) não pegou.
4. `print` das posições da ref no step 1: `pos[...,0] == 1.0`, `off_h/off_w` fracionários quando o gap é ímpar.
5. Nenhum `RuntimeError: Inference tensors...` (armadilha §2.3d).
6. 5 valores de `t` amostrados batem com LN(0,337; 1,538) e são de baixa discrepância (não iid).

## 4.2 Pré-filtro grátis, sem treino — **20 min GPU**

Antes de gastar 1 h em qualquer braço, duas sondas no base sem LoRA:

- **Sonda de t da ref:** massa de atenção do alvo sobre as keys da ref, por bloco, em
  t ∈ {0,97; 0,85; 0,6; 0,3; 0,1}, sob `reference_timestep` zero vs target. Se as curvas
  coincidirem, o eixo #7 morre e a decisão por compat fica sem custo. Se sob `target` a
  massa colapsar em t alto, o custo da escolha por compat está quantificado.
- **Sonda ‖Δv‖(t):** não aplicável sem LoRA; adiar para 4.4.

## 4.3 Probe no ds4 — **500 / 1000 / 1500 / 2000 steps, ~65 min GPU**

Dataset: `konrad_ds4` puro (1.247 pares, `num_repeats = 1`), 512px AR buckets, seed fixa.
2.000 steps = **1,6 época** — dentro da janela 0,8–1,0 época onde **todos** os três
resultados aprovados viveram. `save_every_n_steps = 250`.

Isso é o *replay* do probe vencedor com o contrato trocado — o A/B mais informativo que
existe, e o único que não é confundido pela heterogeneidade das 4 fontes.

## 4.4 Protocolo de avaliação — **não é opcional**

Nos nodes do usuário, **v1.2.5, `fit_mode: "fit"`, `grounding_px: 768`**, 11 pares,
seed 76, mesma resolução de saída para todas as colunas.

| coluna | o que gera | mede |
|---|---|---|
| **A** | ref verdadeira | o resultado |
| **B** | ref **embaralhada** (par errado) | `S_ref = RMS(A−B)` — sensibilidade à referência |
| **C** | **span da ref removido** da sequência (não zerado) | é a ref decorativa? |
| **D** | ref verdadeira, **seed 77** | o nulo `S_seed` |

> Coluna C: removendo o span você compara contra o **T2I base**, que é in-distribution.
> Zerar o latente (o que `--reference-guidance` do fork faz, `tools/infer_reference_adapter.py:700-703`)
> mede extrapolação para um ponto que nada no treino produz. É por isso que o
> ref-dropout foi rebaixado a A/B (§5c) em vez de virar requisito.

**Varredura `ref_boost` ∈ {1,0 / 1,5 / 2,5}** (`Krea2EditModelPatch`, `__init__.py:288`)
na coluna A. É o único dial que fala com a referência e ele nunca saiu de 1,0 em nenhum braço.

**Critérios de decisão:**

1. `RMS(A−B) ≈ RMS(A−D)` ⇒ trocar a ref ≡ trocar a seed. **O modelo não lê a ref.** Nem
   geometria nem schedule salvam: a moeda seguinte vai inteira para a fase 2 de captions (§1 item 24).
2. **C tão bom quanto A** ⇒ mesma conclusão, mais forte.
3. `RMS(A−B) ≫ RMS(A−D)` **e** a imagem boa ⇒ apex aprovado, escalar para o run completo.
4. `RMS(A−B)` grande **e** a imagem degradada ⇒ o polo "segue-mas-degrada" sobreviveu ao
   fim do routing ⇒ o problema é cobertura de t / ref_boost, não classe de função. Varrer `ref_boost` para baixo.
5. **Nunca comparar loss entre braços.** Com o schedule apex a loss é comparável *entre
   braços apex* (peso ≡ 1) mas não contra os runs `flux_shift`.

## 4.5 Run completo (só depois de 4.4 passar) — **~3 h 20 GPU**

`konrad` (10.991) + `konrad_ds4` (`num_repeats = 3`) = 14.732/época. 6.000 steps ≈ 0,41
época global, mas **1.525 exposições do ds4** — dentro da janela útil. Avaliar em
1.500 / 3.000 / 4.500 / 6.000. Cauda opcional 768px de ~600 steps com
`init_from_existing`. **Não vá a 15.000.**

---

# 5. Onde os derivadores discordaram — resolvido

**(a) `reference_timestep`: zero vs target.** Geometria prova vantagem estrutural do t=0
per-token (Teorema 2: sob t uniforme, RoPE é a *única* operação dependente de linha, e o
gate está limitado a ≤1,975 nats; sob per-token, a diferença `tvec(t)−tvec(0)` é O(1) em
unidades de feature). Objetivo/condicionamento tratam como aberto.
**Resolvo por compat, não por matemática: `'target'`.** O node calcula um tvec único
(`__init__.py:215-216`); um LoRA t=0 ali falha **em silêncio**, e "sem mismatch" é a única
evidência dura do corpus. A/B declarado: se o usuário decidir manter um node próprio,
o fork já implementa `reference_timestep='zero'` (`krea2_reference.py:47-49`) — 1 linha de
config, 3 h de GPU, avaliado pelas 4 colunas.

**(b) Captions.** Objetivo pede mistura mecânica; condicionamento pede remoção dirigida de
`C_shared`. **Não é conflito, é fase.** Fase 1 = mistura (grátis, entra agora). Fase 2 =
dirigida (exige captionar 12.238 sources — hoje 0 `.txt`). Custo da fase 2: Gemini se
houver chave, ou o Qwen3-VL-4B local com o mesmo system prompt (~5 h GPU, compete com o
treino). Rejeito o híbrido 50/50 descrição/instrução: sob disciplina de ~1 época ele paga
o custo inteiro do recaption e entrega metade das exposições por dialeto.

**(c) Ref-dropout 0,05.** Objetivo classifica como obrigatório antes de varrer
`reference_guidance`. **Resolvo contra, com argumento:** o node não tem caminho de ref
zerada; o dial dele é `ref_boost` (viés de logit). A coluna diagnóstica "sem ref" fica
melhor servida removendo o span. E 5 % dos updates é caro sob ~1 época. Fica como A/B,
default off. *Nota:* só com o TE ao vivo um ref-dropout **coerente** (zerar VAE **e** tirar
o bloco de visão) passa a ser possível — hoje `krea2_edit.py:157-168` proíbe o parcial, e está certo.

**(d) Batch.** `RELATORIO_MATEMATICA` Claim 5 sugere batch 4; objetivo mostra que o
confounder (10 % vs 80 % de época) domina. **Batch 1**, e a variância vem do item #10.

**(e) Rank.** Condicionamento mede rank efetivo ≤6 e sugere testar r=32. **r=64 no run
principal** (3 receitas aprovadas), r=32 como A/B posterior que libera ~1,2 GB para a cauda 768.

---

# 6. ADVERSÁRIO — os 5 modos de falha mais prováveis

### 1. O TE ao vivo estoura VRAM ou o step time, e o run morre de fome de épocas
+8,5 GB residentes e +0,6–0,9 s/step. Se o step for a 2,6 s, 8 h dão 11.000 steps em vez
de 15.000 — e a janela útil do ds4 (1.000 exposições) escorrega. Pior: a 768px pode OOM.
**Mitigação embutida:** o smoke (§4.1) mede VRAM e s/it **antes** de qualquer run longo;
o probe roda a 512px, onde a folga é ~6 GB; `blocks_to_swap` é o primeiro dial;
contingência de TE em fp8 (−4,3 GB) sem tocar no stack CUDA.
**Mitigação que a spec NÃO tem:** se o step passar de 2,5 s a 512px, a decisão certa é
abortar o encode ao vivo e voltar ao cache com **3 escalas** {384, 576, 768} pré-computadas —
73 GB para 12.455 pares em vez de 271 GB (cabe nos 275 G livres), com jitter reduzido a
3 pontos em vez de contínuo. Já sei que é o plano B; é 1 h de código a mais.

### 2. Trocar geometria + schedule + captions + infra de uma vez ⇒ nenhuma atribuição
Se o probe sair ruim, não saberei qual das quatro mudanças causou. É exatamente o erro
que produziu o v2 (5 variáveis simultâneas, forense de 8 h para não concluir nada).
**Mitigação parcial e honesta:** geometria e fit são **no-op provado em 100 % do ds4**
(dims A↔B idênticas em 1.247/1.247) — no probe do §4.3 elas literalmente não mudam um bit,
então o probe é um A/B de **schedule + captions + jitter** contra o probe vencedor
histórico. Não é single-variable, mas as variáveis vivas caem de 4 para 3, e as 4 colunas
separam "não lê a ref" (captions) de "lê e degrada" (schedule).
**Risco residual real:** o probe vencedor **não é reconstruível** — o ds4 foi recaptionado.
A comparação é contra memória visual, não contra um baseline pago. Se isso incomodar,
custa 1 h de GPU rodar um braço-controle com `timestep_law` desligado (`flux_shift=true`)
e captions sem dropout, no mesmo código, mesma seed.

### 3. Quebrar a suficiência do texto amplifica o polo errado
§3.6 do derivador de objetivo: em t>0,8 só k<1,29 está determinado — **tudo** vem do
condicionamento; e o schedule apex põe 24,8 % da massa lá. Se o clause dropout remover
cláusulas de **delta** (pose, ação, enquadramento novo — o que a ref *não pode* suprir), o
piso irredutível sobe e a saída fica genérica **justamente** na banda que decide composição.
**Mitigação embutida:** é por isso que `q·ρ_hi = 0,24` e não 0,50, `keep_tokens=1` protege
o tipo de plano (que é `C_delta` puro), e 55 % dos steps rodam a caption inteira.
**Mitigação que falta:** a fase 2 (remoção dirigida) é a correção real — remover só
`C_shared` sobe `H(B|C)` sem subir `H(B|A,C)`. Enquanto ela não existir, o dropout aleatório
está destruindo `C_delta` em ~metade das cláusulas que remove. **Se o probe piorar em
"segue a instrução", este é o suspeito nº1, e o teste é rodar o mesmo probe com
`clause_dropout_prob = 0`.**

### 4. A lei de timestep está calibrada num espectro medido em latentes esmagados
`P(k)=27,53·k^−2,15` veio de 300 latentes do cache CTX 512 **quadrado**, anisotropia 1,78.
Os buckets AR do `konrad` não são isso. A média radial é robusta a anisotropia, mas não é
idêntica. E a extrapolação para 768/1024 é lei de potência invariante de escala, **não
medida**. Se `A_N` escalar diferente, `m(N)` erra e a 768 eu subcubro ou supercubro o topo.
**Mitigação:** a 512px (o probe e o bulk do run) o número é medido, não extrapolado — o erro
mora só na cauda 768. E o gap entre a tabela do conradlocke e o ótimo medido é 0,004 nats,
ou seja, **o alvo é um platô largo**: errar `m` por ±0,15 custa quase nada.
**O que eu não sei defender:** que fechar o gap de schedule mude o veredito visual. O
`RELATORIO_ELEMENTO_DISTOANTE` põe captions em #1 e schedule em #2, e os dois estão
**acoplados**. Se só houvesse uma moeda, ela iria para captions.

### 5. `txtmlp` + `projector` derivam e envenenam o canal semântico
`txtmlp` acrescenta ~0,8 % de parâmetros mas é o funil **único** por onde os tokens de
visão entram na largura do DiT — se ele aprender um atalho ("ignore a visão, use o texto"),
piora exatamente o que queremos. O `projector` é pior: 12 números com alavancagem
desproporcional sobre *o que o grounding significa*, e gradiente forte.
**Mitigação embutida:** `projector` é P2, desligado por default, 1 linha para tirar;
`txtmlp` entra no probe e o A/B é o §6 item 3 do derivador de condicionamento (1 h GPU,
com vs sem). E o audit de chaves (`krea2_reference.py:246-273`) falha ruidosamente se o
escopo escorregar.
**Risco que sobra e não tem mitigação barata:** o rank efetivo ≤6 medido no step 1500 pode
significar que o LoRA está aprendendo um **shift de domínio global** em vez de um mecanismo
condicional — a assinatura exata de "decorou o look do dataset pela via do texto". Um único
checkpoint não separa isso de "LoRA jovem". O discriminante é repetir a medida de espectro
no step 6.000 do run apex: se o rank efetivo continuar ≤6 **e** `S_ref ≈ S_seed`, a conclusão
é que nenhuma configuração de treino resolve e o problema é o dado.

---

**Próximo passo concreto (10 min):** eu crio `models/krea2_apex.py` com o esqueleto do §2.1
(itens 1, 3, 4, 5, 6) e rodo `--cache_only` no ds4 para confirmar que os 258 alvos aparecem
e que o audit passa. Zero GPU de treino, zero risco. Confirma?