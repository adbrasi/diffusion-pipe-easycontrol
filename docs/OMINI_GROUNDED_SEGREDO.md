# Omini-Grounded — o segredo, explicado com o código real

**Data:** 2026-07-17 · **Branch:** `ic-lora` · **Commits-chave:** `bdf8d9f` (método),
`1b17e4d` (node), `61d8567` (dials separados) · **Adapter do probe:** HF
`AdwolfCzar/scene-continuity-r64/experiments/omini_grounded_step00250.safetensors`

Este documento explica POR QUE o `krea2_omini_grounded` funciona, com os trechos
reais do código. É a receita proprietária do projeto CONTEXTO: nenhum método
público combina estas duas peças (verificado em revisão de literatura — OminiControl
não tem grounding; Qwen-Image-Edit/Kontext não têm routing; a lacuna é nossa).

---

## 1. A ideia em uma frase

> Dois canais de referência com papéis separados: os **tokens VAE** carregam a
> aparência exata e recebem um LoRA **routado só para eles** (o resto do modelo
> fica congelado = zero drift); o **grounding Qwen3-VL** faz o caption resolver
> entidades contra a imagem ("a mesma garota" liga na garota REAL), adaptado por
> um LoRA global no TextFusion.

O omini puro dá fidelidade sem semântica. O edit dual dá semântica com drift.
O Omini-Grounded soma os dois sem os defeitos — porque cada canal tem o SEU
adapter, no SEU lugar.

## 2. A composição (models/krea2_omini_grounded.py)

O pipeline herda de `Krea2EditPipeline` (que já faz o grounding no cache de texto)
e adiciona o router do omini APENAS nos blocks:

```python
class Krea2OminiGroundedPipeline(Krea2EditPipeline):
    name = 'krea2_omini_grounded'
    config_section = 'krea2_omini_grounded'
    adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.')

    def __init__(self, config):
        super().__init__(config)
        section = config.get(self.config_section, {})
        self.condition_only_lora = bool(section.get('condition_only_lora', True))
        self.condition_lora_router = ConditionOnlyLoRARouter(self.condition_only_lora)

    def configure_adapter(self, adapter_config):
        # krea2_edit coverage: every linear in the SingleStreamBlocks plus the
        # TextFusionTransformer (projector excluded).
        super().configure_adapter(adapter_config)
        if self.condition_only_lora:
            # Route ONLY the block deltas to the reference span. The txtfusion
            # LoRA stays global: it runs on the text stream (different
            # sequence), where masking by image span would be meaningless.
            installed = self.condition_lora_router.install(self.diffusion_model.blocks)
```

**Detalhe crucial:** `router.install(self.diffusion_model.blocks)` — instalar no
`.blocks` (não no modelo inteiro) é o que deixa o txtfusion com PEFT padrão
(delta global no stream de texto) enquanto os 224 linears dos blocks ficam
routados. 512 tensores no save: 448 blocks (routados) + 64 txtfusion (globais).

## 3. O routing (models/condition_lora.py) — o coração do omini

O forward substituto de cada Linear PEFT computa base + delta, e **mascara o
delta às rows da referência**:

```python
def _condition_only_lora_forward(module, x, *args, **kwargs):
    result = module.base_layer(x, *args, **kwargs)      # base congelada, todas as rows
    mask = router.mask_for(result)                       # 1 só no span da referência
    ...
    delta = lora_b(lora_a(dropout(adapter_input))) * scaling
    result = result + delta.to(result.dtype) * mask      # delta SÓ nas rows da ref
    return result
```

Consequência: as rows do target e do texto passam pelo modelo **intocado** — todo
o conhecimento do base é preservado (zero drift de estilo/qualidade), e o adapter
só aprende "como a referência deve se apresentar" via K/V dos tokens dela.

O span é setado por bloco pelo `Krea2OminiTransformerLayer`:

```python
def forward(self, inputs):
    combined, target_timestep, tvec, freqs, attention_mask, sizes = inputs
    text_length, target_length = int(sizes[0]), int(sizes[1])
    self.router.set_reference_span(text_length + target_length, combined.shape[1])
    return super().forward(...)
```

## 4. A geometria e o timestep (models/krea2_reference.py)

Sequência `[texto | target ruidoso | ref limpa]` com:

- **width-shift** (convenção OminiControl para tarefas não-alinhadas): a referência
  vive "ao lado" do target no plano 2D do RoPE, frame axis 0:

```python
elif self.position_mode == 'width_shift':
    reference_pos[..., 2] = reference_pos[..., 2] + float(target_grid_w)
```

- **refs modulados a t=0** (per-token; texto+target no t amostrado):

```python
if self.reference_timestep_mode == 'target':
    reference_t = timesteps[:, None].expand(batch, reference_length)
else:
    reference_t = timesteps.new_zeros(batch, reference_length)   # <- usamos este
per_token_timestep = torch.cat([target_t, reference_t], dim=1)
```

⚠️ Lição histórica: um `to_layers()` que não repassava `reference_timestep_mode`
custou um dia de debugging (fix `a66d7ba`). O contrato REAL é o que o código
executa, não o que a config/metadata dizem.

## 5. O grounding (models/krea2_edit.py, herdado)

No cache de texto, cada caption é tokenizado JUNTO com a imagem de referência
(vision block plain, sem prefixo "Picture 1:", maior lado 768):

```python
if self.vl_prompt_style == 'plain':
    text = VISION_BLOCK * len(images) + caption   # <|vision_start|><|image_pad|><|vision_end|>
...
tokens = text_encoder.tokenize(text, images=images, llama_template=KREA2_TEMPLATE)
```

A torre visual do Qwen3-VL produz embeddings *contextualizados pela referência*
— e o TextFusion (treinável, LoRA global) aprende a colapsar esse stack de 12
camadas do jeito que o task precisa.

## 6. A config exata do probe validado

```toml
[model]
type = 'krea2_omini_grounded'
diffusion_model = '.../krea2_raw_fp8_scaled.safetensors'
flux_shift = true                     # mu oficial por resolução

[krea2_omini_grounded]
position_mode = 'width_shift'
reference_timestep = 'zero'
condition_dropout = 0.0               # (proibido no caminho edit; ver melhorias)
condition_only_lora = true
vl_prompt_style = 'plain'
vl_longest_side = 768

[adapter]
type = 'lora'
rank = 64                             # alpha=rank forçado -> escala 1.0

[optimizer]
type = 'AdamW8bitKahan'
lr = 1e-4
```

250 steps, batch efetivo 4, dataset 1255 pares 512px AR-buckets. Loss final
0,067 (a menor de todos os braços do projeto).

## 7. Inferência — node `CtxRush - Krea 2 Omini-Grounded (setup completo)`

All-in-one: constrói o conditioning grounded internamente (CLIP Text Encode comum
NÃO serve — não groundaria) e aplica o LoRA em **runtime bf16** (nunca fundir nos
pesos: merge em fp8-scaled requantiza `W+ΔW` e afoga o delta). Dois escopos:

```python
# txtfusion: delta GLOBAL, só durante a chamada do txtfusion
with _FullLoraScope(fusion_entries, fusion_scale):
    context = m.txtfusion(context, ...)

# blocks: delta MASCARADO ao span da referência, durante o loop de blocks
with _MaskedLoraScope(entries, txtlen + tgtlen, seq_len, seq_len, strength):
    for block in m.blocks:
        combined = block(combined, tvec, freqs, None, ...)
```

Desde `61d8567`, os dois têm dials independentes: **`block_strength`**
(fidelidade à referência) e **`fusion_strength`** (semântica do grounding;
0 = mede quanto vem do built-in do base). Fiação: model+clip+vae+image entram no
node; saem model/positive/negative/latent/steps/cfg para o KSampler.

A referência SEMPRE entra como imagem (crop-fit em pixel + encode nativo) —
redimensionar latente lava o sinal.

## 8. Por que funciona (validado empiricamente)

- Held-out real (aquarela fora do dataset): obedece o prompt novo carregando
  personagem, paleta, estilo E elementos específicos do cenário que o caption
  não menciona (grade, narcisos, bolsa) — grounding e routing somando.
- Forense de pesos: os deltas routados treinam à mesma magnitude do omini puro
  (95%) — o grounding NÃO rouba capacidade; ele muda a *direção* do aprendizado
  (os blocks aprendem o resíduo sobre o que o grounding já entrega).
- Zero drift: rows do target intocadas pelo adapter.

## 9. Melhorias mapeadas para o treino longo

1. `caption_dropout = 0.1` (treinar o uncond do CFG — o probe rodou sem, e a
   resposta ao strength fica amortecida por isso).
2. Jitter de grounding 384–768 no treino (dial de inferência robusto).
3. Se `fusion_strength=0` mostrar txtfusion ocioso → remover txtfusion.
4. Se competição funcional se confirmar → dropout assimétrico (Composer-style,
   maior no canal grounded).
5. Milhares de steps: o rank efetivo colapsou a ~1/64 em 250 steps — há
   capacidade ociosa de sobra.
