# Ideogram 4 × Contexto Rush — debate de métodos (Claude ⇄ Codex gpt-5.6-sol xhigh)

**Data:** 2026-07-19 · **Branch:** `ic-lora` · 3 rodadas (plano → teoria → réplica com evidências).
Objetivo: decidir COM EVIDÊNCIA como ensinar o Ideogram 4 (DiT single-stream 34L, 9,3B,
TE Qwen3-VL-8B 13 camadas concatenadas, VAE Flux.2 32ch×patch2=128ch/token, compressão 16,
flow matching, t_model = 1−σ) a gerar "próxima cena" condicionada por referência.

## Fatos estabelecidos (verificados no código/checkpoint)

1. **Não existe slot nativo de referência.** `embed_image_indicator.weight = [2, 4608]`
   (header do fp8_scaled lido byte a byte). O código embute o booleano
   `(indicator == OUTPUT_IMAGE)`; alvo E referência compartilham o índice 1; o
   "indicator 4" é rótulo lógico (span/máscara/timestep/metadata), nunca índice de
   embedding. Confirmado no oficial (ideogram-oss/ideogram4) e no fork BitPoet.
   ⇒ O sucesso do BitPoet prova que o DiT **aprende** o contrato `[texto | alvo ruidoso |
   ref limpa]`; a distinção ref/alvo vem de: timestep por token (ref pinada em 1.0 via
   AdaLN), offset temporal do MRoPE, e (quando usado) routing.
2. **fp8 do trainer é cast cru.** `base.py`: dequantiza o QuantizedTensor (com escala) e
   faz `.to(float8_e4m3fn)` SEM rescale — segunda quantização destrutiva. **VETADO** para
   os pilotos; só reconsiderar mantendo QuantizedTensor+escalas ou requantizando com escala.
3. **Captions em prosa são suficientes.** Template do TE = chat Qwen simples
   (`<|im_start|>user\n{caption}<|im_end|>...`, no-think). O prior do checkpoint foi
   treinado com captions JSON (doc oficial), mas prosa funciona; JSON vs prosa = ablação
   futura de teto de qualidade, nunca antes de escolher arquitetura.
4. **A torre visual já existe no stack.** `Ideogram4Qwen3VLClipModel` (ComfyUI) processa
   `images=[...]` no tokenizer (placeholders, MRoPE visual, DeepStack). O cache guarda os
   13 taps crus; `llm_cond_proj` roda DEPOIS, dentro do DiT ⇒ LoRA nele recebe gradiente
   mesmo com TE cacheado. Porte do grounding é barato.
5. **MRoPE interleaved:** H e W ocupam 20 pares de frequências cada; os demais 88/128
   pares usam a coordenada temporal ⇒ offset temporal ±1 é um separador FORTE (não
   detalhe). +1 e −1 não são equivalentes (fase de sinal oposto); +1 tem precedente
   empírico (BitPoet); −1 não tem prior de vídeo num modelo T2I. Offset 0 (posições
   compartilhadas) = prior de cópia pixel-aligned — errado para próxima cena.
6. **Omini2 (atenção assimétrica + stride 2) é otimização de inferência, não arquitetura
   de aprendizado** para esta tarefa: ref não contextualizada pelo texto perde seleção de
   entidades ("a mesma garota", "o mesmo casaco"); stride 2 perde rosto/cabelo/figurino em
   anime. Meio-termo futuro: "ilha de conditioning" (texto⇄ref bidirecional, ambos cegos
   ao alvo ruidoso — preserva KV reuse).
7. **AdaLN:** excluir globalmente protege mas pode castrar (com per-token t, AdaLN é o
   pathway que prepara o "regime encoder" da ref em t=1.0). BitPoet treinou tudo e
   funcionou (7k steps). Decisão: **treinado nos pilotos** (P1 global, P2 routado por
   row). Meio-termo documentado p/ refinamento: adaln routado ref-only rank 8-16,
   lr 0,1-0,25×.
8. **Routing condition-only** (y = W₀x + M_ref·ΔWx) = "encoder de referência aprendido +
   decoder congelado". Nota: routing zera o delta nas rows do alvo, mas NÃO o efeito da
   presença dos tokens de ref na atenção base ⇒ scale-0 no packing ≠ modelo base.
   E o vencedor do Krea tinha routing + txtfusion global; o análogo Ideogram do canal
   global é LoRA em `llm_cond_proj` (P3).

## Plano final de pilotos (endossado pelo Codex, vetos incorporados)

Comum: bf16 em tudo (`diffusion_model_dtype='bfloat16'`), rank 64/64, lr 5e-5
AdamW8bitKahan, warmup 50, batch 1×ga4, shift 3, logit_normal, 512px AR buckets,
`condition_dropout 0.1`, offset +1, ref full-res, atenção simétrica, saves a cada 100,
600 steps (~2 épocas), MESMA seed/ordem/buckets, blocks_to_swap 8 (subir só com OOM).
Orçamento realista: 60-90 min/piloto (medir s/step pós-warmup antes de prometer).

| Piloto | Config | Isola |
|---|---|---|
| P1 | `ideogram4_ic_lora` global, `train_adaln_modulation=true` (configs/ideogram4_ctxrush_m1.toml) | Existe sinal aprendível? Packing funciona? (teto de capacidade, paridade BitPoet) |
| P2 | idêntico + `ideogram4_ominicontrol` subject/routing (configs/ideogram4_ctxrush_m2.toml) | Efeito causal do routing (única variável) |
| P3 | P2 + ref na torre Qwen3-VL + LoRA global rank 16 em `llm_cond_proj` | Efeito causal do canal semântico |

P3 — requisitos mandatórios antes de treinar:
- TE detectado como QWEN3VL_8B (pesos `visual.*` presentes), nunca text-only;
- encoder de cache com assinatura de 3 args (utils/dataset.py passa `control_file`);
- `images=[ref]` batch 1 BHWC [0,1] (nunca BCHW [-1,1], nunca batch inteiro);
- attention_mask com comprimento do embedding expandido por tokens visuais;
- dropout Bernoulli ACOPLADO VAE+VL (cache das duas variantes do TE, seleção conjunta);
- uncond grounded: remove caption, MANTÉM referência nos dois canais;
- `llm_cond_proj` incluído nominalmente nos targets + auditoria ajustada (hoje rejeita
  chaves fora de `.layers.`);
- paridade cache↔inferência: mesma caption+refs diferentes ⇒ embeddings diferentes;
  reencode ≈ cache; gradiente não-zero na LoRA do llm_cond_proj; sem ref repetida no batch.

## Protocolo de julgamento (por checkpoint 100/300/600)

Matriz com MESMO ruído/caption/seed/ordem, 8 pares held-out × 2 seeds, guidance 1,
resolução EXATA do bucket de treino (nunca o default 1024!):
1. **base-nativo** (packing T2I stock, sem tokens de ref) ← controle obrigatório;
2. {ref correta, ref embaralhada, sem ref} × adapter_scale {0, 1};
3. em P3, "sem ref" = a representação de dropout treinada, não improviso.

Tabela de sintomas (resumo): correta≈embaralhada ⇒ ref ignorada (shortcut/routing);
muda certo ao trocar ref ⇒ aprendeu (mesmo feio); copia ref e piora 250→500 ⇒ gaps
temporais curtos/overfit; base stock ok mas packed scale-0 degrada ⇒ shift de
distribuição do packing; tudo ruim até base stock ⇒ ferramenta/sampler/dtype — NÃO
julgar o método; 250 > 500 com degradação ∝ scale ⇒ LR/overfit.

Inferência: sampler de deployment = ComfyUI Ideogram (scheduler logit-normal dependente
de resolução) ≠ diffusers linear+shift — reproduzir a versão exata ao julgar. Fallbacks
de geometria (offset −1/0, stride 2, máscara assimétrica) só na rodada seguinte.

## Vetos do Codex (aceitos)
1. fp8 pelo cast cru atual — nunca.
2. P3 sem teste de paridade cache/inferência — nunca.
3. Avaliação sem o controle base-nativo — nunca.

## Adendo empírico (2026-07-19, pós-P1 rodada 1) — o harness estava quebrado

O primeiro julgamento do P1 deu FERRAMENTA-QUEBRADA e a investigação provou três
fatos que INVALIDAVAM qualquer avaliação anterior:

1. **CFG do Ideogram 4 é dual-model**: o deployment oficial (template ComfyUI)
   usa um transformer INCONDICIONAL dedicado (`ideogram4_unconditional_fp8_scaled`,
   9,3 GB) via DualModelGuider, cfg 7 com override →3 no fim, schedule
   `Ideogram4Scheduler` logit-normal (Default: 20 steps, mu 0.5, std 1.75;
   Quality: 48/0.0/1.5). Sampling conditional-only em guidance 1 com Euler
   linear+shift 3 produz papa cinza — não é falha de método, é contrato errado.
2. **O base COLAPSA com captions em prosa**: prosa → cinza chapado (σ≈11);
   o MESMO conteúdo em JSON estruturado → cena de anime correta e aderente.
   A nota oficial confirma: treinado exclusivamente com captions JSON; prosa
   também dispara mais o safety filter embutido (overlay "Image blocked...").
   ⇒ A recomendação da rodada B ("manter prosa") caiu por evidência: as 1255
   captions foram convertidas para o schema JSON oficial
   (high_level_description + style_description + compositional_deconstruction,
   SEM bbox) via gemini-3.1-flash-lite; originais em input_B_prose_backup/.
3. **Ferramentas novas**: tools/infer_reference_batch.py e tools/infer_base_native.py
   (carga única, 2 fases TE→DiT p/ caber em 29 GiB RAM/32 GiB VRAM, schedule
   oficial, CFG dual estagiado 7→3@70%, uncond image-only = referência só no
   positivo, paridade BitPoet); tools/ideogram4_uncond.py (uncond via loader
   fp8-scaled do ComfyUI, sem o recast destrutivo).

Pilotos re-executados com captions JSON + harness corrigido a partir daqui.

## Resultado P3 (2026-07-19, 600 steps, captions completas, dropout acoplado 0.1)

O grounding INVERTEU o modo de falha — evidência causal forte + colapso de reconstrução:

- **Causalidade PROVADA**: com prompt mínimo ("the same woman", zero descrição) o P3
  gera a identidade da referência (elfa OOD: rosto, tatuagem, armadura) — P1/P2
  falharam 100% nisso. Ref embaralhada → segue a nova ref. A resolução de "the
  same X" via torre visual funciona como teorizado.
- **Colapso de reconstrução**: TODAS as saídas ≈ a própria referência "recozida"
  (enquadramento/pose/cena), instrução nova ignorada. Dataset de frames adjacentes
  (alvo≈ref) + tokens visuais descrevendo a ref = reconstruir é o mínimo da MSE.
- **CFG triplo NÃO cura** (sR=5, sT=9): vRT≈vR — o adapter produz a mesma velocity
  com ou sem caption; a cópia está nos pesos, não no guidance. Ferramenta
  implementada em tools/infer_reference_batch.py (--triple-cfg --sr --st) e útil
  para o modelo final, mas não resgata um adapter colapsado.

Leitura de arquitetura: P1/P2 = obedecem texto, ignoram ref (atalho de caption);
P3 = obedece ref, ignora texto (atalho de reconstrução). Os dois atalhos são
propriedades do DADO+objetivo, não dos módulos: captions exaustivas alimentam o
primeiro; pares sem delta alimentam o segundo. A receita de escala (delta-captions
+ modos 70/15/15 + gaps temporais) ataca exatamente os dois. Próximos braços do
fatorial: E2b = routing + caption_dropout 0.25 SEM grounding (referência
necessária sem tokens visuais "copie-me"); depois P3 + delta-captions.

## Encerramento da fase (2026-07-19) — estado FIRA e handoff

Sweep sR×sT completo (24 células): NENHUMA combinação escapa da reconstrução no
P3 — só migalhas de obediência (gaivota pousada, esporos) vazam em sR baixo.
Confirmação final: o colapso está nos pesos, não no guidance.

FIRA v1 implementada e commitada (role_routed_lora.py + ideogram4_identity_nextscene.py
+ configs + dataset fatorado COMPLETO em /workspace/datasets/contexto_rush_fira:
1255 entity crops [1140 bbox gemini + 115 center-crop] + 1255 delta-captions R1
validadas + 1255 change masks; layout de treino em targets/ + change_masks_floored/).

GATE DA CÁPSULA REPROVOU (o preflight do Codex funcionou): capsule = últimos 8
estados do encode identidade → cos(mesma ref, captions≠)=0.9961 OK, mas
cos(refs≠)=0.9453 — o rabo causal é dominado pelo texto do TEMPLATE, não pela
imagem. A sonda de 300 steps NÃO foi treinada (teria sido inválida).

FIX DESENHADO (próxima sessão): extrair a cápsula das ROWS DE VISÃO (localizar o
span do <|image_pad|> expandido no token stream: índice da entry-imagem no
tokenize + n_vision = L_com_img − (T_tokens−1); pooling adaptativo em 4 grupos)
+ 4 âncoras do rabo = 8 tokens. Rows de visão de refs distintas divergem por
construção. Re-rodar o preflight (mesmos thresholds) antes de qualquer cache.

Prioridade de GPU transferida para o treino LTX smooth_rush por ordem do usuário.
