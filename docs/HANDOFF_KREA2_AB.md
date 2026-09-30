# Handoff — Krea 2 A/B (braço A nativo × braço B beta1 corrigido)

**Quando começar:** só depois de fechar o Anima (registre o estado final no
`NEXTSCENE_RUN_LOG.md` e suba os adapters para o HF). Mesma máquina (RTX 5090 32 GB,
~US$0,62/h, disco ~180 GB) e os mesmos padrões do `HANDOFF_AGENTE_GPU.md`:
- smoke antes de tudo e probes curtos;
- upload contínuo para um repo **privado** no HF (sugestão: `AdwolfCzar/krea2-ab-runs`),
  com model card curto;
- log em `docs/KREA2_AB_RUN_LOG.md`, com commit e push na branch
  `claude/elegant-ptolemy-8kywqy`;
- autonomia para corrigir bugs; se a evidência contradisser este plano, registre o porquê.

## 0. Leia primeiro

1. `docs/KREA2_ANALISE_METODO_2026-09.md` — o diagnóstico, **obrigatório**.
2. `models/krea2_native.py` (braço A), `models/krea2_multiref.py` + `krea2_edit.py` +
   `krea2_reference.py` (braço B = o pipeline do beta1).
3. `examples/krea2_ab/*.toml`, `tools/krea2_native_parity.py`, `tools/k2ab_metrics.py`,
   `test/test_krea2_native.py`.
4. Contexto histórico: `docs/AUDITORIA_2026-08-06.md`, `docs/KREA2_EDIT_SAGA.md` §4 e §6,
   `docs/OMINI_GROUNDED_SEGREDO.md`, `docs/KREA2_MULTIREF_RECEITA.md`.

## 1. O que está sendo decidido

O usuário quer a solução mais elegante: **um LoRA de próxima cena que rode no ComfyUI
normal, sem o patch `ctxrush-edit`**, preservando personagem, estilo e ambiente com a
fidelidade do beta1 (`AdwolfCzar/k2-context-rush-ofc-beta1`), mas sem "devolver a mesma
imagem" e sem degradar a qualidade.

| braço | contrato | inferência |
|---|---|---|
| **A — nativo** | `krea2_native`: ref no frame 1 (grid próprio a partir de 0), t=0, ~1 MP, texto do `TextEncodeQwenImageEditPlus`, LoRA global (blocks + txtfusion) | ComfyUI **stock**: `LoraLoaderModelOnly` + `TextEncodeQwenImageEditPlus` + `FluxKontextMultiReferenceLatentMethod('index_timestep_zero')` |
| **B — beta1 corrigido** | `krea2_multiref_grounded` como o beta1 (routing, width_shift, t=0, grounding 384² "image 1:", txtfusion r128), com base bf16 exata, `caption_dropout` 0 e dados limpos | runner `tools/infer_reference_adapter.py` (ou node CtxRush v2 **sem** `K2 Training Base`) |
| C — reserva | A + deslocamento da ref por hook `post_input` (~15 linhas, sem trocar o forward) | só se A copiar |

Tudo igual entre A e B: pares, legendas, 1024 px, 500 steps × 4 amostras, lr 1e-4,
AdamW8bitKahan, rank 64, `flux_shift`, base bf16, seed. Os 500 steps são o pedido do
usuário para o primeiro olhar, com checkpoints em 250 e 500.

## 2. Setup

```bash
git pull   # branch claude/elegant-ptolemy-8kywqy
# Modelos (Comfy-Org/Krea-2): base bf16 EXATA — nunca diffusion_model_dtype='float8'
huggingface-cli download Comfy-Org/Krea-2 diffusion_models/krea2_raw_bf16.safetensors \
  text_encoders/qwen3vl_4b_bf16.safetensors vae/qwen_image_vae.safetensors \
  loras/krea2_turbo_lora_rank_64_bf16.safetensors --local-dir /workspace/models/krea2
```
- Confira os nomes exatos no repo; ajuste os `.toml` se diferirem.
- **ComfyUI stock para avaliar o braço A e medir paridade:** `git clone` do ComfyUI oficial
  em `/workspace/ComfyUI_stock` (≥ `c9602625`; testado em `fb2315f1`), em venv separado, com
  `comfy-kitchen` instalado. É o "ComfyUI normal" do usuário. Nenhum custom node.
- Blueprint oficial de referência para montar o workflow de avaliação do A:
  `blueprints/Image Style Reference (Krea-2 Turbo).json` dentro desse ComfyUI.

## 3. Portões de paridade (antes de cachear qualquer coisa)

1. **Forward** (CPU, segundos):
   `KREA2_STOCK_COMFY=/workspace/ComfyUI_stock python -m pytest -q test/test_krea2_native.py`
   precisa passar os 2 testes de paridade (relL2 ~1e-7 aqui; o controle negativo com
   geometria errada dá ~3e-3).
2. **Encoder de texto** (GPU, pesos reais, 2–3 refs em PNG):
   `tools/krea2_native_parity.py te-stock ...` seguido de `te-fork ...`.
   - Se falhar (esperado: o `submodules/ComfyUI` do fork é de 23/06 e o Qwen3-VL ganhou
     DeepStack/MRoPE depois), atualize o submodule **num worktree separado para o braço A**
     até ficar igual ao ComfyUI stock, e rode de novo os dois portões e os testes do fork.
   - O braço B continua no ComfyUI fixado: é o contrato dele (o node CtxRush imita esse
     encoder).
   - Não compare A e B misturando caches de encoders diferentes.
3. **LoRA carregável:** depois do smoke de 10 steps do A, carregue o `adapter_model.safetensors`
   no ComfyUI stock com `LoraLoaderModelOnly` e confirme **0** "lora key not loaded" no log.

## 4. Dados

- **Os mesmos pares filtrados do Anima** (proxima_cena, `nextscene_pairs.py`, sem
  quase duplicados) e **o mesmo held-out**. Sem pico-banana.
- **Disco:** o cache de texto do Krea 2 custa ~24,5 MB por amostra, e A e B têm caches
  diferentes. Para o probe use um subconjunto de **~1.500 pares** estratificado por subset,
  **uma legenda por par** (tier completo ou curto sorteado com seed fixa, igual nos dois
  braços). São ~37 GB por braço.
  - 500 steps × 4 = 2.000 amostras, ~1,3 época.
  - Apague o cache do braço que perder.
- Layout: `/workspace/k2ab/data/{target,control}` (hardlinks) + `/workspace/k2ab/data/refs/<stem>_1.<ext>`
  (symlinks para o B, cujo loader multi_ref exige o sufixo).
- Registre a proporção SFW/R18 do subconjunto. No beta1, 60% de NSFW virou viés.

## 5. Treino

```bash
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config examples/krea2_ab/A_native.toml
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config examples/krea2_ab/B_beta1_fixed.toml
```
- Smoke de 10 steps em cada braço: s/step, VRAM, audit das chaves, resume.
- `blocks_to_swap` é o primeiro dial; bf16 exato ocupa ~26 GB só de pesos.
  **Não** volte ao float8 para ganhar velocidade: é o bug que causou a degradação.
- Se o bf16 + swap ficar lento demais: avalie manter o `fp8_scaled` **com a escala** (sem
  passar pelo `dequantize()` → `.to(float8)` do `models/base.py`). Isso exige provar que o
  gradiente da LoRA bate com o do bf16 num bloco antes de usar.
- Parar com `touch <run_dir>/save_quit`, nunca SIGTERM. Conferir o LR no log em qualquer resume.

## 6. Avaliação (a decisão)

- 12–16 pares held-out, 1024 px, mesma seed, em **Turbo** (8 passos, CFG 1, é o uso real
  do usuário) e **Raw** (~28 passos, CFG ~4–5,5).
- **Braço A:** renderizar no ComfyUI stock via API. Workflow: blueprint + `LoraLoaderModelOnly`;
  para Raw, o negativo é `TextEncodeQwenImageEditPlus` com prompt vazio + a mesma imagem.
  - Sem `FluxKontextMultiReferenceLatentMethod`, a ref é **ignorada em silêncio**.
  - Faça um smoke de 1 imagem com e sem o node para provar que ele está ativo.
- **Braço B:** runner `tools/infer_reference_adapter.py` (turbo fundida como no runner), sem
  `K2 Training Base`.
- Por checkpoint e condição: `<stem>_true.png` e `<stem>_shuffled.png` (ref de outro par,
  mesmo prompt/seed). Depois:
  `python tools/k2ab_metrics.py --pairs /workspace/k2ab/heldout --dir <pasta> [...]`.
- **Critério:** obedecer à próxima cena (pose/câmera/ação nova) mantendo personagem, estilo e
  ambiente, **sem ser cópia de A**; o shuffle precisa mudar a saída na direção da ref trocada.
  - Métricas para triagem: `ref_gain` > 0, `copy_rate`/`copy_gap` baixos, CCIP.
  - Veredito visual pelo grid.
  - Compare também com o beta1 original (via node CtxRush + Training Base, como o usuário
    usa hoje) nos mesmos held-out: é a régua de fidelidade.
- Qualidade: compare A/B também com o Krea 2 base em T2I puro no mesmo prompt. O LoRA não pode
  degradar pele, traço e detalhe. Essa era a queixa nº 2.

**Leitura esperada:**
- A ≈ B em fidelidade, sem cópia → A vence (elegância).
- B muito mais fiel → fica o B corrigido; o node custom só aplica o LoRA mascarado e as
  posições, sem mexer em fp8.
- **A copia** (saída ≈ A, `copy_rate` alto) → **braço C**: hook `post_input` que soma
  `ref_grid_w` (ou frame 5–10) nos `img_ids` da ref, treinado com a mesma mudança no
  `Krea2ReferenceInitialLayer`. Núcleo RoPE: frame 1 = 0,987; frame 5 = 0,911; width_shift
  512 = 0,806.
- A fraco em identidade → antes de concluir, teste mais steps (1.000–1.500), porque o
  grounding do Krea costuma fixar identidade cedo. E o A/B `vl_grounding` true × false.

## 7. Depois (não agora)

- **Duas referências** (o usuário perguntou): o nativo suporta até 3
  (`image1..3` → frames 1..3, "Picture N:"). É o próximo experimento depois que houver um
  vencedor com 1 ref. Exige estender `krea2_native` para N refs e o `Krea2ReferenceInitialLayer`
  para frames 1..N, com paridade contra o stock, que já faz isso.
- Tiers de legenda com a curta 2×, lei de timestep do apex + 15% em σ∈[0,8; 1], acabamento de resolução.
- **Resgate do beta1 sem treino:** `block_strength` 0,6–0,8 / força do `K2ReferenceGuider`
  0,6–0,9, em uma bateria curta com o usuário.
