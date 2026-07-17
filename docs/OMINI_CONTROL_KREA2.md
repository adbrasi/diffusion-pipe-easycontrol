# OminiControl para Krea 2 — receita funcional, saga e commits

**Data:** 2026-07-17 · **Branch:** `ic-lora` · **Commit de referência (use este): `1b17e4d`**

Este documento registra o caminho até o OminiControl funcional no Krea 2 (o melhor
método validado do projeto CONTEXTO até esta data), os commits exatos de treino e
inferência, e como reutilizar tudo em outras sessões/máquinas.

---

## 1. A saga (resumo honesto, com as lições)

1. **Ponto de partida** — o projeto testava `krea2_edit` (dual conditioning estilo
   Ostris/Conrad). Resultados de consistência insatisfatórios em 250–750 steps; o
   built-in do Krea 2 (conditioning grounded nativo) mascarava avaliações.
2. **Braço OminiControl** (`krea2_ominicontrol`, já existente no fork) foi adaptado
   ao paper original em `1f42c60`: posições **width-shift** (a referência vive "ao
   lado" do target no eixo W do RoPE, convenção OminiControl para tarefas
   não-alinhadas) — novo `position_mode = 'width_shift'` no pipeline de referência.
3. **Treino do probe** (250 steps, lr 1e-4, rank 64, dataset contexto_rush 1255
   pares, base `krea2_raw_fp8_scaled`): melhor fidelidade de referência de todos os
   braços — céu/design/paleta da referência quase exatos em strength 1–2.
4. **BUG descoberto e corrigido (`a66d7ba`)** — o `to_layers()` do
   `Krea2OminiControlPipeline` não repassava `reference_timestep_mode`; o adapter
   treinou com refs modulados a **t=0** apesar da config pedir `target` (e a
   metadata gravou `target` incorretamente). O runner usava o mesmo código →
   consistente → funcionava; o node ComfyUI, fiel à metadata, usava modulação
   uniforme → saídas genéricas. Diagnóstico por bisseção de forward único
   (tensores idênticos nos dois stacks): packing/freqs idênticos, tvec do span da
   ref divergente. **Lição: o contrato real é o que o código executou, não a
   metadata/config.**
5. **Node ComfyUI dedicado** (`cdf734f`, corrigido em `a66d7ba` e `bdf8d9f`) —
   ver §4. Duas armadilhas resolvidas nele:
   - **LoRA condition-only não pode ser fundido nos pesos** (loader padrão aplica
     o delta a todas as rows E requantiza `W+ΔW` em fp8, afogando deltas jovens):
     o node aplica o delta em **runtime bf16, mascarado ao span da referência**.
   - **A referência não pode ser redimensionada em espaço latente**: o node recebe
     IMAGE+VAE e faz crop-fit em pixel + encode nativo (geometria do treino).
   Validação final: comfy ↔ runner com noise idêntico → diff médio 10.9 (residual
   numérico fp8-scaled vs fork; visualmente equivalentes).

## 2. O contrato REAL do adapter funcional (o que importa reproduzir)

| Elemento | Valor efetivo |
|---|---|
| Sequência | `[texto | target ruidoso | referência limpa]` |
| Posições RoPE da ref | width-shift: grade própria com `w += largura_da_grade_do_target`, eixo frame = 0 |
| Modulação (timestep) da ref | **t = 0** (per-token; texto+target no t amostrado) |
| LoRA | rank 64, alpha=rank (escala 1.0), **somente linears dos 28 SingleStreamBlocks**, delta **routado só às rows da referência** (`ConditionOnlyLoRARouter`) |
| Conditioning de texto | **puro** (CLIP/TE nunca vê a imagem; sem vision block) |
| Geometria da ref | crop-fit em PIXEL para o tamanho do target + encode nativo no VAE |
| Base | `krea2_raw_fp8_scaled.safetensors` (treino); inferência validada em Raw e Turbo |
| Schedule de treino | logit_normal + `flux_shift = true` (mu oficial por resolução, endpoints 256→1280px) |

## 3. Treinamento — código e comando exatos

- **Commit:** `1b17e4d` (ou qualquer um ≥ `a66d7ba`, que contém o fix do to_layers).
- **Config:** `examples/krea2_omini_true.toml` (cópia da usada no probe; ajuste
  `dataset`/`output_dir`). Chaves críticas:

```toml
[model]
type = 'krea2_omini_grounded'  # NÃO: para o omini puro use 'krea2_ominicontrol'
```

  **Atenção:** para o OMINI PURO descrito aqui, `type = 'krea2_ominicontrol'` e:

```toml
[ominicontrol]
position_mode = 'width_shift'
reference_timestep = 'target'   # ver nota abaixo
reference_position_offset = 1.0
condition_only_lora = true
condition_dropout = 0.1
```

  Nota sobre `reference_timestep`: o adapter validado foi treinado ANTES do fix
  `a66d7ba`, logo rodou efetivamente com t=0. Pós-fix, a config é obedecida:
  use `reference_timestep = 'zero'` para replicar o adapter validado, ou `target`
  para a variante uniforme (não testada em treino pós-fix).

- **Comando:**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config <sua_config>.toml --regenerate_cache
```

- **Adapter do probe:** HF `AdwolfCzar/scene-continuity-r64` →
  `experiments/omini_true_step00250.safetensors`.

## 4. Inferência

### 4a. Runner (código, referência canônica)

```bash
python tools/infer_reference_adapter.py \
  --config <config_de_treino>.toml \
  --adapter <dir_step250_ou_safetensors> \
  --reference ref.jpg --prompt "..." \
  --width 672 --height 384 --seed 76 \
  --steps 8 --text-guidance 1.0 --reference-guidance 1.0 \
  --reference-fit crop --adapter-scale 2 --krea-variant turbo
```

(Para Raw: `--steps 28 --text-guidance 5.5 --krea-variant raw`.)

### 4b. Node ComfyUI — `CtxRush - Krea 2 Omini Apply (condition-only LoRA)`

- **Commit do node:** `1b17e4d` — arquivo `comfyui_nodes/ctxrush_edit/nodes.py`
  (classe `CtxRushKrea2OminiApply`); instalar copiando a pasta
  `comfyui_nodes/ctxrush_edit/` (ou o zip) para `ComfyUI/custom_nodes/`.
- **Fiação:**

```text
Load Diffusion Model (krea2 raw/turbo fp8) ──→ Omini Apply : model   (SEM Load LoRA!)
Load Image (referência) ─────────────────────→ Omini Apply : image
Load VAE (qwen_image_vae) ───────────────────→ Omini Apply : vae (e no VAE Decode)
CLIP Text Encode comum (prompt novo) ────────→ KSampler : positive
CLIP Text Encode vazio ──────────────────────→ KSampler : negative
Omini Apply : model ─────────────────────────→ KSampler : model
Empty SD3 Latent (672×384) ──────────────────→ KSampler : latent
```

- **Parâmetros do node:** `lora_name` = o adapter; `strength` 1–2 (2 = máxima
  fidelidade no probe de 250 steps); `model_variant` = turbo (8 steps/CFG 1) ou
  raw (28/CFG 5.5); **`reference_timestep = 'zero'`** para o adapter do probe
  (obrigatório — ver §1 item 4); `width`/`height` = tamanho da geração.

## 5. Tabela de commits (branch `ic-lora`)

| Commit | O quê |
|---|---|
| `3717153` | Fixes de base: dequantize (fp8 vazando), flux_shift endpoints oficiais, guards |
| `1f42c60` | `position_mode='width_shift'` + caption-dropout grounded + jitter de grounding |
| `1bb8f68` | Configs dos braços em `examples/` (incl. `krea2_omini_true.toml`) |
| `70076b1` | Runner aceita adapters target-timestep no expected_contract |
| `cdf734f` | Node `CtxRushKrea2OminiApply` (LoRA runtime mascarado, width-shift, mu correto) |
| `a66d7ba` | **Fix do to_layers** (reference_timestep) + node com seletor `reference_timestep` + ref via IMAGE+VAE |
| `bdf8d9f` | `krea2_omini_grounded` (variante com grounding Qwen3-VL — em avaliação) |
| `1b17e4d` | **HEAD recomendado** — node Omini-Grounded + tudo acima |

## 6. Estado da variante Omini-Grounded (para contexto)

`krea2_omini_grounded` (núcleo omini + grounding Qwen3-VL + txtfusion global) é o
segundo melhor resultado: excelente em held-out (carrega elementos do cenário não
citados no caption), mas exige strength 2 e a fidelidade exata é inferior ao omini
puro. Hipótese sob investigação (análise multi-agente em andamento): competição de
canais — o grounding satisfaz a loss antes dos deltas routados aprenderem. NÃO é a
receita recomendada para produção até essa análise concluir.
