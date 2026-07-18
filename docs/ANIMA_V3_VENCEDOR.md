# Anima IC-LoRA V3 — o método vencedor (e a saga do llm_adapter)

**Data:** 2026-07-18 · **Branch:** `ic-lora` · **Veredito do usuário:** ic_lora_v3 é o melhor.
**Método:** `models/ic_lora_dual.py::ICLoraV3Pipeline` · **Config:** `examples/anima_v3_500.toml`

---

## 1. A descoberta em uma frase

> O "segredo de abril" que fazia os adapters Anima antigos serem muito melhores
> era o **escopo largo do LoRA** — em especial o **llm_adapter treinado** (a ponte
> texto→DiT) e o **cross_attn** — que os fixes de maio removeram por precaução
> junto com o adaln (este sim o culpado real do derretimento).

## 2. Arqueologia (como chegamos aqui)

1. Os adapters de abril (`next_scene_iclora_test1_1875steps_8batch` etc., 7/abr)
   foram treinados com o código de 5–7/abr: IC-LoRA V1 **target-first** com LoRA
   em TODOS os lineares (self_attn + cross_attn + mlp + adaln + llm_adapter).
2. O audit de maio (`97fc94c`/`f85e808`) diagnosticou o derretimento como
   double-LoRA no adaln (correto) e cortou TAMBÉM cross_attn e llm_adapter
   (sem evidência) — os braços de julho ficaram só com self_attn + mlp.
3. A/B visual definitivo (18/jul, mesma seed): **com** llm_adapter o modelo
   adota o estilo e a continuidade da referência; **sem**, faz anime chapado
   ignorando a estética da ref.
4. **Armadilha crítica encontrada:** `llm_adapter_lr = 0` nas configs (herdada
   dos exemplos) **congela silenciosamente até o LoRA do llm_adapter**
   (`requires_grad False` → nem treina, nem salva). Abril não tinha essa linha.
   NUNCA usar `llm_adapter_lr = 0` com escopo largo.

## 3. A receita do V3

```toml
[model]
type = 'ic_lora_v3'            # broad: self_attn + mlp + cross_attn + llm_adapter
# SEM llm_adapter_lr!          # (0 congelaria o canal semântico)
sigmoid_scale = 1.0

[ic_lora_full]
ref_first = false              # target-first [alvo T=0 | ref T=1] (contrato do V1 de abril)
condition_dropout = 0.0        # abril fiel: ref presente em 100% dos steps
condition_timestep = 0.0
shifted_logit_normal = false   # logit-normal puro
include_adaln = false          # adaln FORA (a causa real do derretimento)

[adapter]
rank = 32                      # alpha=rank

[optimizer]
type = 'adamw_optimi'
lr = 1e-4                      # batch efetivo 8 (micro 1 × accum 8)
```

340 linears no adapter: 168 aparência (self_attn+mlp) + 136 cross_attn + 60 llm_adapter
(120 params de llm_adapter no otimizador — conferir no log de startup!).
500 steps já mostram o comportamento; probe validado no dataset contexto_rush (1255 pares).

## 4. Inferência

### Runner (referência canônica)
```bash
python infer_easycontrol.py --dit anima-base-v1.0 --vae qwen_image_vae \
  --llm qwen_3_06b_base --lora <v3>/adapter_model.safetensors \
  --mode ominicontrol_subject --control_image ref.jpg --prompt "..." \
  --width 720 --height 368 --steps 30 --cfg 4 --flow_shift 3 \
  --lora_strength 1.0 --ref_cfg 1.0
```
(target-first global == modo `ominicontrol_subject`; guidance de 3 branches com
`--ref_cfg` independente do `--cfg` do texto.)

### ComfyUI — `CtxRush - Anima Next-Scene`
- `lora_name = ctxrush_anima_v3_s500` com **`mode = auto`** (o nome resolve o
  contrato `broad_targetfirst`).
- O node aplica: aparência+cross_attn em runtime no forward; **llm_adapter via
  object patch no `preprocess_text_embeds`** (ele roda ANTES do forward do DiT —
  sem esse patch o canal mais importante não age).
- `ref_cfg`/`expected_cfg` para guidance independente (ou o `Dual Guider` +
  SamplerCustomAdvanced para a matemática exata).
- Settings: CFG 4, shift 3 (default Anima), 30 steps, euler/simple ou er_sde.

## 5. Irmãos do V3 (mesma família, sabores diferentes)

| Braço | Diferença | Quando usar |
|---|---|---|
| `ic_lora_v3` ⭐ | tudo global | herdar continuidade COM o estilo da ref |
| `ic_lora_dual` | aparência routada (zero-drift) + semântica global | manter o estilo do base, herdar só conteúdo/continuidade |
| `ominicontrol_broad` | = v3 com condition_dropout 0.1 | uncond de ref treinado (CFG de ref mais fiel) |

## 6. Contratos de teste padrão (definidos pelo usuário)

- `/workspace/imagem000075.jpg` + prompt com sufixo de continuidade
  ("Character continuity: same character. Background continuity: new view of
  the same background.") — o schema de captions do dataset de abril.
- Sakura (`test_refs/anime_random.jpg`) + "the same girl now stands up and
  smiles brightly, arms spread wide, cherry blossom petals swirling around her
  in the schoolyard".
- Sempre: smoke de 1 imagem ANTES de qualquer batch.

## 7. Commits-chave (branch ic-lora)

| Commit | O quê |
|---|---|
| `5456f17` | fix inferência: preprocessing == treino + guidance 3 branches |
| `12edc84` | braços broad (v3/dual/omini_broad) + runner dual |
| (este) | node com llm_adapter via object patch + relatório |

Adapters e grids: HF `AdwolfCzar/groundedsecret` → `anima_probes/broad_*`.
Runs locais: `/workspace/checkpoints/anima_iclora_v3/` (o run `_archive/v3_sem_llmadapter`
é o A/B sem llm_adapter — manter para referência).

## 8. Próximos passos recomendados

1. Treino longo do v3 no dataset SFW definitivo (batch 8, ~1–2 épocas do
   dataset final, saves frequentes, upload automático).
2. Captions com sufixo de continuidade no dataset novo (o schema de abril).
3. Opcional: sweep de `condition_dropout` {0, 0.1} no escopo largo
   (v3 vs omini_broad já são esse A/B em 500 steps).
