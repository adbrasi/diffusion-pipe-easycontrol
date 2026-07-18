# CtxRush Anima — next-scene (ic_lora_v2 / ic_lora_routed / omini_subject)

Um node all-in-one para testar os adapters Anima do projeto CONTEXTO.

## Fiação

```text
Load Diffusion Model (anima-base-v1.0)   ──► Next-Scene : model   (SEM Load LoRA!)
Load VAE (qwen_image_vae)                ──► Next-Scene : vae
Load Image (cena anterior)               ──► Next-Scene : image
CLIP Text Encode (prompt da cena nova)   ──► KSampler : positive
CLIP Text Encode (negativo)              ──► KSampler : negative
Next-Scene : model                       ──► KSampler : model
Next-Scene : latent                      ──► KSampler : latent
KSampler : LATENT ──► VAE Decode (mesmo VAE)
```

CLIP = `Load CLIP` com `qwen_3_06b_base.safetensors`, type **anima**.

## Parâmetros

| Campo | Valor | Nota |
|---|---|---|
| `mode` | contrato do adapter | `ic_lora_v2`/`ic_lora_routed` (ref_first), `omini_subject` ou `routed_targetfirst` — TEM que casar com o adapter carregado |
| `lora_strength` | 1.0 | 0 = baseline honesto (base + ref sem adapter) |
| `width/height` | bucket do treino | gere no mesmo tamanho configurado aqui (a ref é crop-fit para esse tamanho) |
| `ref_cfg` | 1.0 | guidance da REFERÊNCIA, independente do CFG do texto (3 branches IP2P-style). Sweep sugerido: {0, 0.5, 1, 1.5} |
| `expected_cfg` | 4.0 | DEVE ser igual ao CFG do KSampler (desacopla ref_cfg do texto) |

Nota: com `ref_cfg == expected_cfg` o node roda 1 forward por chunk (equivale ao
CFG clássico com uncond de ref zerada — `zero_ref_in_uncond` implícito); com
qualquer outro valor ele roda o segundo forward para separar as branches.

## Sampling recomendado (report Anima)

- Steps 30 · **CFG 4** · shift 3.0 (default do ComfyUI para Anima — não adicione ModelSamplingSD3)
- Sampler **er_sde** ou res_multistep · scheduler **simple** (nunca SDE/karras — borra modelos de flow)
- Negativo sugerido: `worst quality, low quality, score_1, score_2, score_3, artist name`

## Adapters (models/loras)

`ctxrush_anima_<braço>_<steps>.safetensors` — treinados no dataset contexto_rush,
base v1.0, lr 1e-4. O `mode` do node deve casar com o braço do arquivo.

## Regras que o node já cumpre por você

- referência entra por IMAGE+VAE com crop-fit em pixel (nunca resize de latente);
- nenhum scaling de latente da ref (`ref_weight` não existe aqui de propósito);
- ic_lora_routed aplica o delta SÓ nas rows da referência em runtime (fundir o
  LoRA quebraria o contrato zero-drift);
- CFG negativo com a referência zerada (= uncond treinado).
