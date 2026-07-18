# CtxRush Anima — next-scene (ic_lora_v2 / ic_lora_routed / omini_subject)

Um node all-in-one para testar os adapters Anima do projeto CONTEXTO.

## Fiação recomendada: Dual Guider

```text
Load Diffusion Model (anima-base-v1.0) ──► Next-Scene : model   (SEM Load LoRA!)
Load VAE (qwen_image_vae)              ──► Next-Scene : vae
Load Image (cena anterior)             ──► Next-Scene : image

Next-Scene : model                     ──► Anima Dual Guider : model
CLIP Text Encode (prompt da cena nova) ──► Anima Dual Guider : positive
CLIP Text Encode (negativo)            ──► Anima Dual Guider : negative
Anima Dual Guider : GUIDER             ──► SamplerCustomAdvanced : guider
Next-Scene : latent                    ──► SamplerCustomAdvanced : latent_image
SamplerCustomAdvanced : output        ──► VAE Decode (mesmo VAE)
```

CLIP = `Load CLIP` com `qwen_3_06b_base.safetensors`, type **anima**.

Complete o `SamplerCustomAdvanced` com `RandomNoise`, `KSamplerSelect` e
`BasicScheduler`. O novo guider faz três previsões reais por step:

```text
u = negativo sem referência
t = positivo sem referência
c = positivo com referência
resultado = u + text_cfg * (t - u) + ref_cfg * (c - t)
```

Assim `ref_cfg` não é multiplicado pelo CFG textual. O fluxo antigo com
`KSampler` continua funcionando, mas exige duplicar manualmente seu CFG no
`expected_cfg` do Next-Scene e é mais fácil de configurar errado.

## Parâmetros

| Campo | Valor | Nota |
|---|---|---|
| `mode` | contrato do adapter | `ic_lora_v2`/`ic_lora_routed` (ref_first), `omini_subject` ou `routed_targetfirst` — TEM que casar com o adapter carregado |
| `lora_strength` | 1.0 | 0 = baseline honesto (base + ref sem adapter) |
| `width/height` | bucket do treino | gere no mesmo tamanho configurado aqui (a ref é crop-fit para esse tamanho) |
| `text_cfg` (Dual Guider) | 4.0 | guidance textual |
| `ref_cfg` (Dual Guider) | 1.0 | guidance da referência; sweep sugerido: {0, 0.5, 1, 1.5} |
| `ref_cfg` / `expected_cfg` (Next-Scene) | ignorados com Dual Guider | usados somente na compatibilidade com o KSampler clássico |

No caminho clássico, `expected_cfg` precisa ser idêntico ao CFG do `KSampler`.

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
- guidance de texto e referência desacoplados, sem multiplicar o controle pelo CFG.
