# k2-context-rush-ofc-beta1

LoRA de edição por referência para **Krea 2** (DiT fp8-scaled, TE Qwen3-VL-4B),
treinada com o método `krea2_multiref_grounded` (N=1 referência) do fork
[diffusion-pipe-easycontrol@ic-lora](https://github.com/adbrasi/diffusion-pipe-easycontrol/tree/ic-lora).

## Método

- Sequência `[texto | target ruidoso | referência limpa]`, referência a t=0
- Posições RoPE da referência: width-shift
- LoRA rank 64 (alpha 64), condition-only routing nos 28 SingleStreamBlocks
- Canal semântico: grounding Qwen3-VL a 384² + txtfusion LoRA global rank 128
- caption_dropout 0.1 (uncond grounded per-sample para CFG)
- Otimizador AdamW8bitKahan, lr 1e-4, batch efetivo 4 (micro 1 × accum 4)
- Resolução de treino 512px, AR buckets 0.5–2.0 (7 buckets)
- 30.000 pares, 3 épocas = 22.500 steps, checkpoint a cada 500

## Dataset (30.000 pares A→B, subamostra seed 42 de 43.305)

| fonte | pares | conteúdo |
|---|---|---|
| recortados_dataset_captioned | 6.937 | pares _A control / _B target |
| parents_dataset_captioned | 4.815 | pares _A/_B |
| poxima_cena_v2 | 4.129 | next-scene pairs |
| comikontext | 4.950 | manga/comic context pairs |
| dataset_pares_contexto | 865 | context pairs |
| pico-banana-400k (subset) | 7.831 | 6 categorias: pose, add/remove, expressão, estilo, background, câmera |
| InScene-Dataset | 473 | same-scene cinematic pairs (completo) |

Captions originais de cada dataset, usadas como estão.

## Estrutura do repo

- `checkpoints/stepNNNN/` — adapter LoRA (`adapter_model.safetensors`) por checkpoint
- `samples/stepNNNN/` — 3 samples fixos gerados a cada 500 steps (turbo, 8 steps, seed 76)
- `sampling_inputs/` — os 3 inputs exatos do sampling (referência + prompt do dataset)
- `train.toml` / `dataset.toml` — configuração exata do treino
- `NOTES_KREA2_EDIT_SAGA.md` — notas da sessão (erros, causas, decisões)

## Inferência

```bash
python tools/infer_reference_adapter.py \
  --config train.toml --adapter checkpoints/stepNNNN \
  --reference ref.jpg --prompt "..." \
  --width 512 --height 512 --seed 76 \
  --turbo-lora krea2_turbo_lora_rank_64_bf16.safetensors
```

Treinado em 1× RTX 5090 32GB. Trainer: diffusion-pipe-easycontrol, branch ic-lora.
