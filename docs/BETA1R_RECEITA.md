# BETA1-R: receita (adapter de próxima cena para Krea 2)

Réplica do `AdwolfCzar/k2-context-rush-ofc-beta1` (o melhor adapter da saga) com **só 4 correções**, para tirar a
cópia da imagem de entrada sem perder o que fazia o beta1 funcionar. Iniciado em 02/11/2026 ~02:10 UTC.

Pesos e samples: HF `AdwolfCzar/krea2-beta1r` (`checkpoints/stepN/`, `samples/stepN/` com `grid.jpg` e
`metrics.json`, `configs/`). Configs: `examples/beta1r/train.toml` e `examples/beta1r/dataset.toml`.

## Resultado até agora

| step | copy_rate (8 pares fora do treino) | dHash médio saída × referência |
|---|---|---|
| 250 | 0 / 8 | 30,6 |
| 500 | 0 / 8 | 30,5 |
| 750 | 0 / 8 | 30,25 |

No step 750 a composição segue o prompt em todos os casos, e personagens, estilo e paleta vêm da referência. A leitura
da referência melhora com os steps (personagens da referência que faltavam em 250/500 aparecem em 750). O cenário da
referência às vezes some. Avaliado pelo usuário: "muito bom, realmente muito bom".

## O método (idêntico ao beta1)

- `type = 'krea2_multiref_grounded'`, `max_refs = 1`.
- Sequência `[texto | alvo com ruído | referência limpa]`. Referência com `reference_timestep = 'zero'` e atenção isolada.
- RoPE `width_shift` (`reference_position_offset = 1.0`): a referência fica ao lado do alvo no eixo da largura, no
  mesmo grid do alvo (recorte central + resize).
- Grounding: o Qwen3-VL também vê a referência ("image 1:"), `vl_image_max_pixels = 147456` (384²).
- LoRA **routed** (`condition_only_lora = true`): rank 64 nos 28 blocos, aplicada só nas linhas da referência.
  Mais uma LoRA de rank 128 no txtfusion (`txtfusion_rank = 128`). 512 chaves, 242,35M parâmetros, mesmo formato
  do beta1: carrega nos nodes do `adbrasi/ctxrush-edit` (CtxRush multiref).
- AdamW8bitKahan, lr 1e-4, betas (0.9, 0.99), wd 0.01. Batch efetivo 4 (micro 1 × accum 4, como o beta1).
- `logit_normal` + `flux_shift`, 512px com buckets de proporção (0.5–2.0, 7 buckets).

## As 4 correções

| | beta1 | BETA1-R | por quê |
|---|---|---|---|
| base | `diffusion_model_dtype='float8'` (sem escala: zerava 6–27 % dos pesos) | `base_quant = 'fp8_scaled'` | bug de fp8; o adapter não precisa mais do node K2 Training Base |
| caption dropout | 0.1 | 0.0 | legenda vazia + referência limpa ensina a copiar |
| reference dropout | (nenhum) | `reference_dropout = 0.1` | em 10 % das amostras a referência some **inteira**: sem bloco "image 1:" no Qwen3-VL (legenda mantida) e tokens VAE da referência mascarados como chave de atenção e zerados. A amostra vira texto→imagem puro de B |
| checkpoints | até 13.445 steps sem parada | save + samples + `copy_rate` a cada 250 | escolher o checkpoint antes da cópia; o beta1 ficou "velho" (a cópia cresce com os steps no LoRA routed) |

Outras diferenças forçadas: `live_text_encoding = true` (o reference dropout muda o prompt a cada amostra; o cache de
texto do beta1 tinha 729 GB), warmup 50 (beta1: 100), máscara de padding no txtfusion (no-op com micro 1). Rank e todo o
resto ficaram iguais de propósito ("não mudar tanto").

Teste do reference dropout: `test/test_krea2_reference_dropout.py` (3/3). Amostra descartada num batch misto = texto→imagem
do ComfyUI puro (erro ~2e-5); sem a máscara a referência vaza (controle negativo).

## Dataset: o do beta1 sem pico-banana (20.112 pares)

| fonte | pares | origem |
|---|---|---|
| recortados | 6.937 | HF `recortados_dataset_captioned`, legendas originais |
| parents | 4.815 | HF `parents_dataset_captioned`, legendas originais |
| inscene | 473 | HF peteromallet InScene, legendas originais |
| poxima_v2 | 4.129 | substituto: `proxima_cena_grounded_original_dataset` ds2 (re-legendado) |
| comikontext | 2.893 | substituto: ds3, tudo o que existe (beta1: 4.950) |
| mega4 | 865 | substituto: ds4 |

- pico-banana (edições locais, ~26 % do beta1) foi removido a pedido do usuário: é uma das causas documentadas da cópia.
- Os originais do MEGA (poxima_v2, comikontext, pares_contexto) não têm a chave registrada; os substitutos são os mesmos
  pares, com legendas que descrevem B.
- Amostragem por fonte com seed 42 (a subamostra exata de 30k do beta1 não é reproduzível). Normalização das imagens
  igual ao `prepare_krea2_edit.py` (lado maior ≤ 1024, JPEG q95). Montagem: `tools/build_beta1r_data.py`.

## Como rodar

```bash
# treino supervisionado (save + samples + upload a cada 250, retoma sozinho)
cd /workspace/dpe-beta1r && setsid nohup /workspace/dpe-venv/bin/python -u tools/beta1r_supervisor.py \
  --config examples/beta1r/train.toml --cases /workspace/runs/beta1r_samples_inputs/cases.json --resume \
  > /workspace/runs/logs/beta1r_supervisor.out 2>&1 &
# parar limpo (salva o estado completo via save_quit; nunca SIGTERM)
touch /workspace/runs/beta1r_supervisor.stop
```

Samples: `tools/beta1r_sample.py` (Turbo fundida, 8 steps, seed 76; 8 pares do ds2 fora do treino + os 3 exemplos fixos do
beta1, nos mesmos tamanhos, para comparar com `samples/stepN` do beta1). `copy_rate` = fração com dHash(saída, referência)
≤ 10/64.

## Inferência e paliativo

Nodes `adbrasi/ctxrush-edit` (CtxRush multiref). Se aparecer cópia, `ref_bias` 0.95–0.99 no
`CtxRushKrea2MultiRefControlledApply` tira a cópia do beta1 por completo, com pequena perda de consistência (observação do
usuário: a cópia é um limiar, 1 % menos atenção na referência já sai do "poço").

## Próximos passos

1. Seguir até 3000 steps a 512px, olhando samples e `copy_rate` a cada 250.
2. Escolher o melhor checkpoint e fazer um refino curto a 1024px, como o beta1.
3. Se a cópia aparecer mais tarde: usar o checkpoint anterior, ou testar rank 32 numa rodada curta.
