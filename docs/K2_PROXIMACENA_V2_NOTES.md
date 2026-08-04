# k2_proximacena_v2 — NOTES do run (2026-08-04)

Retreino longo do `krea2_omini_grounded` com captions de **descrição pura** da imagem B
(o contrato do probe original que funcionou), substituindo as captions-instrução do run
de 13k steps que não aprendeu a obedecer instruções.

## Dataset final (12.455 pares únicos; 14.965 amostras/época com repeats)

| subset | pares | origem | corte |
|---|---|---|---|
| ds1_recortados | 2.900 | HF AdwolfCzar/recortados_dataset_captioned (10.063 pares) | random seed 42 |
| ds2_poxima_v2 | 5.400 | MEGA poxima_cena_v2 (6.000 pares — não ~16k como estimado) | 90%, seed 42 |
| ds3_comikontext | 2.900 | MEGA comikontext (7.501 pares) | random seed 42 |
| ds4_contexto_curado | 1.255 | MEGA dataset_pares_contexto (curadoria manual do usuário) | intacto, **num_repeats=3** |

Cortes autorizados pelo usuário em 2026-08-04 ("poxima_cena_v2 é o principal; corte os
outros ao redor dele e apenas 10% dele") — motivo: cache de text-embeddings não cabia.

Publicado: `AdwolfCzar/proxima_cena_grounded_original_dataset` (público).

## Captions (vidcap, preset novo `grounded-scene`, modo novo `image`)

- Descrição rica standalone da imagem B: framing primeiro, corpo membro a membro,
  ambiente/luz; 60–110 palavras; sem trigger words, sem instruções, sem palavras de
  estilo/qualidade; conteúdo explícito nomeado explicitamente.
- Gemini 3.1 Flash Lite em **batch** (metade do preço): 18.255 imagens, ~30 min, ~97% de sucesso.
- 534 recusas (content filter/recitation) → fallback live OpenRouter em cascata:
  `qwen3-vl-30b-a3b-instruct` (pin deepinfra/novita — fora do Alibaba) → `gemma-4-31b-it`
  → `seed-2.0-lite`. 0 falhas restantes, custo ~US$0,20.
- Teste de preço/qualidade dos fallbacks: gemma-4-31b $2,75/10k; qwen3-vl-30b $4,59/10k
  (ambos 3/3); qwen3-vl-235b devolveu vazio 2/3 no DeepInfra; seed-2.0-lite $38,94/10k
  (output tokens enormes). max_tokens=8192 fixado nos dois clientes (evita truncar JSON).

## Config (desvios do probe validado, com motivo)

- `caption_dropout=0.1` (melhoria §9 do SEGREDO), `vl_image_max_pixels=384²` (disco;
  desvio já validado no multiref), **sem** jitter de grounding (disco), grad_accum=1
  (pedido do usuário — batch real 1), 1024px AR buckets, rank 64 uniforme, lr 1e-4.
- max_steps=15000 (~1 época), save+sampling a cada 250 steps.

## Medições

- **Cache**: ~24,5 MB/par de text-embeddings (bf16 legítimo, ~400 tokens × 12 camadas ×
  2560 × 2B) + ~2 MB latents ⇒ ~330 GB para 12.455 pares. Foi O gargalo que forçou o corte
  do dataset (18.255 pares ⇒ ~484 GB > 367 GB livres).
- **Smoke 1024px batch 1**: 8,2 s/step **sem blocks_to_swap**; adapter audit OK (512 keys);
  resume OK (lr correto no log); runner de sampling OK (3/3); upload HF OK (adapter 457 MB).

## Infra do run

- Serviço supervisord `k2train` (sobrevive a reboot; `--resume` na relargada) →
  `tools/k2_proximacena_supervisor.py`: pausa via save_quit a cada 250 steps para samples
  (turbo 8 steps, seed 76, prompts em `/workspace/configs/proximacena_samples.json`),
  upload contínuo p/ `AdwolfCzar/k2-proximacena-grounded-v2-full`, OOM ladder de
  blocks_to_swap, poda de resume states (mantém 2) e de checkpoints locais já enviados
  (mantém 4), guarda de disco por df (18 GB).
- Logs: `/workspace/logs/k2_proximacena_v2/{supervisor.log,train_*.log}`;
  `tail -f /var/log/portal/k2train.log` também funciona.

## Erros e causas-raiz

- `ModuleNotFoundError comfy/hyvideo/torchaudio` no primeiro launch: os submodules do fork
  não vinham no clone (`git submodule update --init --recursive`) e torchaudio não estava
  no requirements (ComfyUI audio VAE importa). Corrigidos.
- Download MEGA duplicado criou artefatos `nome (1).ext` (colisão de re-download): dedupe
  por tamanho idêntico. mega-ls de folder público exige login; mega-get não.
- vidcap live em /venv/main: instalar deps via sudo (venv root-owned).
