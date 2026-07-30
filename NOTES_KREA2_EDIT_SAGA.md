# KREA 2 edit LoRA — saga 2026-07-29

Treino de LoRA de edição/referência (pares A→B, N=1) sobre a infra
`krea2_multiref_grounded` validada em 2026-07-26 (docs/KREA2_MULTIREF_RECEITA.md).

## Ambiente
- RTX 5090 32GB (sm_120), disco 923G (915G livres no início), sem volume persistente.
- torch: índice cu128 do PyTorch falhou (pypi.nvidia.com timeout daqui);
  instalado do PyPI padrão (torch 2.11.0, CUDA bundled ≥12.8, sm_120 ok).

## Datasets (fontes)
1. HF AdwolfCzar/recortados_dataset_captioned (2 zips) — pares _A (control) / _B (target), caption da _B
2. MEGA folder ZVp3QKqB (mega2)
3. MEGA folder cUY3kDgJ (mega3)
4. MEGA file 1FQzwS4L (mega4)
5. HF AdwolfCzar/parents_dataset_captioned (1 zip)
6. apple/pico-banana-400k — 11.359 amostras selecionadas (seed 42) em 6 categorias:
   pose 1833, addremove 2000, expression 1526, style 2000, background 2000, camera 2000.
   Editadas via CDN Apple (URL por arquivo); fontes via Flickr (aria2c).
7. HF peteromallet/InScene-Dataset — 473 pares (parquet, control_image/target_image/prompt)

## Decisões
- Sampling a cada 500 steps: pipelines de referência NÃO têm sampling in-process
  no train.py (--test_sample não empacota a referência). Plano: supervisor
  pausa→infer (tools/infer_reference_adapter.py, turbo)→resume a cada checkpoint.
- Layout do loader (ramo multi_ref): target/<stem>.jpg+.txt, refs/<stem>_1.jpg.
  Ordem NUNCA do filesystem — sufixo numérico é o contrato.
- caching_batch_size=1 obrigatório com multi_ref.

## Erros e causas
- uv install via download.pytorch.org/whl/cu128: deps nvidia hospedadas em
  pypi.nvidia.com → timeout repetido (2×). Fix: PyPI padrão (torch 2.13+cu130,
  sm_120 ok, verificado com get_device_capability()=(12,0)).
- /venv/main pertencia a root; uv falhava com Permission denied no
  site-packages. Fix: sudo chown -R claude:claude /venv/main.
- Flickr originais (pico-banana source): rate limit 429 massivo (6,5k em
  minutos com -j8). Fix: mapear OriginalURL→ImageID via
  train-images-boxable-with-rotation.csv (609MB) e baixar do S3 público
  open-images-dataset.s3.amazonaws.com/train/<id>.jpg — 11.359/11.359
  mapeados, sem rate limit.
- pkill com o padrão na própria linha de comando do shell de background
  mata o próprio grupo (exit 144). Usar padrões que não se auto-casem.

- train.py importa comfy: precisa de `git submodule update --init
  submodules/ComfyUI` (sys.path aponta para submodules/ComfyUI).
- InScene: image_id repete entre train/validation → 79 colisões de stem;
  fix: stem inclui split + índice.
- torchaudio ausente (comfy.sd importa audio_vae) → uv pip install torchaudio.
- AdamW8bitKahan quebrou com bitsandbytes 0.50: config do bnb não traz mais
  'percentile_clipping' (removido) nem 'block_wise' (sempre on). Fix em
  optimizers/adamw_8bit.py: config.get(..., default) equivalente ao antigo.
  Assinaturas de optimizer_update_32bit/8bit_blockwise conferidas — batem.

- INCIDENTE step 500 (17:19 UTC): supervisor pausava com SIGTERM, mas
  save_every_n_steps salva SÓ o adapter; o estado DeepSpeed (global_step* +
  latest) só sai por checkpoint_every_n_minutes(=120) ou saída limpa. O
  resume falhou (assert load_path) e 500 steps foram perdidos. O smoke não
  pegou isso porque lá o treino TERMINAVA (saída limpa salva estado).
  Fix: pausa via arquivo-sinal <run_dir>/save_quit (mecanismo nativo do
  saver: salva checkpoint completo e sai); SIGTERM só como fallback 15 min.
  Run morto preservado em /workspace/checkpoints/krea2_edit_saga_dead/.
  Reinício do zero (custo ~50 min; warmup/momentum íntegros).

## Contagens preparadas (target/*.txt, 0 rejeitados)
- recortados 10.063 · parents 6.984 · mega2 5.990 · mega3 7.181
- mega4 1.255 · pico 11.359 · inscene 473 · TOTAL 43.305

## Contagens (pares válidos no formato bruto)
- recortados: 10.063 · parents: ~6.4k (ext mistas, alguns incompletos)
- mega2/poxima_cena_v2: 6.000 · mega3/comikontext: 7.501 · mega4/ctx: 1.255
- pico: 11.359 selecionados · inscene: 473
