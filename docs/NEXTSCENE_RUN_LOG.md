# Anima NextScene — execução na RTX 5090

## Estado em 2026-09-30 (UTC)

Execução iniciada às ~06:05 UTC. Branch `claude/elegant-ptolemy-8kywqy`.
Handoff e pesquisa lidos integralmente, nessa ordem, antes do setup.
Resultados privados: https://huggingface.co/AdwolfCzar/anima-nextscene-runs

### Marco 0 — ambiente

- RTX 5090, 32.607 MiB, compute capability 12.0; GPU inicialmente livre.
- Disco de 180 GiB sem volume persistente. Uploads são obrigatórios.
- Submodules inicializados. Os avisos de pointers LFS no HiDream não afetam Anima.
- Dependências instaladas em `/venv/main`; torch 2.14.0+cu130.
- Operação real bf16 na GPU passou (matmul). Driver host não foi alterado.
- `python -m pytest -q test/test_anima_nextscene.py test/test_anima_inference_contract.py`:
  **17 passed**, 7,03 s. Ainda não é smoke GPU do pipeline.
- HF autenticado como AdwolfCzar; repo privado criado.
- Download dos três arquivos Base v1.0 e dataset em andamento.
- Manifesto de versões: `setup/environment.txt` no HF.

### Plano imediato

Separar 24 held-out antes dos caches; auditar pares, calibrar filtros olhando
fronteiras; construir captions full/short. Smoke de 10 steps, save/audit,
resume, geração única. Medir batch 1/2/4 antes dos probes de 250–1.000 steps.
Nenhum treino final autorizado pela evidência ainda: primeiro medir leitura de
ref, cópia e qualidade visual em held-out.

### Como acompanhar/retomar agora

```bash
cd /workspace/diffusion-pipe-easycontrol
source /venv/main/bin/activate
tail -n 30 /workspace/nextscene_ops/download.log
python -m pytest -q test/test_anima_nextscene.py test/test_anima_inference_contract.py
```

Comandos exatos dos treinos serão adicionados ao iniciar cada run. Nunca parar
um trainer com SIGTERM; usar `touch <run_dir>/save_quit`.

## Marco 1 — smoke, integração, throughput (~06:25 UTC)

Smoke real: 10 steps @512, ref limpa, `disjoint_w`; loss finita, save e geração
passaram. Resume de 10 para 30 passou, com LR real 5e-5 no log. Adapter: 280
lineares, 112 cross_attn, 560 tensors LoRA; nenhum llm_adapter/adaln.
19 testes CPU passaram após adicionar regressões dos bugs encontrados.

Bugs encontrados antes de aprender qualquer hipótese:

1. Configs fornecidos continham `alpha=64`, rejeitado por `train.py`.
   Removido: o trainer define alpha=rank.
2. `PipelineDataLoader` desempacotava `(target, mask)`; NextScene fornece
   `(target, mask, diff_weight)`. Preservar e sincronizar todos os tensors da label.
3. **PEFT usa suffix matching em listas de targets.** `blocks.0.self_attn.wq`
   atingia também `llm_adapter.blocks.0.self_attn.wq`. O log prometia exclusão,
   mas havia 72 parâmetros treináveis indevidos no bridge. Substituído por regex
   de caminhos completos e guarda de parâmetros treináveis. Regressão reproduz
   o namespace real. Nenhum checkpoint contaminado foi usado em experimentos.
4. A/B não tinha seed inicial explícita no trainer. Adicionado suporte opcional
   a `seed`, usando 42 nos configs desta execução.

| smoke (512 quadrado) | mediana s/step | amostras/s | pico VRAM MiB |
|---|---:|---:|---:|
| micro 1, checkpointing | 0,276 | 3,62 | 9.176 |
| micro 2, checkpointing | 0,516 | 3,88 | 10.096 |
| micro 4, checkpointing | 1,056 | 3,79 | 11.892 |
| micro 4, sem checkpointing | OOM antes do step 1 | — | ~32.000 |

Medições curtas (~30 steps), não equivalem ao throughput de todos os buckets.
Manter activation checkpointing; micro 2 foi ligeiramente mais eficiente.
A geração de smoke é funcional, **não prova o método** (apenas 10 amostras).
Sample: `/workspace/nextscene_artifacts/smoke/sample/20260930-061935_42.png`.
HF: `artifacts/smoke/`; adapters em `checkpoints/smoke_b*/`.

Operação: trainer/eval em fila serial, serviço supervisord `nextscene_worker`;
backup contínuo em serviço `nextscene_sync` (a cada 60s, uploads incrementais de
artefatos e adapters completos). Samples e grids ficam sempre em `/workspace/`.
Fila e estados em `/workspace/nextscene_ops/jobs/`. Resume após reboot só usa
run com arquivo `latest`, nunca reinicia treino com estado completo do zero.

Dados: download HF snapshot entrou em 429 pelo número de HEADs/arquivos.
Migrei para Git LFS: clone sem smudge, reutilização por SHA256 de 12.662 objetos
já presentes e `lfs.concurrenttransfers=32`. Download completo em poucos minutos.
Auditoria DINO agora usa batches de 32 pares (64 imagens), mantendo relatórios
por par e reduzindo overhead de GPU. Bruto em `/workspace/ds`.

Held-out: 24 pares, 6/subset, candidatos com seed 20260930; seleção visual para
incluir câmera/pose novas. Ordem intercalada (limit12 = 3 de cada subset).
Manifesto: `/workspace/heldout/manifest.json` e `artifacts/data/heldout_manifest.json`.
Reservado antes de qualquer cache. Falta ainda remover equivalentes/mesmo vídeo
no build definitivo. A avaliação usa um recorte SFW para facilitar revisão.

Captions: chave OpenRouter existe, mas está **expirada** (HTTP401). Não houve
custo de geração. Meu erro: script de probe continuou 12 chamadas após o primeiro
401; deveria ter abortado ali. Usarei extração determinística de ação/framing,
permitida no handoff, registrando cobertura e limitações; sem bloquear por API.

## Marco 2 — dados e início do E1 (~06:37 UTC)

Download completo. A contagem nominal do handoff não corresponde aos arquivos
publicados: alvos ds1=2.846, ds2=4.600, ds3=2.900, ds4=1.255; no ds2 há 5.400
refs/captions, mas faltam 800 imagens B. Pares efetivamente casados: 2.802,
4.600, 2.893, 1.255 (11.550). Não inventei nem completei pares ausentes.

Auditoria completa DINO-small em batches: ~4m24 total. Grids de fronteira com
20 pares/subset em `/workspace/nextscene_artifacts/data/*_boundary.jpg`.
Mantive dHash<=6 / pixel>=0,97 como filtro de duplicatas. DINO 0,35 descartava
muitos cortes válidos da mesma obra; baixei min_dino para 0,30/0,25/0,30/0,20
(ds1/2/3/4). O ds4 é curado e os casos de câmera nova merecem tolerância maior.
Isso é calibração visual, não uma classificação perfeita de continuidade.
Títulos/cartelas/frames pretos óbvios foram removidos pela caption.

| subset | pares finais | repeats | mediana palavras full/short | pares com short |
|---|---:|---:|---:|---:|
| ds1 | 2.368 | 1 | 93 / 17 | 1.600 |
| ds2 | 2.879 | 1 | 94 / 18 | 1.778 |
| ds3 | 2.506 | 1 | 95 / 17 | 1.910 |
| ds4 | 1.008 | 2 | 94 / 18 | 603 |

Total 8.761 pares; captions full + short onde há ação explícita. Short via
`tools/nextscene_captions.py` extrai ação e framing e evita inventar continuidade.
Limitação: não identifica personagens novos/herdados como um VLM. Casos sem ação
legível conservam só a caption original. Os relatórios guardam exemplos para revisão.

Held-out retirado por stem, SHA256 de A/B e vídeo inteiro no ds1 (44 pares
reservados para excluir vazamento por frames próximos). Em ds2/ds3/ds4 o nome
não dá identificação de vídeo/obra; cenas semelhantes podem persistir, uma
limitação documentada. Eval principal usa prompts curtos revisados manualmente,
sem depender da caption exaustiva. Manifesto dos 24 prompts em
`artifacts/data/heldout_short_manifest.json`.

Rating por classificador anime_rating (amostra seed fixa de 100 B/subset):
SFW/R15/R18 = ds1 23/5/72, ds2 87/3/10, ds3 5/11/84, ds4 85/7/8.
Estimativa R18 ponderada por pares/repeats finais ~44% (amostra do bruto, não
classificação exata do filtrado; incerteza amostral e do classificador). Não
alterei o balanceamento além de repetir ds4 duas vezes.

**E1:** 512 pares/subset = 2.048 pares base, ds4 repeats2. Seed42,
512px com 7 buckets, micro2×accum2 (4 amostras/step), rank64, lr5e-5,
warmup100, saves250. A/B só muda RoPE. Começar com 250 steps (1.000 amostras)
e decidir continuação após grid/métricas. Cache do recorte compartilhado entre
os braços. Nenhum resultado de E1 ainda.

```bash
source /venv/main/bin/activate
cd /workspace/diffusion-pipe-easycontrol
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config examples/anima_nextscene/gpu_20260930/E1_A.toml
# mesmo comando para E1_B.toml; usar --resume_from_checkpoint para continuar
python /workspace/nextscene_ops/eval_latest.py examples/anima_nextscene/gpu_20260930/E1_A.toml 250 /workspace/nextscene_artifacts/E1/A250
```

Fila serial já registrada no supervisord. Scripts operacionais copiados para
`artifacts/setup/ops` no HF. Evaluator corrigido para não sobrescrever step250
entre braços, salvar outputs individuais de resolução completa, prompts/config
exatos e cabeçalho do grid.

### E1 A — step250, 1.000 amostras (06:43 UTC)

Throughput real com buckets: mediana 0,990 s/step, 4,04 amostras/s,
pico 12.143 MiB. n=12, seed76, 512/20 steps, LoRA1/ref_cfg1, prompts curtos.
GT_true0,5099; ref_gain+0,0263; null_gain+0,0265; copy_gap−0,1497;
copy_rate0; CCIP0,4167 (CCIP em páginas/múltiplos personagens é apenas proxy).

Grid: https://huggingface.co/AdwolfCzar/anima-nextscene-runs/blob/main/artifacts/E1/A250/E1_A_20260930_06-36-45_step250/grid.png
Local: `/workspace/nextscene_artifacts/E1/A250/E1_A_20260930_06-36-45_step250/grid.png`.

**Veredito visual provisório:** outputs limpos, ações/framing geralmente seguidos,
mas identidade fraca (hat branco vira personagem de cabelo azul; mecha vermelho
vira azul; garota escura vira loira). Algumas trocas de ref afetam estilo/paleta
(ex. anime vs foto nos espectadores), mas não resolvem identidade/cenário. Sem
atrator de cópia grosseiro neste checkpoint. **Ainda não resolve o objetivo.**
Aguardar B e baseline strength0; continuar curto até1.000 se ambos ainda fracos.
Média de n12/uma seed não estabelece ranking.
