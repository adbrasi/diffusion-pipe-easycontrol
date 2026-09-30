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

### E1 B250 e controles sem LoRA (06:57 UTC)

| checkpoint | GT_true | ref_gain | null_gain | copy_gap | copy_rate | CCIP |
|---|---:|---:|---:|---:|---:|---:|
| base aligned (strength0) | 0,4577 | 0,1307 | 0,2063 | −0,0155 | 0,0833 | 0,4167 |
| A250 | 0,5099 | 0,0263 | 0,0265 | −0,1497 | 0 | 0,4167 |
| base disjoint_w (strength0) | 0,4146 | 0,0179 | −0,0202 | −0,2304 | 0 | 0,1667 |
| B250 | 0,5191 | 0,0543 | 0,0428 | −0,1235 | 0 | 0,4167 |

n12/seed76 para todos. B250 melhora GT (+0,1045), uso de ref (+0,0364) e CCIP
contra seu próprio baseline. A250 fica mais bonito/próximo de B, mas perde muito
ref_gain contra o baseline. Isso é direção inicial a favor de B, **não ranking
conclusivo** nem identidade resolvida.

**Olho corrige a leitura dos baselines:** aligned sem adapter é em geral
quadriculado/degradado e reproduz estrutura de A; o ref_gain alto dele não é
sucesso. O único copy flag é o comic `02_ds3_006895`, A↔B dHash21, portanto não
é falso positivo de B quase igual a A. Disjoint sem adapter gera formas planas,
paleta esverdeada e personagens genéricos. Treino melhora os dois em qualidade.
B250 preserva melhor o look 3D/roupa listrada no exemplo da sala de aula, mas
continua perdendo hat/mecha vermelho/cabelo de outros casos. Seguir ambos até500.

Grids locais/HF sob `artifacts/E1/B250/`, `baseline_A/`, `baseline_B/`.
Todos os outputs true/shuffled/null individuais são salvos, não só thumbs do grid.

Benchmark extra: micro2 **sem** activation checkpointing passou a512 quadrado:
0,379s/step, 5,28 amostras/s, pico29.178MiB (contra0,516/3,88 com checkpointing).
Margem menor; não mudei a comparação E1 no meio. Os adapters de30steps nas duas
modalidades têm cosine0,920/relL2 0,401 nos B da LoRA: não são bit-equivalentes
(RNG/caminho numérico podem divergir). Candidato de eficiência para run de produção,
com novo smoke nos buckets reais; checkpointing continua necessário na alta resolução.

### E1 step500 e grids solicitados (07:10 UTC)

n12, seed76, mesmos prompts/config do step250.

| checkpoint | GT_true | ref_gain | null_gain | copy_gap | copy_rate | CCIP |
|---|---:|---:|---:|---:|---:|---:|
| A500 | 0.5678 | 0.0677 | 0.0798 | -0.1148 | 0.0000 | 0.4167 |
| B500 | 0.5331 | 0.0673 | 0.0216 | -0.0944 | 0.0000 | 0.2500 |

Qualidade/framing melhoram, mas identidade continua fraca: perfil/hat e mecha
ainda viram identidade/paleta genérica. B500 preserva estilo 3D/roupa listrada
no caso da sala de aula, enquanto shuffled/null perdem esse estilo. Esse ganho
isolado não resolve o conjunto; não há vencedor robusto aos500. Continuação
pareada até750/1000, com avaliação e atualização dos grids a cada250steps,
foi enfileirada. Nenhum treino final iniciado.

Pedido do usuário: A, B e resultado de **todos** os testes já concluídos.
Montados6grids E1 de3colunas, uma comparação geral de8colunas/12pares,
versões512px, controles shuffled/null e smoke10 separado (seed42/caption
completa). Os benchmarks de memória/throughput não geraram imagens.

Local: `/workspace/nextscene_artifacts/comparacoes/README.md` e
`TODOS_E1_A_B_resultados{,_full}.png`. HF privado:
https://huggingface.co/AdwolfCzar/anima-nextscene-runs/blob/main/artifacts/comparacoes/TODOS_E1_A_B_resultados.png

A/B são recortados exatamente como o evaluator (512 quadrado); outputs512
são preservados integralmente. O crop corta os frames largos, portanto uma
validação posterior em buckets de aspecto é apropriada, sem alterar E1 atual.
Scripts de montagem arquivados em `artifacts/setup/ops/`.

### Revisão visual simplificada solicitada (07:21 UTC)

Criado `tools/nextscene_review_grid.py`: usa somente outputs existentes; nenhum
sampling adicional. Reúne A250/B250/A500/B500,12pares, original+shuffle
(96linhas), estritamente3colunas A|B|Resultado. Agrupado por par, com
layout/step/condição/nome da referência efetiva em cada linha. No shuffle,
A mostra o input realmente trocado, não a referência original.

Em `/workspace/nextscene_artifacts/comparacoes/`: `REVISAO_250_500_3_COLUNAS.png`,
PDF do mesmo nome (12páginas,1par/8resultados por página), PNGs separados250/500
e manifest com correspondências. Baselines LoRA0 e null continuam nos arquivos
anteriores; esta revisão foca os4checkpoints treinados e seus shuffles solicitados.
Validado:96combinações únicas, todos os inputs/outputs existem, original/shuffle
consistentes, PNGs íntegros e PDF válido. Mantido sync para HF privado.

```bash
python tools/nextscene_review_grid.py
```

### Push solicitado e continuação até1.000 (07:30 UTC)

Usuário autorizou explicitamente push dos códigos e1.000steps em ambos os
layouts. Interpretação:1.000totais por método, retomando os mesmos pesos/optimizer
a partir de750; sem reiniciar do zero nem adicionar novos braços. A1000 em
execução, B1000 enfileirado com resume.

Tempo dos primeiros500: soma dos tempos de iteração A512,15s/B511,48s
(~8min32 de cálculo por método). Somando duração dos processos250+500:
A661,57s (11min02, incluiu primeiro cache), B554,39s (9min14). Avaliações,
esperas na fila e testes de outros braços estão fora desses tempos.

Versionados6scripts operacionais em `tools/nextscene_gpu_ops/`; credenciais
não estão no repo. Finalizado evaluator opcional `--match-target-ar`, com
encode por dimensão e config original preservada. E1 continua512quadrado:
não mudei parâmetros nem avaliação da comparação em andamento. Dimensões
AR verificadas em CPU; flag opcional ainda sem smoke de sampling em GPU.

### Pacote para agente com acesso somente ao repo (07:35 UTC)

Usuário pediu push e imagens250/500/750 para revisão externa. Pacote em
`docs/nextscene_results/2026-09-30/`:6grids de3colunas, layouts/steps/condição
rotulados;6pares cobrindo4subsets, referência original+shuffle,72outputs
existentes. Seleção ilustrativa inclui falhas e progresso, não ranking.
Métricas/metadata incluem todos12pares, prompts e contrato de cada checkpoint.
Sem dependência de HF privado para abrir os PNGs; cópia também em
`/workspace/nextscene_artifacts/review_for_agent/2026-09-30/`.

A750: GT0,5832/ref_gain0,0865/null_gain0,0982/CCIP0,4167/copy_rate0.
B750: GT0,5446/ref_gain0,0386/null_gain0,0394/CCIP0,4167/copy_rate0.
A melhora métricas médias vs500; B conserva o caso3D mas sua média de
ref_gain oscila. Identidade ainda fraca nos exemplos de perfil/mecha; sem
vencedor visual robusto. A1000 terminou treino; avaliação e B1000 na fila.

Validação pré-push:23testes relevantes passaram; scripts operacionais com
sintaxe verificada. Grid/manifest da exportação são verificados antes do commit.

### E2 — receita corrigida e uma época por braço (2026-09-30)

A revisão externa considera E1 cedo demais para julgar o método. Correção
importante da contagem: E1 realmente usou o probe de2.048pares,4.183amostras
de captions/repeats por época;1.000steps/4.000amostras são ~0,96época desse
recorte, não0,25. O número0,25 refere-se ao corpus completo antigo de16.263
amostras, que não foi usado em E1. E2 aumenta cobertura para os8.761pares
completos além da receita nova. E1 é evidência inicial de qualidade e uso
parcial de referência, não refutação; sem vencedor/identidade resolvida.
E1 terminou1.000steps nos dois braços, com n12/seed76.

**Pedido do usuário:** uma época do dataset completo por braço, do zero.
E2 usa8761pares-base filtrados; ds4repetido2; `[full, short, short]` nos pares
com curta, somentefull nos demais. Amostras efetivas:22757,
~5690steps/época antes do ajuste de buckets. Não limitar a5.000:
uma época inteira prevalece sobre o número sugerido pelo agente de pesquisa.

Mesmos parâmetros nos dois braços excetooutput_dir/rope_layout: LR1e-4,
diff_weight=false, rank64, ref_dropout0,1, high_noise_prob0,2, micro2×accum2,
seed42 e warmup100. Diretórios E2 separados; nenhum peso/optimizer deE1 reutilizado.
Curtas agora usam `The same <sujeito>`, preservando o substantivo extraído.
Fallback heurístico permanece: chave VLM expirada; não inferiu continuidade
visual nem resolveu ambiguidades de múltiplos sujeitos. Dataset/caches isolados
em `/workspace/ns_E2/`; E1 preservado.

**Smokes7buckets:** micro2×accum2 semactivationcheckpointing deuOOM em ambos
oslayouts (~31,3GiB). Portanto ambos E2 usamactivationcheckpointing=true.
Smokes AC percorreram uma época de28pares/7buckets,14steps em cada braço.
Losses finitos; adapters560keys finitas,0llm_adapter/adaln. Medianas A/B:
1.019/1.019s/step.23testes relevantes passaram.
Smoke de sampling comtargetAR (2pares/4denoisingsteps) passou antes do run completo;
outputs comdimensões640×400, métricas/grids salvos em `E2/smoke/AR_sampling/`.

Plano supervisionado: treinar Aepoch1→Bepoch1 sem sampling entre os braços;
guardar adapters1000/2000/3000/4000/5000 eepoch1, estados de retomada a cada10min;
depois avaliar1000/2000/3000/5000/epoch1 em24heldouts, seeds76/142,512pixelbudget,
20denoisingsteps, targetAR, ref correta/shuffled/null.24pares existentes revisados
visualmente; prompts idênticos nos braços. Diretório `/workspace/nextscene_artifacts/E2/`.

Sync HF corrigido para incluir também adapters `epoch1`, além de `step*`.
Campanha durável versionada em `tools/nextscene_gpu_ops/e2_campaign.py`, interrompe
etapas dependentes se uma fase falhar, retoma somente o estado do próprio E2.
Registra checkpoints/tempos/custo no log e fazcommit/push automaticamente.
Estimativa acumulada desde06:05UTC neste momento: US$1.24 aUS$0,62/h
(inclui setup, downloads/cache e ociosidade; não é extrato de cobrança Vast).

A comparação E1→E2 muda receita e cobertura do dataset em conjunto; não permite
atribuir o ganho a uma mudança individual. Além disso E1 usa n12/cropquadrado e
E2 n24/targetAR: comparar médias cruas entre protocolos exige cuidado. Os dois
braços dentro deE2 continuam diferindo somente no layout RoPE.

### E2 — 2026-09-30 08:07 UTC

Iniciando `train_A_epoch1`. Log local: `/workspace/nextscene_artifacts/E2/train_A_epoch1.log`.

Custo estimado acumulado desde06:05 UTC: US$1.27 (2.05h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — loader confirmado e treino iniciado (08:19 UTC)

Loader: **5685steps/época ×4 = 22740amostras**, após arredondamento dos
buckets (estimativa anterior22.757). A iniciou updates e está no step225;
mediana atual0.992s/step, losses finitos, GPU observada100%utilização.
Previsão de cálculo:94.0min por braço (~3h10 ambos), além do
cache inicial e ~1h para20avaliações (5checkpoints ×2braços ×2seeds).
Custo estimado acumulado desde06:05UTC agora:US$1.39.

Campanha supervisionada está em execução; B começa automaticamente depois
da época deA. Confirmado push da receita/código e backup HF dos configs,
captions exatas e smokeAR. Uma atualização remota de documentação Krea2 foi
integrada via merge; não alterou código/configs de treino.

### E2 — 2026-09-30 08:32 UTC

`train_A_epoch1` salvou step1000 (4,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_A/20260930_08-15-31/step1000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$1.53 (2.47h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 08:50 UTC

`train_A_epoch1` salvou step2000 (8,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_A/20260930_08-15-31/step2000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$1.71 (2.75h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 09:07 UTC

`train_A_epoch1` salvou step3000 (12,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_A/20260930_08-15-31/step3000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$1.89 (3.04h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 09:24 UTC

`train_A_epoch1` salvou step4000 (16,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_A/20260930_08-15-31/step4000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$2.07 (3.33h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 09:42 UTC

`train_A_epoch1` salvou step5000 (20,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_A/20260930_08-15-31/step5000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$2.24 (3.62h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 09:54 UTC

Concluído `train_A_epoch1` em 106.3min. Último step: 5685; época completa.

Custo estimado acumulado desde06:05 UTC: US$2.37 (3.82h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 09:54 UTC

Iniciando `train_B_epoch1`. Log local: `/workspace/nextscene_artifacts/E2/train_B_epoch1.log`.

Custo estimado acumulado desde06:05 UTC: US$2.37 (3.82h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 10:11 UTC

`train_B_epoch1` salvou step1000 (4,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_B/20260930_09-54-22/step1000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$2.55 (4.11h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 10:28 UTC

`train_B_epoch1` salvou step2000 (8,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_B/20260930_09-54-22/step2000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$2.73 (4.40h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 10:46 UTC

`train_B_epoch1` salvou step3000 (12,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_B/20260930_09-54-22/step3000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$2.90 (4.68h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:03 UTC

`train_B_epoch1` salvou step4000 (16,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_B/20260930_09-54-22/step4000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$3.08 (4.97h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:20 UTC

`train_B_epoch1` salvou step5000 (20,000 amostras vistas). Adapter: `/workspace/checkpoints/anima_nextscene/E2_B/20260930_09-54-22/step5000`. Treino segue até o fim da época.

Custo estimado acumulado desde06:05 UTC: US$3.26 (5.26h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:32 UTC

Concluído `train_B_epoch1` em 98.2min. Último step: 5685; época completa.

Custo estimado acumulado desde06:05 UTC: US$3.38 (5.45h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:32 UTC

Iniciando `eval_A_step1000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step1000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.38 (5.45h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:35 UTC

Concluído `eval_A_step1000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.42 (5.51h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:35 UTC

Iniciando `eval_A_step2000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step2000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.42 (5.51h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:39 UTC

Concluído `eval_A_step2000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.46 (5.57h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:39 UTC

Iniciando `eval_A_step3000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step3000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.46 (5.57h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:42 UTC

Concluído `eval_A_step3000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.49 (5.63h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:42 UTC

Iniciando `eval_A_step5000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step5000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.49 (5.63h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:46 UTC

Concluído `eval_A_step5000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.53 (5.69h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:46 UTC

Iniciando `eval_A_epoch1_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_epoch1_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.53 (5.69h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:50 UTC

Concluído `eval_A_epoch1_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.57 (5.75h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:50 UTC

Iniciando `eval_B_step1000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step1000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.57 (5.75h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:53 UTC

Concluído `eval_B_step1000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.60 (5.81h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:53 UTC

Iniciando `eval_B_step2000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step2000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.60 (5.81h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:57 UTC

Concluído `eval_B_step2000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.64 (5.87h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 11:57 UTC

Iniciando `eval_B_step3000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step3000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.64 (5.87h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:00 UTC

Concluído `eval_B_step3000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.68 (5.93h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:00 UTC

Iniciando `eval_B_step5000_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step5000_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.68 (5.93h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:04 UTC

Concluído `eval_B_step5000_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.71 (5.99h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:04 UTC

Iniciando `eval_B_epoch1_seed76`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_epoch1_seed76.log`.

Custo estimado acumulado desde06:05 UTC: US$3.71 (5.99h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:07 UTC

Concluído `eval_B_epoch1_seed76` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.75 (6.05h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:07 UTC

Iniciando `eval_A_step1000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step1000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.75 (6.05h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:11 UTC

Concluído `eval_A_step1000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.79 (6.11h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:11 UTC

Iniciando `eval_A_step2000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step2000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.79 (6.11h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:14 UTC

Concluído `eval_A_step2000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.82 (6.16h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:14 UTC

Iniciando `eval_A_step3000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step3000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.82 (6.17h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:18 UTC

Concluído `eval_A_step3000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.86 (6.22h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:18 UTC

Iniciando `eval_A_step5000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_step5000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.86 (6.22h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:21 UTC

Concluído `eval_A_step5000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.90 (6.28h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:22 UTC

Iniciando `eval_A_epoch1_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_A_epoch1_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.90 (6.28h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:25 UTC

Concluído `eval_A_epoch1_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.93 (6.34h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:25 UTC

Iniciando `eval_B_step1000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step1000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.93 (6.34h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:29 UTC

Concluído `eval_B_step1000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$3.97 (6.40h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:29 UTC

Iniciando `eval_B_step2000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step2000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$3.97 (6.40h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:32 UTC

Concluído `eval_B_step2000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$4.01 (6.46h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:32 UTC

Iniciando `eval_B_step3000_seed142`. Log local: `/workspace/nextscene_artifacts/E2/eval_B_step3000_seed142.log`.

Custo estimado acumulado desde06:05 UTC: US$4.01 (6.46h × US$0.62/h; inclui setup, cache e ociosidade).

### E2 — 2026-09-30 12:36 UTC

Concluído `eval_B_step3000_seed142` em 3.5min. 

Custo estimado acumulado desde06:05 UTC: US$4.04 (6.52h × US$0.62/h; inclui setup, cache e ociosidade).
