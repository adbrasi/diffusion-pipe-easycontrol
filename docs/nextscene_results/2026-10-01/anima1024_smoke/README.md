# Anima A/aligned — novo treino em 1024, BF16, dataset completo

O usuário escolheu reiniciar uma LoRA do zero sobre Anima Base v1.0,
usando o método A/aligned vencedor do E2. O adapter E2 foi preservado e
usado somente como controle de inferência. Nenhum peso dele inicializa o novo treino.
Os treinos e o worker Krea foram parados; os dois serviços de treino Krea
ficaram com autostart desativado.

## Dataset

Todos os pares completos dos quatro subsets existentes em `/workspace/ds`:

| Subset | Pares |
| --- | ---: |
| ds1_recortados | 2.796 |
| ds2_poxima_v2 | 4.594 |
| ds3_comikontext | 2.887 |
| ds4_contexto_curado | 1.249 |
| Total de treino | **11.526** |

24 pares reservados para avaliação, 51 alvos sem referência inequívoca.
Sem filtros de conteúdo, de similaridade ou limites arbitrários de amostras.
As imagens foram hardlinked, sem modificar os originais.

Cada par mantém sua legenda original completa. Como no E2, quando existe
ação textual extraível, acrescentam-se duas apresentações da legenda curta.
São 26.532 apresentações, com num_repeats=1 para todas as fontes; o peso
extra 2x do ds4 no E2 anterior foi removido. A opção `pad_last_batch=true`
evita descartar as caudas dos buckets: 12 apresentações repetidas completam
os últimos batches. Previsão: **6.636 passos para uma época completa**.

## Receita preparada

- Base original `circlestone-labs/Anima`, revisão
  `f973fc41ec7545364ac9776c2440285f43ff2a30`; hashes dos pesos em
  [base_provenance.json](base_provenance.json).
- BF16 na base e no adapter; LoRA rank/alpha 64, 280 lineares,
  incluindo self-attention, cross-attention e MLP; adaln/llm_adapter excluídos.
- Alvo primeiro, referência limpa com timestep zero, RoPE aligned, frame de referência 1.
- LR 0,0001, AdamW optimi, betas 0,9/0,99, wd 0,01; warmup de 100 passos no run principal.
- Ref dropout 0,1; high_noise_prob 0,2; diff_weight=false, sem flip/ruído extra na referência.
- Área aproximada de 1024², sete buckets AR 0,5–2,0, batch real 4,
  accumulation=1, activation checkpointing ligado, sem block swap.
- Uma época inicial completa; adapter a cada 500 passos, estados de retomada
  a cada dez minutos e no término de cada estágio.

Configuração principal:
[`A_full.toml`](../../../../examples/anima_nextscene/gpu_20261001_1024/A_full.toml).
Dataset:
[`dataset_full.toml`](../../../../examples/anima_nextscene/gpu_20261001_1024/dataset_full.toml).

## Resultado do smoke

**Passou treino, salvamento, retomada e execução da inferência.**
10 passos novos + retomada até 12; LR conferido no log após resume.
Batch 4, média **5,5215 s/passo / 0,7244 amostra/s**.
Pico de VRAM observado por polling: **20.398 MiB** (~19,92 GiB).
560 chaves de LoRA auditadas, 91.750.400 parâmetros treináveis.
40 testes CPU passaram (contrato Anima, cobertura dos buckets e verificação dos backups).
Um mock antigo do loader precisou incluir a nova propriedade
`prepare_inputs_per_microbatch`; não foi necessário alterar a matemática do Anima.

A imagem do adapter novo é do **passo 10**: não serve como veredito de qualidade.
O adapter antigo E2 foi também gerado com o mesmo par em 512 e 1024.
Reexecutando o avaliador original em 512, os dois primeiros resultados com
referência correta coincidiram **pixel por pixel** com os arquivos do E2 de ontem.
Isso verifica recuperação dos pesos e caminho de avaliação nesses dois casos;
não demonstra a qualidade do novo treino em 1024.

Uma época de 6.636 passos é estimada em **10,18 horas de cálculo** nesse smoke,
mais cache, sampling e uploads. A estimativa será atualizada com o run real.

## Operação e preservação

O controlador `tools/anima1024_campaign.py` faz estágios de 500 passos com
resume do optimizer/dataloader. Libera a GPU para avaliação de oito pares
held-out (duas referências por fonte; correta/trocada/nula, seed 76, 30 passos,
CFG 4, shift 3, ref_cfg 1, buckets em 1024). Depois continua automaticamente.
Estados antigos deste novo run só são podados após upload e checksums remotos
verificados; todos os adapters e o estado mais recente permanecem locais.
Se upload ou geração falhar, a campanha não pula a falha silenciosamente.

Backup **público**, conforme autorização explícita do usuário:
https://huggingface.co/AdwolfCzar/anima-nextscene-a-aligned-1024
Os 77 arquivos dos estados/adapters do smoke foram enviados e verificados;
recibo em [smoke_backup_verification.json](smoke_backup_verification.json).
Os repos antigos privados não tiveram a visibilidade alterada.

Única exclusão efetuada nesta retomada: o cache Krea regenerável de 96,55 GiB,
após informar o caminho ao usuário. Checkpoints, imagens, logs e datasets
originais preservados. Dois pares exatos da auditoria Krea foram salvos
separadamente antes da remoção, conforme [cleanup_manifest.json](cleanup_manifest.json).

Pastas de operação:

- `/workspace/nextscene_artifacts/anima1024_20261001/` — logs, relatórios e avaliações.
- `/workspace/checkpoints/anima_nextscene/A_aligned_1024_scratch_20261001/` — novo run.
- `/workspace/anima1024_data_20261001/` — todos os pares e cache novo.

Para parar com salvamento: criar
`/workspace/nextscene_artifacts/anima1024_20261001/stop_campaign`.
O controlador encaminha `save_quit` ao diretório timestampado do trainer.
Serviço supervisor isolado: `anima1024_worker`; arquivos reproduzíveis em `setup/`.
O chat recuperado e o manifesto com todas as legendas permanecem locais.
