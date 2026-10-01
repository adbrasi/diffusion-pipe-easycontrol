# A native — treino novo de 5.000 passos (aprovado)

Pedido atual: iniciar **do zero**, mantendo o A1000 anterior somente como baseline. O novo LoRA e o otimizador iniciam novos; a base pré-treinada Krea2 continua congelada. Não há `init_from_existing` nem `resume_from_checkpoint` na configuração inicial. Smoke10+resume12 aprovado; o treino principal inicia do zero após concluir o cache definitivo. Relatório medido em RELATORIO_SMOKE.md.

## Dados e disco

- Limpeza concluída: **105,2 GiB liberados**, aproximadamente **114,7 GiB livres** imediatamente após a limpeza. Cache A/B e smokes antigos, checkpoints Krea A anteriores/B e base Krea BF16 removidos. A1000 + `global_step1000` + `latest` preservados; estado completo também enviado ao HF privado.
- Quatro datasets originais `/workspace/ds`, **11.526 pares completos elegíveis**, sem regex, classificador, cota por subset, filtro de similaridade ou preferência por dataset. 24 pares antigos permanecem reservados para avaliação; 51 pares não têm referência correspondente. Os `.txt` publicados são as legendas originais; JSONL é fallback de geração, não fonte preferencial.
- Seleção global com seed42: **6.138 pares** (ds1:1523, ds2:2444, ds3:1517, ds4:654). Nomes recebem prefixo da origem para evitar colisões; imagens são hardlinks e legendas permanecem idênticas.
- Limite calculado pelo espaço, sem corte fixo1500. Medida do smoke64: **16,893 MB decimais/par**, cache previsto96,57GiB; margem10% + reserva8GiB para estados, exports temporários e resultados. Quantidade recalculada após medir o smoke64; não foram reduzidos tokens, camadas ou texto para caber. Guard de disco opt-in impede consumir os8GiB de reserva.
- O loader existente arredonda cada bucket para múltiplos do batch: nesta seleção,6136pares/época efetiva,2tails arredondados. Não existe nova seleção por conteúdo. Essa pequena diferença precisa constar no audit do dataloader.

## Configuração para conferir

| Parâmetro | Valor |
|---|---|
| Modelo/contrato | `krea2_native`, referência t=0 (`index_timestep_zero`), VL grounding |
| Inicialização | LoRA aleatório + otimizador novo; sem A1000 |
| Passos | 5.000 do novo run (10.000 amostras com micro2) |
| Resolução | 512 por área,7buckets AR0,5–2,0 |
| Batch | micro2 real, accumulation1; activation checkpointing |
| Precisão | armazenamento FP8 scaled, computação BF16, swap0 |
| LoRA | rank64, BF16 |
| Otimizador/LR | AdamW8bitKahan,4e-4; warmup50 |
| Caption dropout | 0; legenda original única por par |
| Avaliação proposta | 4Turbo512 com adapter e referência certa a cada250passos;8samplingsteps,CFG1,seed76 |
| Caminho novo | `/workspace/k2ab/checkpoints/A_native_fullbudget_fromscratch_5000` |

Aproximadamente1,63passagens pelos pares selecionados; “epochs=1000” é somente teto de configuração, o limite real é max_steps5000. Pelo throughput antigo de0,718amostras/s, o treino puro levaria **~3h52** (~US$2,40 à tarifa histórica deUS$0,62/h); cache, cargas e avaliações acrescentam tempo/custo. Não é medição do novo dataset.

## Execução autorizada

1. Liberar modelos da GPU no ComfyUI do usuário sem derrubar a interface. Preparar cache dos64pares de smoke, cobrindo7buckets; medir bytes reais/par e ajustar a seleção pelo disco disponível antes do cache completo.
2. Smoke novo10passos; audit de chaves, loss finita, amostras/s,VRAM e LR. Retomada de2passos apenas do próprio smoke confirma o estado; esse smoke **não** inicializa o treino principal. Micro4 já deu OOM na campanha anterior; micro2 é a referência medida. Se o novo texto exceder memória, reavaliar batch real, sem accumulation extra.
3. Conferir1geração de smoke em ComfyUI stock para confirmar contrato e carregamento do adapter. Cachear o conjunto definitivo com os embeddings nativos, sem alterações de conteúdo.
4. Primeiro segmento0→250 do zero, depois250→500→…→5000 retomando **somente o run novo**. As configs param nos marcos para permitir sampling serial na mesma GPU;4imagens por marco. Upload privado contínuo dos adapters/grids. Manter2estados recentes do novo run; podar arquivos anteriores só depois da confirmação do backup. Baseline A1000 preservado.

Código de treino em `/workspace/k2ab/native_worktree`, que conserva o ComfyUI nativo validado. Os arquivos core de treino são iguais aos da branch atual; revisão exata do encoder registrada em `source_state_before_cleanup.json`. Especificação inicial `train_job.draft.json` está fora da fila supervisionada. Worker Krea ativo. Controller supervisionado A-only coordena cache, treino serial, sampling e backups; os controllers A/B antigos permanecem desligados.

As falhas visuais A1000 foram preservadas no log: Turbo copia alguns casos; Raw pode alterar identidade/estilo. Um treino maior com mais variedade é um experimento solicitado, e precisa das avaliações para demonstrar melhora.

LR final por correção explícita do usuário: **0,0004**, também no smoke. O A1000 anterior usou0,0001.
