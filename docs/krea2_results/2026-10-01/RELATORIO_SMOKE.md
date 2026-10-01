# Relatório do smoke — A native do zero, LR0,0004

Smoke técnico **aprovado**:10steps novos + retomada11/12, loss finita e nenhum OOM.

| Medida | Resultado |
|---|---|
| LR após warmup e na retomada | 0,0004 |
| Batch real/acumulação | 2 / 1 |
| Throughput mediano | 0,722amostra/s |
| Tempo mediano | 2,7705s/step |
| VRAM pico, medição a cada2s | 24322MiB (23,75GiB) |
| Loss nos10steps | 0,0665–0,1857; última0,0767 |
| Loss na retomada11/12 | 0,0991 / 0,1173 |
| Audit de export | 512chaves corretas |
| Stock ComfyUI | 1PNG688×384,Turbo8/CFG1;0avisos de chaveLoRA não carregada |
| Cache64pares | 1.081.177.189bytes;16,893MB/par |

Text embeddings medidos mantêm BF16 e30720features (=12×2560). Exemplo conferido:261tokens. Legendas originais intactas; a diferença para24,8MB é o tamanho real dos tokens nesta amostra, não uma compressão aplicada pelo agente. Custo de cache total inclui ambas as cenas nos latentes e um embedding incondicional compartilhado.

A imagem é válida e conserva estilo/ambiente, mas ainda copia a referência de costas. **Isso valida execução/carregamento, não o aprendizado da próxima cena**. A primeira avaliação maior será no250. PNGem `/workspace/k2ab/artifacts/fullbudget_20261001/smoke_eval/Turbo/03_ds4_imagem000268_with_lora.png`.

Recalculada seleção global pelos bytes medidos: **6138pares** entre11526completos, sem filtro de conteúdo/cota de origem. ds1:1523,ds2:2444,ds3:1517,ds4:654. Estimativa de cache96.57GiB,10%de margem,8GiBreservados. Guard no cache impede consumir essa reserva e confirma as linhas já escritas antes de parar. Cache completo deve confirmar o tamanho e a contagem finais.

Treino principal continua **do zero**,5000steps,sem usar pesos/otimizador do smoke ou A1000. Smoke exports/evidências enviados ao HFprivado; caches e estados temporários do smoke removidos após verificação do backup. BaselineA1000 preservado. Código/guard,controller serial e configs versionados; os próximos20marcos terão4Turbo512+grid/métricas e backup privado verificado antes da poda.
