# Interrupção no passo 500 — cota do HF privado

A nova campanha LR 0,0001 completou 500 passos e gerou quatro amostras Turbo. Loss final 0,1172; LR 0,0001 confirmado; export audit: 512 chaves. O estado completo e o adapter do passo 500 estão salvos localmente.

A automação parou na etapa de backup: o HF respondeu `Private repository storage limit reached`. O backup completo do passo 500 não foi verificado; a pasta remota pode conter upload parcial. O último milestone com backup completo verificado é o passo 250. A execução está parada; nenhum checkpoint local foi removido após essa falha.

A árvore atual deste repositório privado tem aproximadamente 30,42 GiB de arquivos, incluindo probes e smokes antigos A/B. Isso não mede o uso total da conta nem o histórico. Resolver a cota é necessário para continuar com o backup privado exigido pelo guia do repositório. A remoção de arquivos atuais pode não liberar imediatamente o armazenamento de revisões antigas.
