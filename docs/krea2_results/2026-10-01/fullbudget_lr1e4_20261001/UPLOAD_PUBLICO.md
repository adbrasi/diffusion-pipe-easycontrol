# Continuação do passo 500 com backup público

O usuário autorizou explicitamente uploads em repositório público e pediu continuar até 5.000 passos. Destino: https://huggingface.co/AdwolfCzar/krea2-a-native-lr0001 . O repositório privado anterior não teve sua visibilidade alterada.

O controlador continua a mesma execução LR 0,0001 a partir de seu checkpoint global_step500, sem recomeçar ou alterar o learning rate. O cache de 6.138 pares continua válido. Jobs de backup público usam nomes próprios, preservando o registro da falha de cota privada. Step250 também será enviado ao novo repositório público.

Uploads públicos incluem adapters, estados de otimizador/resume, configs e amostras/métricas de treino. Os manifestos completos de captions originais e auditorias de armazenamento da conta permanecem locais. Cada adapter e estado é verificado por tamanho e SHA256 antes da poda local.
