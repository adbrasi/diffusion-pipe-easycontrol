# A native — reinício do zero, LR 0,0001

Pedido do usuário: parar o treino com LR 0,0004 e voltar ao learning rate antigo, 0,0001, reiniciando do zero.

O treino anterior foi encerrado via `save_quit` no passo 843 (último save regular: 750). O estado completo global_step843 foi enviado ao HF privado e verificado por tamanho e SHA256; o último adapter exportado, step750, foi preservado localmente. Os estados antigos removidos já tinham backup verificado. O A1000 original continua preservado.

A nova LoRA e o otimizador são inicializados do zero em uma pasta separada, sem retomar o checkpoint antigo. O cache validado dos 6.138 pares continua igual, pois o LR não altera embeddings ou latentes. Batch real 2, acumulação 1, FP8 scaled/BF16, 512 px, rank 64 e 5.000 passos. Warmup de 50 passos até 0,0001.

Amostras: quatro casos held-out Turbo a cada 250 passos, seguidas de métricas, backup privado verificado e push dos achados. O controlador usa fila, nomes e status próprios, sem reutilizar os jobs da execução anterior. Novo smoke obrigatório: 10 passos + retomada até 12 e uma geração.
