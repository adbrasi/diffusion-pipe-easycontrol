# Smoke A native — LR 0,0001

Resultado: aprovado. Treino do zero por 10 passos e retomada até 12. LR confirmado em 0,0001 após warmup e na retomada; nenhum estado da execução antiga LR 0,0004 foi carregado.

- Micro batch real: 2; acumulação: 1; FP8 scaled/BF16; swap: 0.
- Velocidade mediana: 0.724 amostras/s; 2.7615 segundos/passo.
- Pico de VRAM amostrado a cada 2 segundos: 24.80 GiB.
- Sem OOM/NaN; loss dos 10 passos: 0.0892 a 0.2826.
- Audit do export: 512 chaves; nenhuma chave LoRA rejeitada na inferência stock.
- Uma imagem Turbo gerada e inspecionada: válida, mas ainda mantém os personagens de costas. O smoke valida execução, sem provar aprendizado com 10 passos.
- Cache completo anterior reutilizado: 6.138 pares. O learning rate não muda embeddings/latentes.

Produção: LoRA e otimizador novos, 5.000 passos com warmup de 50, quatro amostras e backup verificado a cada 250. Não retoma o smoke nem o treino antigo.
