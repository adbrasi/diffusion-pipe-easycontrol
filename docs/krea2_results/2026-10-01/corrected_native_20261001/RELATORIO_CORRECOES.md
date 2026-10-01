# Correções e auditoria Krea 2 A native — 2026-10-01

## Mudanças confirmadas

1. Máscara do texto passada a `txtfusion` com shape B×1×1×L antes dos refinadores. A atenção entre as 12 camadas por token continua sem essa máscara, pois não atravessa tokens da legenda. O DiT mantém sua própria máscara. Corrigido em Krea2 básico, referência única e múltipla.
2. Timestep: tmlp/tproj calculados uma vez para t do alvo e uma vez para t=0 da referência; vetores expandidos por span. Mantém a equação e aproxima o caminho de cálculo do ComfyUI, sem repetir projeções grandes por token.
3. Metadata do adapter registra a correção de padding e quais adapters congelados compõem a base, incluindo Turbo quando presente.
4. Teste de regressão com PEFT, comprimentos 3/9 e checkpointing ligado/desligado verifica outputs e gradientes do batch contra execução individual. Ferramenta reproduzível adicional usa os 13B pesos reais, ScaledFP8Linear, PEFT rank64 e checkpointing.

## Precisão e dados

Todo treinamento/smoke continua em FP8 scaled para armazenamento e BF16 para cálculo, rank64, microbatch2, acumulação1 e LR0,0001. FP32 foi usado apenas em processos de diagnóstico. Não foram alterados alvo B, sinal da velocidade, timestep zero da referência, geometria ou distribuição de ruído. Cache de 6.138 pares preservado; a máscara é aplicada após os embeddings cacheados, portanto não exige recache.

## Evidências

- 18 testes CPU passaram, incluindo paridade de forward contra o ComfyUI stock nos dois métodos de timestep, regressão de padding com gradientes PEFT e contrato de referência.
- Smoke mask_only: 10 passos + resume 11/12, LR 0,0001, ~0,728 amostras/s, 2,7465 s/passo e pico amostrado 24,80 GiB. Export audit: 512 chaves.
- Smoke mask_turbo: mesmos 10+2 passos, ~0,728 amostras/s, 2,747 s/passo e 24,80 GiB. Turbo é congelado via caminho existente `merge_adapters`; 256 linears permanecem em FP8 com escalas após o merge. Export da tarefa: 512 chaves, sem incorporar os pesos congelados Turbo no adapter exportado.

## Paridade dos pesos reais — ainda não aprovada

O teste de diferenças de batch usa dois casos, t fixo 0,63, LoRA A inicializada pelo PEFT e B sintética não nula (std0,001) para exercitar ambas as matrizes; não é uma avaliação do adapter treinado no passo1949 nem uma amostra estatística da população. O gate conservador compara erro L2 de output <3% e gradiente <10% em BF16; em controle FP32 exige output <1e-4 e gradiente <1e-3. Esses limiares são critérios deste audit, não garantias universais.

No teste sintético BF16512: output ~1,31%, gradiente ~20,8%; controle com padding idêntico também divergiu (~19,2% nos gradientes). Portanto, o bug de padding não explica sozinho o desvio medido.

Controle FP32 com pesos armazenados em FP8, sem TF32, SDPA matemático e latentes pequenos: output ~2,82e-5 e gradientes ~1,36% após a reutilização das projeções. Gate estrito não passou. Isso impede declarar paridade completa do caminho real. Os gradientes medidos foram finitos.

Com dois pares cacheados reais a512 em BF16: output ~0,687% e gradientes ~159% relativos; o teste usa a LoRA sintética descrita acima. Controle FP32 nessa resolução excedeu a VRAM de32GB. Não se conclui que o dataset está errado, que o treino precisa de FP32, ou que o núcleo matemático está quebrado com base nesses controles. A divergência exige investigação adicional antes de liberar um treino longo.

As tentativas iniciais de importar o script falharam por conflito com o pacote utils do ComfyUI; a ordem de importação foi corrigida. O teste de contrato antigo inspecionava projeções repetidas internamente e foi ajustado para validar o tvec final expandido por span.

## Turbo e estado operacional

Treinar com Turbo congelado elimina a diferença de base nominal W versus W+T, mas a mesma loss de flow não garante preservar a destilação de oito passos. Os smokes verificam execução e inferência; não provam ganho de qualidade. Configurações de probes250 foram preparadas, sem execução automática ou seleção de vencedor. Todas as avaliações continuam Turbo.

A campanha anterior parou de forma segura via save_quit no passo1949. Estado completo1949 e último adapter1750 preservados e enviados ao HF público. Estados locais anteriores foram removidos somente após confirmação de backup. O A1000 antigo continua preservado. O treinamento5000 permanece parado por critério técnico; não foi reiniciado com uma receita experimental.

## Inferência dos smokes

Uma geração stock Turbo por variante foi concluída e inspecionada, sem chave LoRA rejeitada. Ambas continuam com os espectadores de costas; n=1 por variante após10 passos, sem conclusão de qualidade. Na receita Turbo congelada em FP8, a ordem de requantização do merge pode introduzir diferenças adicionais frente ao merge conjunto Turbo+adapter na inferência; esse ponto também requer paridade antes de uso longo.
