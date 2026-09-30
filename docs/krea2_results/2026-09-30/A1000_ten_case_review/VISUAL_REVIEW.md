# Inspeção dos 10 casos — A1000, Turbo vs Raw, 512

Mesmo adapter A/native1000, seed76, prompt e referência em ambos. Turbo8/CFG1 com LoRA Turbo oficial; Raw28/CFG5.5 sem LoRA Turbo. Estes20PNGs passaram na auditoria de dimensões,workflow,sigmas,seed,prompts e força1 do adapter. Captions fonte permanecem intactas. Apenas referência correta: não medimos causalidade da referência ou ganho sobre base semadapter neste lote.

Turbo conserva melhor paleta/estilo nesta inspeção; Raw altera mais a cena, com mais simplificação de estilo, mudanças de cor/identidade e divisão indevida em painéis. Não há ruído checkerboard generalizado visível nas duas folhas de review. Não selecionar receita final só com10casos/uma seed. Alguns alvos B mostram outro personagem de um shot-reverse-shot; sem identificador explícito, o prompt "the same character" não escolhe univocamente a pessoa em uma referência com vários personagens. Similaridade contra B tem essa limitação.

| Caso | Turbo | Raw |
|---|---|---|
|01 perfil|Quase copia close dos olhos; não vira perfil.|Vira perfil, mas divide em dois quadros/personagens e muda olhos/estilo.|
|02 expressão cética|Close frontal e sobrancelhas céticas; cabelo/identidade diferem do alvo loiro.|Close cético, porém estilo muito simplificado.|
|03 menino/energia verde|Abre braços e mantém paleta/roupa, sem arma clara acima.|Muda pose mais, mão acima, verde mais saturado e roupa/cabelo diferentes.|
|04 choque|Close do personagem loiro com tapa-olho da ref; alvo B é outro homem de cabelo escuro.|Dois quadros, olhos amplos, mantém mão parcialmente; estilo e identidade mudam.|
|05 quadrinhos/carro|Mantém manga, mas dois painéis em vez de três e enquadramento/ação distintos.|Três painéis, contexto/personagens mudam e texto é inventado.|
|06 grito subaquático|Close do mesmo personagem/chápeu/paleta, olho ainda fechado.|Grito e close mais fortes, perde chapéu e muda estilo/cor.|
|07 macro olho|Faz macro, mantém clima escuro, desenho de íris distinto.|Faz macro, troca iluminação/paleta escura por clara e olho azul.|
|08 castelo visto de cima|Próximo de copiar ref frontal, mantém multidão.|Vista elevada externa, mas outra arquitetura/cor e mais de três silhuetas.|
|09 mulher surpresa|Close e expressão pedidos, verde/paleta herdados.|Close e surpresa, mas paleta e olhos mudam.|
|10 meninos/papel|Quase copia ref, não faz a ação sobre saco de papel.|Altera composição e faz menino se inclinar sobre saco, mas ambiente/figurantes/rosto diferem.|

Métricas de triagem (n10):

| Modo | DINO contra B | copy_rate dHash | copy_gap | CCIP contra B |
|---|---:|---:|---:|---:|
|Turbo|0.5832|0.30|0.1410|0.50|
|Raw|0.5437|0.00|-0.0549|0.70|

CCIP agrega personagem parecido com B, não confirma herança do personagem de A ou ação correta. Raw copy_rate0 não garante identidade/ambiente. Turbo detecta três cópias por dHash, coerentes com casos01/08/10; veja métricas por par para conferir.

Tempo dos jobs10outputs: Turbo46.45s,Raw165.61s; metrics12.16s. Não é benchmark de denoising isolado (inclui carregamento,encode,API e unload). AproximadamenteUS$0.039 para esses três jobs aUS$0.62/h,fora startup/inspeção/ociosidade. Treinos pausados; B1000 preservado. Comfy stock auxiliar18819 desligado após gerar; Comfy do usuário8818 não foi alterado.
