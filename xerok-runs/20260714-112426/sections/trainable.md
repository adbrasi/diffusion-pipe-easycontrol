## Veredito técnico, arquitetura e rota treinável ControlLite-I4

### Veredito executivo

**Sim, condicionalmente:** é plausível treinar um adaptador leve de controle para os pesos locais do Ideogram 4, mas não é plausível carregar diretamente um ControlNet-LLLite publicado para SDXL. O LLLite oficial é experimental, é treinado para SDXL e visa alvos de `CrossAttention`; seus pesos, hooks e encoder de condição não são uma ABI compartilhada com o Ideogram 4. [61][63] A entrega viável é, portanto, um artefato novo: encoder de condição, mapper de módulos, pesos e wrapper de inferência próprios para I4. [70]

O limite é importante: o que está verificado permite implementar e testar esse adaptador no checkout local; não demonstra antecipadamente fidelidade espacial, retenção de identidade, VRAM, velocidade ou compatibilidade de pesos entre famílias. ControlNet clássico é evidência forte para controles espacialmente definidos — edge, depth, segmentação e pose — em U-Net, mas não prova que o mesmo mecanismo preserve identidade, estilo, fundo ou semântica completa de uma referência arbitrária em um DiT single-stream. [36]

### Arquitetura I4 que delimita o desenho

Ideogram 4 é um DiT de flow matching **single-stream**: tokens de texto e imagem são concatenados e atravessam os mesmos **34 blocos**. [53][35] Isso elimina os pontos naturais de cópia/injeção do encoder e mid-block de uma U-Net ControlNet; qualquer residual precisa declarar em qual fronteira do bloco I4 entra e preservar a semântica da sequência. [35]

Cada bloco tem hidden size **4.608** (`18 × 256`), MLP SwiGLU de **12.288**, AdaLN de **512**, entrada de token de **128** e texto de **53.248** features. [54] O VAE produz latentes de 32 canais, que o pipeline patchifica em 2×2 para chegar aos 128 canais de token. [57] A implementação ComfyUI organiza a sequência como `[texto, imagem]`, usa indicador 3 para texto e 2 para imagem de saída, aplica máscara block-diagonal e reconstrói o latente após a patchificação. [60] MRoPE é 3D e desloca posições de imagem em **65.536** para não colidir com texto; o cálculo não pode sofrer autocast bf16. [58]

Há ainda dois transformers de pesos separados, condicional e incondicional, combinados por CFG assimétrico. [56] Um controle não está completamente especificado se disser apenas “injeta no transformer”: deve definir se o mesmo adaptador entra nos dois ramos, só no condicional, ou com escalas distintas, e deve manter essa escolha no checkpoint e no node de inferência.

**Correção de parametrização para o MVP:** no ComfyUI I4, cada bloco possui uma única `qkv` Linear **4.608 → 13.824**, não três módulos Q/K/V separados. Assim, uma LoRA de rank 16 sem bias em todas as 34 `qkv` contém `34 × 16 × (4.608 + 13.824) = 10.027.008` parâmetros treináveis. Este é um cálculo de projeto a partir das dimensões verificadas, não uma métrica publicada. [54]

### Famílias treináveis: o que transfere como princípio, não como peso

| Família | Mecanismo e custo publicado | Leitura para I4 |
|---|---|---|
| ControlNet-LLLite | Encoder da condição e rede residual pequena anexada a Linear/Conv; no código original, `down → concat(embedding) → mid → up`, com `up` zerado e residual somado antes do módulo-base. [61][62] | O padrão de zero-init e modulação local é aproveitável, mas alvos SDXL não carregam no I4. [63] |
| ControlNet-XS | Variantes de **491M/55M/14M**; a menor equivale a **1,6%** do SD base de 865M. Em benchmark SDXL, a documentação reporta 20–25% mais velocidade e cerca de 45% menos memória versus ControlNet regular. [4][5] | Precedente de ramo compacto, não estimativa transferível de desempenho para I4. |
| ControlNeXt | Relata até **90%** menos parâmetros treináveis que ControlNet ao integrar módulo compacto no backbone. [6] | Bom argumento para injeção in-block; também alerta que zero-conv pode convergir lenta ou instavelmente. [38] |
| OminiControl / EasyControl | OminiControl injeta condições dentro de DiTs com processadores de atenção multimodal e cerca de **0,1%** de parâmetros extras. [12] EasyControl propõe módulos LoRA de injeção por condição e treinamento position-aware. [16] | São os precedentes conceituais mais próximos para DiT; foram desenhos e treinos específicos, não pesos I4 reutilizáveis. [42] |
| IP-Adapter | Cross-attention desacoplada para texto e imagem; o repositório informa **22M** parâmetros com backbone congelado. [10][11] | Melhor família para referência visual rica; exige projeções/atenção I4 e não substitui controle geométrico alinhado. |

Essas famílias também separam duas tarefas que não devem ser fundidas no primeiro experimento: mapa estrutural alinhado e referência visual rica. OminiControl mostra que um DiT pode cobrir ambas, mas o resultado publicado depende de design específico e de Subjects200K com mais de **200 mil** imagens consistentes por identidade. [13][42]

### Design exato proposto: ControlLite-I4 estrutural

**Escopo do MVP.** Treinar primeiro um controlador para Canny, depth, pose ou segmentação, não para “fidelidade completa” de uma imagem de referência. O input é um par `(imagem-alvo, mapa-condição)` com mesmo basename, resolução e grade latente; random crop fica desabilitado para preservar alinhamento. [64] Quando possível, gerar pares sintéticos no estilo do modelo-base reduz a pressão para que um módulo pequeno reaprenda estilo. [65]

**Encoder de condição.** Converter o mapa estrutural para o grid do alvo e produzir features por token I4. O encoder pode seguir a forma LLLite — downsample, fusão com embedding da condição, bloco intermediário e projeção `up` zero-inicializada — mas a saída precisa ser indexada somente aos tokens de imagem de saída, não aos de texto. [62][60] A projeção final deve ter gate/escala por camada inicializados em zero: preserva a função do backbone no passo zero, mas exige demonstrar influência não nula depois do treino; um smoke sem regressão sozinho não valida o controle. [37]

**Alvos e mapper.** Em cada um dos **34** blocos, conectar um módulo low-rank condicionado à `qkv` combinada `4608→13824`. A formulação mínima é um delta condicionado sobre a projeção, com rank 16, escala por camada e gate zero-inicializado. O orçamento é **10.027.008** parâmetros para os adaptadores QKV, antes do encoder e de normalizações: `34 × 16 × (4.608 + 13.824)`. É uma estimativa computada da proposta. [54] AdaLN fica fora do MVP, porque seu input é o embedding de timestep e não codifica diretamente a relação condição→alvo; reativá-la deve ser ablação explícita. [68]

Uma variante mais fiel ao LLLite pode ter um pequeno MLP por bloco que gera delta a partir de hidden state e feature de condição. Com `hidden=4.608`, `mlp=64` e `cond=32`, o orçamento é aproximadamente **604.928 parâmetros por bloco**, ou **20.567.552** nos 34 blocos, além do encoder. Também é cálculo de proposta, não número publicado. Ela é mais flexível que LoRA pura, mas deve ser comparada contra o baseline QKV rank-16 antes de ampliar o escopo.

**Máscara de linha-alvo e sequência.** O forward precisa aceitar uma sequência que contém texto e imagem e pode conter referência. A regra é: encoder e mapper podem ler o contexto permitido pelo contrato, mas o delta estrutural deve ser aplicado apenas às linhas/tokens de **imagem-alvo**; não alterar linhas de texto nem, quando presente, linhas de referência limpa. A máscara de atenção block-diagonal, indicadores e coordenadas MRoPE devem sobreviver ao patch. [60][58] Para o caminho IC-LoRA já existente, o layout é `[texto | alvo ruidoso | referência limpa]`; referência usa indicador 4, timestep limpo 1,0 e offset temporal, enquanto a perda sai somente da fatia alvo. [66][67] Logo, “máscara target-row” não é detalhe de loss: é o contrato que impede o controlador estrutural de corromper texto/referência e que torna possível compor os dois caminhos mais tarde.

**Loss, checkpoint e inferência.** Calcular loss de velocidade somente sobre tokens do alvo; serializar encoder, pesos dos módulos, rank, escalas, lista de alvos, política de ramos CFG, versão de máscara/MRoPE e preprocessador. O node ComfyUI precisa gerar o mapa, codificá-lo, aplicar o wrapper nos mesmos módulos e reproduzir a política condicional/incondicional. Para referência-token, carregar apenas LoRA não basta: o ComfyUI stock não anexa referência, nem preserva seus metadados de indicador, MRoPE, timestep e fatia de saída. [69][40]

**Gates de validação.** (1) escala zero reproduz o base; (2) escala positiva produz mudança mensurável e sensível ao mapa; (3) mapas trocados alteram estrutura sem degradar sistematicamente o prompt; (4) checkpoint reaberto em ComfyUI reproduz o contrato de treino. Como zero-init pode mascarar um caminho que nunca aprendeu, perda plana ou nenhuma sensibilidade à condição é falha de arquitetura/otimização a investigar, não evidência de que o MVP funciona. [37][38]
