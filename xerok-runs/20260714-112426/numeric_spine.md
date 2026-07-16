# Espinha numérica e roteiro — ControlLite para Ideogram 4

## 1. Resposta curta e arquitetura verificável

**Veredito: sim, condicionalmente.** É tecnicamente plausível construir um adaptador leve de controle para os pesos locais do Ideogram 4, mas **não** carregar ou treinar diretamente o ControlNet-LLLite publicado para SDXL. O artefato novo precisa implementar pontos de injeção, encoder de condição, formato de pesos e wrapper de inferência próprios para o DiT single-stream. [61][63][70]

### Números que a evidência realmente publica

| Item | Valor verificado | Implicação de projeto |
|---|---:|---|
| Backbone | DiT de flow matching, single-stream | Texto e imagem percorrem os mesmos blocos; não há os sites U-Net encoder/mid-block do ControlNet clássico. [53][35] |
| Profundidade | **34 blocos** | Um adaptador por bloco é um ponto de partida natural, mas deve preservar sequência e máscara. [53][54] |
| Atenção | **18 heads × 256 = hidden 4.608** | Q/K/V são lineares de largura 4.608; são alvos possíveis de patch/LoRA. [54][59] |
| MLP | SwiGLU **12.288** | MLP é outro alvo possível, com custo de adaptador materialmente maior que só Q/K/V. [54][59] |
| AdaLN | dimensão **512** | É modulação de timestep; o fork local a exclui da LoRA de referência por padrão. [54][68] |
| Entrada de token | **128 canais** | Vem de patch 2×2 de latente VAE de 32 canais. [54][57] |
| Texto | Qwen3-VL-8B-Instruct; concatenação de **13** camadas intermediárias; feature **53.248** | O novo caminho de controle não deve assumir CLIP/T5/Flux. [54][55] |
| Posição | MRoPE 3D e offset de texto **65.536** | Qualquer token de condição/mosaico tem de preservar semântica posicional e executar MRoPE sem autocast bf16. [58] |
| CFG | dois transformers separados, condicional/incondicional; CFG assimétrico | O contrato de controle deve especificar se e como a condição entra em cada ramo. [56] |

O ComfyUI local mantém a sequência `[texto, imagem]`, indicador 3 para texto e 2 para imagem de saída, máscara block-diagonal e reconstrução após patch 2×2; esses são requisitos de ABI, não detalhes dispensáveis. [60]

### Estimativas computadas para proposta — **não publicadas**

Assumindo LoRA de rank `r=16`, sem bias, em matrizes lineares `m × n`, o número de parâmetros é `r × (m+n)` por linear. Usando somente os tamanhos verificados acima [54]:

| Escopo proposto | Cálculo | Estimativa |
|---|---:|---:|
| Q/K/V nos 34 blocos | `34 × 3 × 16 × (4.608 + 4.608)` | **30.081.024** parâmetros treináveis (~30,1M) |
| Q/K/V + projeções MLP (SwiGLU up `4.608→24.576` e down `12.288→4.608`) | anterior `+ 34 × [16×(4.608+24.576) + 16×(12.288+4.608)]` | **55.148.544** (~55,1M) |
| Uma projeção residual densa `4.608→4.608` por bloco | `34 × 4.608²` | **721.944.576** (~721,9M) |

Essas contas **não estimam VRAM, FLOPs, qualidade ou velocidade**: elas ignoram encoder de condição, normalizações, projeções adicionais, otimizador e ativações. Elas mostram por que o MVP deve começar com Q/K/V de rank baixo, e por que um residual denso por bloco deixa de ser “lite”.

## 2. Métodos treináveis e design concreto ControlLite-I4

### Referências publicadas de custo/método

| Família | Número publicado | Leitura correta para Ideogram 4 |
|---|---:|---|
| ControlNet-XS | variantes **491M / 55M / 14M**; menor = **1,6%** de SD base 865M; documentação reporta **20–25%** mais velocidade e ~**45%** menos memória vs. ControlNet regular em SDXL | Referência de eficiência para SDXL, não benchmark transferível ao Ideogram. [4][5] |
| ControlNeXt | até **90%** menos parâmetros treináveis vs. ControlNet | Precedente para ramo compacto, não prova compatibilidade de módulo/peso. [6][7] |
| T2I-Adapter | ~**77M** parâmetros; arquivo ~**300MB** | Alternativa estrutural leve para U-Net, não implementação DiT-I4. [8][9] |
| IP-Adapter | **22M** parâmetros, backbone congelado | Referência forte para imagem de referência via atenção desacoplada, mas requer atenção/projeções I4 próprias. [10][11] |
| OminiControl | ~**0,1%** de parâmetros adicionais; Subjects200K com **>200 mil** imagens | O precedente mais próximo de controle DiT multiuso; ainda é um design/treino específico de DiT. [12][13] |
| IC-LoRA publicado | exemplo recomenda GPU de pelo menos **24GB** | É dado de FLUX, não previsão de memória do I4. [18] |

### ControlLite-I4: desenho de MVP proposto

1. **Condição estrutural primeiro:** Canny/depth/pose/segmentation alinhado ao alvo, normalizado no mesmo grid do latente; LLLite original usa condição RGB, convoluções estridadas, `down → concat(condition embedding) → mid → up` com `up` zerado e residual somado antes do módulo-base. [61][62]
2. **Mapper explícito:** aplicar adaptadores inicialmente em `q/k/v` de cada um dos 34 blocos, com escala por camada e `up`/gate inicializado em zero. Não tocar AdaLN no MVP: ela recebe somente embedding de timestep no contrato local. [59][68]
3. **Contrato de treino:** par `(imagem-alvo, mapa-condição)` com basename sincronizado, mesma resolução/grid; sem random crop para não quebrar alinhamento. O precedente LLLite recomenda cache/checkpointing e dados sintéticos do modelo-base para não exigir que módulo pequeno reaprenda estilo. [64][65]
4. **Loss/checkpoint:** loss de velocidade somente na fatia de tokens-alvo, checkpoint contendo mapper, encoder de condição, ranks/escalas e versão do contrato MRoPE/máscara. Zero-init exige validação de influência não-zero; loss plano não basta. [37][38]
5. **Inferência/ComfyUI:** node de pré-processamento + encoder/patch de condição + wrapper que injeta nos mesmos módulos em ambos os transformers/ramo definido pelo contrato; carregar um LoRA isolado não torna a inferência funcional. [40][56]

Para **referência visual rica**, priorizar o IC-LoRA já existente no checkout: ele empacota `[texto | alvo ruidoso | referência limpa]`, requer grids idênticos e só calcula perda no alvo. A referência usa indicador 4, timestep limpo 1,0 e offset temporal MRoPE padrão +1. [66][67] Isso dobra os tokens de imagem e altera a ABI de inferência; o ComfyUI stock não o suporta sem node positivo customizado. [39][44][69]

## 3. Métodos sem treino: LatentUnfold e DirectEdit

| Método | Número/contrato publicado | Uso correto | Situação no Ideogram 4 |
|---|---|---|---|
| LatentUnfold | zero-shot: sem dados, treino ou fine-tuning; mosaico `M×N`; demonstra grid **3×3** em SD3; estudo humano de **1.500** respostas | Sujeito/logo/try-on sob prompt novo por completion de mosaico. [22][25][26] | **Port experimental**, não opção plug-and-play: código é `FluxInpaintPipeline`/CLIP+T5/Flux transformer e patcha atenção/blocks Flux. Reimplementar máscara, MRoPE, QKV e mosaico I4 primeiro. [47][48][72][74] |
| DirectEdit | training-free; scripts para SD3.5-medium e FLUX.1-dev; knob `attn_ratio` | Edição da própria imagem-fonte; máscara preserva fundo. [27][28][50] | **Não é referência→nova cena** e não é plug-and-play; precisa inversão/reconstrução e controladores de atenção I4 novos. [51][52] |

Conclusão operacional: testar LatentUnfold somente como experimento de port depois do MVP treinável; usar DirectEdit apenas se o produto for edição localizada da imagem de entrada, não condicionamento de referência independente.

## 4. Diffusion-pipe / ComfyUI: implementação e validação

| Rota | Arquitetura/loader | Dados/preprocessamento | Treináveis/checkpoint | Inferência/validação |
|---|---|---|---|---|
| **A. IC-LoRA de referência (agora)** | Reusar `ideogram4_ic_lora`; `[texto|alvo|ref]`, grids iguais. [66] | Pares referência/alvo; dropout; separar held-out por identidade/estilo/fundo para evitar cópia. [46][67] | LoRA excluindo AdaLN por padrão; salvar flags de referência/MRoPE. [68] | Node positivo ComfyUI que VAE-encoda ref, anexa só no positivo e preserva grid 128c. Smoke: ABI, OOM e sensibilidade à referência. [69] |
| **B. ControlLite-I4 estrutural (MVP recomendado)** | Novo mapper nos Q/K/V dos 34 blocos; encoder de mapa no grid I4. [54][59][70] | Pares alvo+edge/depth/pose, mesmo grid/resolução, sem crop aleatório. [64] | Encoder + adaptadores QKV + metadados de alvos/rank/escala. | Node de preprocessor e wrapper do transformer; testar saída com escala 0, depois >0, e curvas de sensibilidade por condição. [37] |
| **C. LatentUnfold-I4 (P&D)** | Portar mosaic completion + hooks de atenção/posição, não reutilizar classes Flux. [47][48] | Sem dataset/treino. [71][74] | Sem checkpoint adaptador. | Primeiro benchmark de inpainting/mosaico e reconstrução; só então comparar sujeito/logo. [73][75] |
| **D. DirectEdit-I4 (edição)** | Portar inversão/controlador de atenção I4. [51] | Imagem-fonte + prompt fonte/alvo (+ máscara se preservação exata). [50][52] | Sem adaptador treinável. | Gate por erro de reconstrução e preservação mascarada antes de alegar edição fiel. [52] |

### Matriz de decisão ranqueada

Escala qualitativa: 5 = melhor; para dependência do fornecedor, 5 = menor dependência. As pontuações são recomendação de engenharia, não métricas medidas.

| Rank | Rota | Viabilidade imediata | Estrutura | Referência rica | Custo de treino | Dependência fornecedor | Decisão |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | IC-LoRA de referência existente | 5 | 2 | 4 | 4 | 4 | Fazer primeiro; fecha referência e ABI ComfyUI. [66][69] |
| 2 | ControlLite-I4 QKV para edge/depth/pose | 3 | 5 | 2 | 3 | 4 | MVP treinável seguinte; engenharia nova, mas tensores e módulos estão expostos. [54][70] |
| 3 | Combinar IC-LoRA + ControlLite-I4 | 2 | 5 | 5 | 2 | 4 | Só após matriz de compatibilidade de escalas/LoRAs; composição não é automática. [45] |
| 4 | LatentUnfold-I4 | 1 | 2 | 3 | 5 | 4 | Pesquisa sem treino para sujeito/logo; requer port completo. [47][75] |
| 5 | DirectEdit-I4 | 1 | 3 | 1 | 5 | 4 | Apenas trilha de edição; não resolve nova composição com referência. [50][51] |
| 6 | API-only | 5 | 1 | 1 | 5 | 1 | Útil como produto hospedado, mas não treina/injeta estados internos nem remove bloqueio de fornecedor. |

### Outline do relatório final

1. **Resposta e arquitetura:** veredito “sim, condicionalmente”; superfície verificável do I4, números/ABI e limites do que a arquitetura prova.
2. **Métodos treináveis e ControlLite-I4:** LLLite/ControlLoRA/XS/NeXt/OminiControl/EasyControl/IP-Adapter; design QKV de MVP, dados, treino, checkpoints e riscos.
3. **Métodos sem treino:** LatentUnfold, DirectEdit e baselines de atenção/inversão; caso de uso, dependência de hooks e gates de port.
4. **Implementação e validação diffusion-pipe/ComfyUI:** delta por rota, contrato de inferência, smoke tests, métricas de sensibilidade e critérios go/no-go por fase.

### Limites que o relatório deve preservar

Os números de ControlNet-XS, ControlNeXt, IP-Adapter, OminiControl e IC-LoRA foram publicados em outros backbones/experimentos; não estimam velocidade, VRAM ou qualidade do Ideogram 4. Não há evidência de pesos LLLite transferíveis para I4, nem de que uma API fechada exponha os estados internos exigidos. Os valores de 30,1M/55,1M/721,9M acima são contas de projeto explícitas, não benchmarks.
