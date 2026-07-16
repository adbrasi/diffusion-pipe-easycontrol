# ControlLite/ControlNet-LLLite para Ideogram 4: viabilidade, contrato real e plano de implementação

## Executive Summary

1. **Sim, condicionalmente.** Com os pesos e o `forward` locais, é plausível criar um ControlLite novo para Ideogram 4; não é plausível carregar diretamente um checkpoint ControlNet-LLLite de SDXL. O obstáculo é o contrato do transformer e da inferência, não uma ausência genérica de capacidade de treino. [1][2][3]
2. O Ideogram 4 é um **DiT single-stream** de 34 blocos: texto e imagem compartilham a mesma sequência e os mesmos blocos. A implementação local do ComfyUI preserva o ABI `[texto, imagem]`, indicadores 3/2, máscara segmentada e reconstrução após patchificação. [4][5]
3. O alvo correto do MVP é a `qkv` **combinada** `4608 → 13824` de cada bloco, não três lineares Q/K/V independentes. LoRA rank 16 nessas 34 matrizes soma **10.027.008** parâmetros por transformer; esse é um cálculo de projeto baseado nas dimensões verificadas, antes do encoder de condição. [6]
4. O Ideogram usa **dois transformers separados**, condicional e incondicional, no CFG. O MVP deve injetar o controle inicialmente apenas no transformer condicional; isso evita duplicar módulos/parâmetros e cria uma ablação clara. Só se adiciona o ramo incondicional se a validação demonstrar necessidade. [7]
5. Para **referência visual rica**, a rota concreta já existente é IC-LoRA por tokens de referência. Para **estrutura** (canny/depth/pose/segmentação), a rota seguinte é ControlLite-I4. Elas devem ser avaliadas separadamente antes de serem compostas. [8][9][10]
6. **LatentUnfold** e **DirectEdit** são adições úteis, mas não substitutos drop-in: o primeiro é uma trilha training-free de mosaico para sujeito/logo; o segundo edita a própria imagem via inversão. Ambos exigem port do runtime interno para Ideogram 4. [11][12][13][14][15]

## Scope

### What this report establishes

Este relatório estabelece a viabilidade de engenharia com base no código/pesos locais verificáveis, no contrato público do Ideogram 4 e nas implementações oficiais dos métodos comparados. Ele separa uma rota executável de referência por IC-LoRA, um MVP estrutural ControlLite-I4 e experimentos training-free, incluindo o delta necessário em `diffusion-pipe` e ComfyUI. [16][17][1][6][5]

### What it does NOT establish

Não estabelece qualidade, VRAM, latência ou fidelidade de identidade antes de treino e medição locais. Também não prova compatibilidade de checkpoints SDXL/FLUX com Ideogram 4, acesso a uma API hospedada fechada, nem que condicionar ambos os ramos CFG seja superior ao ramo condicional. Números de outros backbones são precedentes, não benchmarks transferíveis. [18][19][20][21]

| Termo pedido | Nome canônico resolvido | Base | Cobertura | Ressalva |
|---|---|---|---|---|
| Ideogram 4 | pesos locais + transformer/pipeline verificáveis | DiT local, não somente API | arquitetura e inferência | extensão de controle continua experimental |
| ControlNet-LLLite/ControlLite | princípio de adaptador condicionado leve | LLLite SDXL e proposta I4 | mecanismo e delta | pesos/módulos não são intercambiáveis |
| imagem de referência | identidade, aparência, estilo ou composição | tokens de referência | IC-LoRA | não equivale a mapa espacial alinhado |

## Arquitetura real e veredito de compatibilidade

O modelo é um DiT de flow matching single-stream com 34 blocos. O transformer expõe 18 cabeças de 256 dimensões (`hidden=4608`), MLP SwiGLU de 12288, AdaLN de 512 e entrada de token de 128; o VAE de 32 canais é patchificado em 2×2. O condicionamento textual vem de Qwen3-VL, e MRoPE 3D aplica offset de texto/imagem que não pode ser calculado em autocast bf16. [6][22][23][24]

O ComfyUI local não é um detalhe periférico: ele define sequência `[texto, imagem]`, indicador 3 para texto, 2 para imagem de saída, máscara block-diagonal e a reconstrução posterior do latente. Qualquer node novo deve manter esse ABI. [5]

Portanto, o veredito é: **sim como implementação nova, não como reaproveitamento direto de pesos LLLite**. O LLLite oficial é experimental, atende SDXL e acopla módulos leves a alvos de CrossAttention; seu contrato não é o do DiT I4. [2][25][3]

## Métodos treináveis: o que reaproveitar como princípio

| Família | Mecanismo/custo publicado | Leitura para Ideogram 4 |
|---|---|---|
| ControlNet clássico | cópia treinável do backbone ligada por zero-convolutions | forte para mapas espaciais, mas pesado e U-Net-específico [26][27] |
| ControlNet-LLLite | encoder de condição + residual leve em Linear/Conv | padrão útil de zero-init; pesos SDXL não reutilizáveis [2][25] |
| ControlNet-XS | variantes 491M/55M/14M; documentação reporta 20–25% mais velocidade e ~45% menos memória vs. ControlNet em SDXL | precedente de eficiência, não previsão I4 [18][19] |
| ControlNeXt | módulo compacto; até 90% menos parâmetros treináveis alegados | precedente de injeção in-block [28] |
| OminiControl/EasyControl | controle DiT/multicondição; OminiControl relata ~0,1% de parâmetros extras | precedentes mais próximos, mas exigem desenho e treino próprios [20][21] |
| IP-Adapter/reference tokens | atenção de texto/imagem desacoplada; IP-Adapter relata 22M com backbone congelado | melhor analogia para referência rica, não para geometria exata [29][30] |

| Família pedida | Referência | Estrutura | Custo publicado utilizável | Exige pesos/estados internos | API fechada |
|---|---|---|---|---|---|
| LLLite | baixo a médio | alto para mapas | sem número I4; módulo leve | sim | não |
| ControlLoRA | depende do desenho | alto quando treinado para mapa | sem variante/fonte primária comparável no banco; não extrapolar | sim | não |
| ControlNet-XS | baixo | alto | 14M/55M/491M em SD, não I4 | sim | não [18][19] |
| ControlNeXt | baixo | alto | até 90% menos treináveis alegados, não I4 | sim | não [28] |
| OminiControl | alto | alto | ~0,1% adicional, em seu DiT avaliado | sim | não [20] |
| EasyControl | alto | alto/múltipla | sem número I4 no banco | sim | não [21] |
| IP-Adapter/reference tokens | alto | indireto | IP-Adapter relata 22M, não I4 | sim | não [29][30] |

Na rota **API-only**, a aplicação só pode usar os parâmetros documentados pelo fornecedor; sem pesos, latentes e estados de bloco expostos pelo contrato escolhido, não há onde treinar ou injetar ControlLite. Este relatório não verificou um endpoint público que substitua essa superfície interna; portanto essa rota é produto hospedado, não integração direta.

### ControlLite-I4 estrutural proposto

O primeiro experimento deve aceitar um único mapa alinhado — canny, depth, pose ou segmentação — pareado ao alvo com mesmo basename, resolução e grid. Random crop deve ficar desligado para não quebrar o registro. O LLLite recomenda justamente pares alinhados, cache/checkpointing e, quando disponível, dados sintéticos do estilo do modelo-base. [31][32]

O ponto de injeção inicial é a Linear `qkv` combinada de cada um dos 34 blocos. Para LoRA rank `r=16`, sem bias, a conta é:

`34 × r × (in + out) = 34 × 16 × (4608 + 13824) = 10.027.008`.

Esse total é **por transformer** e exclui encoder, normalizações e otimizador. No MVP ele é contado uma vez, pois o controle entra somente no transformer condicional; não há segundo conjunto no ramo incondicional. [6][7]

Uma alternativa mais próxima do LLLite é um bloco por camada: `down 4608→64`, `mid (64+32)→64`, FiLM `32→128`, `up 64→4608`, com biases. A conta proposta é `(4608×64+64) + (96×64+64) + (32×128+128) + (64×4608+4608) = 604.928` parâmetros por bloco; em 34 blocos, **20.567.552**, mais o encoder de condição. É cálculo de arquitetura proposta, não resultado publicado. [6][25]

Em ambos os desenhos, o `up`/gate começa em zero, o delta só atinge as linhas de imagem-alvo, e AdaLN fica fora do MVP. Zero-init preserva o base no passo inicial, mas pode esconder um caminho sem influência: o teste obrigatório é sensibilidade a controles trocados e escala zero versus escala positiva. [33][34][24][5]

## Referência visual, estrutura e rotas sem treino

O **In-Context LoRA da Ali-ViLab** é o precedente que separa as duas famílias de implementação: ele compõe condição e alvo em um canvas e usa uma relação em linguagem natural para definir a tarefa. O `examples/ideogram4_stitched_ic_lora.toml` é o análogo direto desse baseline: os painéis são gerados juntos no canvas composto. Isso é útil para aprender a relação, mas uma referência externa exata não fica automaticamente fixa; é preciso clamp/inpainting do painel de condição ou migrar para tokens de referência separados. [35]

O `models/ideogram4_ic_lora.py` persegue a mesma tarefa de condicionamento por imagem, mas não é o canvas clássico: implementa `[texto | alvo ruidoso | referência limpa]`, com indicador próprio, timestep interno limpo, offset MRoPE e loss somente no alvo. Arquiteturalmente, ele fica mais próximo de condicionamento por reference tokens — e dos princípios de OminiControl — do que do In-Context LoRA costurado. É a rota mais direta para preservar uma referência externa, mas aumenta a sequência de imagem e requer node de inferência correspondente. [20][8][9][36]

**LatentUnfold** adiciona uma rota training-free relevante: codifica a referência no latente, a replica em mosaico `M×N` e completa um tile alvo, com máscara/ruído consistente e Cascade Attention. Foi demonstrado para FLUX/SD3, inclusive em grade 3×3; não é uma classe reutilizável no I4, pois o código patcha pipeline, blocos e processors de atenção do Flux. Trate-o como spike de P&D para sujeito, logo e try-on, após smoke de mosaico/reconstrução. [37][38][39][40][11][12][13]

**DirectEdit** é a adição complementar para editar a própria imagem: ele inverte a fonte, compartilha atenção/features durante a recuperação e pode preservar área não editada com máscara. Isso não resolve referência→cena nova; um port I4 precisa definir inversão, scheduler, MRoPE, CFG e controller de atenção equivalentes. [41][42][14][43][15]

## Delta de implementação no diffusion-pipe e ComfyUI

| Área | IC-LoRA de referência, já existente | ControlLite-I4 estrutural, novo |
|---|---|---|
| Arquitetura/loader | pipeline especializado que empacota ref/alvo | encoder de mapa + mapper de `qkv` nos 34 blocos [16][6] |
| Dados | referência/alvo com grids idênticos | alvo + mapa alinhado, sem crop aleatório [31][8] |
| Treináveis | LoRA e metadados de packing | encoder + adaptadores/gates QKV; política CFG no checkpoint [9][36] |
| Loss/checkpoint | supervisiona apenas alvo | mesma fatia alvo, com módulo-alvo, escala e preprocessador serializados [8][9] |
| Inferência | node positivo anexa referência e preserva ABI | preprocessor + wrapper/hook que injeta somente no condicional inicialmente [7][10] |

No ComfyUI, carregar apenas um `.safetensors` é insuficiente para IC-LoRA: o node deve VAE-encodar a referência, concatená-la, preservar indicador, offset MRoPE, timestep limpo e slice de saída. Para ControlLite, deve fixar normalização do mapa, resolução, lista/ordem de blocos, ponto de soma, escalas e política CFG. [5][10]

### Plano de três fases e gates propostos

| Fase | Rota | Entrega mínima | Critério mensurável proposto | Decisão |
|---|---|---|---|---|
| 1 | API-only / produto | validar apenas o contrato público disponível | nenhuma alegação de hook interno; registrar limites de imagem/estrutura expostos | não usar como caminho de treino |
| 2 | prova local em backbone aberto | smoke do IC-LoRA stitched e do caminho de tokens separados | mesma seed: trocar a referência deve alterar aparência; prompt deve ainda alterar cena | avançar somente se não houver cópia literal |
| 3 | integração I4 | ControlLite QKV condicional para um mapa | escala 0 reproduz base; escala >0 e mapas trocados alteram estrutura | incluir ramo incondicional apenas se o holdout melhorar |

Esses são critérios de engenharia propostos, não benchmarks publicados. O risco de bloqueio é máximo na rota API-only e passa a ser risco de ABI/dados nas rotas locais.

## Comprehensive Analysis

### 1. Qual é a superfície local verificável do Ideogram 4 e qual decisão ela permite tomar?

Há pesos/forward locais, arquitetura DiT documentada e caminho IC-LoRA no checkout. Isso permite implementar e validar adaptadores internos; não prova ControlNet pronto nem compatibilidade do ComfyUI stock com checkpoints de referência. [16][17][1]

### 2. O que a evidência pública estabelece — e não estabelece — sobre o denoising?

Ela estabelece single-stream, 34 blocos, Qwen3-VL, VAE patchificado, MRoPE e módulos de atenção/MLP. Não estabelece transferência automática de hooks, pesos ou qualidade de SDXL/FLUX para Ideogram 4. [4][6][22][23][24]

### 3. Quais pontos de injeção e pré-requisitos ControlLite exige?

LLLite usa encoder de condição e residual local `down → concat → mid → up`, com saída zero-inicializada. No I4, os alvos precisam ser mapeados explicitamente para a `qkv` combinada ou fronteiras do bloco, preservando RMSNorm, MRoPE, máscara e AdaLN. [44][2][25]

### 4. É possível treinar e conectar diretamente um ControlLite?

**Não diretamente por checkpoint; sim como extensão nova.** Exige encoder, mapper, formato de pesos, wrapper de treino e node ComfyUI próprios. O acesso ao forward torna a hipótese engenheirável; os gates de controle sensível decidem se ela funciona. [3][45]

### 5. Como LLLite, ControlLoRA/XS/NeXt diferem em custo e portabilidade?

LLLite e LoRA minimizam o delta treinável, XS/NeXt propõem ramos compactos e ControlNet clássico duplica muito mais do backbone. Os números publicados para XS e NeXt são específicos das famílias avaliadas; portabilidade para I4 é conceitual, não binária por arquivo de pesos. [26][18][19][28]

### 6. Como OminiControl, EasyControl, IP-Adapter e reference tokens se comparam?

OminiControl/EasyControl são precedentes de DiT e multicondição; IP-Adapter e reference tokens favorecem referência rica. O baseline `ideogram4_stitched_ic_lora.toml` é análogo ao In-Context LoRA de canvas composto, enquanto `ideogram4_ic_lora.py` usa referência limpa separada e loss alvo-only, mais próximo da família de reference tokens. Para o checkout, o segundo é a rota de referência externa; ControlLite cobre a lacuna de estrutura alinhada. [29][20][21][35][8]

### 7. Qual delta concreto há no diffusion-pipe?

O baseline stitched cobre canvas composto e relação em linguagem natural; o IC-LoRA de tokens separados cobre packing, VAE, loss alvo-only e metadados. Para referência externa exata, o stitched requer clamp/inpainting ou deve ceder ao caminho de tokens separados. ControlLite acrescenta preprocessador, encoder, hooks QKV, checkpoint versionado e testes de efeito; não é uma configuração TOML isolada. [35][16][8][9][36]

### 8. Qual contrato de inferência ComfyUI e como validar o MVP?

Reproduzir exatamente packing, indicadores, máscara, MRoPE, timestep de referência, slice alvo e CFG. Validar primeiro ABI/round-trip; depois escala zero=base, escala positiva=efeito e mapas trocados=alteração espacial. Começar pelo transformer condicional evita dupla contagem e isola a ablação; só incluir o incondicional se melhorar o holdout. [7][24][5][10]

### 9. Qual ranking e plano em fases equilibra viabilidade, custo e utilidade?

1. **IC-LoRA/reference tokens:** MVP de referência, pois já existe contrato local. 2. **ControlLite-I4 QKV condicional:** MVP estrutural após o primeiro smoke. 3. **Híbrido IC-LoRA + ControlLite:** somente após matriz de escalas/compatibilidade. 4. **LatentUnfold-I4:** P&D sem treino para sujeito/logo. 5. **DirectEdit-I4:** trilha de edição, não de nova composição. [46][10][11][47][15]

## Limitations and Caveats

### Base de medição

Os parâmetros de 10.027.008 e 20.567.552 são contas explícitas de módulos propostos; não incluem ativações, otimizador, encoder nem medem VRAM/latência. [6][25]

### Escopo de backbone

ControlNet-XS, ControlNeXt, OminiControl, EasyControl, IP-Adapter, LatentUnfold e DirectEdit foram publicados em outros backbones/contratos. Eles validam princípios ou casos de uso, não uma implementação I4. [18][28][20][21][11][15]

### Dados e avaliação

Pares parecidos podem induzir cópia literal no IC-LoRA. O holdout deve separar identidade, estilo, fundo e pose; a loss não substitui ensaio de sensibilidade ao controle. [46]

### Risco de integração

MRoPE, máscara e os dois ramos CFG são partes do ABI. Um checkpoint sem o node/wrapper correspondente pode carregar e ainda produzir condicionamento incorreto ou nulo. [7][24][5][10]

## Conclusion

O caminho recomendado é **IC-LoRA para referência rica primeiro**, seguido de **ControlLite-I4 estrutural em `qkv 4608→13824` apenas no transformer condicional**. É uma aposta técnica plausível e mensurável: 10.027.008 parâmetros QKV rank-16, ou 20.567.552 para a variante LLLite proposta, ambos antes do encoder. LatentUnfold e DirectEdit ampliam o mapa de experimentos sem treino, mas só após ports internos e gates de reconstrução/ABI. O go/no-go deve vir da validação no holdout, não da compatibilidade nominal entre DiTs. [6][7][5][8][11][15]

## Sources

[1] Ideogram AI, Ideogram 4 FP8 model card
https://huggingface.co/ideogram-ai/ideogram-4-fp8

[2] kohya-ss, ControlNet-LLLite training documentation
https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/docs/train_lllite_README.md

[3] kohya-ss, ControlNet-LLLite training documentation
https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/docs/train_lllite_README.md

[4] Ideogram, Ideogram 4 model architecture
https://github.com/ideogram-oss/ideogram4/blob/main/docs/model_architecture.md

[5] ComfyUI, Ideogram4 model source
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/ideogram4/model.py

[6] Hugging Face Diffusers, Ideogram4 transformer source
https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_ideogram4.py

[7] Hugging Face Diffusers, Ideogram4 pipeline source
https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/ideogram4/pipeline_ideogram4.py

[8] diffusion-pipe, Ideogram4 IC-LoRA source
https://github.com/adbrasi/diffusion-pipe/blob/main/models/ideogram4_ic_lora.py

[9] diffusion-pipe, Ideogram4 IC-LoRA source
https://github.com/adbrasi/diffusion-pipe/blob/main/models/ideogram4_ic_lora.py

[10] diffusion-pipe, Ideogram4 IC-LoRA RunPod guide
https://github.com/adbrasi/diffusion-pipe/blob/main/docs/ideogram4_ic_lora_runpod.md

[11] LatentUnfold paper
https://arxiv.org/abs/2504.11478

[12] LatentUnfold pipeline source
https://raw.githubusercontent.com/bytedance/LatentUnfold/main/latent_unfold/latent_unfold.py

[13] LatentUnfold runtime attention registration
https://raw.githubusercontent.com/bytedance/LatentUnfold/main/latent_unfold/register.py

[14] DirectEdit Flux inversion implementation
https://github.com/Tr1stesse/DirectEdit/blob/main/inversion/flow_direct_correction_inv_flux.py

[15] DirectEdit README
https://github.com/Tr1stesse/DirectEdit

[16] diffusion-pipe-easycontrol, Ideogram4 IC-LoRA implementation
https://github.com/adbrasi/diffusion-pipe-easycontrol/blob/b51b045e82dd6bb35682f860dd6a4eb2518db97c/models/ideogram4_ic_lora.py

[17] diffusion-pipe-easycontrol, Ideogram 4 IC-LoRA runbook
https://github.com/adbrasi/diffusion-pipe-easycontrol/blob/b51b045e82dd6bb35682f860dd6a4eb2518db97c/docs/ideogram4_ic_lora_runpod.md

[18] vislearn, ControlNet-XS project repository
https://github.com/vislearn/ControlNet-XS/blob/main/index.html

[19] Hugging Face Diffusers, ControlNet-XS pipeline docs
https://huggingface.co/docs/diffusers/main/api/pipelines/controlnetxs

[20] OminiControl paper
https://arxiv.org/abs/2411.15098

[21] EasyControl paper
https://arxiv.org/abs/2503.07027

[22] Ideogram AI, Ideogram 4 NF4 model card
https://huggingface.co/ideogram-ai/ideogram-4-nf4

[23] Hugging Face Diffusers, Ideogram4 pipeline source
https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/ideogram4/pipeline_ideogram4.py

[24] Hugging Face Diffusers, Ideogram4 transformer source
https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_ideogram4.py

[25] kohya-ss, ControlNet-LLLite source
https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/networks/control_net_lllite.py

[26] lllyasviel, ControlNet official repository
https://github.com/lllyasviel/ControlNet

[27] Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models
https://arxiv.org/abs/2302.05543

[28] JIA-Lab, ControlNeXt official GitHub
https://github.com/JIA-Lab-research/ControlNeXt

[29] IP-Adapter paper
https://arxiv.org/abs/2308.06721

[30] Tencent AI Lab, IP-Adapter repository
https://github.com/tencent-ailab/IP-Adapter

[31] kohya-ss, ControlNet-LLLite training documentation
https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/docs/train_lllite_README.md

[32] kohya-ss, ControlNet-LLLite training documentation
https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/docs/train_lllite_README.md

[33] Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models
https://arxiv.org/abs/2302.05543

[34] ControlNeXt paper
https://arxiv.org/abs/2408.06070

[35] Ali-ViLab, In-Context-LoRA repository
https://github.com/ali-vilab/In-Context-LoRA

[36] diffusion-pipe, Ideogram4 reference contract
https://github.com/adbrasi/diffusion-pipe/blob/main/models/ideogram4_reference_contract.py

[37] ByteDance, LatentUnfold project page
https://bytedance.github.io/LatentUnfold/

[38] ByteDance, LatentUnfold repository
https://github.com/bytedance/LatentUnfold

[39] LatentUnfold Flux-specific pipeline implementation
https://github.com/bytedance/LatentUnfold/blob/main/latent_unfold/latent_unfold.py

[40] LatentUnfold Flux attention registration code
https://github.com/bytedance/LatentUnfold/blob/main/latent_unfold/register.py

[41] DirectEdit official repository
https://github.com/Tr1stesse/DirectEdit

[42] DirectEdit inference scripts and setup
https://github.com/Tr1stesse/DirectEdit

[43] DirectEdit Flux attention controller
https://github.com/Tr1stesse/DirectEdit/blob/main/controller/attn_norm_ctrl_flux.py

[44] Hugging Face Diffusers, Ideogram4 transformer source
https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_ideogram4.py

[45] kohya-ss, ControlNet-LLLite source
https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/networks/control_net_lllite.py

[46] diffusion-pipe-easycontrol, Ideogram 4 IC-LoRA runbook
https://github.com/adbrasi/diffusion-pipe-easycontrol/blob/b51b045e82dd6bb35682f860dd6a4eb2518db97c/docs/ideogram4_ic_lora_runpod.md

[47] LatentUnfold paper
https://arxiv.org/abs/2504.11478
