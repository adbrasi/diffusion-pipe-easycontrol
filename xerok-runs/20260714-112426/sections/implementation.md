## Decisão de implementação: referência rica primeiro, estrutura como adaptador opcional

**Veredito: sim, condicionalmente.** Há superfície local suficiente para treinar um condicionador de imagem para os pesos locais do Ideogram 4, mas não para reutilizar um checkpoint ControlNet-LLLite de SDXL. Ideogram 4 é um DiT de flow matching **single-stream** de 34 blocos; texto e imagem atravessam os mesmos blocos, em vez dos sítios encoder/mid-block de um U-Net que o ControlNet clássico copia. [35][53] O LLLite publicado é experimental, tem encoder de condição e pequenos residuais anexados a Linear/Conv, e seu suporte SDXL visa caminhos de CrossAttention específicos. [61][62][63]

Portanto, a recomendação é um **híbrido em duas vias**, não um único “ControlLite” tentando fazer tudo:

1. **Via principal — tokens de referência + LoRA (IC-LoRA local):** usar uma imagem rica para identidade, material, estilo e semântica visual ao concatenar seus tokens latentes com os do alvo. É a rota que já tem contrato de treino no checkout e melhor corresponde a “usar a referência para uma imagem nova”.
2. **Via complementar — ControlLite estrutural:** treinar um encoder de canny/depth/pose/segmentação que produza resíduos por bloco ou tokens de condição espaciais. Só adicioná-lo se a referência-token não entregar layout/pose reproduzível. Estrutura e retenção de referência são objetivos distintos; o precedente DiT do OminiControl também trata condições de sujeito e alinhadas como tarefas separáveis, não como equivalentes. [12][13][42]
3. **Ladder experimental sem treino:** antes de gastar em novo adaptador, testar mosaico/referência em latente e hooks de atenção em cópia isolada. LatentUnfold mostra a hipótese training-free para FLUX, mas seu código é acoplado a FluxInpaintPipeline, CLIP/T5, classes de transformer e attention processor do Flux; “ambos são flow DiTs” não torna o port compatível. [71][72][74][75]

### Matriz de decisão

| Rota | Referência rica | Estrutura explícita | Pesos/estados internos | Treino no checkout | Compatibilidade hoje | Decisão |
|---|---|---|---|---|---|---|
| API hospedada somente | limitada ao que a API expuser | limitada ao que a API expuser | não | não | somente contrato público | não é rota para ControlLite próprio |
| IC-LoRA de reference tokens local | alta, se os pares evitarem cópia literal | indireta; não substitui pose/depth | sim | sim, caminho já implementado | requer node ComfyUI próprio | **MVP recomendado** |
| ControlLite DiT novo | alta se combinado à via acima | alta para canny/depth/pose | sim | condicional: módulos novos | requer wrapper novo | **fase 2, apenas após MVP** |
| OminiControl/EasyControl reimplementado | potencialmente alta | alta e multi-condição | sim | pesquisa/port | não é drop-in | posterior; desenho mais amplo [12][15][16] |
| LatentUnfold/atenção training-free portado | sujeito/estilo experimental | fraca a experimental | sim | não | não é plug-and-play | spike barato, sem promessa [47][48][49] |

Essa matriz não atribui VRAM, velocidade ou qualidade sem medição local. O ponto objetivo é acesso: a rota API não fornece os tensores necessários; as três últimas exigem controle do forward do DiT. Para o local, os números que delimitam o projeto são 34 blocos, hidden size 4.608, 18 cabeças de 256 e MLP SwiGLU de 12.288. [54]

### Delta concreto no diffusion-pipe e no ComfyUI

O checkout já contém `models/ideogram4_ic_lora.py` e a configuração `examples/ideogram4_ic_lora.toml`. Sem editar nada nesta pesquisa, a leitura local confirma o contrato `[texto | alvo ruidoso | referência limpa]`, shape idêntico entre os latentes de alvo/referência, dropout de condição, offset MRoPE e exclusão padrão de `adaln_modulation`; o exemplo começa em 512 px e há config de smoke. Isso é implementação existente, não prova de qualidade nem de suporte no ComfyUI stock. A evidência externa/coletada para o mesmo fork descreve o mesmo contrato. [66][67][68]

| Área | IC-LoRA já presente | Delta para ControlLite estrutural |
|---|---|---|
| Arquitetura/loader | pipeline Ideogram4 especializado e camada inicial que empacota referência | `Ideogram4ControlLite` com mapper explícito dos `nn.Linear` de atenção/MLP dos 34 blocos; encoder de condição que chega a grade/token compatível com patch 2×2 e latente de 32 canais [57][59][70] |
| Dados/preprocessamento | `control_path` é a referência pareada, mesmo basename | `control_path` passa a ser canny/depth/pose/segmentation derivado do alvo; sem crop aleatório se o sinal tiver de permanecer registrado; separar pares de referência de pares estruturais [64] |
| Treináveis | LoRA nos linears selecionados; AdaLN fica fora por padrão | encoder do sinal + down/mid/up ou projeções residuais zero-inicializadas por bloco; começar em Q/K/V ou boundary de bloco, não assumir sites U-Net [62][70] |
| Loss/checkpoint | somente fatia do alvo é supervisionada; metadados registram layout, indicador, offset e timestep da referência | salvar pesos do encoder, mapa de módulos, escala por bloco, tipo/resolução do controle e versão do ABI; teste deve provar influência não nula, pois zero-init também passa smoke sem controlar [37][38] |
| Inferência/exportação | LoRA não basta: preprocessar referência no VAE e empacotar conditioning positivo | node deve criar o sinal estrutural, codificá-lo, instalar hooks antes do sampler e aplicar as escalas em ambos os ramos CFG conforme contrato definido |

O ControlLite não pode reutilizar cegamente o esquema de LLLite: o Ideogram calcula Q/K/V com RMSNorm, MRoPE e AdaLN em cada bloco. [58][59] Escolher residual **antes** ou **depois** da projeção, e se ele entra em Q/K/V, saída de atenção ou boundary de bloco, muda o ABI e exige checkpoint próprio. Um residual denso 4.608→4.608 por bloco seria aproximadamente 721,9M parâmetros; como proposta de ponto de partida, LoRA rank 16 só em Q/K/V nos 34 blocos dá aproximadamente 30,1M parâmetros. São cálculos de escopo, não benchmarks de VRAM ou qualidade. [54]

### ABI de inferência que não pode divergir

Para **referência rica**, o node ComfyUI deve: (a) VAE-encodar alvo e referência no mesmo grid latente; (b) patchificar e concatenar `[texto, alvo, referência]`; (c) marcar texto=3, alvo=2, referência=4; (d) manter a referência em timestep interno limpo e aplicar seu offset temporal MRoPE; (e) manter máscara/segmentos e extrair somente tokens do alvo para a previsão; (f) anexar referência somente ao conditioning positivo, conforme o contrato do fork. [60][66][67][69] Stock ComfyUI não aceita esses tokens: carregar o `.safetensors` LoRA sem este node é um artefato incompleto. [69]

Para **estrutura**, o ABI adicional deve fixar: tipo e normalização do mapa, resolução e registro com o alvo, output do encoder, lista/ordem dos blocos alvo, ponto exato de soma, escala por condição e regra CFG. A implementação oficial usa transformers condicional e incondicional separados com CFG assimétrico; é necessário testar explicitamente se o sinal estrutural entra nos dois ramos, e com qual escala, em vez de copiá-lo do SDXL. [56] MRoPE 3D usa offset de texto 65.536 e exige cálculo sem autocast bf16; alterações de comprimento/posição sem essa disciplina tornam o resultado inválido mesmo que o sampler não falhe. [58]

### Ladder de validação e go/no-go

| Fase | Entrega mínima | Critério de passagem | Go/no-go |
|---|---|---|---|
| 0. ABI | teste de packing, shape, máscara, posições e checkpoint round-trip | saída do forward no shape esperado; carregamento reproduz metadados; referência zeroada preserva o caminho T2I | **no-go** se ComfyUI e treino empacotam diferente |
| 1. IC-LoRA | smoke existente em 512 px + pequeno holdout de pares | variação de referência muda identidade/aparência de modo observável sem copiar literalmente; prompt ainda muda a cena | **go** para referência; **no-go** se só reconstrói a fonte ou ignora a referência [46] |
| 2. Estrutural | um único sinal, preferencialmente canny ou depth, e um único ponto de injeção | mesma seed/prompt com controles distintos produz mudança espacial correspondente; escala zero reproduz base e escala não-zero tem efeito | **go** só com sensibilidade ao controle; smoke sem regressão não basta por zero-init [37] |
| 3. Híbrido | referência IC-LoRA + ControlLite estrutural | matriz base/referência/estrutura/style-LoRA avaliada em holdout | **no-go** se combinação satura ou uma via anula a outra; composição de LoRAs não é automática [45] |
| 4. Training-free | spike de mosaico ou hook de atenção em branch separado | reconstrução/inpainting e identidade melhoram contra base sem quebrar MRoPE/máscara | descartar se o port não superar o IC-LoRA simples; não promover como compatível por analogia com Flux [47][48] |

A avaliação deve ter prompts e pares de holdout que variem separadamente **identidade/objeto**, **estilo**, **fundo/câmera** e **layout/pose**. Loss de treino não é gate suficiente: pares vizinhos podem ensinar atalho de cópia. [46] Para estrutura, medir aderência ao mapa e preservar prompt; para referência, comparar a mesma cena com duas referências e a mesma referência com duas cenas. Essas são metas de validação propostas, não métricas publicadas.

### Comprehensive Analysis — respostas 1:1

### 1. Superfície local verificável

Há forward/pesos locais do Ideogram 4 e um IC-LoRA já implementado no checkout. Isso permite decidir por um MVP de referência-token; não prova que exista ControlNet pronto nem que o ComfyUI stock execute o checkpoint. [39][40]

### 2. Mecânica pública do Ideogram 4

O que está estabelecido é DiT single-stream, 34 blocos, VAE Flux2, tokens de texto Qwen3-VL e MRoPE. O que não está estabelecido é que qualquer mecanismo SDXL/Flux de controle se transfira sem novo forward e novo ABI. [53][54][55][57][58]

### 3. Contrato ControlLite

LLLite codifica condição e soma um pequeno residual nos módulos alvo; para Ideogram os alvos têm de ser novos linears/boundaries do DiT e respeitar Q/K/V, MRoPE e AdaLN. [61][62][59]

### 4. É possível conectar diretamente?

**Não diretamente por peso; sim como implementação nova condicional.** Pesos LLLite SDXL não servem. Com acesso ao forward local, encoder, mapper, checkpoint e node próprios, a hipótese é engenheirável e deve passar os gates acima. [63][70]

### 5. Métodos leves versus completos

ControlNet clássico copia backbone; XS publica variantes de 491M/55M/14M e ControlNeXt afirma reduzir até 90% de parâmetros treináveis. São referências de custo/arquitetura, mas seus números e pesos pertencem aos backbones avaliados, não ao Ideogram. [3][4][6][7]

### 6. Referência versus estrutura

IP-Adapter separa atenção de texto e imagem; OminiControl/EasyControl são precedentes DiT para condição de imagem e múltiplas condições. No checkout, IC-LoRA é a rota concreta de reference tokens; um ControlLite deve ficar opcional para sinal espacial, não substituir a referência rica. [10][12][15][17]

### 7. Delta diffusion-pipe

IC-LoRA já cobre loader, VAE, packing, loss alvo-only e save metadata. ControlLite adiciona encoder/preprocessador, hooks por bloco, estado de pesos e testes de efeito; não é configuração TOML isolada. [66][67][70]

### 8. Inferência compatível

O contrato ComfyUI precisa reproduzir exatamente packing, indicador, MRoPE, timestep limpo e CFG. Para estrutura precisa ainda reproduzir o ponto de injeção e a escala. Sem node customizado, a LoRA de referência não funciona corretamente no ComfyUI stock. [56][58][69]

### 9. Ranking e plano

1) IC-LoRA/reference tokens, por já existir no checkout; 2) ControlLite estrutural pequeno combinado ao anterior, após sucesso do MVP; 3) Omini/EasyControl como pesquisa posterior; 4) spike training-free isolado, sem compromisso de produto. O bloqueio real é ABI/validação, não apenas capacidade de treino; a rota API permanece fora do escopo de adaptadores internos.
