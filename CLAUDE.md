# EasyControl Anima — Instruções para Claude Code

## Situação

Você é uma nova instância do Claude Code rodando numa máquina GPU alugada no Vast.ai. Outra instância do Claude Code (na máquina local do usuário) fez toda a pesquisa e implementação. Seu trabalho agora é **configurar o ambiente e rodar o treinamento**.

### Sua máquina
- **GPU**: 1x RTX 5090 (32GB VRAM)
- **CPU**: Intel i7-12700K, 20 cores
- **RAM**: 64GB
- **Disco**: ~200GB SSD
- **CUDA**: 13.0
- **OS**: Provavelmente Ubuntu/Debian no Vast.ai

### O que foi feito (na máquina local)
- Pesquisa profunda com 11+ agentes especializados sobre controle espacial para Anima
- Implementação do EasyControl como plugin do diffusion-pipe
- 6 reviews independentes por agentes Opus (adversarial, matemático, pipeline, etc.)
- Todos os bugs críticos corrigidos
- Fork publicado no GitHub

### O que você precisa fazer
1. Clonar o fork do diffusion-pipe
2. Instalar dependências
3. Baixar modelos do Anima (DiT, VAE, text encoder)
4. Preparar ou baixar um dataset (canny edges + imagens + captions)
5. Configurar o TOML de treinamento
6. Rodar o treinamento
7. Monitorar e reportar resultados

---

## Passo 1: Clone o repositório

```bash
git clone https://github.com/adbrasi/diffusion-pipe-easycontrol.git
cd diffusion-pipe-easycontrol
```

Este é um fork do diffusion-pipe (https://github.com/tdrussell/diffusion-pipe) com nosso plugin EasyControl adicionado. Os arquivos que adicionamos/modificamos:

- `models/easycontrol.py` — **O arquivo principal** (~780 linhas). Contém toda a lógica EasyControl.
- `train.py` — 3 linhas adicionadas para dispatch + correções para save/block swap.
- `examples/anima_easycontrol.toml` — Config de treinamento (template).
- `examples/anima_easycontrol_dataset.toml` — Config de dataset (template).

Todo o resto é o diffusion-pipe upstream intacto.

---

## Passo 2: Instalar dependências

```bash
pip install -r requirements.txt
# Pode precisar também:
pip install deepspeed einops safetensors transformers accelerate peft datasets tqdm pillow opencv-python
```

O diffusion-pipe usa **DeepSpeed** (não Accelerate) para treinamento. Certifique-se que o DeepSpeed compila corretamente para sua CUDA.

---

## Passo 3: Baixar modelos Anima

Os modelos estão em https://huggingface.co/circlestone-labs/Anima

Arquivos necessários:
```bash
# Crie um diretório para os modelos
mkdir -p models_anima

# Baixe os 3 arquivos (use huggingface-cli ou wget)
huggingface-cli download circlestone-labs/Anima split_files/diffusion_models/anima-preview2.safetensors --local-dir models_anima
huggingface-cli download circlestone-labs/Anima split_files/vae/qwen_image_vae.safetensors --local-dir models_anima
huggingface-cli download circlestone-labs/Anima split_files/text_encoders/qwen_3_06b_base.safetensors --local-dir models_anima
```

Os paths finais devem ser algo como:
- DiT: `models_anima/split_files/diffusion_models/anima-preview2.safetensors`
- VAE: `models_anima/split_files/vae/qwen_image_vae.safetensors`
- LLM: `models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors`

---

## Passo 4: Preparar dataset

### Opção A: Usar dataset do HuggingFace
O usuário mencionou que encontrou um dataset no HuggingFace. Procure datasets com pares de imagens + canny edges (ou depth maps). Exemplos de busca: "canny edge dataset", "controlnet dataset".

### Opção B: Criar dataset próprio
Estrutura necessária:
```
dataset/
  target_images/
    img001.png          # imagem original
    img001.txt          # caption (texto descritivo)
    img002.png
    img002.txt
  condition_images/     # canny edges (mesmos nomes das imagens)
    img001.png
    img002.png
```

Para gerar canny edges a partir de imagens:
```python
import cv2, os, glob
for img_path in glob.glob('target_images/*.png'):
    img = cv2.imread(img_path)
    edges = cv2.Canny(img, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    basename = os.path.basename(img_path)
    cv2.imwrite(f'condition_images/{basename}', edges_rgb)
```

**IMPORTANTE**: Os arquivos de condição devem ter o **mesmo nome** (stem) que os target.

---

## Passo 5: Configurar treinamento

Edite `examples/anima_easycontrol.toml`:

```toml
output_dir = '/root/easycontrol_output'
dataset = 'examples/anima_easycontrol_dataset.toml'

epochs = 15
micro_batch_size_per_gpu = 1       # RTX 5090 tem 32GB, começar com 1
gradient_accumulation_steps = 4     # batch efetivo = 4
gradient_clipping = 1.0
warmup_steps = 100
save_every_n_epochs = 1
activation_checkpointing = true
pipeline_stages = 1

[model]
type = 'easycontrol'
transformer_path = '/root/models_anima/split_files/diffusion_models/anima-preview2.safetensors'
vae_path = '/root/models_anima/split_files/vae/qwen_image_vae.safetensors'
llm_path = '/root/models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors'
dtype = 'bfloat16'

[control]
rank = 128
alpha = 128.0
cond_size = 512

[optimizer]
type = 'adamw_optimi'
lr = 1e-4
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8
```

Edite `examples/anima_easycontrol_dataset.toml`:

```toml
resolutions = [512]               # Começar com 512 para RTX 5090 (32GB)
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 9

[[directory]]
path = '/root/dataset/target_images/'
caption_ext = '.txt'
control_path = '/root/dataset/condition_images/'
```

### Considerações de VRAM (RTX 5090 = 32GB)

O modelo Anima 2B em bf16 ocupa ~4GB. O EasyControl LoRA adiciona ~120MB (rank 128). Com activation checkpointing, o treinamento deve caber em 32GB com:
- `micro_batch_size_per_gpu = 1`
- `activation_checkpointing = true`
- `resolutions = [512]` (começar pequeno)

Se der OOM, tente:
- Reduzir `resolutions` para `[384]`
- Adicionar `blocks_to_swap = 10` no TOML (swap blocos para CPU)
- Reduzir `rank` para 64 ou 32

---

## Passo 6: Rodar treinamento

```bash
python train.py --config examples/anima_easycontrol.toml
```

O primeiro run faz cache de VAE latents e text encoder outputs (pode demorar). Runs subsequentes são mais rápidos.

### O que esperar
- Loss deve começar alto e diminuir gradualmente
- levzzz (referência) treinou com ~500 imagens, 15 epochs, rank 32, e conseguiu resultados funcionais
- Checkpoints salvos a cada epoch em `output_dir/`

---

## Passo 7: Resumir treinamento

Se precisar parar e continuar:

```toml
[control]
rank = 128
alpha = 128.0
init_from_existing = '/root/easycontrol_output/<run_dir>/adapter_model.safetensors'
```

---

## Arquitetura Técnica (Referência)

O EasyControl injeta tokens de condição no self-attention do DiT via LoRA:

1. Condição → VAE encode → PatchEmbed (compartilhado) → condition tokens
2. LoRA com binary mask em Q, K, V, output_proj — só afeta tokens de condição
3. Causal attention mask: noise vê tudo, condição vê só a si mesma
4. Condição usa timestep=0 (imagem limpa, sem ruído) para AdaLN
5. Condição mantém residual stream próprio: self-attn + MLP, pula cross-attn
6. RoPE interpolado: posições da condição mapeadas para coordenadas do noise

### Detalhes do modelo Anima
- 28 blocos transformer, 2048 dims, 16 heads, head_dim=128
- VAE: Qwen-Image, 16 canais, 8x downscale
- Text encoder: Qwen3-0.6B + LLM Adapter (6 layers)
- Flow matching: target = noise - latents (rectified flow)
- **bf16 obrigatório** (fp16 causa NaN)

### Referências
- Fork original: https://github.com/tdrussell/diffusion-pipe
- EasyControl oficial (FLUX): https://github.com/Xiaojiu-z/EasyControl
- levzzz (Anima canny que funciona): https://civitai.com/models/2443202
- Modelo Anima: https://huggingface.co/circlestone-labs/Anima
