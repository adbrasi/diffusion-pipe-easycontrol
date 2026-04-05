# Anima Control Training — Instruções para Claude Code

## Situação

Você é uma instância do Claude Code rodando numa máquina GPU alugada (Vast.ai/RunPod). Outra instância do Claude Code (na máquina local do usuário) fez toda a pesquisa e implementação. Seu trabalho é **configurar o ambiente e rodar treinamentos**.

### Histórico do projeto
- Pesquisa profunda com 11+ agentes sobre controle espacial para Anima
- **EasyControl**: implementado, treinado com sucesso (~3000 steps), inferência funcionando. Produz resultados bons. Nodes ComfyUI criados e publicados.
- **levzzz temporal concat**: implementado, funciona mas resultados inconsistentes
- **IC-LoRA**: implementação NOVA (branch `ic-lora`), ainda NÃO treinado. É o próximo treinamento a fazer.

### Sua máquina
- **GPU**: 1x RTX 5090 (32GB VRAM) ou similar
- **OS**: Ubuntu/Debian

---

## O repositório

```bash
git clone https://github.com/adbrasi/diffusion-pipe-easycontrol.git
cd diffusion-pipe-easycontrol
```

Fork do diffusion-pipe (https://github.com/tdrussell/diffusion-pipe) com 3 tipos de treinamento:

### Branch `main` — EasyControl + levzzz
- `models/easycontrol.py` — EasyControl (LoRA customizado com máscara binária, ~780 linhas)
- `models/cosmos_predict2.py` — Base Anima + levzzz temporal concat
- `infer_easycontrol.py` — Inferência para todos os modos
- `examples/anima_easycontrol.toml` — Config EasyControl
- `examples/anima_canny_lora.toml` — Config levzzz

### Branch `ic-lora` — IC-LoRA (NOVO)
- `models/ic_lora.py` — IC-LoRA pipeline (~140 linhas, herda de cosmos_predict2)
- `examples/anima_ic_lora.toml` — Config IC-LoRA
- `examples/anima_ic_lora_dataset.toml` — Config dataset

---

## IC-LoRA — O que é

IC-LoRA (In-Context LoRA) é uma abordagem mais simples que EasyControl:
- **Reference frame (T=1)**: imagem de referência, SEM noise (timestep=0)
- **Target frame (T=0)**: imagem alvo, COM noise (timestep=sigma)
- **Concat temporal**: `cat([noisy_target, clean_reference], dim=T)`
- **Per-token timestep**: `t = [sigma, 0]` shape (B, 2) — o modelo sabe qual frame é ref e qual é target
- **Loss**: APENAS no target (referência excluída)
- **LoRA**: PEFT padrão rank 32, sem máscara binária, atenção bidirecional completa

Baseado em: IC-LoRA (Alibaba), LTX-2 IC-LoRA trainer, Sync-LoRA.

---

## Para treinar IC-LoRA

### 1. Checkout da branch
```bash
git fetch origin
git checkout ic-lora
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
pip install deepspeed einops safetensors transformers accelerate peft datasets tqdm pillow opencv-python
```

### 3. Baixar modelos Anima
```bash
mkdir -p models_anima
huggingface-cli download circlestone-labs/Anima split_files/diffusion_models/anima-preview2.safetensors --local-dir models_anima
huggingface-cli download circlestone-labs/Anima split_files/vae/qwen_image_vae.safetensors --local-dir models_anima
huggingface-cli download circlestone-labs/Anima split_files/text_encoders/qwen_3_06b_base.safetensors --local-dir models_anima
```

### 4. Preparar dataset
Estrutura:
```
dataset/
  target_images/     # imagens alvo
    img001.png
    img001.txt       # caption
  condition_images/   # referências (canny, depth, etc.) — mesmo nome que target
    img001.png
```

Para gerar canny edges:
```python
import cv2, os, glob
for img_path in glob.glob('target_images/*.png'):
    img = cv2.imread(img_path)
    edges = cv2.Canny(img, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(f'condition_images/{os.path.basename(img_path)}', edges_rgb)
```

### 5. Editar configs
```bash
# Editar paths em:
nano examples/anima_ic_lora.toml          # paths dos modelos
nano examples/anima_ic_lora_dataset.toml   # paths do dataset
```

### 6. Rodar treinamento
```bash
python train.py --config examples/anima_ic_lora.toml
```

### 7. Inferência (após treinamento)
```bash
python infer_easycontrol.py \
  --dit models_anima/split_files/diffusion_models/anima-preview2.safetensors \
  --vae models_anima/split_files/vae/qwen_image_vae.safetensors \
  --llm models_anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --lora ic_lora_output/adapter_model.safetensors \
  --control_image canny.png \
  --mode ic_lora \
  --prompt "1girl, standing"
```

---

## Diferenças entre os 3 tipos de treinamento

| | EasyControl | levzzz | IC-LoRA |
|---|---|---|---|
| Config type | `easycontrol` | `anima` | `ic_lora` |
| Branch | `main` | `main` | `ic-lora` |
| LoRA | Customizado (máscara binária) | PEFT padrão | PEFT padrão |
| Noise | Ambos frames ruidosos | Ambos frames ruidosos | **Assimétrico** (ref limpo) |
| Timestep | Único + t_emb separado para cond | Único para ambos | **Per-token** (target=sigma, ref=0) |
| Attention | Causal mask | Bidirecional | Bidirecional |
| Loss | Target inteiro | Target + guard de shape | **Só target** (ref excluída) |
| Complexidade | ~780 linhas | 28 linhas | ~140 linhas |
| Status | ✅ Treinado, funciona | ✅ Funciona | ⏳ Não treinado ainda |

---

## Detalhes do modelo Anima
- 28 blocos transformer, 2048 dims, 16 heads, head_dim=128
- VAE: Qwen-Image, 16 canais, 8x downscale
- Text encoder: Qwen3-0.6B + LLM Adapter (6 layers)
- Flow matching: target = noise - latents (rectified flow)
- 3D RoPE (temporal + spatial) — suporta T>1 nativamente
- **bf16 obrigatório para treinamento** (fp16 pode causar NaN)

---

## Problemas conhecidos e soluções

### `batch_encode_plus` não encontrado
O Qwen3Tokenizer não tem `batch_encode_plus`. Já corrigido para `tokenizer(...)`.

### DeepSpeed launcher
```bash
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed --config examples/anima_ic_lora.toml
```
Se `deepspeed` não funcionar, tente `python train.py --config ...` direto.

### Git submodules
```bash
git submodule update --init --recursive
```

### OOM
- Reduzir `resolutions` para `[384]`
- Adicionar `blocks_to_swap = 10` no TOML
- Reduzir rank para 16

---

## Referências
- Fork: https://github.com/adbrasi/diffusion-pipe-easycontrol
- Custom nodes ComfyUI: https://github.com/adbrasi/comfy_control_nimanima
- EasyControl oficial: https://github.com/Xiaojiu-z/EasyControl
- IC-LoRA oficial: https://github.com/ali-vilab/In-Context-LoRA
- LTX-2 IC-LoRA trainer: https://github.com/Lightricks/LTX-2
- diffusion-pipe upstream: https://github.com/tdrussell/diffusion-pipe
- Anima: https://huggingface.co/circlestone-labs/Anima
