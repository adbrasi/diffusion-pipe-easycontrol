# Anima NextScene para ComfyUI

Nodes instalados em `Anima/NextScene`:

- **Anima NextScene — Adapter**: recebe o MODEL do UNETLoader nativo, aplica o LoRA e lê automaticamente a geometria do checkpoint. `strength=1` e `reference_guidance=1` são os defaults E2; strength zero mantém apenas base + geometria.
- **Anima NextScene — Reference**: recebe positive/negative e o LATENT de VAEEncode da imagem A; fornece os condicionamentos ao KSampler. `image/keep` é o default; `null` usa zeros depois da normalização Wan, e `zero` remove a referência apenas do negativo.

Use os loaders nativos: `anima-base-v1.0.safetensors`, `qwen_3_06b_base.safetensors` (CLIPLoader stable_diffusion), `qwen_image_vae.safetensors`. Redimensione a imagem antes de VAEEncode para o mesmo tamanho do EmptyLatentImage. Comece com Euler, simple, 20 steps, CFG 4, denoise 1.

LoRAs em `models/loras/anima_nextscene/`: A = aligned; B = disjoint_w. Os finais E2 são `step5685_epoch1`. O loader resolve o symlink para ler também `adapter_config.json` e preservar alpha/rank. Não aplique o mesmo LoRA novamente com outro loader.

Instalação nesta máquina: `python tools/install_nextscene_comfy.py`. Não altera o core ComfyUI. Usa ModelPatcher, condicionamento normal, operações e sampler nativos. Não precisa de dependências novas no ambiente ComfyUI.

A instância temporária `comfy_nextscene` usada no smoke foi parada e teve autostart desativado a pedido do usuário. A configuração original da porta 8818 foi restaurada; inicie seu ComfyUI pelo Arrakis habitual. Outputs em `/workspace/comfy/ComfyUI/output/NextScene/`, sincronizados pelo serviço externo `nextscene_sync` ao HF privado existente. Worker de treinamento parado e autostart desativado.

Validado em ComfyUI 0.38.0 / RTX 5090: A e B a 512²/20 steps, 280 lineares LoRA carregadas integralmente, null, reference guidance 1.5, negativo zero, strength zero e batch 2. RoPE target igual à nativa e offset da referência igual ao treino. Workflows opcionais e históricos de validação em `/workspace/nextscene_artifacts/ComfyUI/`.
