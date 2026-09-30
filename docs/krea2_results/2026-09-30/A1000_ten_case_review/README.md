# Review A/native step1000 — Turbo e Raw em 512

Treinos pausados a pedido do usuário; B está encerrado e preservado. A e B já chegaram a1000steps. Não continuar A74. Review atual:10pares held-out, mesma seed76 e prompt nos dois modos, apenas adapter treinado/referência correta;20outputs no total. Sem1024 neste lote. A geração Raw está em andamento; grid completo será adicionado no próximo commit.

Contrato A comprovado no **ComfyUI stock** commit `fb2315f11db0ebfaafa9099a5df5227dc6bb42bc`, sem custom node. Veja `workflow_Turbo.json` e `workflow_Raw.json`: payloads reais enviados à API, com nomes de modelos/LoRA da instância. Os arquivos `manifest_512.json` e os workflows preservam prompts/dimensões/seeds.512 significa área dos buckets:688x384 em9casos,416x624 em1; nenhuma imagem1024 é solicitada.

- Adapter: `A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000/adapter_model.safetensors`,512chaves,rank64; nome no Comfy: `A_native_fp8_512_micro2_probe_step1000.safetensors`,strength1.
- Base: `krea2_raw_fp8_scaled.safetensors`, loader default com escala. Treino computa BF16; armazenamento FP8scaled,swap0.
- Encoder: `qwen3vl_4b_bf16.safetensors`,type krea2; VAE `qwen_image_vae.safetensors`.
- Turbo: LoRA oficial `krea2_turbo_lora_rank_64_bf16.safetensors` strength1 + adapter A strength1,Euler8steps,CFG1,mu1.15.
- Raw: apenas adapter A,sem LoRA Turbo,Euler28steps,CFG5.5 (guidance4.5 na convenção Krea),mu dinâmica. Sigmas explícitos calculados por `tools/krea2_sampling.py`; não substituir por scheduler arbitrário.
- Texto positivo e negativo recebem a mesma imagem original para grounding; negativo com prompt vazio. Referência para VAE redimensionada ao bucket alvo,bicubic/cropcenter; `ReferenceLatent` em ambos e `FluxKontextMultiReferenceLatentMethod`=`index_timestep_zero`.
- Noise seed76 e latent vazio,no img2img. Nada de custom node,patch de base ou adapter B para avaliar A.

Reprodução: `tools/k2ab_eval_stock.py --adapter <adapter> --manifest <manifest_512.json> --limit 10 --adapter-only --variant Turbo --variant Raw --base-model krea2_raw_fp8_scaled.safetensors --reference-pixels target --out <out>`. Servidor stock local18819; Comfy do usuário8818 preservado. Builder `tools/k2ab_compare_variants.py` compõe imagemA|alvoB|Turbo|Raw com nomes/prompts.

Resultados originais: `/workspace/k2ab/artifacts/fp8_512_micro2/A1000_ten_case_review/sampling512/{Turbo,Raw}`. Backup contínuo privado: https://huggingface.co/AdwolfCzar/krea2-ab-runs . Adapters ficam em `adapters/`; resultados em `artifacts/`. Acesso depende da conta do usuário. Histórico e grids A/B250/500/750/1000 já estão no diretório pai deste review e em `docs/KREA2_AB_RUN_LOG.md`.
