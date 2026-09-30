# Krea 2 A/B — execução GPU

## 2026-09-30 16:09 UTC — Setup

Usuário autorizou executar o handoff: A/krea2_native × B/beta1_fixed, 500 steps, saves250/500, smoke antes dos probes, base bf16 exata e paridade forward/TE antes dos caches. Git pull sem alterações pendentes; documentos lidos na ordem solicitada. Anima fechado; 63,88GiB liberados, 30 adapters preservados e SHA256 idêntico ao HF privado. Modelos Krea2 confirmados pelo API do Comfy-Org/Krea-2: raw_bf16 26.28GB, Qwen3VL4B8.88GB, turboLoRA0.469GB; VAE0.254GB compartilhado e preservado. Stock ComfyUI separado fixado em fb2315f1 para paridade/avaliação. Não treinar antes dos portões.

## 2026-09-30 16:25 UTC — Portões e integração

Forward CPU passou nos modos index_timestep_zero e index (11 testes iniciais). Correções: bootstrap dos imports do pipeline nativo; máscara de atenção convertida a bool no empacotamento; ferramenta te-stock isolada dos imports de treino para não colidir com utils do Comfy stock. B agora aplica rank_pattern/alpha_pattern ANTES do único get_peft_model: evita segunda injeção PEFT depois de instalar o router. Teste adicional confere ranks txtfusion/blocos e delta apenas na referência.

Encoder antigo 0ba903bd versus stock fb2315f1: relL2 0.305/0.945/1.12 em três PNGs, gate FALHOU (limite 0.02). Nenhum cache criado. Próximo passo é worktree A com stock fb2315f1; B mantém encoder histórico. Arquivos completos em /workspace/k2ab/artifacts/te_{stock,fork_old}.log e te_stock.pt.

Subconjunto: 1.500 pares SFW, rating safe >=0.7 nas DUAS imagens e captions sem conteúdo explícito, retirados dos pares Anima já filtrados/sem heldout. Manifesto seed42; uma legenda sorteada por par, repetida identicamente entre A/B. ds1=50, ds2=1060, ds3=31, ds4=359. R18=0. Seleção proporcional ao pool SFW elegível, para evitar o viés NSFW histórico; difere da mistura original e limita comparação ao domínio SFW. Hardlinks target/control e refs numeradas; sem cache TE/VAE até gates.

HF privado AdwolfCzar/krea2-ab-runs criado e sync contínuo via supervisor k2ab_sync ativo. Base/TE/Turbo BF16 baixados; VAE compartilhado. Artefatos locais em /workspace/k2ab/artifacts.
