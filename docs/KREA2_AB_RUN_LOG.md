# Krea 2 A/B — execução GPU

## 2026-09-30 16:09 UTC — Setup

Usuário autorizou executar o handoff: A/krea2_native × B/beta1_fixed, 500 steps, saves250/500, smoke antes dos probes, base bf16 exata e paridade forward/TE antes dos caches. Git pull sem alterações pendentes; documentos lidos na ordem solicitada. Anima fechado; 63,88GiB liberados, 30 adapters preservados e SHA256 idêntico ao HF privado. Modelos Krea2 confirmados pelo API do Comfy-Org/Krea-2: raw_bf16 26.28GB, Qwen3VL4B8.88GB, turboLoRA0.469GB; VAE0.254GB compartilhado e preservado. Stock ComfyUI separado fixado em fb2315f1 para paridade/avaliação. Não treinar antes dos portões.

## 2026-09-30 16:25 UTC — Portões e integração

Forward CPU passou nos modos index_timestep_zero e index (11 testes iniciais). Correções: bootstrap dos imports do pipeline nativo; máscara de atenção convertida a bool no empacotamento; ferramenta te-stock isolada dos imports de treino para não colidir com utils do Comfy stock. B agora aplica rank_pattern/alpha_pattern ANTES do único get_peft_model: evita segunda injeção PEFT depois de instalar o router. Teste adicional confere ranks txtfusion/blocos e delta apenas na referência.

Encoder antigo 0ba903bd versus stock fb2315f1: relL2 0.305/0.945/1.12 em três PNGs, gate FALHOU (limite 0.02). Nenhum cache criado. Próximo passo é worktree A com stock fb2315f1; B mantém encoder histórico. Arquivos completos em /workspace/k2ab/artifacts/te_{stock,fork_old}.log e te_stock.pt.

Subconjunto: 1.500 pares SFW, rating safe >=0.7 nas DUAS imagens e captions sem conteúdo explícito, retirados dos pares Anima já filtrados/sem heldout. Manifesto seed42; uma legenda sorteada por par, repetida identicamente entre A/B. ds1=50, ds2=1060, ds3=31, ds4=359. R18=0. Seleção proporcional ao pool SFW elegível, para evitar o viés NSFW histórico; difere da mistura original e limita comparação ao domínio SFW. Hardlinks target/control e refs numeradas; sem cache TE/VAE até gates.

HF privado AdwolfCzar/krea2-ab-runs criado e sync contínuo via supervisor k2ab_sync ativo. Base/TE/Turbo BF16 baixados; VAE compartilhado. Artefatos locais em /workspace/k2ab/artifacts.

## 2026-09-30 16:30 UTC — Paridade A aprovada, smoke liberado

Worktree /workspace/k2ab/native_worktree em 79e7a3a com submodule ComfyUI stock fb2315f1; demais submodules reutilizados via symlink, root/B continua 0ba903bd. TE real: relL2=0 nas três referências PNG. Forward/contrato/routing: 12 testes passando no worktree atualizado. Outros testes: 28 passaram, uma fixture histórica sem position_mode foi corrigida e seu arquivo passou (2/2). Prova dos ranks/routing do B incluída no teste novo.

Subconjunto contém seis buckets AR (0.5/0.63/0.79/1.26/1.59/2.0; sem imagens quadradas elegíveis). Smoke com quatro pares por bucket, 24 pares; 10 steps, saves/resume states em 5/10; BF16, swap16, micro1×accum4. Worker serial supervisor k2ab_worker iniciado. Configs e monitor de GPU em /workspace/k2ab/artifacts; nenhum probe de 500 enfileirado antes dos smokes.

## 2026-09-30 16:56 UTC — Smokes e resumes aprovados

Bug GPU no A: collator recebe batches de acumulação (4 amostras) ANTES do split em micro1, e ref tem grids distintos. Prepare_inputs_per_microbatch passou a preparar cada microbatch antes de empilhar refs/pad de texto. Evita tanto stack inválido quanto padding de texto diferente do stock. Teste reproduz referência variável e texto de comprimentos diferentes; demais pipelines preservam caminho antigo. Gate completo: 30 testes passaram.

A10: 25.03s/step mediano, pico31759MiB (swap16). A resume até12 com swap20:25.33s/step, pico26293MiB, LR1e-4 preservado. B10:35.75s/step, pico27216MiB (swap20); B resume até12:36s/step, pico30176MiB, LR1e-4. Ambos 512 chaves permitidas; B224 block linears routados +32 fusion linears r128. A stock LoraLoaderModelOnly:256 patches, ZERO chaves não carregadas; sampling Turbo certa/trocada completo. B Turbo runner completo,8steps CFG1, CPU seed76 igual ao stock. Tempo previsto só compute500: A3h29/B4h58; eficiência e custo serão registrados com tempos reais, sem troca para float8.

Runner Turbo corrigido para aplicar também os 7 diff_b oficiais (antes ignorados);264 LoRAs+7bias=271 módulos/patches. Nova opção noise-device=cpu preserva default histórico cuda e permite seeds iguais ao Comfy stock. Avaliação em lote reutiliza DiT e encoder, sem reload do arquivo por imagem. Stock usa ManualSigmas nativo com os exatos timesteps do runner, evitando diferença de schedule do ModelSamplingFlux por resolução.

Heldout:13 pares neutros revisados visualmente,1024px com AR do target, seed76, incluindo hat/night binocular/red mecha. Review A/B em /workspace/k2ab/artifacts/heldout_neutral_review.jpg. Rating automático rejeitou frames noturnos neutros: mantidos após inspeção visual; captions explicitamente sexualizadas excluídas. Zero interseção de stems entre treino e TODOS os24heldout originais. Método de referência ablation em andamento. Outputs e métricas smokes em /workspace/k2ab/artifacts/smoke; evidência de integração, não ranking de qualidade.

Custo Krea estimado desde16:09UTC até16:56UTC:US$0.49 (47min aUS$0.62/h, inclui setup/ociosidade); acumulado da instância desde06:05UTC:US$6.73. Não é extrato da Vast.

## 2026-09-30 17:07 UTC — Probe A iniciado

A500 do zero iniciado após gates e smokes; native worktree atualizado para38c94bf com stockfb2315f1,13testes nativos passaram. Config runtime /workspace/k2ab/artifacts/configs/A_native_probe.toml:swap20;demais parâmetros da receita preservados. Cache concluído;373steps/época,1492amostras após arredondamento dos buckets;500steps≈1.34época. OITO pares dos1500não entram no loader por divisibilidade/AR. Probe B terá os mesmos targets/captions/buckets.

Ablation com/sem FluxKontextMultiReferenceLatentMethod: saídas diferentes,MAEpixel0.2842,same seed/prompt, prova de caminho VAE ativo. Arquivos /workspace/k2ab/artifacts/smoke/A_native_no_method. Smokes foram pontuados em apenas UM par e não servem para selecionar método; grids de4colunas locais. A30testes+worktreeA13testes passaram.

Controller serial supervisor k2ab_campaign:segueA500→B500→A250/500stock→T2Ibase→B250/500runner→metrics. C só após revisão visual/evidência; não enfileirado. Stop_campaign interrompe agendamento futuro; trainer ativo deve sair via save_quit. Mantém dois optimizerstates recentes e latest;smokes5/10states obsoletos removidos depois do resume12validado. Checkpoints/outputs preservados. Cache A só será removido se o disco não comportarB (reprodutível,dados/configs mantidos). HFupload contínuo ativo.
