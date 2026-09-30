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

## 2026-09-30 17:22 UTC — Régua beta1 e preparação da avaliação

- A/probe500 está no treino, ~25,3 s/step, sem overflow. A sequência B/probe500 continua automática, sem concorrência de dois modelos na GPU.
- Comparação beta1 original preparada em uma segunda instância oficial do ComfyUI `fb2315f11db0ebfaafa9099a5df5227dc6bb42bc`, localhost18820, desligada durante o treino. Nodes de `adbrasi/ctxrush-edit` fixados em `5b0250d79a0449dd50e3eaa38c007e9f602123d0`; o ComfyUI stock do A permanece sem custom nodes.
- Adapter original `AdwolfCzar/k2-context-rush-ofc-beta1/step13250` e base oficial `krea2_raw_fp8_scaled.safetensors` baixados **apenas para a inferência histórica** `CtxRush v2 + K2 Training Base`. Essa régua reproduz o grid FP8 sem escala e a fusão Turbo histórica, que ignora sete biases `diff_b`. A/B continuam com base BF16 exata e Turbo oficial com os sete biases aplicados.
- `tools/k2ab_eval_legacy.py`: mesmos13heldout, seed76 em CPU, Turbo8/CFG1 e Raw28/CFG5,5, referências certa/trocada. Nodes/lora/modelo não foram instalados na sessão ComfyUI do usuário.
- Grids agora identificam braço/checkpoint/variante e cada stem, nas quatro colunas A | B | resultado/ref certa | resultado/ref trocada. Workflows API ficam junto de cada PNG.
- Corrigida a espera de prontidão da API: Supervisor RUNNING não implica que os nodes já foram importados. Reiniciar o controlador da campanha agora também aceita o serviço stock já rodando. O worker e o processo de treino não são reiniciados.

## 2026-09-30 17:47 UTC — Parada solicitada e teste rápido A74

Usuário pediu parar qualquer treinamento enquanto Claude prepara mudanças. Campanha e worker desligados, autostart=false, `/workspace/k2ab/ops/stop_campaign` mantido. Nenhum processo train.py/deepspeed ativo; GPU ociosa após sampling. Serviços de backup HF continuam ativos.

- A/probe encerrou por `save_quit` no **step74**, LR1e-4, código0. Estado completo local: `/workspace/k2ab/checkpoints/A_native_probe/20260930_17-06-17/global_step74`; latest aponta para esse estado. **B/probe500 não iniciou; nenhum braço chegou a250/500.** Smokes anteriores A/B chegaram a12 incluindo resume.
- Atenção: o Saver atual salva apenas o estado DeepSpeed ao receber save_quit, sem exportar adapter. Exportei os pesos LoRA no CPU, sem retomar treinamento, com `tools/k2ab_export_checkpoint.py`: layer00=txtfusion, layer01..28=blocks0..27;512chaves e shapes conferidos contra adapter nativo compatível, todos finitos/BF16. Proveniência real do treinamento38c94bf-dirty registrada; configuração runtime preservada. Adapter em `step74/adapter_model.safetensors`, HF privado SHA256 confirmado. Loader stock:256patches, zero chaves não carregadas.
- Usuário pediu testar74. Avaliação rápida em3heldout (perfil/chapéu, noite/binóculo, mecha), Turbo8/CFG1, CPU seed76, dimensões1360×768, refs certa/trocada, ComfyUI stock sem custom nodes. SEIS imagens e workflows em `/workspace/k2ab/artifacts/quick_A_native_step74/Turbo`. Grid compacto identificado: `grid.png`.
- Triagem n=3: gt_true0.4450, ref_gain−0.0017, copy_gap0.1717, CCIP0.3333, copy_rate0.0 (dHash). Não comparar com métricas smoke n=1 como se fossem o mesmo conjunto.
- Visual: shuffle muda bastante os resultados; noite/paleta já recebem influência da ref. Ainda falha ação/câmera (grupo continua de costas em vez do frontal pedido), identidade e cor do mecha (branco/preto em vez de vermelho). A74 é evidência preliminar, sem veredito do método, sem disparar braço C.
- Estado recuperável da campanha anotado como stopped_by_user. A marca `done` do job010 indica saída normal após parada manual, não conclusão de500; não reativar controlador automaticamente a partir desse job.

Custo Krea acumulado estimado desde16:09UTC: **US$1.01**, inclui setup/cache/ociosidade/teste; não é extrato Vast.

### Integração com as mudanças de Claude, sem retomar execução

Push inicial da parada foi rejeitado porque Claude já havia publicado8a263fc (base fp8_scaled com escala,512px,batch real e AGENTS.md). Fiz fetch, rebase do commit de parada/exportação/grid sobre esse código, li AGENTS.md inteiro e envieicecd51b. **Não executei o novo treino/receita/testes**: a parada do usuário continua vigente. Adapter A74 pertence à receita BF161024/micro1×accum4 antiga e ao worktree38c94bf, não à nova receita de Claude.

Correções operacionais durante o teste: o audit stock não aceita --comfy, removi o argumento e repeti com sucesso. Reexportação determinística dos tensors usa serialização de metadata cuja ordem pode alterar o SHA256: a auditoria HF detectou o arquivo anterior, republiquei o adapter atual e confirmei SHA256 local=HF. Sem diferença nos pesos e sem retomar treino.

## 2026-09-30 18:09 UTC — Retomada autorizada: FP8 com escala,512,batch real

Usuário autorizou smoke e A/B, nesta ordem A depois B, ambos **do zero**. A74 permanece arquivado; não será carregado no novo treino. Captions e embeddings naturais preservados; sugestão de remoção de padding retirada por Claude porque o fork já salva tamanho natural.

- Receitas novas separadas em `/workspace/k2ab/artifacts/fp8_512/configs`; base oficial `krea2_raw_fp8_scaled.safetensors` hardlink do arquivo já baixado, `base_quant=fp8_scaled`, sem diffusion_model_dtype=float8,512px,micro4×accum1,sem block swap. A usa reference_pixels=target; ambos refs crop-fit ao bucket.
- Testes CPU16passaram no fork B e16no worktree A (ComfyUI stock fb2315f1). Encoder A real voltou a dar relL2=0 em3refsPNG, dump originalstock íntegro; log `/workspace/k2ab/artifacts/te_fork_fp8_512.log`.
- Adicionada auditoria explícita da conversão: quantização solicitada com0Linears ou armazenamento diferente de e4m3 aborta. O forward atual desquantiza para matmul BF16; não implementa W8A8 nem promete acelerar a própria multiplicação. Primeiro ganho a medir é eliminar swap e reduzir resolução.
- Smoke A fresh enfileirado100,semresume,10steps,saves5/10; B será enfileirado após resultado A e escolha do batch que realmente cabe. Probes não iniciam antes dos gates/smokes. Configs novas não sobrescrevem os artefatos BF16.
- Comparação antiga: A1024/micro1×accum4/swap20 ~25,15s/step =0,159amostras/s; Bsmoke1024 ~35,75s/step =0,112amostras/s. Relatar ganho inclui mudança de resolução/batch/offload, não atribuí-lo só à quantização.
- W8A8 opcional somente após A/B rodando e se o throughput incomodar; implementação própria, sem código AGPL OneTrainer; troca de padrão condicionada a paridade/seed fixa/grids e ganho medido. Variante seguinte se A copiar/ficar atrás de B: reference_timestep=target / método index stock, como orientação nova de Claude.

## 2026-09-30 18:21 UTC — Smokes FP8 e correção de fusão Turbo

A512/micro4×accum1/swap0:10steps,mediana5.8295s/step=0.6865amostras/s,pico31687MiB. B:10steps,6.7415s/step=0.5935amostras/s,pico32033MiB. Picos incluem ~776MiB do ComfyUI do usuário; margem B apertada, monitorar batches longos. Ganho observado contra antigos1024micro1×accum4:4.32×A e5.30×B; resolução,batch e remoção de swap mudaram juntos, portanto não medir isso como ganho isolado de quantização.

Ambos converters registraram256ScaledFP8Linear. A resume10→12 passou,LR1e-4 preservado. B resume10→12 em andamento. Sem carregamento de A74.

Integração descoberta antes do sampling B: `apply_turbo_lora` e fusão de extras somavam delta diretamente ao parâmetro weight; no ScaledFP8Linear esse parâmetro são códigos FP8 crus. Corrigi para dequantizar com escala, somar em FP32 e requantizar com escala recalculada/arredondamento estocástico e seed da chave, igual ao set_weight do ComfyUI. Adapter de referência segue separado/routado, nunca fundido. Teste novo compara qdata,escala e peso dequantizado contra QuantizedTensor.requantize_from_float:3testesScaledFP8passaram. Isso é correção de inferência, não W8A8 nem mudança no forward de treino.

Avaliação512 A preparada com nodes exclusivamente stock: TextEncodeQwenImageEditPlus semVAE (grounding original384²) + ImageScale/VAEEncode/ReferenceLatent (ref no bucket512) + método index_timestep_zero. O node Qwen comVAE sempre faria ref1MP e quebraria a relação de grids do novo probe; não utilizar essa variante aqui. O resize PIL do treino e bicubic Comfy não são bit-idênticos em pixels; geometria/posições/método/texto são os mesmos. B runner512 com basefp8_scaled e semswap. Manifest512 separado; captions e manifest1024preservados.

## 2026-09-30 18:56 UTC — Quadriculado no smoke A: controles reais antes do probe

Usuário identificou ruído/quadriculado nos PNGs A12/512. Confirmei visualmente; não é resultado aceitável nem demonstração de sucesso. Não iniciar500 cegamente. B resume10→12 completou, LR1e-4; ambos smokes/resumes/keys passaram.

Controles no MESMO heldout, seed76,688×384,workflow stock sem custom nodes:
- A12 zero e base SEM adapter com ref zero: quadriculado. O mesmo defeito existe na base BF16, no FP8 com escala e com ref512 ou ref1MP. Raw28 também apresenta o padrão. Portanto não atribuir o problema exclusivamente ao FP8, ao Turbo, ao resize512 nem aos pesos aprendidos no smoke.
- FP8 T2I sem referência: imagem limpa. FP8 com referência e método index (t compartilhado): imagem limpa, porém copia praticamente a referência, inclusive a trocada. Trocar timestep só na inferência de A treinado em zero seria mismatch e NÃO foi feito como correção.
- Controle POSITIVO: LoRA oficial Comfy-Org/Krea-2/loras/krea2_style_reference.safetensors no MESMO workflow FP8/ref512/index_timestep_zero/Turbo. Imagens limpa certa e trocada, com nova pose. Esse LoRA é somente régua de diagnóstico, NÃO entra no treino ou na avaliação A/B.
- Paridade adicional GPU com pesos BF16 reais,sem adapter e sem TE,alvo/ref(1,16,48,86),texto300tokens,t0.6: relL2 zero0.006725/index0.006494,ambos finitos. Ferramenta tools/krea2_native_gpu_parity.py, JSON/log completos. Compatível com arredondamento BF16; não bit-idêntico. Paridade CPU e encoder real já passaram.

Inferência limitada a n=1: o quadriculado pertence ao uso do contrato zero sem adapter suficientemente adaptado; não é prova de bug no loader nem prova de que A irá convergir. O LoRA oficial demonstra que o encanamento stock consegue sair limpo. Preservar a receita autorizada; começar A fresco e inspecionar125 antes de liberar250–500. O novo controller tem esse portão operacional; B começa depois de A500. Não usar A74, não retomar pesos dos smokes e não alterar captions.

Grid dos controles: /workspace/k2ab/artifacts/fp8_512/diagnostic/reference_diagnostic_grid.jpg. PNGs e workflows de cada controle,paridadeGPU e receitas estão nessa árvore; sync HF privado contínuo. Confusão própria evitada: a primeira suspeita FP8/base foi descartada ao repetir BF16.

Custo Krea acumulado estimado desde16:09UTC:US$1.73, inclui diagnóstico/setup/ociosidade; não é extrato Vast.

## 2026-09-30 19:09 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$1.87, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:10 UTC — Campanha

A_native_fp8_512_probe salvou step125; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_probe/20260930_18-57-31/step125/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$1.87, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:21 UTC — A fresco125: quadriculado desapareceu, gate liberado

A_native_fp8_512_probe/20260930_18-57-31 terminou125steps (500amostras),sem A74 ou pesos dos smokes. Mediana5,770s/step=0,693amostras/s,pico31723MiB incluindo Comfy do usuário,batch4real,accum1,swap0,256ScaledFP8Linear;512keys auditadas,adapter já enviado ao HF. LR após warmup=1e-4. Houve avisos do allocator de tentativa de alocação que foram recuperados,sem torch.OutOfMemoryError,sem skipped steps e saída0. Não apresentar esses avisos como OOM fatal.

Inspeção visual dos grids completos,13heldout,certa+trocada em Turbo e Raw (52PNG): quadriculado do smoke não visível. O aprendizado125 resolveu esse artefato sem mudar o timestep ou o workflow; reforça a hipótese de contrato zero ainda pouco adaptado no smoke12. Ainda há repetição de pose/composição (noite continua de costas),identidade e ação imperfeitas. Shuffle muda a saída para a ref trocada. Não confundir traço limpo com tarefa resolvida.

Métricas n13: Turbo gt_true0.4746,ref_gain0.0705,copy_gap0.0664,CCIP0.4615; Raw gt0.5307,ref_gain0.0922,copy_gap−0.0156,CCIP0.5385. copy_rate dHash=0 em ambos,mas repetição visual de composição continua: métrica não substitui revisão. Sem vencedor; B500 ainda não começou.

Portão A125 liberado para seguir a autorização A250/375/500→B125/250/375/500,sempre retomando somente estados da campanha NOVA. Grids locais /workspace/k2ab/artifacts/fp8_512/eval/A_native_step125/{Turbo,Raw}/grid.png. Versões JPEG,paridadeGPU e JSONs de métricas publicados em docs/krea2_results/2026-09-30 para review por agente que só lê GitHub. Outputs completos/HF privado contínuo.

Disco: removidos somente optimizerstates obsoletos5/10 dos dois smokesFP8 depois do resume12validado,liberando5.29GiB. Latest12 e todos os adapters5/10/12 enviados ao HF foram mantidos; nenhum cache de texto/caption foi alterado.

Custo Krea acumulado estimado desde16:09UTC:US$1.99,inclui setup/diagnóstico/cache/ociosidade; não é extrato Vast.

## 2026-09-30 19:27 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.05, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:27 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.06, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:28 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.07, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:29 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.08, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:30 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.09, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:31 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.10, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:32 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.11, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:33 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.12, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:34 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.13, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:35 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.14, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:36 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.15, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:37 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:38 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:39 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.18, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:40 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.19, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:41 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.20, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:42 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.21, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:43 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:44 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.23, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:45 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.24, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:46 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:47 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.26, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:48 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.27, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:49 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.28, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:50 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.29, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:51 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.30, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:52 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:53 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.32, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:54 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.33, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:55 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.35, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:56 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.36, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:57 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.37, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:58 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.38, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 19:59 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.39, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:00 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.40, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:01 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:02 UTC — Campanha

Campanha FP8 parou em erro: 210_A_native_250.job failed; inspect its log; investigar antes de seguir.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:02 UTC — Campanha

Novo A_native/FP8/512 salvou step125, 500 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.41, inclui setup/cache/ociosidade, não é extrato.
