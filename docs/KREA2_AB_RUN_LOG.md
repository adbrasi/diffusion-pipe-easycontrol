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

## 2026-09-30 20:32 UTC — Falha fatal no batch4 e erro próprio do controlador

Após A125validado e retomada normal, o treino chegou ao163 e FALHOU no backward do164 às19:27:28. Erro cuDNN SDPA: memória CUDA fora do allocator PyTorch insuficiente para shape/JIT. É OOM fatal, diferente dos avisos recuperados anteriores. Último estado seguro125. B ainda não começou.

Erro operacional meu: configurei autorestart=unexpected, mas o controlador tratava erro terminal com raise e, na reinicialização, não respeitava status failed. Assim repetiu o job já marcado failed,publicou notas/commits repetidos e manteve GPU OCIOSA até a pergunta do usuário20:24. Cerca de57min (~US$0.59) de ociosidade paga desnecessária. Parei o controlador; não houve avanço oculto nem treino B. Desculpa não substituir isso por uma narrativa de progresso.

Corrigido: status failed/concluído é terminal; captura de falha grava JSON e encerra com exit0 esperado pelo supervisor; milestones de segmento são idempotentes. Foram condensados1164blocos de notas repetidas, mantendo log integral em /workspace/k2ab/artifacts/fp8_512/controller_restart_loop_full_log.md e o histórico git sem reescrita/forcepush.

Fallback autorizado por AGENTS/handoff: microbatch2 REAL,accum1,512,scaledFP8,sem swap. Dois braços novos do zero,sem A74,smokes ou A125carregados,para A/B homogêneo.500steps×2=1000amostras por braço (não2000). Novo namespace fp8_512_micro2; experimento4 arquivado. Primeiro smoke10 +resume12 em ambos,com duas legendas EXISTENTES mais longas por cada um dos6buckets AR. Não alterar captions da fonte. Depois A500→B500,grids/metrics a cada125.

Custo Krea acumulado estimado desde16:09UTC:US$2.73,inclui a falha/ociosidade; não é extrato Vast.

## 2026-09-30 20:40 UTC — Stress smoke micro2 aprovado e probe fresco liberado

Smokes10 em12pares (2legendas existentes mais longas por cada6buckets),micro2×accum1,512,scaledFP8,swap0: A mediana2.8470s/step=0.7025amostras/s,pico25747MiB; B 3.2925s/step=0.6070amostras/s,pico25985MiB. Picos incluem776MiB da sessão Comfy do usuário,que não foi interrompida. Nenhum erro fatal. Ambos resume10→12,LR1e-4,512keys auditadas. StockLoader A real:256patches,zero unloaded keys.

Teste operacional: lançar controlador antigo em estado failed termina com código0 e não agenda/commita; não há restart loop. Novo controller distingue namespaces/batches e idempotência por estágio. Micro2 usa o mesmo contrato de inferência já validado nos52PNGs A125/micro4,sem outro portão manual duplicado; geração Turbo/Raw e métricas continuam a cada125. Probes micro2 começam do zero,500steps=1000amostras cada,~0.67época,sem nenhum adapter antigo ou smoke carregado. Captions fonte SHA256 inalterado. Marcadores de parada anteriores revogados pela autorização de retomada; caminho antigoA74 continua apenas arquivo.

Artefatos novos: /workspace/k2ab/artifacts/fp8_512_micro2; checkpoints em /workspace/k2ab/checkpoints/{A_native,B_beta1_fixed}_fp8_512_micro2_probe. Ordem A500→B500; backups HF ativos.

## 2026-09-30 20:47 UTC — Campanha

Novo A_native/FP8/512/micro2 salvou step125, 250 amostras; nenhum peso do A74 utilizado. Avaliando antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$2.87, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 20:47 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step125; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step125/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$2.88, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:09 UTC — Auditoria Raw e comparação de configuração, A125 micro2

Pedido do usuário: investigar Raw abaixo do Turbo, incluindo CFG. Conferido contra krea-ai/krea-2 commit db3984fbc6e13b34c0064990fc2d95ac64d00058, sampling.py e README, e a documentação oficial Diffusers. CFG Krea usa cond+guidance*(cond-uncond); Comfy usa uncond+cfg*(cond-uncond). Logo guidance4.5 Krea = CFG5.5 Comfy, sem erro de off-by-one. Raw28/5.5 corresponde aos defaults oficiais; README também apresenta Raw52/guidance3.5 (=CFG4.5 Comfy). Turbo8/CFG1 é correto.

13 workflows auditados: Raw sem TurboLoRA, mesma referência no conditioning positivo/negativo, método index_timestep_zero, scheduler dinâmico pelo número de tokens do alvo. Divergência máxima dos sigmas vs implementação oficial8.46e-8; fórmula de guidance1.91e-6 (float32). Sem evidência de erro de CFG/schedule ou Turbo aplicado no Raw.

Sweep stock, checkpoint A125 NOVO micro2, seed76/prompt/referência/Euler/baseFP8scaled idênticos, três casos críticos (perfil, noite/binóculo, mecha): Raw28 CFG1/3/4.5, Raw52 CFG4.5 e Raw52 CFG4.5 mu1.15; comparados aos baselines Turbo8/CFG1 e Raw28/CFG5.5.15 imagens novas,309.54s (~US$0.053 GPU pelo tempo do job). Cada PNG acompanha workflow API JSON; manifest contém os sigmas/mu. Grid em /workspace/k2ab/artifacts/fp8_512_micro2/raw_settings_check/A_native_step125/grid.jpg, cópia versionada docs/krea2_results/2026-09-30/A125_micro2_Raw_settings_grid.jpg.

Leitura visual limitada a n=3: CFG1 degradou o mecha para a cena larga de fogo; CFG3 manteve melhor tons quentes nesse exemplo mas mudou o desenho.52steps não melhorou consistentemente a28; mu1.15 pouco mudou52. Nenhuma alternativa resolveu a falta de mudança de pose na cena noturna. Não selecionar receita a partir desse n pequeno: manter Raw28/5.5/dynamicmu para comparabilidade A/B; não afirmar que Turbo superior prova bug ou treino ruim. Turbo continua distinto por sua LoRA de destilação.

Erro operacional meu nesta auditoria: a pausa do controlador parou stock enquanto o job de avaliação A125 ainda estava ativo, deixando3 imagens Raw faltantes. Snapshot da falha preservado, job reexecutado sem sobrescrever49 imagens existentes,3 faltantes recuperadas,52PNGs e métricas completos. Corrigido wait_job: flag de pausa aguarda o job ativo terminar antes de desmontar o serviço; estado stopped terminal e SystemExit normal evitam loop no supervisor. Teste com job temporário running→done comprovou espera e saída stopped. Sem treino perdido.

Retomar A micro2 do seu próprio125 até500, depois B fresco500. Proibido continuar A74; parâmetros de treino/captions intocados. Custo Krea acumulado estimado desde16:09UTC:US$3.11, inclui setup/cache/ociosidade/auditoria, não é extrato Vast.

## 2026-09-30 21:16 UTC — Campanha

Novo A_native/FP8/512/micro2 salvou step250, 500 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$3.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:16 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step250; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step250/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.17, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:16 UTC — Sampling reduzido conforme pedido do usuário

Novo contrato explícito: apenas Turbo nos steps250 e500 de A eB. Só comparação COM/SEM adapter treinado, mesma imagem de referência, prompt e seed; sem shuffle e sem Raw. Dois casos fixos (noite/binóculo e mecha),2 versões cada=4PNGs por checkpoint. A125 já produzido preservado. Manifest reduzido separado heldout_manifest_quick.json, sem alterar captions/dados/manifest original. Grids A|B alvo|com LoRA treinada|sem LoRA treinada. Métrica da diferença entre versões chama-se adapter_gain; não inventar ref_gain sem shuffle. Baselines extras T2I/beta1original desativados nesta campanha para economizar.

Stock A: versão sem adapter omite nó LoRA treinada, mantém Turbo oficial e conditioning da mesma imagem. Runner B: contexto PEFT disable_adapter cobre todos os módulos treinados, DiT e text-fusion; mantém Turbo oficial fundida e o mesmo contrato B. Roteadores existentes respeitam disable_adapters. Teste CPU confirmou bypass/restauração PEFT; teste dos graphs stock confirmou condicionamentos, noise e schedule iguais, só LoRA treinada difere.

Treino não interrompido: só scheduler reiniciado,worker/GPUjob A250 permaneceu rodando. Próximo segmento A250→500 direto (sem pausa375), checkpoints de recuperação/upload continuam125. B250 inicia fresco e segue até500. Dry-run de agendamento validou reutilização do job A250, B fresco, intervalos corretos e ausência de Raw/baselines.

Esclarecimento ao usuário: ~0.72 no log era AMOSTRAS/s; microbatch2 resulta~0.36steps/s (~2.8s/step,~23min/500steps computação). Além do treino, A125 consumiu52PNGs Turbo/Raw e auditoria adicionou15Raw; também houve inicialização/cache e incidentes previamente registrados. Redução atual elimina a maior parte do sampling.

## 2026-09-30 21:23 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step375; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step375/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.25, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:23 UTC — Apenas adapter treinado

Usuário retirou também a geração sem adapter. Campanha agora Turbo8/CFG1, com adapter treinado e referência correta APENAS, steps250/500. Mantidos dois casos fixos:2imagens por checkpoint. Sem shuffle, sem versão sem adapter, sem Raw nem baselines extras. Arquivos históricos preservados para rastreabilidade. Grid novo3colunas A|B alvo|resultado com LoRA. Métricas gt_true,copy_gap,copy_rate,CCIP; sem ref_gain/adapter_gain porque suas contrapartes não serão geradas.

Artefatos sem adapter já tinham aparecido em stock sem adapter na investigação anterior; Turbo oficial e base quantizado/conditioning continuam ativos nessas imagens. Causa exata não isolada, não atribuir automaticamente ao treino/quantização. Não gastar GPU em novo diagnóstico contrário à redução pedida.

Treino A500 permaneceu ativo durante ajuste; só scheduler reiniciado. Suporte --adapter-only stock/runner e metrics validado por compile e teste funcional3imagens de entrada A/B/resultado, sem contraparte, grid3colunas e ausência de gains não calculáveis. Publicar código/log, manter upload HF contínuo.

## 2026-09-30 21:29 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step500; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step500/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:29 UTC — Campanha

Novo A_native/FP8/512/micro2 salvou step500, 1000 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$3.31, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:40 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step125; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step125/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.43, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:47 UTC — Campanha

Novo B_beta1_fixed/FP8/512/micro2 salvou step250, 500 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$3.50, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:47 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step250; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step250/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.50, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:57 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step375; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step375/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.60, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 21:58 UTC — Meta1000steps e sampling512/1024

Usuário ampliou CADA braço até1000steps e pediu sampling a cada250 na resolução de treino e1024. A500 concluído; B continuava seu segmento250→500 (~step334 ao início da alteração). Não interromper/descartar esse treino: após completar o segmento ativo, backfill das avaliações1024 e continuação dos mesmos runs A500→750→1000, B500→750→1000. Sem A74/smokes ou restart do zero.1000steps×micro2=2000amostras (~1.34épocas) por braço. Max_steps dos configs-base elevado1000; LR1e-4, warmup50, captions, dataset,512,micro2,accum1,rank64,FP8scaled/storage+BF16compute,swap0 inalterados.

Sampling: Turbo8/CFG1/mu1.15, dois casos (noite/binóculo e mecha), somente LoRA treinada/referência certa, checkpoints250/500/750/1000. Bucket512 tem688×384;1024 significa dobra das dimensões=1376×768, mantendo AR e~1MP.4imagens por avaliação no total, sem shuffle/versão sem adapter/Raw. Manifest1024 separado,mesmo prompt/ref/seed.512 existente reaproveitado;1024 também será feito para250/500 já disponíveis. Outputs maiores em eval/<arm>_step<N>/resolution_1024/Turbo;512 continua eval/<arm>_step<N>/Turbo.

Dry-run validou16jobs de avaliação (2braços×4saves×2resoluções),continuação do run correto, geometria exatamente2×,seeds/prompts/ref iguais,Turbo+adapteronly em todos. Compilação ok; primeiro1024 real fica atrás do segmento B ativo na fila GPU serial,sem declarar paridade/VRAM1024 antes de executá-lo. Grids identificam treino512/sampling512ou1024,checkpoint,batch. Configs/manifest snapshots versionados em docs/krea2_results/2026-09-30/micro2_1000steps.

## 2026-09-30 22:04 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step500; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step500/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.67, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:12 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step625; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step625/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.76, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:18 UTC — Campanha

Novo A_native/FP8/512/micro2 salvou step750, 1500 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$3.82, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:18 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step750; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step750/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.82, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:18 UTC — Monitoramento após extensão1000

Pedido explícito do usuário: acompanhar até1000. A500/B500 salvos; A retomado no mesmo run500→750,LR1e-4 confirmado,~2.8s/step (~0.72amostras/s). Primeiras inferências1024 stock passaram: A250/1024: pico24305MiB,2PNGs1376×768 completos; A500/1024: pico24305MiB,2PNGs1376×768 completos. Grids respectivos confirmados no HF privado. Sem alterar receita ou captions.

Inspeção A500/1024: saída limpa de checkerboard; cena noturna preserva ambiente mas continua com três personagens de costas, apesar do prompt frontal. Mecha muda enquadramento, porém troca desenho/paleta; n=2 e mesma seed não permitem eleger método. Métricas DINOgt500/512=.4814,500/1024=.4911; não interpretar como herança completa. Sem ref_gain/adapter_gain por retirada expressa das comparações. Avaliações512/1024 previstas em750/1000 e B250/500/750/1000.

## 2026-09-30 22:20 UTC — A750: cópia na resolução de treino

Checkpoint A750 e4PNGs completos,512/1024,somente adapter treinado/Turbo. Inspeção visual:512 quase reproduz A nos dois casos (noite ainda de costas; mecha ainda cena larga de fogo, em vez do perfil pedido). dHash copy_rate=1.0 em n=2,copy_gap=.4367,DINOgt=.4520. Em1024 mecha muda para perfil,mas estilo/identidade divergem; noite continua próxima de A. dHash1024=0 não contradiz cópia visual do cenário: métricas isoladas não bastam. DINOgt1024=.5205,copy_gap=.1853,CCIP=.5; mesma seed/dois exemplos,sem generalizar para dataset inteiro.

Há sinal de polo de cópia no A nativo t_ref0 conforme hipótese do handoff. Manter a autorização atual1000 por braço para comparação homogênea; não iniciar automaticamente variante reference_timestep='target' nem modificar LR/captions. A750→1000 retomado no mesmo run. Grids em /workspace/k2ab/artifacts/fp8_512_micro2/eval/A_native_step750/Turbo/grid.png e resolution_1024/Turbo/grid.png.

## 2026-09-30 22:26 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step875; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step875/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.90, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:32 UTC — Campanha

Novo A_native/FP8/512/micro2 salvou step1000, 2000 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$3.96, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:32 UTC — Campanha

A_native_fp8_512_micro2_probe salvou step1000; adapter local: /workspace/k2ab/checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$3.96, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:34 UTC — A1000 concluído: mudança de pose funciona no exemplo noturno

A1000 salvo às22:31:54UTC,512keys auditadas,saída normal,LR1e-4. Mesmo run iniciado20:41,2000amostras,~1.34épocas,sem A74.4PNGs Turbo completos em512/1024. Inspeção visual: cena noturna agora FRONTAL,binóculo no centro e ambiente/paleta noturnos conservados nas duas resoluções; em512 também conserva logo do uniforme. Identidade/detalhes dos personagens diferem de B (e faces não eram visíveis na referência A de costas). Mecha agora perfil pedido,porém amarelo/gold em vez de vermelho: herança de cor/desenho segue falhando. Não declarar sucesso global nem vencedor com n=2/seed76.

DINOgt A1000/512=.6628 e1024=.6022,contra A750 .4520/.5205. dHashcopy_rate=0 em ambos,noitecopy_gap=.1510/.0274; transição750→1000 mostra por que não extrapolar fracasso definitivo de checkpoint intermediário. CCIP512=.5,1024=1,sem tomar isso como prova automática de identidade. Grids/JPEG e metricsJSON A1000 nas duas resoluções versionados para review; adapter/HF em sincronização contínua. B500 preservado,runner em avaliação B250/1024 antes de B500/1024 e continuação500→1000.

## 2026-09-30 22:36 UTC — Campanha

Novo B_beta1_fixed/FP8/512/micro2 salvou step500, 1000 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$4.01, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:41 UTC — Reduzir recargas na avaliação B

B250/1024 passou,2PNGs1376×768 sem erro. B512 leva~164s/job para2imagens,embora o denoise512 seja~3.5s por imagem; grande parte é preparação/fusão da base Turbo no CPU,com GPU ociosa. Iniciar um processo por resolução duplicava esse custo em cadacheckpoint.

Runner agora aceita --manifest-1024 e reúne as duasresoluções em uma pipeline/carregamento/fusão,sem mudar matemática de forward,adapter,prompt/ref,seed76,Euler8/CFG1/mu1.15,basequant ou preprocessing. Arquivos existentes são ignorados ANTES de criar pipeline; B250 completo não recarrega o modelo. B500512 estava ativo ao editar,foi preservado; job combinado de B500 gera apenas os1024 faltantes. B750/B1000 gerarão4imagens com uma única preparação de base cada. Stock A inalterado,pois a preparação já era rápida.

Verificação funcional mockada: uma create_pipeline/setup para4PNGs,dimensões688×384 e1376×768,seed76/steps8/CFG1 em todos; sem shuffle ou semadapter. Reexecução com4arquivos existentes não chama create/setup. Compilação ok. Não alegar ganho medido ou paridade numérica a partir do mock; confirmar execução real/tempo nos próximos jobs. LR e treinamento mantidos.

## 2026-09-30 22:46 UTC — B500 avaliado e retomado501

B250/B500 têm4PNGs cada,512/1024;1024 real passou no runner. Primeiro job combinado B500 apenas completou1024 faltante,tempo180.59s; não é benchmark de4imagens com uma carga,aguardar B750. B250 combinado com4PNGs existentes encerrou sem preparação de modelo.

Inspeção B500: noite ainda de costas em ambasresoluções;512 eleva o binóculo acima da cabeça,1024 altera contagem para4personagens. Mecha muda perfil/composição mas ainda difere no desenho;512 mantém vermelho/preto melhor que A1000amarelo,porém são checkpoints diferentes,não comparar como seleçãofinal. DINOgt B500512=.5885 e1024=.5195,copy_rate=.5/0 respectivamente;copy_rate0 não significa ação correta. Grids/JPEG e metrics versionados para review. B500→750 resume501,LR1e-4 confirmado,~3.15s/step; A1000 concluído e preservado. Continua monitoramento até ambos1000.

## 2026-09-30 22:51 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step625; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step625/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$4.16, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:57 UTC — Campanha

Novo B_beta1_fixed/FP8/512/micro2 salvou step750, 1500 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$4.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 22:57 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step750; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step750/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$4.22, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 23:03 UTC — B750 e primeira avaliação das duas resoluções com uma carga

B750 salvo às22:57:31UTC,512keys auditadas,sem erro. Job novo gerou4PNGs688×384/1376×768 em204.16s com uma única preparação/fusão de base; todasas dimensões/seed/CFG/steps do manifest preservadas. Não é A/B numérico contra reload com pesos idênticos,mas código do forward/preprocessing permanece igual; ganho operacional é evitar uma segunda preparaçãoCPU. B750→1000 resume751,LR1e-4,~3.2s/step.

Visual B750:512 dois personagens frontais,terceiro ainda de costas;1024 três frontais,com binóculo no centro,mas expressões/identidade/iluminação continuam divergindo. Mecha tem vermelho nas laterais, porém cabeça/desenho não herda fielmente B; gera subtítulos em1024. DINOgt512=.7150 e1024=.7214,copy_rate0 ambos,CCIP1/.5. Melhora em relação aB500;n=2 seed76 não bastam para seleçãogeral. A750 eB750 não têm o mesmo comportamento: A750 copiava ambos512;B750 muda cena parcialmente. Avaliar comparaçãohomogênea1000 antes de qualquer variante/decisão. Grids versionados no Git/HF.

## 2026-09-30 23:09 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step875; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step875/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$4.34, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 23:15 UTC — Campanha

Novo B_beta1_fixed/FP8/512/micro2 salvou step1000, 2000 amostras; nenhum peso do A74 utilizado. Avaliando apenas Turbo antes do próximo segmento.

Custo Krea acumulado estimado desde16:09UTC:US$4.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 23:15 UTC — Campanha

B_beta1_fixed_fp8_512_micro2_probe salvou step1000; adapter local: /workspace/k2ab/checkpoints/B_beta1_fixed_fp8_512_micro2_probe/20260930_21-33-41/step1000/adapter_model.safetensors. Sync HF contínuo ativo.

Custo Krea acumulado estimado desde16:09UTC:US$4.41, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 23:19 UTC — Campanha

A/B FP8/5121000/micro2 completo:2000amostras por braço, grids/metrics Turbo250/500/750/1000,512 e1024 em /workspace/k2ab/artifacts/fp8_512_micro2/eval. Decisão visual do usuário pendente; sem iniciar variante target ou W8A8 automaticamente.

Custo Krea acumulado estimado desde16:09UTC:US$4.45, inclui setup/cache/ociosidade, não é extrato.

## 2026-09-30 23:22 UTC — A/B1000 concluído, acompanhamento encerrado

A1000 concluído22:31:54UTC; B1000 concluído23:15:23UTC,ambos512keys auditadas,sem erro nos novosmicro2. Cadaum1000steps/2000amostras (~1.34épocas),mesma receita/captions,retomadas do próprio run; não A74. Teste final confirmou32PNGs solicitados (2braços×4checkpoints×2resoluções×2casos),16grids/metrics completos,dimensões688×384/1376×768 corretas,Turbo comadapter/referência certa. Hash captions f2a13e2f8913b44824578cb41cbcc9df63f24de39b650e8c3f18f21825ee36a4 permanece idêntico. JSON final_validation e training_1000_summary em artifacts/fp8_512_micro2 e cópias versionadas para review.

Medianas efetivas1000iterações: A2.787s/step=.7176amostras/s,46min35.9s soma de iterações; B3.194s/step=.6262amostras/s,53min30.1s soma. Milsteps verificados1..1000,mas carregamento/cache/sampling não entram nessa soma. VRAMpico A30193MiB/B30411MiB inclui776MiB da sessão Comfy do usuário. ComputaçãoBF16,armazenamentoFP8scaled,swap0; não W8A8. Custo Krea acumulado estimado desde16:09UTC:US$ 4.48,inclui setup/cache/diagnósticos/ociosidade e falha micro4 histórica;não extrato Vast.

B1000 visual:512 centrofrontalcombinóculo,dois laterais de costas;1024 três de costas. Regrediu do B750/1024frontal nesses mesmos seed/prompts. Mecha conserva vermelho/preto melhor que A1000amarelo,mas desenho de cabeça/identidade continua divergente em ambos. A1000cumpre melhor açãofrontal da noite512/1024; não definir vencedor geral com doiscasos/umaseed,sem shuffle/null por retirada expressa do usuário. Copy_rate0 em A1000/B1000 não garante seguir ação. DINOgt A1000 .6628/.6022 e B1000 .6800/.5766(512/1024)não substitui inspeção; B512scoremaior não torna sua pose correta.

Grids compactos: review/A_B_step1000_Turbo_512_1024.jpg e review/A_B_steps250_500_750_1000_Turbo_512_1024.jpg,mesmas entradas,6colunas refA|alvoB|adapterA512|A1024|adapterB512|B1024. PNGsoriginais nas pastas eval. Adapters/grids/log HF privado,grids/metrics/código/log versionados com push. Controller terminou em awaiting_user_visual_verdict,sem novos jobs/variante/W8A8; GPU livre de nossos treinos. Comfy do usuário preservado.

## 2026-09-30 23:32 UTC — B encerrado, review A1000 Turbo/Raw512 e handoff Comfy

Usuário encerrou B e pausou qualquer treino; ambos já completaram1000. Controller EXITED,stop_campaign presente,state stopped; nenhum treino retomado. Pedido atualizado:10Turbo+10Raw,apenas512 nos mesmos10casos/seed76/prompts originais. Não há job1024 neste lote; plano anterior1024 substituído antes de executar.10Turbo prontos (46.45s); Raw em andamento com28steps/CFG5.5,sem LoRA Turbo,sigmas oficiais. Nenhum shuffle ou semadapter por pedido do usuário.

Push intermediário solicitado para Claude planejar uso no Comfy: código atual,manifest10casos e payloads API reais Turbo/Raw versionados em docs/krea2_results/2026-09-30/A1000_ten_case_review. A já roda no Comfy stock com nodes padrão; contrato é index_timestep_zero,reference VAE no bucket alvo,grounding VL na referência original. Não inventar dependência de custom node para A. Builder de grid novo compila; grid completo/inspeção/métricas aguardam término dos10Raw. Outputs/backupHF permanecem em /workspace.

## 2026-09-30 23:36 UTC — Review A1000:10Turbo+10Raw512 concluído

20outputs prontos,9casos688x384+1caso416x624 por modo,seed76,mesmas entradas,adapter A1000strength1. Turbo8steps/CFG1/mu1.15 com LoRA oficial; Raw28steps/CFG5.5/mu dinâmica sem TurboLoRA. Validação de20PNGs e20payloads passou:modelo,adapter,força,prompts positivo/negativo,referência,native index_timestep_zero,seeds,dimensões/sigmas e captions hash original. Primeira checagem contou grid.png como amostra; corrigida para contar apenas *_with_lora.png,sem regenerar imagens. Manifest1024 não utilizado removido para evitar confusão.

Grids4colunas imagemA|alvoB|Turbo|Raw completos e duas folhas5casos,com nomes/prompts,em artifacts/fp8_512_micro2/A1000_ten_case_review/review; cópias versionadas para Claude em docs/krea2_results/2026-09-30/A1000_ten_case_review. Inspeção detalhada10casos em VISUAL_REVIEW.md. Turbo preserva estilo/paleta mas copia01/08/10; Raw altera pose/composição mais,porém muda identidade/estilo/paleta e cria quadros duplos01/04. Nenhum checkerboard generalizado observado; não é sucesso consistente ainda. Alvos01/04 são shot-reverse-shot com outro personagem e prompts same character ambíguos; métricas vsB não validam por si só seguir A.

Métricasn10:Turbo DINOgt.5832/copy_rate.30/copy_gap.1410/CCIP.50;Raw.5437/.00/-.0549/.70. Sem ref_gain/null_gain por retirada expressa de shuffle/base pelo usuário. JobsTurbo46.45s,Raw165.61s,metrics12.16s (~US$0.039,sem startup/ociosidade). Custo Krea acumulado estimado desde16:09UTC:US$4.62,não extrato. HF privado/upload contínuo; pushfb3121b intermediário já entregue,código novo executou grids reais. Stock auxiliar desligado,B e todos os treinos parados; Comfy usuário preservado.

## 2026-09-30 23:42 UTC — Modelos A disponíveis no ComfyUI do usuário

Conferência no Comfy usuário /workspace/comfy/ComfyUI (8818): faltavam links da baseFP8,Qwen3VL,TurboLoRA e adaptersA; VAE já estava presente e válido. Criados symlinks para modelos centrais e A/native micro2 steps250/500/750/1000,sem copiar pesos ou expor runs antigos. API object_info do processo usuário confirmou todos os modelos atuais nos dropdowns (UNETLoader,CLIPLoader,VAELoader,LoraLoaderModelOnly),sem reiniciar sua sessão. Modelo base krea2_raw_fp8_scaled.safetensors,encoder qwen3vl_4b_bf16.safetensors,adapter A_native_fp8_512_micro2_probe_step1000.safetensors. Turbo usa também krea2_turbo_lora_rank_64_bf16.safetensors; Raw sóadapterA. Manifest de links em artifacts/fp8_512_micro2/user_comfy_symlinks.json. Treinos continuam pausados.


## 2026-10-01 — Recuperação da conversa e preparação A native longo

Transcrição anterior recuperada de /root/.codex/sessions; exportação local privada em /workspace/k2ab/artifacts/recovery_20261001/conversa_anterior.txt. Git principal limpo em cfd5456, idêntico à branch remota. Worktree native em 6355674: diferenças locais são a revisão de ComfyUI e symlinks dos demais submodules, não código perdido; registradas em source_state_before_cleanup.json. Scripts operacionais externos e configs dos checkpoints preservados neste commit antes da limpeza.

Confirmado no tools/k2ab_prepare_data.py: seleção anterior usou /workspace/ns_E2, regex de legenda, classificador safe>=0.7 e corte rígido em 1500. O handoff anterior de probe prescrevia ~1500; pedido atual substitui essa seleção por todos os datasets originais, sem filtro de conteúdo/subset, limitada apenas por disco e integridade do trio A+B+legenda. Legendas originais não serão reescritas.

Usuário autorizou apagar cache antigo e checkpoints, mantendo apenas último A native. Preservar step1000, global_step1000, latest e metadados do run micro2 20260930_20-41-00; evidências, imagens de avaliação, dados originais e modelos necessários permanecem. Adapter A1000/B1000 e grid10 verificados existentes no HF privado AdwolfCzar/krea2-ab-runs. Estado global de retomada permanece local e será protegido. Novo treino de 5000 passos requer apresentação da configuração ao usuário antes de execução; não iniciado.

Achados anteriores preservados: A1000 ainda copia 3/10 em Turbo e Raw pode mudar identidade/estilo; aumento de passos por si só não demonstra correção dessas falhas. Novo smoke10 com audit/resume e amostras/s exigido antes do treino maior. Micro4 anterior falhou por memória; baseline medido micro2, accumulation1, FP8scaled com BF16, swap0, LR1e-4, rank64,512 com7buckets.


## 2026-10-01 — Limpeza concluída; A native novo, não continuação

Usuário corrigiu explicitamente: treino DO ZERO. Preservado A1000 anterior como baseline, não como inicialização. Foram removidos 105,202GiB (manifest versionado), ficando114,747GiB livres imediatamente após limpeza. Hash do adapterA1000 e captions originais JSONL verificados intactos. Último estado globalA1000+latest enviado e confirmado no HF privado. Links250/500/750 removidos do Comfy usuário;1000 permanece válido.

Planejador novo tools/krea2_prepare_budget_data.py usa os4datasets originais e captions.txt publicados (fallback JSONL), sem filtro semântico/SFW/classificador, sem cotas porsubset e sem corte1500. Seleção provisória4201de11526pares completos pelo disco;24heldout preservados,51sem referência. Hardlinks/captionsmaterializados,64pares de smoke preparados, configsnovo5000/smoke10/resume12/segmentos250 produzidas fora da fila. Todos4201captions/hardlinks checados contraorigem. Dois testesCPU do planejador passaram, incluindo regressão>1500, determinismo, colisões, heldout, captionspublicados e recusa de reutilizar árvore. Loaderatual arredonda3tails de buckets:4198pares/época efetiva,documentado. SmokeGPU/paridade do novo cache ainda pendentes de conferência do usuário; nenhuma geração/treino novo iniciado.

Plano/configs em docs/krea2_results/2026-10-01/PLANO_A_NATIVE_5000.md. Cacheestimado97GiB,overhead10%,reserva8GiB; número deve ser recalculado pelo cache real do smoke,não por redução de texto/tokens. Novo LoRA aleatório,baseFP8scaledcongelada,rank64,512/7buckets,micro2×accum1,LR1e-4,warmup50,swap0,activationcheckpointing. Retomar no futuro só segmentos do runnovo,sampling4Turbo512a cada250. Tempo puro~3h52 extrapolado do runantigo,não medido neste dataset.


## 2026-10-01 — LR solicitado0,0005

Pedido explícito: aumentar LR para0,0005. Conferência: A1000/receitaoriginal usou0,0001, não0,0004. Todas configs novas de treino5000, segmentos e smoke/resume do própriosmoke agora têm optimizer.lr=0,0005; nenhumpeso/estado do A1000 será carregado no início. LR5×receita anterior será validado no smoke antes do runmaior. Nenhuma execução iniciada.


## 2026-10-01 — Correção final LR0,0004

Usuário confirmou que, sendo o LR anterior0,0001, deseja0,0004. Isso substitui o pedido imediatamente anterior de0,0005. Todas configs novas e smoke agora em0,0004(4×o LR do A1000), com warmup50 no treino principal. Nada executado naGPU; aguarda conferência do plano.


## 2026-10-01 — Smoke aprovado e cache recalibrado

Novo smoke10do zero:0,722amostra/s,2,7705s/step,VRAM24322MiB,lossfinita0,0665–0,1857;LR0,0004 apóswarmup. Resume11/12passou,LRinalterado. 512chavesauditadas,stockComfy1imagem688×384Turbo8CFG1semchavenãocarregada/semcorrupção; ainda copia referência,não alegar aprendizado em10steps. Cache64medido16,893MB/par,12camadasBF16 intactas,legendasverificadasoriginais.

Seleção recalibrada6138pares (ds1:1523,ds2:2444,ds3:1517,ds4:654),seed42global,semcotas/filtros. Reserva8GiB+10%margem;Cache.add agora tem guard opt-in viaKREA2_CACHE_MIN_FREE_BYTES,sem mudar tensores. Teste confirma parada por disco preserva shard/linhascommitadas. Smokeexports/evidências confirmados noHFprivado antes de removerstates/cachetemporários. ControllerA-only novo fará cachecompleto,countaudit,5000passos novos,4Turbo512por250,HFadapters+estadosverificados,retenção2estadosnovos,pushmilestones. Nunca carrega A1000/smoke no treino principal.


## 2026-10-01T04:09:41.644470+00:00 — A native novo step250

Do zero, dataset completo por orçamento de disco, LR0,0004 confirmado; 500amostras vistas, lossfinal0.1612. 4Turbo512+grid/métricas e adapter/estado completos enviados ao HF privado; backup verificado antes da poda. Não é retomada do A1000 antigo.


## 2026-10-01T04:23:07.199388+00:00 — A native novo step500

Do zero, dataset completo por orçamento de disco, LR0,0004 confirmado; 1000amostras vistas, lossfinal0.1342. 4Turbo512+grid/métricas e adapter/estado completos enviados ao HF privado; backup verificado antes da poda. Não é retomada do A1000 antigo.


## 2026-10-01T04:36:28.482179+00:00 — A native novo step750

Do zero, dataset completo por orçamento de disco, LR0,0004 confirmado; 1500amostras vistas, lossfinal0.1218. 4Turbo512+grid/métricas e adapter/estado completos enviados ao HF privado; backup verificado antes da poda. Não é retomada do A1000 antigo.


## 2026-10-01T04:44:32.338264+00:00 — reinício solicitado com LR 0,0001

O usuário interrompeu a campanha LR 0,0004. Encerramento seguro via save_quit no passo 843; checkpoint final completo verificado no HF privado. Nova execução do zero, LR 0,0001 (valor original), mesmo cache de 6.138 pares, output/fila separados e smoke novo antes de 5.000 passos. Os primeiros 250 passos da campanha anterior terminaram corretamente; o controlador tinha parado ao interpretar `loss: loss_fn` do dump do modelo, corrigido para ler somente as linhas de métricas `steps:`. A geração e os backups de 250, 500 e 750 foram concluídos depois da correção.


Smoke LR 0,0001 aprovado: 10 passos + retomada até 12, 512 chaves exportadas e uma imagem stock gerada sem chave LoRA rejeitada. Mediana 0,724 amostras/s, 2,7615 s/passo; pico amostrado 24,80 GiB. Backups privados de step10 e step12 verificados antes da limpeza dos checkpoints temporários. Produção LR 0,0001 começou no passo 1, com LoRA e otimizador novos, sem argumento de resume; warmup de 50 passos. Novo serviço supervisionado `krea2_lr1e4`, execução antiga parada.
