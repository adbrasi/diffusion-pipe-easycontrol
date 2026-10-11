# Diagnóstico — por que o Krea 2 Turbo 1024 falhou, os erros que se repetem e o plano (2026-10-11)

Sessão sem GPU. Base da análise:
- leitura de todos os `.md` do repo e das branches `krea2-probe-512`, `krea2-known-good-a1000` e `ic-lora`;
- auditoria do código realmente usado no run (`krea2-probe-512`, ComfyUI `fb2315f1`);
- pesquisa externa (ai-toolkit, card do adapter do ostris, krea2edit-trainer, README oficial da Krea);
- **medições diretas nos adapters publicados no HF** (`AdwolfCzar/krea2-turbo-1024` step500/1500/3000 e `AdwolfCzar/krea2-probe512-turbo` step1000), comparados com os pesos do `krea2_raw_bf16` e com a LoRA Turbo oficial (`Comfy-Org/Krea-2`).

A ferramenta usada nas medições é `tools/lora_delta_audit.py`. Ela roda só com numpy e lê a base por range request, sem baixar os 26 GB.

Legenda: **[MEDIDO]** = número que calculei nesta sessão. **[CÓDIGO]** = verificado lendo o código. **[FONTE]** = doc ou fonte externa citada. **[HIPÓTESE]** = raciocínio ainda não testado na GPU.

---

## 0. Resposta curta

1. **Não há bug matemático no treino.** [CÓDIGO] Foram verificados:
   - o shift (é igual ao `flux_time_shift` do ComfyUI com mu 1,15);
   - `x_t=(1-t)x0+tε`, alvo `ε-x0`, t em [0,1];
   - ref com t=0 por token (igual ao `index_timestep_zero`), ref no frame 1 com grid a partir de 0;
   - loss só no alvo;
   - merge das LoRAs (alpha/rank certo, as 7 `.diff_b` aplicadas, requantização FP8 com escala recalculada e arredondamento estocástico);
   - cosseno sobre 5.178 passos.

   O problema está nas **decisões de método**, não na implementação.
2. **Causa principal: treinar sobre "Turbo + adapter de treino do ostris".** Três evidências independentes:
   - [FONTE] A Krea recomenda **treinar LoRA no Raw e aplicar no Turbo**. O card do adapter diz que ele é "um hack" para runs **curtos** de estilo e personagem, e que "runs longos vão quebrar a destilação, causando artefatos quando o adapter é removido".
   - [MEDIDO] No step3000, a nossa LoRA move os pesos dos blocos iniciais e intermediários **4–6× mais do que toda a destilação Raw→Turbo**. É exatamente o regime em que o "hack" deixa de valer.
   - [visual, grids do HF] O **A1000** (treinado no Raw) continua nítido no Turbo. O **probe Turbo+adapter** já começa a ficar **lavado e nebuloso** no step500–750. No 1024 isso explode a partir do step1500.
3. **Causas que se somam:**
   - excesso de treino (o mesmo erro do beta1);
   - distribuição de timestep com **50% da massa em t>0,8 e só 1,9% em t<0,3**;
   - avaliação que não podia detectar o problema: 3 casos, 1 seed, só Turbo, e o prompt era a própria legenda do dataset.
4. **O que o probe acertou e deve ser mantido:**
   - buckets de AR reais (16:9);
   - dataset novo;
   - LR com decaimento.

   O "probe melhor que o A1000" mudou **7 variáveis de uma vez**. A melhora em seguir a ação provavelmente veio dos dados, dos buckets e do timestep. A névoa veio da base Turbo+adapter.
5. **Plano:** antes de qualquer treino, rodar ~1,5 h de **testes decisivos na GPU** (§6), que confirmam ou derrubam o item 2 de forma barata. Depois, um treino **nativo no Raw** com as correções (§7.1). O beta1 corrigido (§7.2) fica como segunda opção.

---

## 1. Linha do tempo resumida do Krea 2 (para não precisar reler tudo)

| run | base / método | resultado |
|---|---|---|
| **beta1** (`k2-context-rush-ofc-beta1`, 13.445 passos) | Raw com fp8 re-quantizado **sem escala** (até 27% dos pesos zerados); LoRA **roteada** só nas linhas da ref; `width_shift`; grounding 384² bicubic; 30k pares (26% pico-banana, 23% frames quase iguais, 60% NSFW); caption_dropout 0,1; LR 1e-4 constante (a fase 2 em 1024 ficou em 1e-4 por bug de resume) | sempre respeita a ref; muitas vezes **devolve a própria imagem**; LoRAs empilhadas degradam; precisa do node `ctxrush-edit` |
| groundedsecret (250 passos × batch 4) | mesma família, parado em ~0,8 época, grounding 768 | não copiava (é o "beta1 jovem") |
| apex | LoRA global, ref t='target', legendas completas | polo oposto: bonito, **sem relação com a ref** |
| **A1000** (30/09) | `krea2_native` no **Raw** fp8_scaled correto, 512, batch 2, r64, LR 1e-4 constante, flux_shift dinâmico, 1.500 pares | "known good"; no Turbo copia em 3 de 10 casos, nítido |
| A scratch (01/10) | igual, 6.138 pares | copy_rate sobe de 0 (step 500) a 1,0 (step 1500); com LR 4e-4 isso acontece já no step 750 |
| probe512 Turbo (08–09/10) | Raw+LoRA Turbo+**adapter ostris** fundidos; shift 3,16 + 15% em U[0,8;1]; r32 cosseno; buckets reais; dataset-2 (1.726 pares) | o usuário achou melhor que o A1000; parado no step 1000 |
| **Turbo 1024** (09/10) | o mesmo do probe, em 1024 e batch 1 | "começando a queimar" no step 3362; ignora prompt, copia ou inventa, perde qualidade |

Detalhes: `KREA2_EDIT_SAGA.md`, `AUDITORIA_2026-08-06.md`, `KREA2_ANALISE_METODO_2026-09.md`, `KREA2_AB_RUN_LOG.md`,
`RELATORIO_2026-10-08_09.md`.

---

## 2. Evidências do run Turbo 1024

### 2.1 Os grids mostram degradação progressiva, não aleatória

Grids no HF: `AdwolfCzar/krea2-turbo-1024/evidence/eval/*_step{500..3000}/grid.png`.
- **step500:** nítido e saturado.
- **step1500:** já lavado, acinzentado, com névoa e traço "esboçado".
- **step3000:** contraste e detalhe piores ainda.

No comparativo `AdwolfCzar/krea2-probe512-turbo/evidence/compare_A1000_vs_probe_250_500_750.png`:
- o A1000 é nítido;
- o probe step250 é nítido;
- o probe step500 e step750 **já mostram o início da mesma névoa**, sobretudo no caso 1.

Ou seja, o sintoma estava no probe que foi aprovado. Ele só era pequeno.

### 2.2 Tamanho do delta da LoRA [MEDIDO]

Medida: `|ΔW|₂ / |W|₂`, a norma espectral do delta da LoRA dividida pela do peso do Raw bf16. A coluna "Turbo" é a LoRA oficial `krea2_turbo_lora_rank_64_bf16`, ou seja, **toda a destilação**. A norma da base foi conferida com SVD exata (59,4437 contra 59,4436).

| módulo | Turbo (destilação) | 1024 s500 | 1024 s1500 | 1024 s3000 | probe512 s1000 |
|---|---|---|---|---|---|
| blocks.0.attn.wq | 0,019 | 0,066 | 0,091 | **0,107** | 0,071 |
| blocks.0.attn.wk | 0,009 | 0,032 | 0,060 | **0,079** | 0,025 |
| blocks.0.mlp.down | 0,059 | 0,142 | 0,277 | **0,355** | 0,177 |
| blocks.14.attn.wq | 0,028 | 0,017 | 0,049 | **0,107** | 0,081 |
| blocks.14.mlp.down | 0,071 | 0,070 | 0,146 | **0,202** | 0,126 |
| blocks.27.attn.wq | 0,245 | 0,037 | 0,103 | 0,157 | 0,092 |
| blocks.27.mlp.gate | 0,330 | 0,059 | 0,171 | 0,234 | 0,139 |

Leitura:
- **A destilação Turbo mexe sobretudo no fim da rede** (bloco 27: 25–33%) e quase nada no início (bloco 0: 1–6%).
- **A nossa LoRA mexe sobretudo no início.** No step3000, o bloco 0 `mlp.down` está em 35,5% contra 5,9% da destilação, 6× mais. O `wq` e o `wk` do bloco 0 ficam 5–9× acima da destilação.
- O adapter do ostris pressupõe que a LoRA é uma **perturbação pequena** em torno do ponto des-destilado. Aqui ela é muito maior que a própria distância Raw↔Turbo. [HIPÓTESE forte] É por isso que, sem o adapter (na inferência), o Turbo sai lavado. É o "artefato quando o adapter é removido" do card.
- A norma total cresce sem parar: Σ‖ΔW‖_F = 250 → 561 → 806 (steps 500 → 1500 → 3000). O probe512 aprovado (s1000) tem tamanho equivalente ao do 1024 entre o step500 e o step1000.
- **Picos rank-1:** σ1/σ2 chega a ~60 no `wq`/`wk`/`mlp.gate` do bloco 0 desde o step500. É o mesmo fenômeno da issue ai-toolkit #964 (LoRAs de Krea 2 criam picos rank-1 que crescem com os passos, sobretudo em `mlp.gate`). [HIPÓTESE] No bloco 0, uma direção rank-1 em `wq`/`wk` é candidata natural a um "interruptor de copiar a ref", porque os tokens da ref são separáveis pelo t=0. Vale medir na GPU a massa de atenção alvo→ref por bloco.

### 2.3 A base de treino não era nem o Turbo nem o Raw [CÓDIGO + FONTE]

- **Treino:** Raw FP8 + `krea2_turbo_lora_rank_64` + adapter, fundidos e requantizados em FP8.
- **Inferência:** Raw FP8 + LoRA Turbo, sem o adapter.
- **O adapter do ostris foi ajustado contra o Turbo de verdade** (`krea/Krea-2-Turbo`). O ai-toolkit carrega o checkpoint Turbo completo, não "Raw + LoRA Turbo".
- **O Comfy-Org publica o Turbo completo** (`krea2_turbo_bf16`, `krea2_turbo_fp8_scaled`). A LoRA rank 64 é só uma aproximação dele.

Resultado: a base de treino era uma composição para a qual o adapter nunca foi feito. A LoRA teve de absorver esse descasamento. E, sobretudo, o **Raw (a base de onde o Turbo saiu) estava disponível**. O adapter existe para quem não tem a base não destilada (Z-Image Turbo, FLUX schnell); para o Krea 2, ele não resolve nada que o Raw não resolva melhor.

### 2.4 Distribuição de timestep [MEDIDO, 2 milhões de amostras]

| receita | t>0,8 | t<0,3 | t<0,15 | mediana |
|---|---|---|---|---|
| A1000 512 (flux mu≈0,58) | 21% | 7,6% | 1,0% | 0,64 |
| beta1 fase 2 / flux mu 0,91 (1024) | 32% | 3,9% | 0,4% | 0,71 |
| **Turbo 1024 (shift 3,16 + 15% de banda)** | **50%** | **1,9%** | **0,2%** | 0,80 |
| Anima v2 (LN sem shift + 20% de banda) | 27% | 15,8% | 3,3% | 0,58 |
| lei `krea2_apex` em 512 | 25% | 22% | 8,9% | 0,58 |
| lei `krea2_apex` em 1024 | 37% | 17% | 7,0% | 0,69 |

- Em 3.362 passos, só ~65 amostras caíram em t<0,3.
- A LoRA é global e atua em **todos** os t na inferência, mas quase não recebeu gradiente na banda que decide textura e nitidez. É compatível com "perde detalhe e ganha névoa". [HIPÓTESE]
- Com metade dos passos em t>0,8, onde quase nada do alvo sobrevive, o minimizador da MSE é "a média de B dado ref + texto". Com uma ref limpa em t=0, essa média puxa para **a ref**: o "copia ou entrega o que acha que a imagem seria" que você descreveu. [HIPÓTESE, mas é o mesmo mecanismo documentado em `AUDITORIA_2026-08-06.md` §2]
- O `shift = e^1,15` foi justificado como "igual ao Turbo/ComfyUI". Mas o shift **de amostragem** do sampler não é a distribuição **de treino**. Usar shift de inferência em cima de logit-normal desloca a massa duas vezes.

### 2.5 Excesso de treino [MEDIDO / CÓDIGO]

| | atualizações | exposições por par | "massa de LR" (Σlr/1e-4) |
|---|---|---|---|
| A1000 | 1.000 (batch 2) | 1,34 | ~975 |
| probe512 @1000 | 1.000 (batch 2) | 1,16 | ~873 |
| **Turbo 1024 @3362** | **3.362 (batch 1)** | **1,95** | **~2.413** |

- O relatório de lançamento dizia que "5.178 passos de batch 1 = 2.580 de batch 2 em amostras vistas". Isso é verdade, mas **dobra o número de atualizações** com o mesmo LR.
- Todos os resultados aprovados do projeto ficam entre 0,5 e 1,3 época (`AUDITORIA_2026-08-06.md`).

### 2.6 A avaliação não podia pegar o problema [CÓDIGO]

`tools/krea2_v2_eval.py` como foi usado:
- 3 pares, sempre os mesmos 3 (os primeiros em ordem alfabética), seed única;
- só Turbo (`--raw` nunca foi passado);
- **o prompt era a própria legenda longa do held-out** ("The same woman with long red hair… The same background."), no mesmo dialeto do treino;
- sem coluna "base sem LoRA", sem "sem ref", sem métrica;
- e nenhuma loss de validação foi registrada, embora o trainer suporte `eval_datasets` com quantis fixos de t.

Com legenda de treino como prompt, o texto quase descreve B inteiro, então "respeita o prompt" fica mascarado. Você usa prompts próprios e mais curtos, e aí o modelo cai na ref ou ignora o pedido.

### 2.7 Pontos menores [CÓDIGO]

- `reference_pixels="target"`: a ref é cortada no bucket do alvo (PIL `fit`, bicúbico). O node stock manda ~1 MP no AR da própria ref, sem corte. Já o VL recebe a imagem inteira. É pequeno, mas é uma divergência de contrato.
- Em batch>1 o código do A1000 passa `mask=None` no `txtfusion` com texto preenchido com zeros (o código novo corrige isso). Afeta o probe512 (batch 2), não o 1024 (batch 1).
- Ninguém conta as chaves casadas no `merge_adapters`; o ComfyUI só registra no log `lora key not loaded`. Vale conferir no log do run.
- O treino faz a matmul em bf16; o ComfyUI stock quantiza ativações para fp8 em parte das camadas. O probe também tinha isso, então não é o diferencial.

---

## 3. Erros que se repetem (história × agora)

| padrão de erro | antes | agora |
|---|---|---|
| **Treinar até "queimar" em vez de parar pela avaliação** | beta1 13k passos (~10× a janela boa); sauce 1500@1024 "já é cópia"; A scratch copia a partir de 1.500 | Turbo 1024 planejado para 3 épocas, com o sintoma visível desde o step1500; o `A1000code_1024` estava planejado para **6 épocas** com LR constante; Anima v2 com 5 épocas e LR constante |
| **Base numérica diferente da inferência** | beta1: fp8 sem escala (pesos zerados), e a LoRA aprendeu a corrigir uma base mutilada; daí "LoRA degrada" e "precisa de patch" | base composta Raw+LoRA Turbo+adapter, diferente tanto do Raw quanto do Turbo da inferência |
| **Mudar várias variáveis de uma vez e dar o crédito à mais vistosa** | apex: tirou o routing e o t=0 e trocou as legendas de uma vez; caiu no polo "ignora a ref" | probe: base, adapter, shift, banda alta, rank, LR, buckets e dataset mudaram juntos; o crédito foi para "Turbo+adapter" |
| **Fome de ruído baixo (textura)** | beta1 fase 2: 3,9% em t<0,3 (`AUDITORIA` já apontava "degrada quando segue") | 1,9% em t<0,3: a metade do que já tinha sido apontado como problema |
| **Avaliação pequena e no dialeto do treino** | A/B com n=2 e seed única; o próprio log diz que as métricas são grosseiras | n=3, seed única, prompt = legenda do dataset, só Turbo |
| **LR alto demais para o número de atualizações** | fase 2 do beta1 rodou com 1e-4 em vez de 7,5e-5; LR 4e-4 copiou no step 750 | batch 1 com o mesmo LR de batch 2: 2,8× a massa de LR do probe |
| **Usar o truque de alguém fora do caso de uso dele** | node do conradlocke: o apex copiou o contrato sem a parte que fazia funcionar | adapter do ostris: feito para runs curtos de estilo e para quem não tem o Raw |

Princípio comum: **cada decisão nova foi validada no próprio caso que a motivou (3 imagens, 1 seed, prompt de treino)**, nunca contra um controle. O plano abaixo corrige isso primeiro.

---

## 4. O que funcionou e deve ser mantido

- Contrato nativo do ComfyUI: `TextEncodeQwenImageEditPlus` + `index_timestep_zero` + LoRA global com `LoraLoaderModelOnly`. Roda sem node custom, é o "método elegante" (o modelo lê a ref pelo próprio Qwen3-VL) e a paridade com o código foi verificada.
- Base Raw `fp8_scaled` **com escala** (o A1000 é nítido no Turbo).
- Encoder de texto `fb2315f1`/`f856877e` (diferença 0,00 contra o ComfyUI do usuário), com o gate `te-fork` antes de cachear.
- Buckets de AR reais (16:9 era 71% dos dados e era cortado).
- Dataset v2 / v11.
- LR com decaimento, rank 32–64.
- Checkpoints frequentes.

---

## 5. Anima "próxima cena" (v2 1024): diagnóstico

[CÓDIGO] **Sem bug matemático.** Foram verificados: alvo `ε-x0`, loss só no frame alvo, `t=[σ,0]`, sem shift duplo, o `compile_blocks` e o `uncheckpointed_blocks` (numérica igual ou melhor que eager), e a paridade do node (0,42% / 3,2%). A queda geral de qualidade se explica melhor pela receita:

1. **LR 1e-4 constante, sem decaimento, por 5 épocas** (cada imagem vista 10×, 1.726 pares, rank 64). A pesquisa de setembro pedia 5e-5 (`ANIMA_NEXTSCENE_PESQUISA_2026-09.md`). É o mesmo padrão "treinar demais" da tabela do §3.
2. **Legendas sem tags de qualidade, score ou artista, com LoRA na cross-attention** (112 dos 280 módulos leem o texto). A LoRA associa "texto sem tags" à estética média do dataset (frames de anime e páginas de mangá). Na inferência você usa `masterpiece, best quality, score_7` e artistas, que ficam fora da distribuição dessa cross-attention adaptada. Isso dá o "modelo pior no geral". [HIPÓTESE com bom suporte]
3. **O tier de legenda curta ("The same X…") foi retirado no v2.** Com legendas de ~94 palavras descrevendo B inteiro, o texto basta e a ref perde peso. Explica o "nem sempre respeita a ref".
4. Menores: CFG 4 com um ramo negativo nunca treinado; 74% dos dados em AR 1,59 (pouco quadrado ou retrato).

**Teste barato (GPU, ~20 min):** grid seed fixa com base sem LoRA, step 500 / 1000 / 2000 / 4325, prompt sem tags × prompt com tags de qualidade, e strength 0,6 / 0,8 no step 4325.
- Se o 500–1000 for bom e piorar depois, a causa é o item 1.
- Se a qualidade voltar com as tags, a causa é o item 2.

**Próximo treino Anima:**
- `lr_scheduler="cosine"`, LR 5e-5–1e-4, ~2 épocas, rank 32;
- tags de qualidade nas legendas (ou tirar a LoRA da cross-attn);
- tier curto de volta;
- `eval_datasets` com held-out.

---

## 6. Testes decisivos na GPU — ANTES de qualquer treino (~1,5 h)

Todos com adapters que já existem (linkados em `loras/krea2_nextscene/`). Regras para todos os testes:
- **8 held-out**, 2 seeds;
- **3 tipos de prompt**: a legenda do dataset, uma instrução curta no estilo do usuário e prompts reais do usuário;
- colunas: base sem LoRA | ref certa | ref trocada | sem ref.

| # | teste | o que decide |
|---|---|---|
| T1 | Turbo1024 s500/s1000/s1500/s3000, probe s250/s1000 e A1000 renderizados no **Raw (28 passos, CFG 3–4,5)** e no **Turbo (8 passos, CFG 1)** | se no Raw saem nítidos e no Turbo lavados, o item 2.2/2.3 está confirmado: a névoa é a destilação quebrando, e treinar no Raw resolve |
| T2 | Turbo oficial (`krea2_turbo_fp8_scaled`) × "Raw + LoRA Turbo r64", sem LoRA de tarefa, mesma seed | se o Turbo composto que você usa é igual ao oficial; define com qual Turbo avaliar e usar |
| T3 | strength 0,5 / 0,75 / 1,0 no Turbo1024 s1500/s3000 e no probe s1000 | resgate sem treino; quanto do problema é magnitude |
| T4 | prompts curtos e do usuário × legenda do dataset no A1000 e no probe s1000 | se o "ignora o prompt" vem do dialeto das legendas (a decisão sobre um tier curto depende disso) |
| T5 | loss de validação nos held-out em t ∈ {0,1; 0,3; 0,5; 0,7; 0,9} para cada checkpoint | curva objetiva de overfit por banda de ruído |
| T6 | `grep "lora key not loaded\|NOT LOADED"` no log do run Turbo 1024 | se o merge carregou 100% das chaves |
| T7 | (opcional) massa de atenção alvo→ref por bloco, base × s500 × s3000 | confirma ou derruba o "interruptor rank-1 de copiar" do bloco 0 |

Critério: se o T1 mostrar o Raw nítido, a decisão "treinar no Raw" fica fechada sem precisar de mais um run de teste.

---

## 7. Métodos de treino propostos

### 7.1 Principal — "nativo no Raw" (o método elegante, sem o adapter)

Código: `krea2-probe-512`, que é o código do A1000 + `high_noise_prob`. Mudanças em relação ao run Turbo 1024:

| item | Turbo 1024 | **proposto** | por quê |
|---|---|---|---|
| base | Raw fp8 + LoRA Turbo + adapter | **Raw `fp8_scaled`, sem `merge_adapters`** (ou Raw bf16 + `blocks_to_swap` se o T1 apontar numérica) | recomendação oficial; o A1000 nítido; §2.2–2.3 |
| timesteps | LN + shift 3,16 + 15% de banda | **`timestep_law = "krea2_apex"`** (já implementado, sem shift) | cobre as duas pontas (512: 25% em t>0,8 e 22% em t<0,3) |
| batch | 1 | **micro 1 × accum 2** em 512 | sem padding de texto (`mask=None` igual ao stock) e com batch efetivo 2 |
| LR | 1e-4 cosseno até 3 épocas | 1e-4 cosseno, **planejado para ~1,5 época**, parada pela avaliação | todos os vencedores ficam entre 0,5 e 1,3 época |
| resolução | só 1024 | 512 para o grosso e 1024 como acabamento curto (10–20%, LR 5e-5, batch 1) | receita do krea2edit-trainer e da saga |
| checkpoints | a cada 500 | **a cada 125** em 512 e a cada 100 no acabamento | a janela boa é estreita |
| avaliação | 3 casos com legenda do dataset | protocolo do §6 (Raw + Turbo, 3 tipos de prompt, controles) + `eval_datasets` com loss por quantil | §2.6 |
| guarda-corpo | nenhum | `tools/lora_delta_audit.py` a cada checkpoint; atenção se o bloco 0 `mlp.down` passar de ~0,18 (o tamanho do probe aprovado) | heurística medida aqui, não lei |
| rank | 32 | 32 (ou 64 igual ao A1000) | os picos rank-1 mostram que o rank não é o gargalo |
| ref | `reference_pixels="target"` | manter em 512; testar `node_1mp` no acabamento 1024 | contrato exato do node |

Mantém do probe: buckets reais, dataset v2, rank 32, cosseno. Mantém do A1000: base Raw.

### 7.2 Alternativa — "beta1 corrigido"

Você pediu para reaproveitar o método do beta1. As correções documentadas (`examples/krea2_ab/B_beta1_fixed.toml` e `AUDITORIA_2026-08-06.md` Fase 1):
- base `fp8_scaled` **com escala**;
- TE `fb2315f1`;
- grounding 768 área (como o groundedsecret) no lugar de 384² bicúbico;
- caption_dropout 0;
- dataset v2 (sem pico-banana e sem quase duplicados);
- lei apex;
- LR 1e-4 com cosseno;
- **parada em 0,5–1,2 época** com checkpoints a cada 125.

Ressalvas que você precisa pesar:
- A LoRA **roteada** (só nas linhas da ref) **nunca carrega num loader padrão**. Ela continua exigindo o node `ctxrush-edit` (sem o Training Base) e não é uma LoRA "normal" para empilhar com outras.
- O B_beta1_fixed já foi testado por 1000 passos (30/09): **não copiou, mas seguiu pouco a pose e a identidade** (n=2).
- A teoria dos "dois polos" é uma hipótese. O A750 nativo também copiou, então o routing não é o único caminho para a cópia, e o tempo de treino parece pesar mais.

**Recomendação:** rodar o 7.1 primeiro. O 7.2 só entra se o 7.1 mostrar cópia nos testes com ref trocada.

---

## 8. Ordem de execução na próxima sessão com GPU

1. Setup:
   - checkout `krea2-probe-512` (o código usado) + submódulo ComfyUI `fb2315f1`;
   - baixar Raw `fp8_scaled`, Turbo `fp8_scaled`, LoRA Turbo, VAE e Qwen3-VL;
   - rodar o gate `te-fork`.
2. T1–T6 (§6), com o veredito visual do usuário no T1 e no T4.
3. Smoke do 7.1 (~50 passos):
   - conferir a distribuição de t no log;
   - conferir que não há `merge_adapters`;
   - conferir a VRAM;
   - conferir que as manchas somem por volta do step 125–250, que é esperado (`RELATORIO_2026-10-08_09.md` §2).
4. Run 7.1 em 512, com avaliação a cada 125 passos e parada pela janela.
5. Acabamento em 1024 a partir do melhor checkpoint de 512.
6. Opcional: média dos 2–3 checkpoints vizinhos do melhor (o krea2edit-trainer relata que costuma ganhar de qualquer um isolado).
