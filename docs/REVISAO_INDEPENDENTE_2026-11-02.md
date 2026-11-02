<!-- Relatório de um Opus com contexto limpo (só leitura, sem GPU), recebido em 02/11/2026, sem edição de conteúdo.
Scripts de análise dele: /tmp/claude-1002/-workspace/93d46e3a-0174-4b5b-80b7-c7a0e6a17f28/scratchpad/independent_review/
(tab.py, overlap.py, overlap_shuf.py, loss.py). -->

# Revisão independente: copy collapse no BETA1-R

## 0. Resumo
1. **O que é a cópia.** É um atalho de "cópia alinhada" (shortcut learning), não overfitting. No step 1000 o modelo viu 4.000 amostras, cerca de **0,2 época** do dataset de 20.112 pares.
2. **Os dados ensinam a cópia em grande escala.**
   - Em média, **37,7 % dos pixels do alvo são cópia pixel-alinhada da referência** (|A−B|<0,08 a 64 px, com a ref recortada no centro como no trainer). Com pares embaralhados, a taxa de acaso é **~10,5 %**.
   - Em **~27,5 % dos pares, mais da metade da imagem é idêntica e alinhada**. Por acaso, isso ocorre em 0–2 % dos pares.
   - Pelo próprio critério da avaliação (dHash≤10), **15 % dos pares de treino (3.034) já são "cópias"**.
   - O peso por par do R2b (dHash global) só pega a ponta disso. O dHash global não detecta cópia parcial: fundo e personagem iguais com um atributo mudado.
3. **O método torna o atalho barato e o amplifica.**
   - (a) O `width_shift` dá à ref o mesmo grid e **frame=0**, colada à direita do alvo. É literalmente um díptico (`krea2_multiref.py:229-241`, `krea2_reference.py:421-425`). `reference_position_offset=1.0` é ignorado em width_shift.
   - (b) Copiar exige um único deslocamento relativo fixo de RoPE, (0,+W). A LoRA routed só altera K/V das linhas da ref (`condition_lora.py:51-53,81`; `krea2_omini_grounded.py:50`), e essa é justamente a operação de que o atalho precisa.
   - (c) A cópia se decide no **1º passo (σ=1)**, que o treino quase nunca vê. O treino usa mu=0,581 a 512 px (`krea2.py:166`), o que dá P(t>0,957)=0,6 % e P(t>0,98)=0,05 %. A inferência Turbo usa mu=1,15, com sigmas 1,0 / 0,957 / 0,905 / ….
   - (d) A Turbo (8 passos, CFG 1) é fundida no base só na inferência (`infer_reference_adapter.py:525+`). O adapter foi treinado no Raw. Nunca foi testado se o Raw com 28 passos e CFG também copia.
4. **Veredito:** a causa primária está nos dados (o alvo supervisionado contém a cópia), e o método a habilita e acelera. Confiança alta no mecanismo e média no peso relativo de cada fator.
5. **A cópia depende do caso.** Acontece quando o prompt é compatível com a ref.
   - b0 e b1 são prompts de edição com "same background" e "same woman". **O b1 está no treino** (`recortados_udvuxwn25f_image_0002`).
   - No h000263, a ref já contém a loira com a mão no queixo que o prompt descreve. Vi a imagem: o step 750 faz uma cena nova, o 1000 devolve a ref, o 1000 com skip k=1 volta à cena nova e o R2b 750 sai híbrido.
   - Os seeds dentro de um caso são muito correlacionados. No 1000, h000263 copia em 4/4 seeds e os outros held-out em 0/28.
6. **`ref_bias` 0,99 equivale a −0,01 nos logits.** É matematicamente irrelevante, e o copytest confirma: 4/48 tanto com 1,0 quanto com 0,99. O "0,99 cura" do beta1 original deve ser acaso ou efeito colateral do caminho de código: com bias≠1 a máscara vira float densa (B,1,L,L) em vez de bool (B,1,1,L), o que pode trocar o backend de atenção ou o tratamento de padding no ComfyUI. Dá para testar com bias 1,0 contra 0,9999 na mesma seed.

## 1. Código: bugs e problemas de desenho
Não achei bug que cause a cópia. Encontrei problemas que a facilitam ou confundem os testes:
- O offset de posição é ignorado em width_shift, então a ref fica com o mesmo frame do alvo (díptico). A config e a doc enganam.
- O mu do treino difere do da inferência, e o `high_noise_prob` (`krea2.py:171-176`) está desligado. Não há supervisão onde a cópia é decidida.
- Os pesos de loss são por amostra inteira (`krea2_multiref.py:97-106`), mas o problema é espacial. A máscara por pixel já é suportada (`krea2.py:118-121` e base.loss_fn).
- O zoom-crop interpola latentes de VAE bilinearmente (`:109-132`), o que gera latentes borrados fora da distribuição. Isso confunde o teste de "desalinhamento".
- O R3 (`:499-504`) reintroduz o dropout parcial (VAE mascarado com o Qwen3-VL vendo a imagem) que o próprio código proíbe em `:581-583`. É aceitável por ser consistente com o skip na inferência.

## 2. Crítica dos experimentos
- **Todos os runs usam seed=42 e são pareados** (mesmos pares, timesteps e ruído). A correlação da loss por step é 0,96 entre BETA1-R e R2b e 0,9997 entre BETA1-R e R3.
  - Mesmo assim, o R2b (só pesos anti-cópia) começou a copiar no 750, antes do BETA1-R (1000).
  - Isso refuta "os 783 idênticos explicam a cópia".
  - Também mostra que o momento de início da cópia varia **±250 steps ou mais** com mudanças pequenas.
- **A unidade estatística é o caso, não a geração.** 4/48 é na prática 1/8 casos, e 5/9 nos beta1 vem de ~2 casos. "Limpo no step N" com 1 run e 8 casos não separa configs.
- **"Zoom-crop acelera" é sugestivo, não provado.** O R2a tem dHash médio held-out de 24,2, contra 30,6 (BETA1-R) e 31,3 (R2b). É um deslocamento amplo, mas com n=1 e o confundidor do latente interpolado.
- **Ruído e token-drop "não pioram"** não tem poder estatístico. O ruído na ref só na inferência é fora da distribuição de qualquer forma.
- **O skip k=0/1/2 é o melhor experimento do projeto.** No step1000 com k=1 dá 0/32 held-out e 0/12 beta1, sem retreino. Falta medir a perda de consistência: paleta e identidade se decidem em σ alto.
- **O R3 mascara a ref VAE em só ~3,1 % das amostras** (t>0,92 com mu 0,58). No treino é quase uma réplica do BETA1-R. O ganho vem do skip na inferência, que já funciona no BETA1-R, e a pressão de cópia vinda dos dados continua.
- A loss de treino fica plana (0,091–0,093 entre os steps 250 e 1115) e não sinaliza a transição.
- **Avaliação proposta:**
  - 64–128 casos held-out, estratificados por compatibilidade prompt↔ref, sem casos que estão no treino.
  - Métricas contínuas (dHash, DINOv2, LPIPS) contra a ref **e** contra o alvo.
  - Consistência: DINO/CLIP de personagem, distância de paleta e um juiz VLM.
  - Uma sonda barata a cada 100 steps: similaridade do x0 previsto em σ=1 e σ=0,957 com a ref, mais a massa de atenção alvo→ref no deslocamento (0,+W).
  - Comparar Turbo 8 passos contra Raw 28 passos + CFG.
  - Bootstrap sobre casos e ≥2 seeds de dados nas decisões importantes.

## 3. Literatura: correção real ou paliativo
- **FLUX Kontext** (2506.15742) usa o mesmo grid com offset no eixo de frame e nenhum truque anti-cópia. É um modelo de edição, onde copiar é desejado.
- **OminiControl** (2411.15098) põe a posição da condição fora do footprint do alvo em tarefas não alinhadas (desenho real) e oferece bias γ (paliativo).
- **UNO** (2504.02160) viu que refs recortadas do próprio alvo levam a colar a ref no resultado. A solução foram pares cross-cena sintéticos com filtro DINO/VLM: correção real nos dados.
- **DreamO** (2504.16915) descarta ~95 % dos tokens da ref num refino: meio-termo.
- **Captain Cinema** (2507.18634), **LCT** (2503.10589) e o noise-aug de I2V (SVD, CogVideoX) usam contexto com ruído: correção parcial, porque ainda dá para copiar o layout.
- **WithAnyone** (ICLR 2026) usa pares de mesma identidade com variação real mais perda contrastiva: correção real.
- **Geirhos et al. 2020 (shortcut learning)**: ou o atalho sai do sinal de treino, ou o modelo é impedido de implementá-lo.
- **Correções reais:**
  - Tirar a cópia do alvo supervisionado: máscara espacial de loss, filtro de pares, pares cross-cena.
  - Tirar a vantagem da cópia alinhada: eixo de frame ou jitter de posição, ref em escala e recorte diferentes feitos na imagem, não no latente.
  - Supervisionar σ alto.
- **Paliativos:** ref_bias/γ, força da LoRA, early stop no 750, skip da ref nos primeiros passos (útil como cinto de segurança), ruído só na inferência, pesos por dHash global.

## 4. Recomendações priorizadas
**P0: sem treino, horas**
1. Usar o step1000 e o R3 com `REF_SKIP_FIRST=1`, medindo a consistência contra o step750.
2. Montar a avaliação acima: ~½ dia de código/CPU e 10–15 min de GPU por checkpoint.
3. Testar Turbo contra Raw (28 passos, CFG 3–4) no step1000: ~30 min.
4. Testar no node bias 1,0 contra 0,9999.

**P1: run "R4" (~3 h por 1.500 steps a 6–8,6 s/step)**
1. A partir do BETA1-R, adicionar uma **máscara espacial de loss**: peso ~0,1–0,2 nos pixels do alvo idênticos e alinhados à ref, calculados por par na resolução do latente (base: `overlap.py`). Alternativa mais simples: filtrar ou subamostrar os ~27 % de pares com mais de 50 % de área alinhada.
2. Somar **cobertura de σ alto**: `high_noise_prob` 0,15–0,25 em U[0,85; 1], ou mu de treino ≈1,15.
   - Não fazer isso sem o item 1: supervisionar σ alto sem a máscara supervisiona a média condicional, que hoje aponta para A.
3. Braços: R4a (só máscara) e R4b (máscara + σ alto), idealmente com 2 seeds de dados no R4b. São ~12 h no total, com sonda a cada 100–250 steps até 2.000+.

**Quanto ao R3 em andamento:** deixar rodar até ~1.250 como quase-réplica pareada e avaliar com e sem skip. Se a cópia migrar para o 3º passo (σ 0,905), confirma que o skip é paliativo.

**P2: dias**
- Pares cross-cena de verdade com filtro por banda de similaridade (DINO ~0,5–0,9) e menos peso em parents, poxima_v2 e inscene.
- Testar `slot_axis='frame'` ou jitter de posição.
- LR em queda depois de ~500 steps ou EMA do adapter, como medida secundária.

**Não fazer:** mais pesos por dHash global, ref_bias como solução, zoom no latente, ruído só na inferência, decisões com 1 run e 8 casos.

## 5. Confiança
- Mecanismo de cópia alinhada decidida em σ alto: alta.
- Dados como causa primária: média-alta. A máscara espacial é o teste decisivo que falta.
- Papel do mu diferente e da Turbo: média, ainda não testado.
- "Zoom acelera": baixa a média.
