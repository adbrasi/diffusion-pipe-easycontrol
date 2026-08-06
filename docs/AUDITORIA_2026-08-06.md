# Auditoria — a saga "próxima cena" no Krea 2 (2026-08-05/06, ~13h+)

Escrita após leitura integral de: OMINI_GROUNDED_SEGREDO, OMINI_GROUNDED_CANAL_SEMANTICO,
KREA2_MULTIREF_RECEITA, KREA2_EDIT_SAGA, KREA2_APEX_SPEC, RELATORIO_ELEMENTO_DISTOANTE,
RODADA2_ARM_C (Anima), ACHADO_REF_CFG_ALTO (Anima), NK2E NOTES/ROADMAP/README,
README + código do krea2edit-trainer (conradlocke), configs e logs reais dos runs.

---

## O mapa: todos os runs caem em dois polos

O adapter tem dois canais para satisfazer a MSE: o **texto** (que o base já domina — é
T2I treinado) e a **referência** (que exige aprender leitura cross-imagem, cara).
A escolha de arquitetura decide qual polo o treino alcança:

| polo | mecanismo | runs que caíram nele |
|---|---|---|
| **"bonito, sem relação com a ref"** | LoRA global: o gradiente satisfaz a loss pelo caminho do texto; o delta não instala leitura nova da ref | apex, NK2E global, armB, armC(braços saga) |
| **"segue a ref, mas copia/degrada"** | LoRA routado (condition-only): o único grau de liberdade é inflar a saliência da ref — a catraca | beta1 13k, sauce 1500@1024, saga512 tardio |

Os únicos resultados aprovados vivem **entre** os polos, todos com a mesma assinatura:
**routing + grounding + parada em ~0,5–1,2 época** (groundedsecret 250 steps·batch4,
NK2E step1250, EXP3 500, saga512 cedo).

---

## 1. Por que o apex decepcionou

O apex foi projetado para maximizar **compatibilidade com o node do conradlocke**
(LoraLoaderModelOnly + Krea2EditModelPatch), e para isso removeu de uma vez os dois
mecanismos que fazem os teus vencedores usarem a referência:

1. **Removeu o routing** (LoRA global em 258 módulos) — sem a catraca, o gradiente
   escoa pelo canal do texto.
2. **Trocou ref t=0 por t='target'** (exigência do node: um tvec único) — a matemática
   do próprio spec favorecia t=0 (Teorema 2); a escolha foi por compat, não por mérito.
3. A aposta substituta para forçar leitura da ref era o **regime de captions
   (55/40/5 clause-dropout) + schedule largo**. Insuficiente: 55% dos steps rodam a
   caption completa, e no CTX ela determina quase toda a imagem B — a função de treino
   continua sem nenhum termo que *obrigue* a leitura da ref.
4. Resultado: o LoRA aprendeu o "look" do dataset pela via do texto. Samples de treino
   pareciam ok (prompts = captions completas do dataset, o texto basta); com ref nova e
   prompt curto → **zero relação com a referência**. Exatamente o polo previsto.

**Foi treinado e usado corretamente?** Sim, verificado: smoke 3× (258 alvos, jitter
vivo por step, posições frame_fit corretas, VRAM 29,2 GB, 1,45 s/step), LoRA
comprovadamente aplicando (A/B strength 0 vs 1 difere), geometria bit-a-bit com o node
v1.2.5. Não é bug de execução — **o contrato é que está errado para a tarefa**.

**Agravante só no caminho ComfyUI:** o trainer do fork descarta o `weight_scale` do
fp8_scaled e requantiza (models/base.py:547, RCA da SAGA §4.1: sozinho isso deu
v relL2 0,58). O workflow que montei usa UNETLoader com fp8_scaled **sem** o
`K2 Training Base` — divergência numérica real em cima do problema de contrato.
O runner (paridade de treino) também deu "zero relação", então isso é agravante,
não causa raiz.

**Autocrítica:** o apex foi síntese minha. Otimizei o eixo errado — fusibilidade no
node stock — e paguei com os dois mecanismos que tinham evidência empírica tua a favor.
O RELATORIO_ELEMENTO_DISTOANTE já dizia: com uma moeda só, aposte na criação de
*necessidade* de ler a ref, não em geometria/schedule.

---

## 2. Por que o beta1 (e o omini grounded longo) te devolve a própria referência

**O mecanismo — a catraca do routing.** No condition-only, o delta do LoRA só age nas
rows da referência; a função com que o alvo LÊ (query/gate/MLP do alvo) fica congelada
(condition_lora.py). Consequência estrutural: **o único movimento que o otimizador tem
é aumentar a saliência da ref** (inflar K/V dela). Nenhum step consegue ensinar
"leia seletivamente". Cada step gira a catraca na mesma direção.

Em "próxima cena", A e B compartilham quase tudo (mesmo personagem/figurino/cenário) —
então **reconstruir A é uma solução de loss baixa** na maior parte dos timesteps. Com
milhares de steps a catraca satura: nos passos de alta-t (onde composição é decidida,
3 dos 8 passos do turbo), a atenção do alvo é dominada pela ref → a composição trava
na de A → o resto da integração reconstrói A. **Saída = tua imagem de input.**
"Quando funciona é perfeito" = quando o prompt empurra forte o suficiente para escapar
do atrator; "toda hora me devolve a ref" = o atrator.

**Agravantes específicos do beta1:**

1. **13k+ steps.** Todo resultado routado que você aprovou vivia em 250–1.250
   exposições (~1 época). O beta1 passou 10× disso. A sauce mostrou a mesma seta:
   já em 1.500 steps @1024, "praticamente uma cópia" — a pressão de cópia cresce
   monotonicamente com os steps, independente do dataset (por isso é método, não dado).
2. **width_shift tem atração de costura** (+2,0 nats na correspondência da emenda,
   13/24 pares de fase descoerentes a 512, 15/24 a 1024): a geometria ativamente puxa
   queries do alvo para a ref — combustível extra para a catraca.
3. **Fase 2 a 1024 com flux_shift@1024**: ~50% da massa de t no miolo, 4% abaixo de
   t=0,3 e 0,1% abaixo de 0,1 — a banda que resolve pele/tecido/textura quase não
   treinou → o "degrada quando segue".
4. **O lr da fase 2 nunca caiu** (resume do DeepSpeed restaurou 1e-4; o 7,5e-5 da
   config foi ignorado em silêncio — documentado na SAGA §6.1). A fase 1024 inteira
   rodou 33% mais quente do que você pediu.

**Por que o groundedsecret NÃO faz isso:** é o mesmo mecanismo, parado no ponto certo —
250 steps × batch 4 ≈ 0,8 época, antes da catraca saturar — e com o canal semântico
inteiro (grounding longest-side 768; o beta1 usou 384², que carrega ~2,2× menos
informação semântica da ref). O groundedsecret não é um método diferente do beta1;
é o beta1 jovem e com o grounding cheio.

---

## 3. O que NUNCA foi exercido (e não custa treino)

1. **O dial de fidelidade na inferência.** Precedente duro do Anima
   (ACHADO_REF_CFG_ALTO): identidade (adorno, marcas do figurino) só aparece em
   ref_cfg 2–3; avaliar em 1,0 mede clima/paleta. TODA a avaliação do dia rodou com
   reference_guidance/ref_cfg = 1,0.
2. **`block_strength` < 1 no beta1** (node CtxRush tem dials separados). A catraca
   inflou o delta da ref; **escalar o delta para baixo (0,5–0,8) na inferência
   desanda a catraca sem retreinar** — é o antídoto direto do "me devolve a ref".
3. **`fusion_strength` > 1** (1,5–3): o canal semântico carrega ~1% da energia do
   adapter; superamplificá-lo é seguro e barato (CANAL_SEMANTICO §0).
4. **Colunas de controle para teu olho**: mesma seed, (a) ref verdadeira, (b) ref
   embaralhada, (c) seed diferente — separa "usa a ref" de "decorou", sem métrica
   nenhuma, veredito 100% teu.

---

## 4. Konrad / ostris / NK2E — o que cada um confirma

| fonte | o que faz | o que confirmamos |
|---|---|---|
| **conradlocke** | LoRA global + captions-instrução + grounding 768 jitter/step + fit AR + uniform+peso de t + **sampling em treino desligado por princípio** (o preview mentiria) | o contrato dele só funciona *inteiro*: global sem instrução (apex) cai no polo texto. Grounding 768 + jitter é consenso com o teu probe vencedor |
| **ostris (ai-toolkit builtin)** | edit mode próprio: "Picture N", resize por área, **ref t=0** | t=0 tem dois trainers públicos independentes usando; 'target' só existe por limitação do node |
| **NK2E** | global simples, latent-cache only, resume atômico, fusível no loader padrão | infra é a melhor das três; EXP3 (grounding+t0+shift) foi teu "melhor do dia" — o shift de schedule tem efeito visual real |

---

## 5. Plano de correção (mira: consertar groundedsecret + beta1, depois UM retreino)

### Fase 0 — resgate sem treinar (~40 min GPU)
Varrer nos adapters JÁ EXISTENTES (beta1 step13250 e groundedsecret step250/1000),
mesma seed, mesmos pares:
- `block_strength` ∈ {0,6 / 0,8 / 1,0} × `fusion_strength` ∈ {1 / 2} (node CtxRush);
- `ref_cfg`/`reference_guidance` ∈ {1,0 / 1,5 / 2,5} no runner;
- colunas ref-verdadeira / ref-embaralhada / seed-77 para teu veredito.
Se `block_strength` 0,6–0,8 matar o "devolve a ref" mantendo o "quando funciona é
perfeito", o beta1 vira utilizável HOJE.

### Fase 1 — o retreino (receita groundedsecret + 4 correções mecânicas)
Base: contrato groundedsecret intacto (routing + txtfusion global + grounding 768 +
ref t=0 + width_shift + lr 1e-4 + rank 64). Mudanças, cada uma com mecanismo:

1. **Disciplina de época**: checkpoints/250, avaliação na janela 0,5–1,5 época do
   dado curado; **nunca** deixar a catraca passar disso. O melhor checkpoint é
   escolhido pelo teu olho, não pelo último step.
2. **Vacina anti-cópia — dropout de aparência (Composer-style)**: em ~25% dos steps,
   zerar os tokens VAE da ref mantendo o grounding. Sem a aparência para copiar, a
   loss só cai via canal semântico → a catraca ganha um contrapeso estrutural.
   (Mecanismo já mapeado em CANAL_SEMANTICO §4; é mudança de runtime, não de cache.)
3. **caption_dropout 0,1** (uncond grounded treinado → CFG/ref_cfg vira dial
   confiável — pré-requisito da Fase 0 funcionar no adapter novo).
4. **Cobertura de t**: shift por resolução com alargamento (a lição do EXP3, que você
   elegeu "melhor do dia", + a tabela do conradlocke): garantir massa nas duas pontas
   (t>0,8 E t<0,3) para não trocar aderência por textura.

### O que NÃO mudar
Geometria (width_shift provada nos teus dois melhores — frame axis fica como A/B
futuro), rank, otimizador, lr, batch 1 sem accum, turbo como protocolo de avaliação.

---

## Estado da bateria comparativa
16 gerações (8 métodos × instrução/descritivo), input `/workspace/000004.jpg`,
512, seed 76, turbo, tudo plano em `AdwolfCzar/apex-ctx-1/testes/000004_todos/`.
`k2-proxima-cena-grounded-v1` está vazio no HF (só .gitattributes) — sem gerações dele.
