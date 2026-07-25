# Krea 2 multi-referência — Macro-Dataset sobre o Omini-Grounded

**Pesquisa de 2026-07-25, v2.** Esta é uma reescrita completa. A v1 foi
escrita com leitura rasa e continha uma evidência inventada, uma tabela de
custo errada por até 2.2×, três diagnósticos de código errados e uma seção
inteira construída sobre uma premissa falsa. O que sobreviveu está marcado;
o que caiu está no §10, para não ser reintroduzido por engano.

**Pergunta:** dá para treinar o Krea 2 com N referências separadas, usando o
`Azily/Macro-Dataset`, partindo do `krea2_omini_grounded`?

**Resposta:** sim. Existe **um** bloqueio arquitetural de verdade, ele tem
correção de uma linha, e quase todo o resto é encanamento. Os custos reais
são bem menores do que a v1 dizia; o gargalo verdadeiro é disco, não FLOPs.

---

## 1. O achado central: uma constante que torna as referências indistinguíveis

`models/krea2_reference.py:359`, o `width_shift`:

```python
reference_pos[..., 2] = reference_pos[..., 2] + float(target_grid_w)
```

Isso soma uma **constante**. A implementação mais natural de multi-referência
— um laço sobre N reusando essa linha — dá a **todos os N spans posições de
RoPE idênticas**. E aí:

> **Teorema.** Se dois spans de referência de mesmo comprimento recebem
> posições de RoPE idênticas, a saída do DiT sobre o target é *exatamente*
> invariante à troca dos dois.

A prova é direta e vale para qualquer quantidade de dados:

1. atenção é soma sobre keys ponderada por softmax — permutation-invariant
   sobre o conjunto de keys, dadas as rotações que eles carregam;
2. `tvec` é o mesmo para todas as linhas de referência (`krea2_reference.py:335`,
   zeros);
3. a máscara é a mesma para todas (`:364-367`, ones; ou `:375`, bloco uniforme);
4. o router LoRA aplica a mesma máscara a todas (`condition_lora.py:51-52`,
   sem granularidade por slot);
5. MLP e norms são pointwise.

Logo trocar os dois sub-spans é uma permutação de tokens com metadados
idênticos → saída idêntica. **Não é "difícil de aprender". É impossível em
princípio.**

**A correção é uma linha:** offset cumulativo por índice, slot *i* em
`w += (i+1) * ref_grid_w`. Como o loader já força toda referência ao grid do
target (§4), na prática é `(i+1) * target_grid_w`. Com isso os spans deixam de
ser permutáveis e **não sobra nenhum bloqueio arquitetural**.

Este era o único conteúdo indispensável da v1, e a v1 passou ao lado dele
enquanto debatia se `<image 1>` precisava de um mecanismo dedicado.

---

## 2. `<image N>` não precisa de mecanismo — o precedente está no projeto

`<image 1>` é texto no caption. O binding é aprendido por correlação, como
qualquer convenção. Isto não é especulação: **o projeto já fez exatamente
isso.**

`docs/ANIMA_SAGA_COMPLETA.md` §7 — o dataset de abril do Anima ensinou três
**verbos-operação** (`create the next scene, same character` /
`create a different scene, different characters` / `Change the scene to…`) por
pura correlação. ~21k pares, ~1950 steps a batch 8 (~15.600 exposições, menos
de uma época). O adapter que saiu daí é o que o usuário considera o melhor que
já teve, e o prefixo virou trigger de inferência.

`<image 1>` ↔ primeiro span é a mesma classe de problema, com 400k amostras
(19× mais). E o IC-LoRA original já usa esse formato: *"A composite of N
images. **[IMAGE1]** description 1 **[IMAGE2]** description 2…"*
(`/workspace/research/2026-04-05-ic-lora-deep-research.md` §1.2).

Três ressalvas que o mesmo corpus impõe:

1. **Escala.** O probe de 250 steps que validou o Omini-Grounded não serve para
   isso. A regra do projeto: *"o número que importa não é `max_steps`, é
   quantas vezes o modelo viu a rota difícil"* (`ANIMA_FUTURO_TREINO.md` §7).
2. **Posição do marcador no caption.** `ANIMA_TRAINING_GUIDE.md`: o Qwen3 tem
   atenção causal, *"trigger words no fim de prompts longos perdem
   efetividade"*. Nos prompts do Macro, `<image 2>` costuma aparecer no meio ou
   no fim de frases longas. Mitigação barata: prefixar.
3. **Tokenização.** `<image 1>` vira `<`,`image`,`1`,`>`. Não impede o
   aprendizado (sequência estável repetida 400k vezes), mas é o argumento a
   favor de reescrever para `Picture 1` — que é o vocabulário que o Qwen3-VL
   instruction-tuned usa nativamente e que `build_vl_image_prompt`
   (`krea2_edit.py:44`) já emite no canal de grounding.

E há um caminho ainda mais curto que ninguém tinha notado: no caminho grounded,
`<image 1>` no caption e `Picture 1:` no bloco de visão passam **pelo mesmo
encoder Qwen3-VL, na mesma sequência**. O binding pode se resolver dentro do
TE, sem atravessar o DiT. Isso torna os knobs do canal semântico
(`docs/OMINI_GROUNDED_CANAL_SEMANTICO.md`) um **pré-requisito**, não um
refinamento — ver §6.

---

## 3. O dataset

400k amostras, **média de 5,44 referências**, até 10 por amostra. CC BY 4.0.
Paper: arXiv 2603.25319 (HKU-MMLab). Índice JSON:

```json
{"task": "customization", "idx": 1,
 "prompt": "Create an image of the modern glass and metal interior from <image 2>, applying the classical oil painting style from <image 1> globally across the entire scene.",
 "input_images": ["…/image_1.jpg", "…/image_2.jpg"],
 "output_image": "…/image_output.jpg"}
```

Quatro tarefas (customization, illustration, spatial, temporal), bracketizado
por contagem de referências: `1-3`, `4-5`, `6-7`, `≥8`.

**Ablação do paper, que contradiz o plano da v1:** o desempenho *degrada*
conforme N cresce em todos os modelos testados, e *"upweighting large-input
samples substantially boosts high-input performance"*. Ou seja, o problema **é**
o N alto — treinar só o bracket 1-3 não o ataca, e 1-3 é a cauda fácil, não o
corpo do dataset.

O paper faz fine-tune de **Bagel, OmniGen2 e Qwen-Image-Edit** — todos com
capacidade multi-imagem nativa. Nenhum inventa mecanismo de endereçamento.
O Krea 2 é diferente: o **DiT** nunca viu N referências, mas o **text encoder**
(Qwen3-VL-4B) é multi-imagem nativo. Essa assimetria é a coisa mais importante
a entender sobre este projeto.

### As captions do Macro contra os dois atalhos conhecidos

O antagonista nº1 documentado no projeto é o **atalho de caption** (a caption
descreve tudo, a referência vira supérflua); o nº2 é o **atalho de
reconstrução** (o alvo é quase igual à referência, copiar minimiza a MSE).

- Os prompts de `customization` são razoavelmente delta-style — descrevem a
  operação, não o resultado. Bom sinal.
- A tarefa **`spatial`** (*"`<image 1>` is the left view… Generate the back-left
  view"*) é **exatamente a configuração do colapso de reconstrução do
  Ideogram4 P3**, onde *"nenhuma combinação de CFG escapa da reconstrução — o
  colapso está nos pesos, não no guidance"* (`IDEOGRAM4_DEBATE_METODOS.md`).
  Tratar `spatial` com cuidado, ou deixar fora do primeiro probe.

---

## 4. Auditoria: o que já funciona, o que quebra

### Funciona sem mudança

**Router de LoRA.** `condition_lora.py:23-26` marca um span contíguo;
`krea2_ominicontrol.py:76-77` define `set_reference_span(text_length +
target_length, seq_len)`. Com N spans contíguos no fim, continua correto por
construção. E `mask_for` (`condition_lora.py:46-50`) valida o comprimento da
sequência, então mudanças são auto-verificadas.

**Slots de tamanho fixo — já são o status quo.** `utils/dataset.py:1124` passa
o `size_bucket` **do target** ao preprocessador, e `models/base.py:151` faz
center-crop + resize. Toda referência já sai com exatamente o grid do target.
A v1 apresentava isso como decisão de design a tomar; é fato a documentar. O
custo real é o oposto do que a v1 dizia: o pipeline **obriga** as N refs ao
tamanho do target, destruindo o AR de cada uma — perda de conteúdo silenciosa,
não crash.

**RoPE tem banda suficiente.** `axes_dim=[32,48,48]` (frame,h,w), θ=1000,
headdim=128. Períodos máximos: frame 4080, h/w 4712. Mas o que importa não é
wraparound — é quais frequências discriminam um deslocamento de slot Δw=32.
Para 8 slots não-ambíguos (extensão 256) é preciso ω·256 < 2π → **11 pares de
24 ficam inequívocos**, com fase adjacente entre 0.76 e 0.043 rad. Suficiente e
não-degenerado.

> Limitação que decorre disso: no `width_shift`, "qual imagem" e "onde dentro
> da imagem" ficam no **mesmo eixo, na mesma escala**. Os pares intermediários
> servem às duas funções e o modelo tem que desemaranhá-las de uma coordenada
> 1-D. Não é impossível — é o que qualquer modelo de concat faz — mas é caro
> comparado a um eixo dedicado.

### Funciona no arquivo, morto no sistema

**O grounding aceita lista** — `krea2_edit.py:250`:
```python
files = control_file if isinstance(control_file, (list, tuple)) else [control_file]
```
e `:265-268` emite `VISION_BLOCK * len(images)` ou `build_vl_image_prompt(n)`.
**Mas nada a montante jamais produz uma lista.** `utils/dataset.py:721` guarda
uma string por linha. É branch morto, nunca exercitado. Dizer "zero linhas de
mudança" é verdade sobre o arquivo e falso sobre o sistema — a v1 se
contradizia, listando o loader como bloqueio na seção seguinte.

### Quebra (inventário)

**Bloqueios duros:**

| local | o quê |
|---|---|
| `krea2_reference.py:294` | `raise` — "supports one target and one reference frame" |
| `krea2_reference.py:85` | `get_call_vae_fn` aceita 1 ou 2 args, `raise` em 3+ |
| `krea2_reference.py:122-132` | exige `reference.shape == target.shape` |
| `utils/dataset.py:1125-1126` | `assert len(control_items) == 1` |
| `utils/dataset.py:715` | `raise` se o stem não casar 1:1 |
| `krea2_edit.py:144-147` | **proíbe `condition_token_stride != 1`** |
| `krea2_edit.py:151-157` | **proíbe `condition_dropout != 0`** |

**Geometria que assume uma ref silenciosamente:** `krea2_reference.py:296-297,
302-306, 312-316, 322-323, 333-335, 346-353, 359, 364-367, 373-375, 382-385`.
Mecânico, mas volumoso.

**Dataset:** `utils/dataset.py:678, 713-716, 721, 1123-1127` — uma string por
stem, por construção.

**Avaliação: não existe.** `models/base.py:188` —
`prepare_sample_test(prompt, negative_prompt, cfg)`, sem control. `train.py:588`
chama só com prompt. **Não há amostragem de eval funcional para nenhum pipeline
de referência do Krea 2**, nem com uma referência. E os nodes ComfyUI recusam
N>1 explicitamente (`ctxrush_edit/nodes.py:89` e `:336`), e
`tools/infer_reference_adapter.py:75` tem `--reference` singular.

### Risco de ordem — real, mas concentrado numa linha

O acoplamento entre canais está seguro: ambos leem a mesma linha do mesmo
dataset de metadados, e o join final é por chave (`image_spec`), não por
índice. Shuffles reordenam amostras, nunca conteúdo dentro de uma amostra.

O ponto de falha é `utils/dataset.py:678`:
```python
control_file_stems = {path.stem: path for path in self.control_path.glob('*') if path.is_file()}
```
`Path.glob` retorna em ordem de `os.scandir` — **arbitrária e dependente do
filesystem**. Note que a linha `:672-674` ordena explicitamente os arquivos de
imagem (`files.sort()`) mas **não** os de controle. Hoje é inofensivo (dict com
casamento 1:1). No momento em que a ordem do glob virar a ordem dos slots, o
binding `<image 1>`→slot 1 embaralha por amostra e o aprendizado por correlação
morre.

**Para o Macro isso some de graça:** o manifesto JSON já traz `input_images`
como array ordenado. Regra: **nunca derivar ordem de filesystem; derivar do
manifesto.** E assertar que `len(control_files)` do canal VAE bate com
`len(images)` do canal grounding, na mesma amostra.

---

## 5. Custo real — a v1 errou por até 2.2×

A v1 apresentou ratios de `L²` como se fossem custo total. Não são.

- Atenção: `28 × 4 × L² × 6144 = 6.881e5 · L²` FLOPs
- Lineares (blocos): `2 × 12.156e9 × L = 2.432e10 · L` FLOPs
- **Cruzamento em L = 35 344 tokens** — muito acima de qualquer configuração
  de 1 a 8 referências.

Ou seja: o custo é **dominado pelo termo linear**, não pelo quadrático. A
atenção é 7% do custo em N=1 e ainda só 28% em N=8.

Comprimento de sequência com target 512px, grounding a 768 (576 tokens por
imagem no canal de texto — a v1 assumiu texto constante, outro erro):
`L(N) = 1088 + 1600N`

| N | L | custo real vs N=1 | a v1 dizia |
|---|---|---|---|
| 1 | 2 688 | **1.00×** | 1.0× |
| 2 | 4 288 | **1.66×** | 2.0× |
| 3 | 5 888 | **2.38×** | 3.2× |
| 5 | 9 088 | **3.95×** | 6.8× |
| 8 | 13 888 | **6.69×** | 14.4× |

**Consequência: "o bracket ≥8 é outro projeto" está errado.** É ~6.7× por
step. Caro, não proibitivo. Isso importa porque a ablação do paper diz que o N
alto é justamente onde está o problema.

### VRAM — RTX 5090 32 GB, B=1, fp8, `blocks_to_swap=8`, checkpointing

O monstro de memória é o **`tvec` de shape `(B, L, 36864)`** — 6× o hidden
state — produto do timestep per-token que é a base de todo o contrato de
referência. A v1 não o menciona.

| N | L | total estimado |
|---|---|---|
| 1 | 2 688 | ≈ 16.2 GB |
| 3 | 5 888 | ≈ 18.6 GB |
| 5 | 9 088 | ≈ 21.2 GB |
| 8 | 13 888 | ≈ 24.8 GB |

**A 512px tudo cabe, até N=8.** A 768px (`L(N) = 2368 + 2880N`), N=3 dá ≈21.6
GB (ok) e N=8 dá ≈31.6 GB (**OOM na prática**).

Alavancas, por eficácia: `blocks_to_swap` maior (~0.43 GB por bloco);
`vl_image_max_pixels=384²` em vez de `vl_longest_side=768` (corta o texto de
576 para 144 tokens por referência — a 8 refs são −3456 tokens, ~25% da
sequência).

**A alavanca que a v1 propôs é ilegal:** `condition_token_stride` levanta
`ValueError` em `krea2_edit.py:144-147`, e o grounded herda. Reforçado por
veto de qualidade independente: *"stride 2 perde rosto/cabelo/figurino em
anime"* (`IDEOGRAM4_DEBATE_METODOS.md`).

### O gargalo verdadeiro é disco, e é o cache de embeddings

A v1 só olhou disco de imagem. O que já estourou o disco neste projeto uma vez
(commit `f94067e`, *"incidente de disco cheio 2026-07-19"*) foi o cache de
text-embeddings.

`reference_adapters.md` mede: *"(caption + ~144 vision tokens) × 12 layers ×
2560 × 2 bytes ≈ **8-14 MB de cache por par**"* — isso com **uma** referência a
384px. São ~60 KiB por token.

- a 768px: ~576 tokens/ref → **~35 MB por referência**
- Macro com média 5,44 refs → ~190 MB por amostra
- 400k amostras → **~76 TB**

Com **142 GB livres**, mesmo restrito ao bracket 1-3 com grounding baixado a
256px (~64 tokens, ~4 MB/ref), são ~10 MB/amostra → **~14k amostras**. Este é
o blocker prático nº1 e determina o tamanho do experimento muito mais do que
FLOPs ou VRAM.

---

## 6. O risco arquitetural que a v1 não mencionou

O plano é construído sobre o **routing condition-only**. A v1 vendeu isso como
"✅ zero linhas de mudança" e não disse que:

- **o routing PERDEU na bateria do Anima** (armC), com efeito grande e
  consistente;
- o mecanismo da derrota é exatamente o que multi-ref exige. De
  `RODADA2_ARM_C_ROUTING.md`: *"o delta só é somado nas rows da referência
  […] os pesos que o **ALVO** usa para LER essa informação nunca são
  modificados"*. Ou, como ficou em `ANIMA_FUTURO_TREINO.md`: **"ele enriquece o
  que é escrito e não ensina ninguém a ler."**
- o Krea 2 só escapou disso porque tem canal semântico em paralelo — e esse
  canal carrega **1,13% da energia do adapter**, com **rank efetivo colapsado
  a ~1/64 em 250 steps** (`OMINI_GROUNDED_CANAL_SEMANTICO.md`,
  `OMINI_GROUNDED_SEGREDO.md` §9).

Com N referências e um caption que pede seleção (*"a mulher de `<image 1>`, o
estilo de `<image 2>`"*), **ler seletivamente é a tarefa inteira**. Apostar num
método cujo modo de falha documentado é "não treina a leitura", com o canal de
leitura a 1% de capacidade, é o risco central deste plano.

**Consequência prática:** os knobs de `OMINI_GROUNDED_CANAL_SEMANTICO.md`
deixam de ser refinamento opcional e viram pré-requisito. Em ordem de custo:

1. `rank_pattern={'txtfusion': 128}` no PEFT — *"fácil, recomendado"*; save/load
   e o node já leem shapes do próprio tensor, ranks mistos funcionam sem
   mudança na inferência;
2. `txtfusion_lr` como param group separado (precedente: `llm_adapter_lr` do
   caminho Anima);
3. incluir `.projector.` nos targets — ele comprime 30720→6144 e é o gargalo
   de informação do canal.

Alvo declarado no doc original: sair de ~1% para **5-15%** sem colapsar a
aparência. Aviso do mesmo doc: **não** subir o lr global para isso — os blocks
aceleram junto e o equilíbrio não muda.

### `condition_dropout`: proibido aqui, e já resolvido no Ideogram 4

`krea2_edit.py:151-157` levanta `ValueError`. O motivo está no comentário:
zerar os latentes VAE não zera o que o Qwen3-VL viu — *"an incoherent partial
dropout"*. O grounded herda a proibição.

Mas o achado do Anima diz que uma tarefa de **atribuição** precisa de dropout
(`ANIMA_FUTURO_TREINO.md` §5: *"arm1 acopla, armD atribui"*), e um caption com
N referências endereçadas é a definição de atribuição.

**A saída já existe e está implementada — do outro lado.**
`models/ideogram4_omini_grounded.py:247-274` faz um Bernoulli único que troca
*simultaneamente* o embedding grounded pelo text-only **e** zera os latentes
VAE, com guarda contra grounding morto. Custo: cache dobra — sobre um cache
que já é o gargalo (§5).

O substituto barato que o projeto já construiu para o Krea 2 é o
**`caption_dropout`** (`krea2_edit.py:135-144`): caption vazio mantendo a
referência nos dois branches, que é o incondicional exato que o CFG usa. A
auditoria do Ideogram4 (`ddd90a8`) é enfática: *"uncond_fraction/caption_dropout
NUNCA esteve ativo nos pilotos — só a referência era dropada, **o inverso do
necessário**"*. Se o binding `<image N>` é aprendido por correlação texto↔span,
`caption_dropout` tem que estar **no probe**, não na fase 4 como a v1 propunha.

### `independent_condition` — a otimização que ninguém testou

`krea2_reference.py:369-378`: com ele, as linhas de referência não veem texto
nem target. Combinado com `reference_timestep='zero'`, o forward do span de
referência fica **invariante ao passo de sampling** — pode ser computado uma
vez e cacheado para todos os passos. Com N=8 é a diferença entre pagar 8192
tokens por passo e pagá-los uma vez. É o que o OminiControl2 faz
(`krea2_ominicontrol2.py:12`).

Nunca foi testado em Krea 2 (default `False`, ligado só num `.toml` sem
registro de execução). Contra: a máscara aditiva densa é `L²` —
69 MB a L=5888, mas **1.29 GB a L=25408**. E qualquer máscara não-`None`
desliga o backend flash do SDPA (~20-30% de throughput) — o que já vale para
todo treino Krea 2 deste fork.

---

## 7. Plano

### Fase 0 — pré-requisitos (sem GPU)
- **Os modelos do Krea 2 não estão nesta máquina.** Confirmado duas vezes.
  Baixar o DiT fp8-scaled e o Qwen3-VL-4B.
- Baixar `customization` (não `spatial`, §3) do bracket 1-3, dimensionado pelo
  orçamento de cache do §5 — na ordem de 10-15k amostras, não 400k.
- Decidir `vl_longest_side`: 768 é o contrato validado, 384 corta o cache por
  ~4×. Provavelmente 384 por necessidade, e registrar como desvio do contrato.

### Fase 1 — loader por manifesto
`control_path` ganha alternativa `manifest_path` lendo os JSON do Macro.
Preserva `input_images` na ordem do manifesto. Assert de casamento entre
canais. Bucketiza por N.

### Fase 2 — geometria multi-ref
Laço sobre N em `Krea2ReferenceInitialLayer.forward`, **offset cumulativo
`(i+1)·W`** (§1). Router e grounding intocados.

> Armadilha documentada, aplicável aqui: `to_layers()` já repassa 5 parâmetros
> de geometria à mão, e um parâmetro esquecido nesse repasse
> (`reference_timestep_mode`) já custou *"um dia de debugging"*
> (fix `a66d7ba`). Adicionar N cria a 6ª oportunidade do mesmo bug. Lição do
> doc: **o contrato real é o que o código executa, não o que a config diz.**

### Fase 3 — avaliação ANTES do treino
Não existe eval para pipelines de referência do Krea 2, nem com uma referência
(§4). Construir primeiro. O histórico é explícito sobre o custo de não fazer
isso: modo de inferência errado produz *"ruído puro, fácil de confundir com 'o
método falhou'"* (commit `2b52409`), e no Ideogram4 *"o harness estava
quebrado […] invalidava qualquer avaliação anterior"*.

Critério: com 2 referências, **trocar a ordem tem que trocar quem é quem na
saída**. É o teste de referência embaralhada do Anima adaptado, e é o único que
separa endereçamento de mistura. Aplicar as regras que o projeto já adotou por
escrito: controle base-nativo obrigatório, ≥3 seeds, ~10 exemplos held-out,
métrica só para triagem, **decisão visual do usuário**.

> Nota sobre o controle: com routing, *"zerar o delta nas rows do alvo NÃO
> zera o efeito da presença dos tokens de ref na atenção base"* — logo
> "base sem refs" e "packing com scale 0" são controles **distintos**
> (`IDEOGRAM4_DEBATE_METODOS.md`, veto Codex #3).

### Fase 4 — probe
Contrato do Omini-Grounded (`width_shift` cumulativo, `reference_timestep='zero'`,
`condition_only_lora=true`), **mais** `vl_prompt_style='picture_n'`,
**mais** `caption_dropout=0.1` desde o início (§6), **mais**
`rank_pattern={'txtfusion': 128}` (§6). Steps na ordem de milhares, não 250 (§2).

### Fase 5 — só então decidir sobre N alto
A ablação do paper diz que o problema é o N alto e que *upweighting* de
amostras grandes ajuda. Com o custo real de 6.7× (não 14.4×), isso é uma opção
viável — mas só depois que a Fase 3 provar que o endereçamento emerge em N=2.

---

## 8. Riscos e incógnitas

1. **Disco de cache** (§5) é o limitante real e limita o experimento a ~14k
   amostras. Tudo mais é secundário a isso.
2. **O routing pode não ensinar a ler** (§6). É o risco arquitetural central e
   não tem mitigação barata além de fortalecer o canal semântico.
3. **`spatial` é armadilha de reconstrução** (§3).
4. **Um bug pré-existente que multi-ref agrava:** `krea2.py:220`,
   `krea2_reference.py:318` e `model.py:252` chamam `txtfusion(context,
   mask=None)` — a máscara de padding do texto **nunca chega aos
   `refiner_blocks`**. Tokens de padding entram na self-attention com peso
   softmax uniforme não-nulo e diluem tokens reais. Hoje é ruído pequeno; com N
   variável e 576 tokens de grounding por referência, o padding pode virar
   metade do stream de texto.
5. **VRAM é estimativa, não medição** — e os pesos do Krea 2 não estão aqui
   para medir.
6. **Nenhum veredito de Krea 2 neste projeto passou pelo protocolo do Anima**
   (10 exemplos × ≥3 seeds × ref embaralhada × decisão visual do usuário).
   Todos foram por loss, um held-out e forense de pesos.

---

## 9. Uma divergência a resolver com o usuário

O usuário afirmou que o `ominigrounded` foi *"o melhor treinamento krea 2
disparado, praticamente o único que funcionou de verdade"*.
`docs/OMINI_CONTROL_KREA2.md` §6 diz o contrário: *"segundo melhor resultado
[…] a fidelidade exata é inferior ao omini puro […] **NÃO é a receita
recomendada para produção**"*, e chama o **omini puro** de *"o melhor método
validado do projeto"*.

Os docs foram escritos por mim em sessões passadas; o usuário viu as imagens.
Não há nenhuma citação literal do usuário sobre Krea 2 em nenhum doc ou commit
— toda a assimetria de rigor entre o lado Anima (veredito visual do usuário em
cada rodada) e o lado Krea 2 (veredito do Claude) está aí.

**Isso não bloqueia o plano.** O omini puro dá *"fidelidade sem semântica"* e
uma tarefa de N referências endereçadas por texto é semântica quase por
definição — o grounded é a base certa para este problema independentemente de
quem venceu no anterior. Mas vale resolver antes de fixar a receita.

---

## 10. O que a v1 errou — para não ser reintroduzido

1. **Evidência inventada.** Afirmei que `position_mode='subject'` *"já foi
   testado contra `width_shift` neste projeto e perdeu"*. **Esse A/B não
   existe** em nenhum doc ou commit. O que existe é uma comparação entre
   métodos que diferem em ~5 variáveis. Pior: o eixo de frame é o **contrato
   público do Krea Edit** (`krea2_edit.py:9`: *"clean latents, t=0, RoPE frame
   1"*), o default do código ainda é `'subject'` (`krea2_reference.py:50`), e
   o único precedente de pesquisa do projeto para multi-ref usa o eixo temporal
   (HunyuanCustom, índice −k). Rejeitei por OOD o eixo que o modelo usa em
   produção.
2. **Tabela de custo errada por até 2.2×** — tratei ratio de `L²` como custo
   total (§5).
3. **Assumi texto constante** — no grounded o texto cresce 576 tokens por
   referência (§5).
4. **`utils/dataset.py:877` "descarta o resto"** — diagnóstico errado, aquele
   `[0]` é a dimensão de batch de um `map` batched. O bloqueio real é `:713-716`.
5. **"Referências de tamanhos diferentes quebram"** — não quebram; o loader já
   força tudo ao bucket do target (§4).
6. **"Grounding: zero linhas"** — verdade sobre o arquivo, falso sobre o
   sistema; o branch de lista é código morto (§4).
7. **`condition_token_stride` como mitigação** — é `raise` no caminho grounded.
8. **O §5 inteiro** ("como `<image 1>` vira endereçável") era elaboração sobre
   premissa falsa. Sobreviveram: a observação de que o texto está na origem do
   sistema de coordenadas (e por isso lê endereços absolutos), e nada mais.
9. **Não mencionei que o routing perdeu no Anima** (§6) — a omissão mais grave.
10. **Não considerei o cache de embeddings** (§5) — o gargalo real.
11. **`caption_dropout` na fase 4** — tem que estar no probe (§6).
12. **"Bracket ≥8 é outro projeto"** — é 6.7×, e é onde está o problema (§5).
