# Anima "próxima cena", método A/aligned: como o treino funciona (para quem vai escrever as legendas)

Este documento descreve o treino que a gente **realmente executa**: modelo `anima_nextscene`,
`rope_layout = "aligned"`. Ele foi escolhido no veredito visual do usuário em 2026-09-30 e é a
base do run 1024 v2. Tudo aqui foi conferido no código desta branch (`models/anima_nextscene.py`,
`models/cosmos_predict2.py`, `models/llm_adapter.py`, `utils/dataset.py`, `infer_easycontrol.py`),
no model card oficial do Anima (revisão `f973fc41`) e em medições feitas em 2026-10-08.
Onde algo é **inferência** e não fato, isso está marcado.

---

## 1. A tarefa em uma frase

Dada uma imagem **A** (referência) e uma legenda, o modelo gera **B**, a "próxima cena": mesmo
personagem e mesmo mundo, outro momento, pose, enquadramento ou lugar. No treino, cada amostra é
um par (A, B) e uma legenda **de B**.

---

## 2. O que entra no modelo a cada passo de treino

```
imagem B  ──VAE──► latente B (16 canais, 1/8 da resolução) ──+ ruído(σ)──► frame 0 (alvo ruidoso)
imagem A  ──VAE──► latente A                                  (limpo)  ──► frame 1 (referência)
legenda de B ──► Qwen3-0.6B (congelado) ──► llm_adapter (congelado) ──► cross-attention do DiT
```

Fatos do contrato (`anima_nextscene.py`, `contract()` gravado em cada adapter):

| item | valor no A/aligned |
| --- | --- |
| empacotamento | `x = [B_ruidoso (T=0) | A_limpo (T=1)]`, concatenados no eixo temporal |
| timestep por frame | `t = [σ, 0]`: o alvo tem ruído σ, a referência é marcada como **limpa** (t=0) |
| RoPE (posição) | **aligned**: o token (h, w) da referência fica na mesma posição espacial (h, w) do alvo, só com tempo 1 em vez de 0 |
| loss | MSE da velocidade **só no frame do alvo**; a referência nunca é "prevista" |
| `ref_dropout = 0.1` | em 10% das amostras a referência vira **zeros** (a legenda continua); isso treina o "sem referência" usado no ref-CFG |
| `high_noise_prob = 0.2` | 20% das amostras usam σ entre 0,8 e 1,0, onde quase nada de B sobrevive e o modelo depende de **A + texto** para compor a cena |
| `diff_weight = false` | todas as regiões de B pesam igual na loss |
| A e B | sempre no **mesmo bucket de tamanho**; a área fica em ~1024² e o aspecto vai de 0,5 a 2,0 |

Como A está "alinhado" ponto a ponto com B, a atenção trata A como um quadro vizinho no tempo,
como dois frames seguidos de um vídeo. A pesquisa do projeto mediu que isso cria uma tendência
natural a **copiar A** (`ANIMA_NEXTSCENE_PESQUISA_2026-09.md`, P1). Mesmo assim, o usuário
preferiu visualmente o aligned ao disjoint. A legenda é uma das principais forças que puxam
contra a cópia, porque é ela que diz **o que muda**.

---

## 3. O que é treinado e o que é congelado

| componente | treinado? | detalhe |
| --- | --- | --- |
| VAE (Qwen-Image) | não | os latentes de A e B ficam em cache |
| **Qwen3-0.6B** (encoder de texto) | **não** | `requires_grad_(False)`; as saídas ficam **em cache** (`cache_text_embeddings` = true por padrão) |
| **llm_adapter** (6 camadas) | **não** | proibido por auditoria: o treino aborta se alguma chave de LoRA cair nele. O próprio model card manda não treinar |
| `adaln_modulation` | não | excluído (o Anima já tem uma LoRA interna ali) |
| **DiT, 28 blocos** | **LoRA rank 64** | 280 lineares, 91,75 M parâmetros: self-attention (q, k, v, out), **cross-attention (q, k, v, out)** e MLP |

Dos 280 lineares com LoRA, **112 são de cross-attention**, que é onde a imagem lê o texto.
Na cross-attention, `k_proj` e `v_proj` recebem diretamente os embeddings do texto (dimensão
1024) e têm LoRA.

**Consequência (fato):** o texto chega ao DiT como um embedding **fixo**, porque nem o encoder
nem o adapter mudam. Já **a forma como o DiT interpreta esse embedding é treinada**. A frase
"o modelo não aprende palavras novas" é verdade para o encoder e falsa para o modelo como um
todo: a LoRA aprende a reagir de forma nova a frases que o encoder já representa. Uma frase
repetida milhares de vezes, como "the same girl", vira um padrão que a cross-attention aprende
a associar a "olhe a referência". Quem aprende isso é o DiT, não o Qwen.

---

## 4. Como o texto vira condição (detalhe do caminho)

1. A legenda é tokenizada **duas vezes**:
   - pelo tokenizador do **Qwen3**, que alimenta o encoder;
   - pelo tokenizador do **T5**, só para obter IDs, sem pesos do T5.

   Os dois cortam em **512 tokens**, completam com padding até 512 e zeram as posições de padding.
2. O **Qwen3-0.6B base** (não é a versão instruct) recebe o texto cru, **sem chat template e sem
   BOS**. Usa-se o `last_hidden_state`.
3. O **llm_adapter** pega os IDs do T5, transforma em embeddings aprendidos (que servem de
   perguntas) e faz cross-attention nos estados do Qwen3 (que servem de memória). Ele tem
   self-attention entre os tokens T5 e RoPE próprio. A saída é **uma sequência alinhada aos
   tokens T5**, que imita o embedding T5 que o Cosmos-Predict2 original esperava.
4. Essa sequência entra na cross-attention de **todos os 28 blocos**. Tanto os tokens do alvo
   quanto os da **referência** fazem cross-attention com a mesma legenda (a de B), porque os dois
   frames passam pelos mesmos blocos.

**Tamanho na prática** (medido nas legendas do subset ds4 antigo):

| tier | palavras (mediana) | tokens Qwen (mediana / máx.) | tokens T5 (mediana / máx.) |
| --- | ---: | ---: | ---: |
| completa | 94 | 111 / 150 | 133 / 192 |
| curta | 19 | 23 / 56 | 27 / 67 |

O limite de 512 está longe. O T5 gera ~20% mais tokens que o Qwen. Corte só aconteceria acima
de ~350 palavras.

---

## 5. Como as legendas entram no dataset

- `target/captions.json`: `{"arquivo.png": ["legenda 1", "legenda 2", ...]}`.
  **Cada legenda da lista vira uma amostra separada**, com o mesmo par (A, B) e texto diferente,
  e cada uma é vista **uma vez por época**.
  - Isso multiplica o custo: no run anterior, 11.526 pares viraram **26.532 apresentações por
    época** (receita full + short + short), cerca de 2,3× mais passos.
  - Ou seja, **o número de legendas por par é decisão de custo de GPU**, não só de formato.
- **Não existe caption dropout** neste pipeline. Legendas vazias são **puladas**
  (`skip_empty_caption = true`), então o modelo **nunca treina com texto vazio**.
- Mudar o texto exige **refazer o cache de texto**, mas o cache é barato (Qwen 0,6B).
- Não há gatilho (trigger word). **Quem "liga" o comportamento é a LoRA carregada junto com a
  referência.** Nenhum run do projeto usou trigger.

### O que o treino A/aligned aprovado (E2) realmente viu

Este é o dado mais importante para comparar formatos.

- **Tier completo:** a legenda original do dataset. São ~94 palavras descrevendo B **inteiro**,
  inclusive aparência, roupa, fundo e luz. Formato misto: enquadramento no começo, depois frases
  separadas por vírgula. **Sem** tags de qualidade, **sem** tag de classificação (safe/nsfw),
  **sem** gatilho.
- **Tier curto (×2):** gerado por regex (`tools/nextscene_captions.py`), sem LLM. Saiu em cerca
  de 58% dos pares do ds4 (720 de 1.251, contagem de 2026-10-08); nos outros pares só existe a
  legenda completa. Exemplo real:
  > `close-up from a side profile view. The same man is looking intently toward the right, running down from his temple along his jaw.`

  Ele usa **"The same <sujeito>"** como ponteiro para A e às vezes sai truncado (o fragmento
  "running down from…" do exemplo).
- **Prompts da avaliação held-out:** os mesmos tiers curtos, também com "The same …".

Logo, o comportamento que o usuário aprovou veio de **legendas completas e longas + legendas
curtas com "The same X"**. Qualquer formato novo é uma **mudança em relação ao validado**. Pode
ser melhor, mas não está testado.

---

## 6. Inferência (o que o usuário vai digitar e como é combinado)

`infer_easycontrol.py --mode nextscene` e os nodes do ComfyUI usam o mesmo contrato.

- O CFG de texto padrão é **4**, com o negativo `"worst quality, low quality, blurry, jpeg artifacts"`.
  - A referência fica nos **dois** ramos (`uncond_ref = keep`), então o CFG empurra só **o texto**.
- O ref-CFG é opcional: `pred = n + cfg·(c − n) + (ref_cfg − 1)·(c − z)`, onde `z` é a predição
  com a referência zerada (o "nulo" treinado pelo `ref_dropout`).
  - Regra do usuário: o adapter tem que ficar bom com `ref_cfg = 1`.
- Como o treino nunca viu texto vazio, a legenda é **sempre** a principal condição de texto, e o
  formato que o usuário digitar precisa se parecer com o formato do treino.

---

## 7. O que o model card oficial do Anima diz sobre prompts (fato, sobre o modelo base)

- Ele foi treinado com **tags Danbooru, linguagem natural e as duas misturadas**.
- Tags em minúsculas, com espaço no lugar de underscore (as tags `score_*` são a exceção).
- Ordem das tags: `[qualidade/meta/ano/segurança] [1girl/1boy/...] [personagem] [série] [artista] [tags gerais]`.
- Tags de segurança: `safe, sensitive, nsfw, explicit`.
- Positivo recomendado: `masterpiece, best quality, score_7, safe,`.
- Negativo recomendado: `worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration`.
- Foi treinado com **tag dropout aleatório**, então legendas incompletas são normais para ele.
- Em linguagem natural pura: "quanto mais descritivo melhor, pelo menos 2 frases; prompts
  curtos demais dão resultados inesperados". Isso vale para **T2I sem referência**.
- Com vários personagens, recomenda descrever a aparência de cada um. Também vale para T2I; no
  nosso caso, a aparência vem de A.
- Pode gerar conteúdo indesejado com prompts curtos; a defesa são as tags de segurança.
- Artistas com `@`. Datasets não-anime levam uma tag de dataset na primeira linha.

---

## 8. Implicações para o formato da legenda

Separando o que é fato do que é hipótese.

**Fatos que restringem o formato:**

1. **O texto não vê a imagem.** O Qwen e o adapter são congelados e só leem a legenda. O único
   jeito de o modelo "saber" algo que não está no texto é **ler A** pela atenção.
2. **A loss só fecha se o que falta no texto vier de A.** Esse é o mecanismo de ensino.
   - Legenda exaustiva: B pode ser reconstruído pelo texto e a referência fica dispensável
     (atalho 1, polo "bonito sem relação").
   - Sem texto, ou com pares sem mudança: copiar A é o ótimo (atalho 2, polo "copia").
     Projeto externo (AnimaRefLora) mediu que dropout da legenda inteira **antecipou a cópia**.
   - Fonte: `ANIMA_NEXTSCENE_PESQUISA_2026-09.md`, P6.
3. **Cada legenda extra por par é um passo extra por época.** O formato e a quantidade de tiers
   definem o tempo de GPU.
4. **512 tokens não é limite prático**, então o tamanho é decisão de método, não de capacidade.
5. **Não precisa de gatilho.** Se existir, ele aparece em 100% das amostras e não carrega
   informação; só vira obrigação de digitar.
6. **A cross-attention com LoRA aprende frases recorrentes.** Uma convenção fixa ("the same
   girl", "the girl", etc.) se torna um sinal aprendido pelo DiT. A escolha da convenção importa
   e **precisa ser a mesma que o usuário vai digitar**.

**Hipóteses (não testadas; precisam de A/B para virar fato):**

- *Proibir "same/still/now".*
  - O E2 aprovado foi treinado justamente com "The same X".
  - Argumento contra o "same": na inferência o usuário talvez não escreva assim.
  - Argumento a favor (`PROPOSTA_CAPTIONS_v3.md` §2.1): é um ponteiro curto e estável e é o que
    se digita.
  - Correção àquele doc: quem aprende o ponteiro é a cross-attention do DiT (LoRA), **não o
    Qwen3**, que é congelado.
  - Seja qual for a escolha, ela tem que ser **consistente entre treino e uso**.
- *Tag de classificação (safe/nsfw…).*
  - O Anima base entende essas tags, e o dataset antigo tinha ~44% R18 (estimativa amostral).
  - Incluir a tag provavelmente dá controle na inferência.
  - As legendas do E2 **não** tinham tag nenhuma, então seria uma diferença de dialeto em relação
    ao validado.
- *Tags de qualidade (`masterpiece`, `score_7`).*
  - As legendas de treino nunca tiveram, e o usuário pode usá-las na inferência.
  - O tag dropout do base sugere que tanto faz incluir ou não, mas não foi medido no nosso
    adapter.
- *"Pelo menos 2 frases".* É conselho do card para T2I. No nosso caso, a parte "descritiva" vem
  de A, e não há evidência de que um delta curto precise de 2 frases.

**Recomendação prática para o teste de formato:**

- Gere as legendas candidatas para os **pares held-out** e rode o mesmo adapter com cada formato.
  - Use `tools/nextscene_eval.py` com referência correta, trocada e nula.
  - Isso mede qual formato **o adapter atual** obedece melhor, sem treinar nada.
- O teste do Muse com página de revisão mede outra coisa: se o **captioner** escreve o formato
  com precisão.
- Os dois testes são complementares. Só um A/B de treino curto diz qual formato **treina** melhor.

---

## 9. Referências

- Código:
  - `models/anima_nextscene.py`: contrato, `ref_dropout`, `high_noise`, loss e auditoria da LoRA.
  - `models/cosmos_predict2.py`: `_tokenize` (512), `_compute_text_embeddings`, `LLMAdapterLayer`.
  - `models/llm_adapter.py`
  - `utils/dataset.py`: `captions.json`, `skip_empty_caption`.
  - `infer_easycontrol.py`: `sample_nextscene` e `combine_nextscene`.
- Configs:
  - `examples/anima_nextscene/gpu_20261001_1024/A_full.toml`: campanha anterior.
  - `examples/anima_nextscene/gpu_20261008_1024_v2/A_full.toml`: run v2.
- Docs:
  - `docs/ANIMA_NEXTSCENE_PESQUISA_2026-09.md`: P1 (geometria e cópia), P6 (atalhos de legenda).
  - `docs/PROPOSTA_CAPTIONS_v3.md`
  - `docs/NEXTSCENE_RUN_LOG.md`: E2 e veredito A/aligned.
- Model card: https://huggingface.co/circlestone-labs/Anima (revisão `f973fc41`).
