# Anima — futuro treino

Fechamento da bateria exploratória de 2026-07-25 e definição do que vem
depois. Este documento é a referência para quando o dataset recaptionado
estiver pronto.

Grids de avaliação: `/workspace/outputs/GRIDS_TODOS/EVAL10__*.png`

---

## 1. O que estava em jogo

O Anima é um DiT de vídeo de 2B (derivado do Cosmos-Predict2) que recebe a
referência como **um frame temporal extra**, concatenado ao latente ruidoso:

```python
noisy_latents = torch.cat([noisy_latents, control_latents], dim=2)
```

Não existe caminho dedicado para a referência — ela é lida por self-attention,
como se fosse mais um quadro do vídeo. O texto vai por outro caminho
(Qwen3-0.6B → `llm_adapter` → cross-attention) e **nunca vê a referência**.

A pergunta da bateria era: *dado esse desenho, qual escopo de LoRA e qual
contrato de treino fazem o modelo realmente LER a referência?*

O antagonista é o **atalho de caption**: se o caption já descreve o alvo por
completo, a loss é satisfeita sem olhar a referência, e o adapter aprende a
ignorá-la. Foi medido com o **teste de referência embaralhada** — gerar a
mesma imagem trocando a referência por outra, mantendo caption e seed. Se a
saída não muda, o adapter não está usando a referência.

---

## 2. Protocolo de avaliação

Cada grid tem 10 exemplos (linhas) × 6 colunas:

| coluna | o que é |
|---|---|
| referência | a imagem dada como condição |
| alvo real | só existe para o ex1, que vem de um par do dataset |
| sem ref | LoRA aplicado, sem referência — baseline |
| **lora 1.0** | **O CRITÉRIO.** É aqui que o braço é julgado |
| lora 1.0 + ref_cfg 1.75 | headroom do dial, não critério |
| ref EMBARALHADA | mesmo caption, referência de outro exemplo |

Regra fixada pelo usuário: **se o adapter só fica bom com `ref_cfg` alto, ele
não está bom o suficiente.** A coluna do meio é que decide.

O conjunto foi de 3 para 10 exemplos porque os próprios dados pediram: a
variância **entre exemplos** é maior que entre seeds (o arm1 na seed 2024 deu
0.390 e 1.159 no mesmo run). Adicionar exemplos reduz o erro padrão mais
rápido, por geração gasta, do que adicionar seeds.

Ferramentas: `tools/eval10_all_arms.sh` (roda tudo), `tools/eval_examples.sh`
(os 10 exemplos), `tools/battery_grid_assemble.py` (monta o grid com o
cabeçalho de identificação), `tools/battery_metrics.py` (triagem).

### Armadilha que custou caro, documentada aqui para não repetir

**O modo de inferência precisa casar com o contrato do treino.** Errar isso
produz ruído puro, que parece "o método falhou":

| contrato de treino | modo de inferência |
|---|---|
| `ic_lora_v3`, `ref_first = false` (target-first) | `ominicontrol_subject` |
| `ic_lora_v3`, `ref_first = true` | `ic_lora_full` |
| `ic_lora_v3` + `include_adaln = true` | `ominicontrol_subject --skip_adaln` |
| `ic_lora_dual`, `condition_only_lora = true` | `ic_lora_dual` |

O armC foi avaliado uma vez no modo errado e produziu ruído. **Sempre fazer
smoke de 1 imagem antes de rodar a bateria inteira.**

---

## 3. Os métodos avaliados

Todos partem da mesma base: LoRA de escopo largo (`self_attn` + `mlp` +
`cross_attn` + `llm_adapter` nos alvos), rank 32, lr 1e-4, batch efetivo 8.
O que muda é uma variável por braço.

### M1 — `ic_lora_v3` target-first, `llm_adapter` congelado ⭐
`arm1_broad_llm_frozen` · `armB_dropout` · `armD_dropout_2000`

A linha de base. Concat `[alvo | ref]`, `llm_adapter_lr = 0`, adaln fora do
LoRA. As três variantes diferem apenas em `condition_dropout` (0.0 vs 0.1) e
duração (1000 vs 2000 steps).

### M2 — `llm_adapter` treinável
`arm3_broad_llm_full`

Idêntico ao M1, mas `llm_adapter_lr = 1e-4`. Testa a "eureka de julho" — a
hipótese de que treinar a ponte T5→Qwen3 ajudaria o canal semântico.

### M3 — adaln dentro do LoRA
`armA_adaln_in`

`include_adaln = true`. O Cosmos-Predict2 já tem LoRA interna no adaln
(`use_adaln_lora=True, adaln_lora_dim=256`), então isso empilha uma segunda
LoRA por cima — amplificação multiplicativa.

### M4 — routing condition-only (`ic_lora_dual`)
`armC_routed`

`condition_only_lora = true`: o delta da LoRA é somado **apenas nas rows da
referência**, deixando o alvo com os pesos base.

### M5 — `ref_first = true` — **NUNCA RODADO**
`examples/round2_2026-07-25/armE_reffirst.toml`

Inverte a ordem do concat para `[ref T=0 | alvo T=1]`, a convenção do LTX-2 —
a referência recebe posições temporais de RoPE mais baixas. Config pronta, o
treino nunca foi disparado.

---

## 4. Resultado

Métricas do conjunto de 10 exemplos, checkpoint mais treinado de cada braço.
**Isto é triagem.** A decisão foi visual e do usuário.

| braço | método | steps | sensibilidade | fidelidade |
|---|---|---|---|---|
| **armD** | M1, dropout 0.1 | 2000 | **0.921** | **0.346** |
| arm3 | M2, llm_adapter treinável | 1000 | 0.718 | 0.214 |
| **arm1** | M1, dropout 0.0 | 1000 | 0.657 | 0.121 |
| armA | M3, adaln dentro | 1000 | 0.597 | 0.142 |
| arm1 s500 | M1, dropout 0.0 | 500 | 0.592 | 0.137 |
| armB | M1, dropout 0.1 | 1000 | 0.589 | 0.148 |
| armC | M4, routing | 1000 | 0.587 | 0.166 |

> A métrica de **fidelidade não deve ser usada como critério**. Ela compara
> paleta e estrutura contra a referência, então gerações escuras e degradadas
> pontuam alto quando a referência é escura — ela chegou a premiar um
> checkpoint com artefato de painel duplicado. Aviso está dentro de
> `tools/battery_metrics.py`.

### VENCEDORES: armD e arm1

Veredito visual do usuário: *"arm1 e armB são bons, próximos, parecidos. armD
é tão bom quanto arm1 só que de forma diferente. armD é o melhor na verdade."*

Ambos são o **mesmo método M1**. A diferença é o par acoplado
`condition_dropout` + duração.

### Os perdedores, e por quê

- **M2 (`llm_adapter` treinável) perde.** Bate com a recomendação oficial do
  criador do Anima ("nunca treine o llm_adapter"). A "eureka de julho" que
  dizia o contrário estava contaminada: as sondas que a produziram tinham
  `llm_adapter_lr = 0` por bug, ou seja, o canal semântico estava congelado
  justamente nos runs que supostamente provavam que treiná-lo ajudava.
- **M3 (adaln dentro) perde.** Artefato de instabilidade observado
  diretamente, sem ganho compensatório. Coerente com a dupla-LoRA.
- **M4 (routing) perde.** Efeito grande e consistente. O mecanismo explica: o
  delta só é somado nas rows da referência, então o **alvo nunca treina a
  capacidade de LER** essa informação. Ele enriquece o que é escrito e não
  ensina ninguém a ler.

---

## 5. O insight central: acoplamento vs atribuição

O diff entre arm1 e armD são **duas linhas**:

```diff
- condition_dropout = 0.0    → 0.1
- max_steps         = 1000   → 2000
```

### O que o dropout faz de verdade

Em `models/ic_lora_full.py:143-146` ele **não remove** a referência, ele a
**zera**:

```python
drop_mask = torch.rand(bs) < self.condition_dropout
control_latents[drop_mask] = 0.0
```

O frame continua concatenado, com o mesmo timestep 0 e as mesmas posições de
RoPE. A referência não some — ela fica **em branco**. Consequência: o armD
tem um **nulo calibrado**, e o `ref_cfg` é uma CFG cujo branch incondicional é
exatamente `f(prompt, ref=0)`. No arm1 esse ponto está fora da distribuição de
treino; no armD ele foi treinado. *(Mecanismo coerente com a construção, mas
não isolado experimentalmente — é interpretação, não medição.)*

### As duas linhas são acopladas, não independentes

| | steps | sensibilidade |
|---|---|---|
| armB — dropout 0.1 | 1000 | 0.589 |
| armD — dropout 0.1 | 2000 | 0.921 |

Mesma receita. O dropout bloqueia a rota fácil, então o modelo aprende **mais
devagar** — a 1000 steps ele ainda não chegou, e o armB fica atrás até do
arm1. **Copiar o dropout sem aumentar os steps é estritamente pior que não
usar dropout.** As duas linhas andam juntas ou nenhuma delas.

### Acoplamento vs atribuição

O arm1 viu a referência em 100% do treino, então pôde construir uma solução
onde referência e alvo estão **fundidos** — nunca precisou ser robusto à
ausência dela. Saídas se misturam mais suavemente com o prompt.

O armD aprendeu que a referência *pode* estar em branco, então a trata como
entrada **separável, que ele consulta**. Transferência mais forte e mais
literal — no ex4 ele reproduz a placa 会社員(35) que o armC inventa do zero —
mas às vezes menos graciosa (no ex6 o armD saiu escuro e de costas).

> **arm1 acopla. armD atribui.**

Isso conecta direto com o dataset novo: o contrato *"se eu não especificar,
herda"* **exige atribuição**. O modelo precisa saber quais partes vêm da
referência para poder herdar seletivamente. Um modelo que funde não faz
herança seletiva de forma limpa. Ou seja, **o dropout tende a ser MAIS
importante com delta-captions do que é com as captions exaustivas atuais.**

---

## 6. Limites honestos deste resultado

1. **A célula que falta do 2×2 é arm1 com 2000 steps** (sem dropout, treino
   longo). Sem ela não dá para separar "o dropout é bom" de "2000 steps é
   bom". Custo: ~2h de GPU (o armD levou 1h56 para 2000 steps).
2. **Todos os braços foram treinados no dataset com captions exaustivas.** O
   atalho de caption estava ativo o tempo todo. É plausível que o ranking mude
   com delta-captions — em particular, o M4 (routing) pode ter sido penalizado
   por um problema que o dataset novo remove.
3. **Nenhum ranking fino sobrevive à estatística.** Entre arm1, armB e armD no
   conjunto de 3 exemplos, o teste de Welch com 6 seeds deu t=1.74 contra
   crítico 2.23 — não conclusivo. O gap do conjunto de 10 é maior, mas não foi
   replicado em múltiplas seeds.
4. **Cinco autocorreções nesta bateria.** O padrão é sempre o mesmo: concluir
   cedo demais com amostra pequena. Regras adotadas: nenhum veredito sem ≥3
   seeds; métrica só para triagem; **a decisão final é visual e do usuário**;
   fazer a conta antes de afirmar.

---

## 7. PRÓXIMO PASSO

**Treino completo, de verdade, com todos os métodos únicos**, no dataset
recaptionado.

Variação de `condition_dropout` entre 0.0 e 0.1 **não conta como método
diferente** — é um parâmetro, ainda que dos mais relevantes. O padrão passa a
ser **`condition_dropout = 0.1`**, com a duração escalada junto (ver §5: as
duas coisas são acopladas).

Métodos únicos a treinar:

| # | método | config base | status |
|---|---|---|---|
| M1 | `ic_lora_v3` target-first, llm_adapter congelado | `armD_dropout_2000.toml` | vencedor da bateria |
| M2 | `llm_adapter` treinável | `arm3_broad_llm_full.toml` | perdeu, revalidar |
| M3 | adaln dentro do LoRA | `armA_adaln_in.toml` | perdeu, revalidar |
| M4 | routing condition-only | `armC_routed.toml` | perdeu — mas é o que mais pode mudar com delta-captions |
| M5 | `ref_first = true` | `armE_reffirst.toml` | **nunca rodado** |

Por que revalidar os perdedores em vez de só treinar o M1: os três perderam
sob o atalho de caption. O dataset novo remove o atalho, que era o antagonista
principal — o ranking pode não se preservar. O M4 em particular tem um
mecanismo que só faz sentido quando o alvo *precisa* ler a referência.

### Pré-requisito

O dataset recaptionado. O usuário está escrevendo o system prompt final; a
proposta de desenho está em `docs/PROPOSTA_CAPTIONS_v3.md` (5 decisões
abertas na §9). O benchmark de VLMs já está feito, com custo real medido —
para 1255 pares o intervalo é $0.98 (gemma-4-31b) a $6.91
(gemini-3.1-flash-lite). **Custo não é restrição; o critério é qualidade.**

### Ao escalar

O número que importa não é `max_steps`, é **quantas vezes o modelo viu a rota
difícil**. Se o dataset novo for maior que os 1255 pares atuais, escalar os
steps na mesma proporção.

---

## 8. Estado do trainer

**A receita vencedora está inteiramente implementada. Nada a construir.**

| botão | onde é lido |
|---|---|
| `type = 'ic_lora_v3'` | `train.py:357` → `models/ic_lora_dual.py:ICLoraV3Pipeline` |
| `llm_adapter_lr` | `models/cosmos_predict2.py:586` |
| `condition_dropout`, `ref_first` | `models/ic_lora_full.py:71-72` |
| `include_adaln` | `models/ic_lora_dual.py:95` |
| `condition_only_lora` (M4) | `models/ic_lora_dual.py:142` |

`examples/round2_2026-07-25/armD_dropout_2000.toml` serve de template direto:
trocar o `path` do dataset e treinar.
