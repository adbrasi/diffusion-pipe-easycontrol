# Krea 2 multi-referência — viabilidade do Macro-Dataset sobre o Omini-Grounded

**Pesquisa de 2026-07-25.** Pergunta: dá para treinar o Krea 2 no dataset
`Azily/Macro-Dataset`, cujos prompts endereçam várias referências por índice
(`<image 1>`, `<image 2>`, `<image 3>`), partindo do `krea2_omini_grounded` —
que foi disparado o melhor treino de Krea 2 do projeto e praticamente o único
que funcionou de verdade?

**Resposta curta: sim, e o caminho é mais curto do que parece.** Dois dos três
canais do Omini-Grounded já suportam N referências sem tocar em nada. O que
falta é geometria, dataset loader e o custo de atenção.

---

## 1. O que o Omini-Grounded é, em três canais

| canal | o que carrega | adapter |
|---|---|---|
| **tokens VAE** | a aparência exata da referência | LoRA **routado** só nas rows da ref (`models/condition_lora.py`) |
| **grounding Qwen3-VL** | resolve entidades do caption contra a imagem | LoRA **global** no txtfusion |
| **base congelada** | todo o conhecimento do modelo | nada — zero drift |

Sequência: `[texto | target ruidoso | ref limpa]`, referência com
`position_mode = 'width_shift'` e modulada a `t = 0`.

O routing é o que dá drift zero: as rows do target passam pelo modelo
**intocado**, e o adapter só aprende "como a referência deve se apresentar" via
K/V dos tokens dela.

---

## 2. O dataset Macro — estrutura real

~400k amostras de treino, 4k de avaliação, CC BY 4.0. Índice em JSON:

```json
{
  "task": "customization",
  "idx": 1,
  "prompt": "Create an image of the modern glass and metal interior from <image 2>, applying the classical oil painting style from <image 1> globally across the entire scene.",
  "input_images": ["data/final/customization/train/1-3/data/00022018/image_1.jpg",
                   "data/final/customization/train/1-3/data/00022018/image_2.jpg"],
  "output_image": "data/final/customization/train/1-3/data/00022018/image_output.jpg"
}
```

Quatro tarefas (customization, illustration, spatial, temporal) e — **isto é o
que salva o projeto** — já vem **particionado por número de referências**:
brackets `1-3`, `4-5`, `6-7`, `≥8`, em
`data/filter/{task}/{split}/{bracket}/`.

Os três exemplos que você citou caem todos no bracket **1-3**:

- 1 ref — *"A waist-up portrait of the woman from `<image 1>` standing in a misty bamboo forest"* → customization
- 2 refs — *"the woman from `<image 1>` and the woman from `<image 2>` standing together"* → customization
- 3 refs — *"`<image 1>` is the left view, `<image 2>` is the front-left view... Generate the back-left view"* → spatial

---

## 3. Auditoria peça por peça

### ✅ JÁ FUNCIONA — o grounding aceita N imagens

`models/krea2_edit.py:250`:

```python
files = control_file if isinstance(control_file, (list, tuple)) else [control_file]
...
if self.vl_prompt_style == 'plain':
    text = VISION_BLOCK * len(images) + caption
else:
    text = build_vl_image_prompt(len(images)) + caption   # "Picture 1: <vis>Picture 2: <vis>..."
```

O canal semântico inteiro — o que faz `<image 1>` significar alguma coisa — já
está escrito para uma lista. **Zero linhas de mudança.**

E `build_vl_image_prompt` (`krea2_edit.py:44`) já emite `Picture 1:`,
`Picture 2:` … ou seja, o estilo `picture_n` já dá ao Qwen3-VL uma âncora
explícita por índice. O probe do Omini-Grounded usou `plain` porque tinha uma
referência só; **para o Macro o `picture_n` passa a ser o default natural.**

### ✅ JÁ FUNCIONA — o routing da LoRA

`ConditionOnlyLoRARouter.set_reference_span(start, end)` marca **um span
contíguo**. Com N referências concatenadas em sequência, o span continua sendo
`[text_length + target_length, fim]`. **Zero linhas de mudança.**

### ✅ JÁ FUNCIONA — o RoPE tem espaço de sobra

`headdim = 6144/48 = 128`, `axes_dim = [32, 48, 48]` (frame, h, w), `theta = 1000`.
Calculado a partir de `comfy/ldm/flux/math.py:rope`:

| eixo | dim | pares | freq mínima | período |
|---|---|---|---|---|
| frame | 32 | 16 | 0.001540 | **4080** posições |
| h | 48 | 24 | 0.001334 | **4712** posições |
| w | 48 | 24 | 0.001334 | **4712** posições |

A 512px cada imagem é um grid de 32×32 tokens. O `width_shift` empurra a
referência para `w + 32`. Com N referências lado a lado, a última fica em
`N × 32` — com 8 referências, posição 256, contra um período de 4712.

**Não há colisão nem wraparound. O `width_shift` generaliza para N sem tocar em
uma linha de RoPE.** Caberiam ~147 imagens antes de a fase dar volta.

### ❌ BLOQUEIO EXPLÍCITO — o forward recusa mais de um frame

`models/krea2_reference.py:294`:

```python
if target.shape[2] != 1 or reference.shape[2] != 1:
    raise ValueError('Krea2 reference training currently supports one target and one reference frame')
```

E o resto do método assume uma referência só: um `reference_grid_h/w`, um
`reference_tokens`, um `reference_pos`. Precisa virar um laço sobre N.
É a mudança mais volumosa, mas é mecânica.

### ❌ BLOQUEIO — o dataset loader só carrega uma referência

`utils/dataset.py:523` tem **um** `control_path`, casado por stem do arquivo.
Pior, `utils/dataset.py:877` descarta explicitamente o resto:

```python
ret['control_file'] = [example['control_file'][0]]
```

Para o Macro o caminho certo não é multiplicar `control_path` — é um **loader
por manifesto JSON**, porque o dataset já vem indexado com `input_images` de
tamanho variável. O `control_file` já trafega como lista no pipeline, então o
encanamento a jusante coopera.

### ❌ ATRITO — referências de tamanhos diferentes

`prepare_reference_latents` (`krea2_reference.py:127`) exige que a referência
tenha exatamente a forma do target. Com N referências de aspect ratios
diferentes isso cai. Ver §5 — a solução tem consequência de design, não é só
remover o assert.

### ❌ ATRITO — N variável quebra o batching

Amostras com 1, 2 e 3 referências têm comprimentos de sequência diferentes.
Duas saídas:

1. **Bucketizar por N.** O Macro já vem bracketizado, então sai quase de graça.
2. **Slots fixos com preenchimento em branco.** Treinar sempre com `N_max`
   slots e **zerar** os não usados.

A opção 2 tem uma justificativa bonita: é exatamente o que o
`condition_dropout` já faz — ele **não remove** a referência, ele a **zera**
(`models/ic_lora_full.py:143`, achado da bateria do Anima). Uma referência em
branco já é in-distribution por construção. O custo é pagar sempre o
comprimento de `N_max`.

**Recomendo a 1 para começar** (mais barata), com a 2 no bolso se a variação de
N se mostrar um problema de generalização.

---

## 4. O custo — é aqui que dói

Atenção é O(L²). A 512px, patch 2 sobre VAE 8×, cada imagem = **1024 tokens**.
Assumindo ~512 tokens de texto:

| refs | sequência | custo de atenção relativo |
|---|---|---|
| 1 (o probe validado) | 2 560 | 1.0× |
| 2 | 3 584 | 2.0× |
| 3 | 4 608 | **3.2×** |
| 5 | 6 656 | 6.8× |
| 8 | 9 728 | 14.4× |

O bracket `1-3` custa até ~3.2× por step contra o probe do Omini-Grounded. É
caro mas viável. O bracket `≥8` a 14× é outro projeto — e é por isso que a
recomendação é **fechar o escopo no bracket 1-3**.

Se apertar: `condition_token_stride` já existe
(`krea2_reference.py:88`) e reduz a resolução dos tokens da referência,
cortando o custo quadraticamente. Referência a metade da resolução do target =
256 tokens em vez de 1024.

---

## 5. O ponto de design mais interessante: como `<image 1>` vira endereçável

Este é o problema de verdade, e vale pensar com cuidado.

Krea 2 é **single-stream**: texto e imagem vivem na mesma sequência e atendem
uns aos outros. Então os tokens de texto que dizem *"the woman from `<image
1>`"* precisam atender ao **span VAE correto**. O que distingue um span do
outro? Só a posição no RoPE.

E tem um detalhe que muda tudo — `krea2_reference.py:360`:

```python
text_pos = combined.new_zeros(batch, text_length, 3)
```

**O texto não tem posição nenhuma.** Fica todo em (0,0,0). Logo o deslocamento
relativo do texto para a referência *i* é exatamente o offset de largura dela.
Se a referência 1 está sempre em `w = 32` e a 2 em `w = 64`, o modelo tem um
sinal estável e aprendível para ligar `<image 1>` ao primeiro span.

**Mas isso só é estável se as referências tiverem largura fixa.** Com ARs
variáveis, o offset acumulado muda de amostra para amostra e "a referência 1
começa em 32" deixa de ser verdade. O endereço vira ruído.

Daí a recomendação concreta:

> **Slots de referência de tamanho fixo.** Todas as referências pré-processadas
> para o mesmo grid de tokens, empilhadas em `width_shift` com offsets
> `i × W_ref`. O endereço de cada slot passa a ser uma constante.

### A alternativa que eu considerei e NÃO recomendo

Usar o **eixo de frame** como índice da referência (ref *i* recebe posição de
frame `i+1`), que é invariante a tamanho e mais elegante. O código já tem essa
capacidade: é o `position_mode = 'subject'` com `reference_position_offset`,
generalizado.

O problema: **`subject` já foi testado contra `width_shift` neste projeto e
perdeu.** O Krea 2 é um modelo de imagem — o eixo de frame provavelmente está
sub-treinado no base, então posições não-zero ali são fora da distribuição. A
evidência empírica do projeto contradiz a elegância teórica; fico com a
evidência.

Vale como A/B barato depois que a linha de base funcionar, não como aposta
inicial.

### E o casamento dos dois canais

O caption diz `<image 1>`; o grounding emite `Picture 1:` antes dos tokens
visuais correspondentes. São dois vocabulários para a mesma coisa. Três
opções, em ordem de agressividade:

1. deixar como está e confiar que o Qwen3-VL amarra sozinho (ele é
   instruction-tuned e lida com "Picture N" nativamente);
2. reescrever `<image 1>` → `Picture 1` no caption;
3. emitir os dois: `Picture 1: <vision> (<image 1>)`.

**Testável e barato.** Eu começaria pela 1, porque é a que não mente para o
modelo sobre o formato que ele verá na inferência.

---

## 6. Plano proposto

### Fase 0 — infra (sem GPU)
1. Baixar **só** `data/filter/customization/train/1-3/` + as imagens
   correspondentes. O dataset inteiro não cabe: são 142 GB livres em
   `/workspace` e o índice sozinho já tem 510 MB.
2. **Os modelos do Krea 2 não estão nesta máquina** (`find` não achou nada em
   `/workspace/models*`). Precisam ser baixados: o DiT fp8-scaled e o
   Qwen3-VL-4B do text encoder.

### Fase 1 — loader por manifesto
`control_path` vira opcionalmente um `manifest_path` apontando para os JSON do
Macro. Preserva `input_images` inteiro em vez de `[0]`. Bucketiza por N.

### Fase 2 — geometria multi-ref
Laço sobre N em `Krea2ReferenceInitialLayer.forward`, slots de tamanho fixo,
offsets `i × W_ref` em `width_shift`. O router e o grounding não são tocados.

### Fase 3 — probe curto
Espelhar o probe que validou o Omini-Grounded: 250 steps, rank 64, lr 1e-4,
512px, `condition_only_lora = true`, `reference_timestep = 'zero'`,
`position_mode = 'width_shift'`, mas com `vl_prompt_style = 'picture_n'`.
Critério: em amostras held-out com 2 referências, trocar a ordem das
referências tem que trocar quem é quem na saída. **Este é o teste de referência
embaralhada da bateria do Anima, adaptado — e é o único que prova endereçamento
em vez de mistura.**

### Fase 4 — só então escalar
E incorporar as melhorias já mapeadas em `docs/OMINI_GROUNDED_SEGREDO.md` §9,
em especial `caption_dropout = 0.1`, que o probe original não teve.

---

## 7. Riscos e incógnitas honestas

1. **O `condition_dropout` é problemático no caminho grounded.** Zerar os
   latentes VAE não zera o que o Qwen3-VL viu — o "nulo" não é nulo. A config
   do probe traz `condition_dropout = 0.0` com a anotação *"proibido no caminho
   edit"*. Com N referências isso piora: dropout por slot pode dessincronizar
   os dois canais. **Provavelmente precisa ser dropout conjunto** (zera o
   latente E remove a imagem do grounding, juntos).
2. **Endereçamento pode simplesmente não emergir em 250 steps.** Ligar
   `<image 1>` ao span certo é uma capacidade mais difícil que "copie esta
   referência". O probe de 250 steps validou a segunda, não a primeira.
3. **400k amostras contra os 1255 pares de todo o histórico do projeto.** Toda
   a intuição de duração de treino acumulada aqui não transfere.
4. **Não medi VRAM.** 3.2× de atenção sobre um modelo de 28 camadas e 6144 dims
   numa 5090 de 32 GB é plausível com activation checkpointing e
   `blocks_to_swap`, mas é estimativa, não medição.
5. **O bracket 1-3 mistura N=1, 2 e 3.** Se bucketizar por N, os buckets são
   desbalanceados e o de N=1 é o mais fácil — risco de o modelo aprender bem o
   caso trivial e mal o interessante.

---

## 8. Veredito

**É viável, e o Omini-Grounded é a base certa.** Os dois canais que dão o
diferencial dele — routing condition-only e grounding Qwen3-VL — **já
generalizam para N referências sem alteração**. O trabalho real está em três
lugares delimitados: loader por manifesto, laço de N na geometria, e a decisão
de slots fixos que torna `<image 1>` endereçável.

O que eu **não** faria: tentar o dataset inteiro, ou os brackets de 4+
referências, antes de a Fase 3 provar que o endereçamento emerge.
