# Krea 2 multi-referência — receita que FUNCIONOU (2026-07-26)

**Método: `krea2_multiref_grounded`.** Krea 2 recebendo N referências
SEPARADAS (não stitched), endereçadas por texto no caption (`image 1`,
`image 2`). Validado com **50 steps de treino** — o modelo já compôs duas
referências com os papéis certos.

Este documento é o suficiente para reproduzir do zero em outra sessão.

---

## 1. O achado que fez funcionar

O `width_shift` do `krea2_reference.py:359` soma uma **constante**:

```python
reference_pos[..., 2] = reference_pos[..., 2] + float(target_grid_w)
```

A implementação óbvia de multi-referência — um laço reusando essa linha —
daria a **todos os N spans posições de RoPE idênticas**. E aí:

> Se dois spans de referência de mesmo comprimento recebem posições
> idênticas, a saída do DiT sobre o target é **exatamente invariante** à
> troca dos dois.

Porque (1) atenção é soma sobre keys ponderada por softmax, logo
permutation-invariant dadas as rotações; (2) `tvec` é o mesmo para toda linha
de referência (zeros); (3) a máscara é a mesma; (4) o router LoRA aplica a
mesma máscara a todas — span contíguo, sem granularidade por slot; (5) MLP e
norms são pointwise.

**Não é "difícil de aprender". É impossível em princípio,** para qualquer
quantidade de dados.

**A correção é uma linha:** offset **cumulativo**, slot *i* em
`w += (i+1) * ref_grid_w`. Feito isso, não sobra nenhum bloqueio
arquitetural — o resto é encanamento.

## 2. `<image N>` não precisa de mecanismo

O binding é aprendido por correlação, como qualquer convenção de caption.
Precedente no próprio projeto: o dataset de abril do Anima ensinou três
**verbos-operação** (`create the next scene` etc.) com ~21k pares e ~1950
steps, e virou o melhor adapter que o usuário já teve.

**Correção importante feita no caminho:** treinar com `<image 1>` fez o
modelo **renderizar o marcador como texto dentro da imagem** (uma saída virou
colagem com "Inage <2" escrito). Os sinais `<` `>` fazem o marcador parecer
glifo a desenhar. A receita final usa **`image 1` sem sinais**, e rotula os
blocos de visão com o **mesmo vocabulário** (`image 1:` em vez de
`Picture 1:`) — assim não há ponte de vocabulário para o modelo aprender.

---

## 3. Reproduzir do zero

### 3.1 Modelos

```bash
hf download Comfy-Org/Krea-2 diffusion_models/krea2_raw_fp8_scaled.safetensors --local-dir /workspace/models/krea2
hf download Comfy-Org/Krea-2 text_encoders/qwen3vl_4b_bf16.safetensors        --local-dir /workspace/models/krea2
hf download Comfy-Org/Krea-2 loras/krea2_turbo_lora_rank_64_bf16.safetensors  --local-dir /workspace/models/krea2
# VAE: qwen_image_vae.safetensors (o mesmo do Anima)
```

### 3.2 Dataset

```bash
hf download Azily/Macro-Dataset --repo-type dataset \
  --include "final_customization_train_1-3_*.tar.gz" --local-dir /workspace/datasets/macro
# extrair, depois:
python tools/prepare_macro.py /workspace/datasets/macro_raw /workspace/datasets/macro_prepared --max-refs 2
```

`prepare_macro.py` faz três coisas que importam:

1. **Normaliza `<image N>` → `image N`** (§2).
2. **Filtra por qualidade** usando as notas do juiz que vêm no próprio Macro
   (`following_score`, `consistency_scores`, padrão ≥9). Descartou 3.476 de
   23.724 — curadoria fraca foi o que estragou o dataset anterior do projeto.
3. Nomeia as referências `<stem>_1`, `<stem>_2` — o **sufixo numérico é o
   contrato de ordem**. Nunca derivar ordem de `Path.glob` (ordem de
   `os.scandir`, arbitrária e instável entre máquinas): se a ordem embaralhar
   entre o canal VAE e o de grounding, o binding é aprendido errado.

### 3.3 Treinar

```bash
NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/macro_multiref/run2_multiref.toml
```

### 3.4 Inferir

```bash
python tools/infer_reference_adapter.py \
  --config examples/macro_multiref/run2_multiref.toml \
  --adapter <checkpoint>/stepNNN \
  --reference ref1.jpg --reference ref2.jpg \
  --prompt "Generate an image of the woman from image 1 holding the bag from image 2." \
  --width 512 --height 512 --seed 76 \
  --turbo-lora /workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors
```

`--reference` é **repetível** e a ordem das flags é a ordem dos slots.
`--turbo-lora` funde a LoRA turbo oficial no base e implica 8 steps / CFG 1.0
/ mu 1.15.

---

## 4. A receita

`examples/macro_multiref/run2_multiref.toml`

```toml
[model]
type = 'krea2_multiref_grounded'
diffusion_model = '.../krea2_raw_fp8_scaled.safetensors'
text_encoders = [{path = '.../qwen3vl_4b_bf16.safetensors', type = 'krea2'}]
diffusion_model_dtype = 'float8'
flux_shift = true

[krea2_multiref_grounded]
max_refs = 2
slot_axis = 'width'          # offset cumulativo (i+1)*W — o achado do §1
position_mode = 'width_shift'
reference_timestep = 'zero'
condition_dropout = 0.0      # PROIBIDO no caminho grounded (raise)
condition_only_lora = true
vl_prompt_style = 'picture_n'
vl_image_label = 'image'     # mesmo vocabulário das captions
vl_image_max_pixels = 147456 # 384*384 — corta o cache ~4x contra 768
caption_dropout = 0.1
txtfusion_rank = 128         # canal de leitura reforçado

[adapter]
type = 'lora'
rank = 64

[optimizer]
type = 'AdamW8bitKahan'
lr = 1e-4
```

`caching_batch_size = 1` é **obrigatório**: as N referências ocupam a
dimensão de batch da chamada do VAE.

### Diferenças para o `krea2_omini_grounded` (o vencedor anterior)

**Igual** — todo o núcleo: `width_shift`, refs a `t=0`, LoRA condition-only
routada nos blocks, txtfusion global treinável, grounding Qwen3-VL, rank 64,
lr 1e-4, AdamW8bitKahan.

**Diferente:**

| | omini_grounded | multiref |
|---|---|---|
| referências | 1 | **N, offset cumulativo** |
| rótulo dos blocos de visão | `Picture 1:` | `image 1:` |
| `caption_dropout` | 0 | **0.1** |
| rank do txtfusion | 64 | **128** |
| grounding | longest-side 768 | 384² |
| resolução | 512/768/1024 | 512 |

Os três últimos desvios têm motivo escrito no cabeçalho do toml. O
`txtfusion_rank` é o mais discutível: o canal semântico carrega ~1,13% da
energia do adapter e o rank efetivo colapsa a ~1/64, e com N referências
**ler seletivamente é a tarefa inteira** — mas é mudança não testada
isoladamente. `txtfusion_rank = 0` reproduz o comportamento antigo.

---

## 5. Código novo

| arquivo | o quê |
|---|---|
| `models/krea2_multiref.py` | **o método.** `Krea2MultiRefInitialLayer` (geometria de N spans com offset cumulativo) + `Krea2MultiRefGroundedPipeline` |
| `tools/prepare_macro.py` | Macro → layout do loader, com filtro de qualidade e normalização de caption |
| `tools/macro_eval.sh` + `tools/macro_eval_grid.py` | avaliação held-out em anime + grid |
| `tools/disk_guard.sh` | guarda de disco (§7) |
| `tools/multiref_order_test.sh` + `tools/multiref_grid.py` | teste de ordem (§6) |

**Modificados:**

- `utils/dataset.py` — ramo `multi_ref`: agrupa `<stem>_<n>`, ordena pelo
  sufixo, empilha as N refs na batch do VAE
- `models/krea2_edit.py` — `build_vl_image_prompt` ganha `label`; config
  `vl_image_label` (default `Picture` = contrato ostris intacto)
- `tools/infer_reference_adapter.py` — `--reference` repetível,
  `--turbo-lora`, tipo registrado
- `train.py:441` — registro do tipo

---

## 6. Como avaliar: o teste de ordem

Gera duas vezes com a **mesma seed e o mesmo prompt**, mudando só a ordem das
referências. Se as saídas forem iguais, o modelo **não endereça — mistura**.
É o teste de referência embaralhada do Anima adaptado: lá se trocava a
referência por outra, aqui se troca a ordem entre duas válidas.

**Resultado no step 50** (`tools/multiref_order_test.sh`): distância
correta-vs-trocada não-zero em 7/7 casos (0.049 a 0.329). E nos dois exemplos
com posição explícita no prompt (*"o homem de image 1 na esquerda, o de
image 2 na direita"*), **trocar a ordem trocou quem estava de cada lado.**
Isso é endereçamento, não "a ordem muda alguma coisa".

Viés observado: o **slot 1 tende a virar o sujeito principal** — coerente com
o `width_shift`, já que o slot 1 fica mais perto do target no RoPE.

Depois que isso ficou provado, a ordem trocada saiu da avaliação de rotina —
dobrava o tempo sem informação nova. Melhor gastar em mais exemplos.

---

## 7. Armadilhas que custaram tempo (todas reais, todas nesta sessão)

1. **Cache de embeddings de texto: ~29-35 MB por amostra.** 20k amostras
   projetam **~690 GB**. É o gargalo real do projeto — não FLOPs, não VRAM.
   **Medir o cache de um run pequeno antes de disparar o grande.**
2. **A guarda de disco falhou DUAS vezes**, as duas por bug meu:
   (a) eu media o cache com `du` para enriquecer a mensagem — num cache de
   110 GB o `du` leva minutos e o loop nunca chegava no `df`. `df` é O(1),
   `du` é O(arquivos). **A instrumentação matou o instrumento.**
   (b) saía na largada por corrida com o launcher do deepspeed (~10s até o
   processo existir).
3. **Não dá para treinar e inferir ao mesmo tempo.** O Krea 2 tem 12,8B e o
   treino ocupa 24 dos 32 GB. Diferente do Anima. Avaliar checkpoints exige
   pausar o treino (retoma com `--resume_from_checkpoint`).
4. **`prepare_sample_test` já embrulha `control_files` em `[ ]`** — passar
   lista-de-lista faz o PIL receber uma `list` em vez de path.
5. **PEFT normaliza `target_modules` para sufixos.** Ler de volta do
   `peft_config` depois do `get_peft_model` não devolve nomes casáveis; os
   nomes completos têm que sair do modelo, antes.
6. **`global_step*` são estados do otimizador** — apagar libera muito disco
   sem perder nenhum adapter (só a capacidade de retomar aquele run).
7. **Validação de contrato**: `expected_contract` mandava tudo que não fosse
   `omini_grounded` para `control_family = 'krea2_edit_dual'` — teria
   reprovado o adapter que o próprio trainer salvou.

---

## 8. Estado e próximos passos

- **run2 pausado no step 500** (loss 0.111 → 0.0776), 2.200 amostras,
  `max_refs=2`. Retomar: `--resume_from_checkpoint`.
- Avaliação held-out em **anime** montada (`tools/macro_eval.sh`, 6 pares do
  dataset do Anima) — domínio totalmente fora da distribuição de treino, que
  é o teste mais duro disponível.
- **Não testado ainda:** `slot_axis = 'frame'` (índice discreto no eixo de
  frame). É o contrato público do Krea Edit (`krea2_edit.py:9`, "RoPE frame
  1") e o default do código é `position_mode='subject'`. Está implementado e
  é o A/B mais interessante que sobrou.
- **Não testado:** `txtfusion_rank = 0` (baseline com rank uniforme 64),
  para isolar se o reforço do canal semântico ajudou mesmo.
