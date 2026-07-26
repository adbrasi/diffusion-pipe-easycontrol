# LTX-2.3 — Concept Slider de timing de animação anime (RECEITA COMPLETA)

**Modelo produzido:** `ltx23_anime_timing_slider_v1.safetensors` (313 MB, formato
ComfyUI). Veredito do usuário: *"ficou incrível"*.

Este documento é a receita **fiel** — o que foi de fato executado, incluindo os
valores medidos e os erros que mudaram o caminho. Seguindo daqui dá para
reproduzir do zero.

---

## 1. O problema

O LTX anima anime **fluido demais**. Movimento contínuo, como material 60fps
interpolado (RIFE/DAIN). Anime de verdade é animado **"on twos"** (12
desenhos/s) ou **"on threes"** (8/s), com frames **segurados** e transições em
pop entre poses-chave.

Não é falta de amplitude de movimento — é **excesso de densidade temporal**: o
modelo preenche os intervalos que a animação deixa vazios de propósito.

**Alvo de uso: i2v, ~100% das vezes.**

## 2. Por que slider e não LoRA comum

Uma LoRA treinada nos vídeos-alvo empurra o modelo numa direção só. O slider
aprende o **eixo**, então o multiplicador dá controle bidirecional e
extrapolável — `+2` para timing escalonado, `-2` para fluido, e valores fora de
`[-1,+1]` extrapolam.

## 3. Ferramenta

Fork [AkaneTendo25/musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner)
branch `ltx-2`, script `ltx2_train_slider.py` (Concept Sliders nativos).

```bash
git clone -b ltx-2 --depth 1 https://github.com/AkaneTendo25/musubi-tuner.git /workspace/musubi
python3 -m venv /workspace/.venv-musubi
/workspace/.venv-musubi/bin/pip install -e /workspace/musubi
/workspace/.venv-musubi/bin/pip install torchaudio   # o cache de latentes importa o VAE de audio mesmo em modo video
```

Modelos:
```bash
hf download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors --local-dir /workspace/models_ltx2
hf download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized --local-dir /workspace/models_ltx2/gemma
```

---

## 4. O dataset

[AdwolfCzar/animateka](https://huggingface.co/datasets/AdwolfCzar/animateka) —
708 clipes de anime de **câmera estática** com caption `.txt` por clipe, já
preparados pelo usuário para LTX-2.3 (832×480, 24 fps).

```bash
hf download AdwolfCzar/animateka --repo-type dataset --local-dir /workspace/datasets/animateka
```

> **Nota de fidelidade:** neste treino o download foi interrompido em **456 de
> 708** vídeos e o treino seguiu com o que havia. Os 456 bastaram.

### Por que este dataset funciona

Ele **é** o alvo estético. Medido no clipe `c_000`: **44,3% dos pares de frames
consecutivos são quase idênticos** (diff média < 0.0002 em luminância 160×90) e
a diff mediana entre vizinhos é **0.00021**. Isso é animação limitada de
verdade, não uma aproximação.

---

## 5. Construir o lado NEGATIVO (o coração do método)

O insight que torna isto barato: **o negativo é o mesmo vídeo com a estrutura
de frames segurados destruída.** Não precisa de RIFE — o ffmpeg basta.

```bash
ffmpeg -loglevel error -y -threads 2 -i "$v" \
  -vf "mpdecimate,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir" \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -an "$out"
```

- `mpdecimate` remove as duplicatas → sobram só os desenhos distintos
- `minterpolate` reconstrói 24fps com compensação de movimento → **todo frame
  passa a ser único e o movimento vira contínuo**

Isso é literalmente o defeito do LTX aplicado ao material do usuário.

**Rodar em paralelo** (a tarefa é trivialmente paralela; serial numa máquina de
32 núcleos levava 1h30, com `-P 8` leva ~15 min):

```bash
ls /workspace/datasets/animateka/videos/*.mp4 | xargs -P 8 -n 1 ./neg_one.sh
```

Descartar clipes com menos de 57 frames (`MIN_FRAMES=57`) — abaixo disso não dá
para formar o par no `target_frames` escolhido. Copiar a caption `.txt` junto: a
**mesma caption** vale para os dois lados, porque a direção vem das imagens.

### Verificação — medir ANTES de escalar

| | duplicatas | diff mediana |
|---|---|---|
| positivo (original) | **44,3%** | 0.00021 |
| negativo (interpolado) | **0,0%** | 0.00256 (12×) |

E o **alinhamento temporal se preserva**: `orig[i]` vs `neg[i]` difere 0.0003 —
*menos* que a diferença entre frames vizinhos do original (0.0022). O par mostra
o mesmo instante em cada índice e difere só na textura temporal. É isso que faz
o slider aprender um **eixo** e não uma troca de conteúdo.

### Trilha de áudio silenciosa é OBRIGATÓRIA

O `-an` remove o stream de áudio, e o loader do `ltx2_cache_latents.py` (PyAV)
acessa o stream de áudio **sem checar se existe** → `IndexError: tuple index out
of range`, que derruba o cache inteiro sem dizer qual arquivo. Os vídeos do
animateka têm stream de áudio (parte é silêncio digital, mas o stream existe),
por isso o lado positivo passava e o negativo quebrava no mesmo código.

```bash
ffmpeg -loglevel error -y -i "$v" -f lavfi -i anullsrc=r=44100:cl=stereo \
  -c:v copy -c:a aac -b:a 32k -shortest "$tmp" && mv "$tmp" "$v"
```

### Checagem de integridade do conjunto (antes de gastar GPU)

Um único arquivo ruim derruba o batch inteiro. Verificar tudo de uma vez:

```bash
for f in $DIR/*.mp4; do
  a=$(ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$f" | grep -c audio)
  [ "$a" = "0" ] && echo "SEM AUDIO: $f"
  [ -f "${f%.mp4}.txt" ] || echo "SEM CAPTION: $f"
done
```

Neste treino isso pegou: **1 vídeo sem áudio** (remux falhou silenciosamente,
removido) e **9 sem caption** (recuperadas do positivo).

---

## 6. Configs de dataset

`animateka_pos.toml`:

```toml
[general]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
caption_extension = ".txt"

[[datasets]]
resolution = [832, 480]
target_frames = [57]
target_fps = 24.0
frame_extraction = "head"
video_directory = "/workspace/datasets/animateka/videos"
cache_directory = "/workspace/datasets/animateka/cache_pos"
num_repeats = 1
```

`animateka_neg.toml` é idêntico, trocando `video_directory` para
`animateka_neg/videos` e `cache_directory` para `cache_neg`.

> **`target_frames` FIXO e `frame_extraction = "head"` nos DOIS lados.** O
> `mpdecimate` encurta o negativo; deixar o comprimento livre daria latentes de
> shapes diferentes e o pareamento por nome falharia.

---

## 7. Cache

```bash
# latentes dos DOIS lados
for side in pos neg; do
  python ltx2_cache_latents.py --dataset_config animateka_${side}.toml \
    --ltx2_checkpoint /workspace/models_ltx2/ltx-2.3-22b-dev.safetensors
done

# texto UMA vez (a caption e a mesma; o slider aponta text_cache_dir para o positivo)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python ltx2_cache_text_encoder_outputs.py --dataset_config animateka_pos.toml \
  --ltx2_checkpoint /workspace/models_ltx2/ltx-2.3-22b-dev.safetensors \
  --gemma_root /workspace/models_ltx2/gemma \
  --gemma_load_in_4bit --batch_size 1
```

**`--gemma_load_in_4bit --batch_size 1` são obrigatórios em 32 GB.** O script
carrega o **LTX 22B e o Gemma juntos**; em bf16 e em 8-bit dá OOM. Em 4-bit
roda em ~6 min para 456 captions.

Tamanho do cache: latentes 246 MB (pos) + 200 MB (neg); texto **22 GB**.

### Precache das amostras (para o Gemma não competir por VRAM no treino)

```bash
python ltx2_cache_latents.py --dataset_config animateka_pos.toml \
  --ltx2_checkpoint ... --precache_sample_latents --sample_prompts slider_sample_prompts.txt --skip_existing

python ltx2_cache_text_encoder_outputs.py --dataset_config animateka_pos.toml \
  --ltx2_checkpoint ... --gemma_root ... --gemma_load_in_4bit --batch_size 1 \
  --precache_sample_prompts --sample_prompts slider_sample_prompts.txt --skip_existing
```

O modo default é *"deferring Gemma encoding until sampling"* — ou seja, ele
carrega o Gemma a **cada** evento de sampling. Com precache, nunca carrega.

---

## 8. PATCH OBRIGATÓRIO no trainer

Sem isto o treino descarta **todos** os pares de vídeo com
`"No text cache for ..."` e morre com `No matched pairs found`.

**Causa:** `_LATENT_BASENAME_RE = r"^(.+)_\d{4}x\d{4}_ltx2\.safetensors$"` segue
a convenção de **imagem**. Todo modo de `frame_extraction` — inclusive `"full"`
— anexa o token de intervalo de frames ao `item_key`
(`image_video_dataset.py:3562`), e o dataset de vídeo compensa resolvendo o
cache de texto com `tokens[:-3]` (`:3712`), enquanto imagens usam `tokens[:-2]`
(`:2794`). O slider gerava o stem `c_000_00000-057` e procurava
`c_000_00000-057_ltx2_te.safetensors`, que nunca existe para vídeo.

Em `src/musubi_tuner/ltx2_train_slider.py`, logo antes de `te_path`:

```python
te_basename = f"{stem}_ltx2_te.safetensors"
if not os.path.exists(os.path.join(self.text_cache_dir, te_basename)):
    video_stem = re.sub(r"_\d{5}-\d{3}(?:-\d{2})?$", "", stem)
    if video_stem != stem:
        te_basename = f"{video_stem}_ltx2_te.safetensors"
```

Confirmar depois: o log deve dizer `found 261 matched pairs` (ou o número que
corresponder ao seu conjunto), e **zero** `No text cache`.

---

## 9. Config do slider

`anime_timing_slider_ref.toml`:

```toml
mode = "reference"
reference_modality = "video"
pos_cache_dir  = "/workspace/datasets/animateka/cache_pos"
neg_cache_dir  = "/workspace/datasets/animateka/cache_neg"
text_cache_dir = "/workspace/datasets/animateka/cache_pos"
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
```

**Convenção de sinal:** multiplicador **positivo** → animação limitada (o
alvo); **negativo** → interpolado (o defeito).

**Âncoras não se aplicam** ao modo `reference` — o trainer as ignora, porque os
pares já se auto-regularizam. Aqui isso é literalmente verdade: os dois lados
*são o mesmo vídeo*.

---

## 10. O comando de treino

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_cpu_threads_per_process 8 --mixed_precision bf16 \
  ltx2_train_slider.py \
  --mixed_precision bf16 \
  --ltx2_checkpoint /workspace/models_ltx2/ltx-2.3-22b-dev.safetensors \
  --fp8_base --fp8_scaled \
  --sdpa \
  --blocks_to_swap 0 \
  --gradient_checkpointing \
  --max_data_loader_n_workers 4 --persistent_data_loader_workers \
  --network_module networks.lora_ltx2 \
  --network_dim 16 --network_alpha 16 \
  --lora_target_preset video_sa_ca_ff \
  --learning_rate 1e-4 \
  --optimizer_type AdamW8bit \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 \
  --max_train_steps 600 \
  --ltx2_first_frame_conditioning_p 0.9 \
  --output_dir /workspace/outputs/slider_ref --output_name anime_timing_ref \
  --slider_config anime_timing_slider_ref.toml \
  --save_every_n_steps 150 \
  --gemma_root /workspace/models_ltx2/gemma --gemma_load_in_4bit \
  --use_precached_sample_prompts --use_precached_sample_latents \
  --sample_prompts_cache /workspace/datasets/animateka/cache_pos/ltx2_sample_prompts_cache.pt \
  --sample_latents_cache /workspace/datasets/animateka/cache_pos/ltx2_sample_latents_cache.pt \
  --sample_prompts slider_sample_prompts.txt \
  --sample_every_n_steps 200 \
  --logging_dir /workspace/outputs/slider_ref/logs
```

**Resultado:** 600 steps, 261 pares, **1h27**, ~4,3 s/step, loss 0.33 → 0.18.

### Por que cada escolha

| flag | valor | motivo |
|---|---|---|
| `--blocks_to_swap` | **0** | **MEDIDO: 0 → 2,21 s/step; 20 → 8,4 s/step.** Swap é tráfego CPU↔GPU. Herdar esse valor de outra fase custou quase 4× em dinheiro de GPU. Começar em 0 e só subir se der OOM. |
| `--gradient_checkpointing` | **ligado** | libera VRAM sem custar tráfego. Sem ele dá OOM com latentes de vídeo reais. |
| `--lora_target_preset` | `video_sa_ca_ff` | **`t2v` DEGRADA o áudio do modelo base**: cria pesos de LoRA nas camadas de áudio e cross-modais que nunca recebem gradiente com sentido, e aplicar a LoRA sobrescreve o áudio com deltas quase-zero. Os presets `video_*` restringem ao ramo de vídeo. |
| `--ltx2_first_frame_conditioning_p` | **0.9** | ancora o frame 0 como condicionamento e o exclui da loss. Como o uso real é i2v ~100% das vezes, o treino tem que estar nesse regime quase sempre. E não se perde nada: o par **compartilha** o frame 0 por construção, então ele não carrega informação de eixo. Os 10% restantes mantêm o eixo válido em t2v. |
| `--network_dim` | 16 | o doc recomenda 8–16 para sliders. |
| `--max_train_steps` | **600** | ver §11 — o eixo só assentou entre 400 e 600. |
| `--sdpa` | — | único backend disponível sem compilar flash/sage. |

---

## 11. Resultado medido

Diff mediana entre frames consecutivos nas amostras do step 600 — mesma seed,
mesmo prompt, mesma imagem inicial, só o multiplicador muda:

| multiplicador | diff mediana | |
|---|---|---|
| **−2.0** | 0.01203 | mais movimento por frame (fluido) |
| −1.0 | 0.00489 | |
| 0.0 | 0.00108 | base |
| +1.0 | 0.00081 | |
| **+2.0** | **0.00052** | menos movimento por frame (escalonado) |

**Monotônico, faixa de 23×.**

Sinal independente da métrica: o **tamanho dos arquivos** cai monotonicamente
(1,5 MB em −2.0 → 1,2 MB em +2.0). Menos mudança entre frames = menos dados
para o codec.

### O eixo emergiu entre os steps 400 e 600

| step | comportamento |
|---|---|
| 200 | ruído |
| 400 | irregular: 0.0039 / 0.0023 / 0.0011 / 0.0023 / 0.0019 (**não monotônico**) |
| 600 | **monotônico** |

**Parar em 400 teria dado um falso negativo.** Norma L2 dos deltas: step300 =
11,01 → step450 = 13,89 → step600 = **17,84**.

> Isso também significa que o `step450` **não** é "uma versão suave do v1" — é
> um eixo possivelmente ainda não assentado. Para efeito mais fraco, baixe o
> `strength` do v1 (que está num eixo medido), não use um checkpoint anterior.

### Métrica: o que serve para quê

- **`dup%`** (fração de frames quase idênticos) serve para caracterizar o
  **dataset**. É **inútil** no material gerado: um modelo de difusão
  praticamente nunca produz frames exatamente iguais, sempre há ruído residual.
- **diff mediana** é a métrica que capta o eixo nas **saídas**.

---

## 12. Como usar

Copiar `anime_timing_ref.comfy.safetensors` (o trainer já salva o formato
ComfyUI) para `ComfyUI/models/loras/`.

**Nenhum custom node é necessário.** ComfyUI tem LTX-2.3 nativo (`LTXAV` em
`supported_models.py`, `av_model.py` com `prompt_adaln_single`), e as chaves da
LoRA (`diffusion_model.transformer_blocks.N.attn1.to_k.lora_A.weight`) casam com
o "generic lora format" do `model_lora_keys_unet`.

O slider é um `LoraLoaderModelOnly` comum — **o `strength` é o slider**, e ele
aceita valores negativos:

| strength | efeito |
|---|---|
| **+2.0** | timing bem escalonado, frames segurados |
| **+1.0** | moderado |
| 0.0 | LTX base |
| **−2.0** | interpolado (o defeito amplificado) |

Começar em **+1.5**. Valores além de ±2 extrapolam (não testado acima de +2).

---

## 13. Armadilhas que custaram tempo (evite repetir)

### De método

1. **`pkill -f <padrão>` mata o próprio shell** quando o padrão aparece no
   comando. Causa de vários "exit code 1/144" inexplicáveis — e chegou a
   impedir a gravação de um arquivo. Use `pkill -f "[p]adrão"`.
2. **Editar um `.sh` enquanto ele roda** corrompe a execução: o bash lê o
   arquivo incrementalmente e passa a executar conteúdo misturado. Aconteceu
   3× nesta sessão.
3. **`| grep ... | head -N`** no pipe de um processo longo: quando o `head`
   sai, o processo leva SIGPIPE e **morre no meio**.
4. **`python` sem `-u`** com stdout redirecionado: o log fica vazio durante
   toda a execução, impossível distinguir "progredindo" de "travado".
5. **Medir o custo do experimento pequeno antes de rodar o grande.** Uma
   config de slider de texto deu 80 s/step porque eu empilhei 5 targets ×
   2 multiplicadores × 3 linhas de âncora = 30 passes com backward por step,
   sem multiplicar antes.

### De configuração

6. **`blocks_to_swap` só quando medido.** Ver tabela §10.
7. **Treino e inferência têm perfis de memória diferentes.** Treino cabe com
   swap 0; a **inferência standalone** funde a LoRA nos pesos e o modelo em
   fp8 ocupa **26,3 GB de 32** — sobra pouco para o `res_2s` (solver de 2ª
   ordem). O sampling **durante o treino** funciona porque lá a LoRA é rede, não
   merge.
8. **`--sample_tiled_vae` é o interruptor**; `--sample_vae_tile_size` é só
   parâmetro. Passar só o parâmetro não liga nada.
9. **OOM na inferência é silencioso**: vem como
   `ERROR ... Sampling failed for prompt, skipping`, o script pula todos os
   prompts e termina imprimindo **PRONTO sem ter gerado nada**. Sempre contar
   os arquivos de saída, nunca confiar na mensagem final.

---

## 14. Fase que FALHOU (e por quê) — não repetir

Antes disto tentei o modo **`text`** do slider: 5 formulações do conceito
(`animated on twos`, `limited animation`, `stop motion`, `12 drawings per
second`, `traditional hand-drawn timing`) contra os negativos correspondentes,
2 âncoras, 200 steps.

**Resultado:** frames segurados 33,3% (mult −2) vs 35,8% (mult +2) — ruído — e
vídeos visualmente indistinguíveis.

**Mecanismo:** o modo texto computa `direction = pred_pos − pred_neg` no modelo
**congelado**. Ele só extrai a direção que o modelo **já associa** aos prompts.
Se "animated on twos" não tem representação distinta no condicionamento do
LTX-2.3, `direction ≈ 0` e não há o que amplificar.

**Diagnóstico barato para qualquer slider de texto:** a **loss inicial** é
essencialmente `‖gs·direction‖²`, porque a LoRA ainda é zero. No modo texto ela
deu **0.0003**; no modo reference, **0.33** — mil vezes maior. Se a loss inicial
for uma ordem de magnitude abaixo do esperado, o conceito não está no modelo e
não vale treinar.

---

## 15. O que ficou por testar

- **Ações rápidas** (corrida, soco, salto). As amostras existentes são de
  movimento sutil — o pior caso para julgar timing, porque "frame segurado" e
  "movimento lento" ficam parecidos. Nunca consegui gerar as de ação por causa
  do problema de memória da inferência standalone (§13.7).
- **Multiplicadores acima de +2** (extrapolação).
- **`batch_all_targets = true`** num refino curto: o argumento de que a média
  entre formulações cancela ruído de contexto continua válido, só é caro.
- **Os 252 vídeos restantes** do animateka (o download parou em 456/708).
