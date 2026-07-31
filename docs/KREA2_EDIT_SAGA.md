# Saga do adapter de edição por referência — KREA2 (2026-07-29 → 07-31)

Registro completo do treino do adapter `krea2_multiref_grounded` de 30.000
pares, da investigação que reconciliou o node ComfyUI com o runner, e dos
incidentes do caminho. O **método** está em `KREA2_MULTIREF_RECEITA.md`; este
documento é sobre a **execução**: o que foi rodado, o que quebrou, o que se
descobriu e o que ficou pendente.

Escrito no fim da saga, com o treino encerrado por decisão do usuário no step
13.445.

---

## 1. O que foi feito

Um LoRA de edição por referência sobre o KREA 2, treinado em 30.000 pares
A→B com uma referência limpa por exemplo, primeiro a 512px e depois adaptado
para 1024px.

| | fase 1 | fase 2 |
|---|---|---|
| resolução | 512 | 1024 |
| steps | 0 → 11.343 | 11.343 → 13.445 |
| lr na config | 1e-4 | 7,5e-5 (**não teve efeito — ver §6.1**) |
| tokens/sequência | 2.267 | 8.411 |
| s/step | ~11 | ~34 |
| blocks_to_swap | 0 | 8 |

Constantes nas duas fases: rank 64 (alpha = rank → escala 1.0), `txtfusion_rank`
128, `condition_only_lora` nos 28 SingleStreamBlocks, referência a t=0
per-token, `width_shift` no RoPE, grounding VL a 384², `caption_dropout` 0.1,
AdamW8bitKahan, bf16 com DiT em fp8, `micro_batch 1 × grad_accum 4`.

**30 checkpoints** (`step500` … `step13250`) e **27 rodadas de sample**, todos
publicados em `AdwolfCzar/k2-context-rush-ofc-beta1`.

---

## 2. O dataset da fase 1 — 30.000 pares

| fonte | pares | % |
|---|---|---|
| pico | 7.831 | 26,1 |
| recortados | 6.937 | 23,1 |
| comikontext (`mega3`) | 4.950 | 16,5 |
| parents | 4.815 | 16,1 |
| poxima_cena_v2 (`mega2`) | 4.129 | 13,8 |
| mega4 | 865 | 2,9 |
| inscene | 473 | 1,6 |

**As pastas `mega2`/`mega3`/`mega4` são nomes de download, não de conteúdo.**
Isso custou tempo depois: `mega2` é o poxima_cena_v2 e `mega3` é o
comikontext. O mapeamento só é recuperável em `/workspace/logs/normalize_all.log`.
Nomear pasta pela origem, não pela ordem de download.

O ponto que mais importou: **~60% do dataset é NSFW** (`recortados` +
`parents` + boa parte do `comikontext`). O adapter aprendeu isso junto com a
tarefa e passou a puxar o conteúdo para lá mesmo com prompt neutro. Foi o que
motivou o dataset 2 (§7).

### Cache

O cache fica em `<fonte>/target/cache/krea2_multiref_grounded` e somou
**729 GB** para os 30k pares — ~24,6 MB por par, dominado pelos embeddings de
texto. As imagens normalizadas são só ~11 GB.

**O cache de texto é indexado por aspect ratio, não por resolução.** Foi o que
tornou a fase 2 viável: mudar de 512 para 1024 reaproveitou 100% dos
embeddings de texto e só regerou os latentes.

---

## 3. Progressive resolution em vez de retreinar

Na virada para 1024 a pergunta foi: recomeçar ou adaptar?

Adaptar, porque a tarefa, o vocabulário das captions e o binding
referência→alvo já estavam aprendidos e são independentes de resolução. O que
muda com a resolução é a coordenada RoPE da referência (o `width_shift` põe a
ref em `w=[32,64)` a 512 e em `w=[64,128)` a 1024) e a densidade de detalhe.

A evidência foi empírica e veio antes da decisão: **o adapter de 512 já
produzia resultado bom a 1024 no runner** — 6 seeds aprovadas pelo usuário.
Precisava de ajuste, não de reaprendizado.

Confirmou-se na prática. O usuário, sobre os primeiros samples a 1024:
*"pior que ficou bom, tu teve uma boa ideia mesmo. em alta resolução tudo fica
bem melhor."*

---

## 4. A investigação do node ComfyUI — o achado principal

O usuário gera pelo ComfyUI, não pelo runner. O node produzia resultado
visivelmente pior: *"ele não mantem a mesma personagem, ele faz uma personagem
proxima"*. Mesmo adapter, mesmo prompt, mesma seed.

Foram ~6 h e 15+ agentes até fechar. **Cinco divergências**, todas medidas
contra o runner:

### 4.1 fp8-scaled vs fp8 cru — a causa dominante

O checkpoint é `fp8_scaled`: pesos fp8 com um `weight_scale` por tensor. O
ComfyUI carrega e usa a escala. **O trainer descarta o `weight_scale` e
re-quantiza no grid fp8 puro** (`models/base.py:547`).

Ou seja: o adapter foi treinado sobre uma base numericamente diferente da que
o ComfyUI usa. Sozinha, essa divergência dava `v relL2 = 0,58`.

Duas armadilhas na correção:

- **`.to(fp8)` sobre um `QuantizedTensor` é no-op.** Só troca o `orig_dtype`.
  Precisa de `dequantize()` antes — a primeira tentativa de correção não fez
  nada por causa disso e pareceu refutar a hipótese.
- Depois de corrigir, é preciso forçar `_full_precision_mm = True`, senão o
  matmul cai no GEMM fp8 nativo e reintroduz erro (0,178 contra 0,031).

### 4.2 Qwen3-VL: MRoPE 3D + DeepStack

O ComfyUI novo aplica MRoPE 3D e DeepStack no encoder de visão; a árvore de
treino não. Contexto divergia com `relL2 1,36`. Corrigido desfazendo
`Qwen3VL.forward` durante o encode.

### 4.3 Atenção GQA

`enable_gqa=True` contra `repeat_interleave` explícito produz resultados
diferentes. Achado do Codex.

### 4.4 As 7 chaves `.diff_b` da turbo

A LoRA turbo tem 7 chaves `.diff_b` que **o runner ignora e o
`LoraLoaderModelOnly` aplica**. Daí a regra de uso: turbo sempre dentro do
`K2 Training Base`, nunca pelo loader padrão.

### 4.5 Ruído

KSampler gera em CPU; o runner usa `torch.randn` em CUDA. Mesma seed, ruído
diferente.

### Resultado

**`v relL2` de 0,58 → 0,027, cos 0,9996.**

O trabalho virou três packs de nodes em `adbrasi/ctxrush-edit`: `K2 Training
Base` (model patcher que reproduz a base fp8 do treino em runtime, sem
checkpoint extra), `K2 Native` e `K2 Runner Bridge` (executa o runner exato em
subprocesso isolado).

---

## 5. O contrato, e por que nome de arquivo não faz parte dele

Todo adapter carrega metadata no header (`control_family`, `position_mode`,
`reference_model_timestep`, `sequence_layout`, `condition_token_stride`,
`vl_image_label`, …). O `expected_contract()` em
`tools/infer_reference_adapter.py:274` compara essa metadata com a config de
inferência e **falha** se divergir.

Duas propriedades que valem registrar:

- **O nome do arquivo do modelo base NÃO é requisito.** `base_model_file` é
  conferência mole: se diferir, só imprime `WARNING`
  (`infer_reference_adapter.py:353`). Renomear o base não quebra nada.
- **A metadata não pode ser censurada.** O `model_type` gravado é comparado
  com o `model.type` da config; alterá-lo faria o adapter só funcionar com uma
  config igualmente alterada. Por isso, quando o usuário pediu para não expor
  o nome do modelo base no HF, a solução foi redigir **a cópia publicada da
  config** (`tools/sanitize_config_for_hf.py`), nunca a metadata.

---

## 6. Problemas, incluindo os meus

### 6.1 O learning rate da fase 2 nunca mudou

A config da fase 2 pede `lr = 7.5e-5` (a redução de 25% que o usuário pediu).
**O treino rodou inteiro a `1e-4`.**

Causa: `--resume_from_checkpoint` restaura o estado do otimizador e do
scheduler do DeepSpeed, que carrega o LR da fase 1. O `lr` da config só vale
na construção do otimizador; num resume ele é ignorado em silêncio.

Verificável em qualquer log da fase 2:
`[Rank 0] step=13496, skipped=0, lr=[0.0001]`.

Consequência prática: nenhuma catastrófica — os samples foram aprovados. Mas
**quem ler a config vai concluir errado sobre o que rodou.** Encontrado pelo
Codex; o usuário decidiu não corrigir no meio do run. Fica aqui como aviso:
num resume, conferir o LR no log, não na config.

### 6.2 Parada abrupta destrói o resume (step 500)

Um SIGTERM matou o trainer sem estado de resume. A correção é usar o arquivo
de sinal: `touch <run_dir>/save_quit` faz o trainer salvar o checkpoint
DeepSpeed completo e sair limpo (`utils/saver.py:158`). **Nunca SIGTERM.**

### 6.3 O disco enchendo a 3,5 GB/h

Cada `global_step*` do DeepSpeed pesa ~2,3 GB e serve só para retomar aquele
ponto. Acumularam 28 deles (64 GB) e a guarda de espaço quase abortou o run.
Corrigido com `prune_resume_states()` no supervisor, mantendo 3.

### 6.4 O supervisor re-sampleando o passado

Ao retomar, o supervisor varria os `step*` antigos e pausava o treino para
gerar samples de cada um — 22 checkpoints × 3 samples antes de treinar um
step. Corrigido marcando os pré-existentes como já sampleados.

### 6.5 O supervisor sem `--resume` recomeçaria do zero

Faltava a flag; ela agora é obrigatória e verifica a existência de `latest`
antes de largar.

### 6.6 Meus erros

Registrados porque custaram tempo do usuário:

- **Teste pareado nulo.** Rodei um comparativo com `KSamplerAdvanced` em
  `add_noise=disable`, que **zera o `x_T`** (`CONST.noise_scaling` faz
  `sigma*noise + (1-sigma)*latent`; com sigma=1 e noise=0, `x=0`). O
  `dump_node.x.npy` tinha std 0,0. **Todas as correlações que reportei
  daquele teste eram lixo.**
- **"Latente idêntico bit a bit".** Comparei o mesmo `vae.encode()` do
  ComfyUI nos dois lados do teste. A diferença real era 6,6%.
- **Correlação de pixel como métrica.** O usuário cortou: *"pare de usar
  codigo para ver 'correlação' isso nao funciona"*. Ele estava certo — não
  discrimina o que importa aqui.
- **Model card inchado.** Enchi o card do HF de detalhe de método. O usuário
  limpou e avisou. Os repos HF são vitrine dele; subir só artefatos.
- **Hipótese errada sobre `blocks_to_swap`.** Previ ganho relevante baixando
  de 16 para 8; medido, foi 1,4% (34,9 → 34,4 s/step).

### 6.7 A instância reiniciou duas vezes

Em 2026-07-31, às ~17:51 e ~21:54 UTC. Sem traceback: log cortado no meio de
um step, `uptime` zerado. Não foi OOM nem erro do trainer — **o container
reiniciou e levou o servidor tmux inteiro**, com as três sessões.

Perda pequena (69 steps na primeira, 17 na segunda) porque a poda mantinha 3
estados de resume. Mas ficou **~3h50 parado** na primeira, sem ninguém
perceber.

**A lição que importa:** monitores e esperas do agente são processos da
sessão — morrem no mesmo reboot que deveriam detectar. Não servem de rede de
segurança. A correção real é registrar o supervisor como serviço do
supervisord do container, que sobe no boot. **Não foi feito** (mexe na infra
da instância e ficou pendente de autorização).

---

## 7. O dataset 2 (SFW) — preparado, não treinado

Com o adapter enviesado para conteúdo sexual, o usuário pediu um segundo
dataset e um treino do zero — **do zero, porque o viés está nos pesos, não no
dataset**.

Composição pedida, ~11.000 pares:

| fonte | pares | regra |
|---|---|---|
| poxima_cena_v2 | 5.990 | todos |
| recortados | 2.500 | amostra aleatória, seed 42 |
| comikontext | 2.500 | amostra aleatória **excluindo tudo abaixo do par 000800** |

Dois achados que economizaram horas:

- **As três fontes já estavam normalizadas em disco** desde a fase 1. Só o
  poxima precisou voltar do MEGA (1.871 pares apagados numa subamostragem
  anterior), e a pasta inteira tinha 2 GB — download seletivo não se
  justificava.
- **`prepare_krea2_edit.py:195` preserva o stem original** no nome
  (`{prefix}_{stem}`), então `comik_000800` é literalmente o par 000800 e a
  regra do usuário corta exatamente onde devia.

A árvore fica em `/workspace/datasets/sfw/` e é montada por
`tools/build_sfw_dataset.py` **por hardlink** — mesmo filesystem, custo zero
de disco, e sobrevive à limpeza dos caches da fase 1.

**Numa árvore separada de propósito:** as pastas de `prepared/` estavam sendo
lidas pelo treino em andamento, e o supervisor relança o trainer a cada 500
steps; arquivo novo ali faria o trainer cachear pares no meio do run.

Estado: **10.990 pares montados e validados**, samples sorteados
mecanicamente (1 por fonte, seed fixa, caption como está), configs escritas em
`examples/krea2_edit_sfw/`. **Nunca lançado** — o usuário encerrou antes.

---

## 8. Estado final e como retomar

- **Treino parado no step 13.445**; último adapter salvo: **`step13250`**.
- Tudo publicado em `AdwolfCzar/k2-context-rush-ofc-beta1`.
- Último estado de resume: `global_step13428` em
  `/workspace/checkpoints/krea2_edit_saga/20260729_17-22-13/`.
- Os 729 GB de cache **não** foram apagados.

Para retomar a fase 2:

```bash
cd /workspace/projects/diffusion-pipe && tmux new-session -d -s saga_train \
 "source /venv/main/bin/activate && python tools/krea2_saga_supervisor.py \
  --config examples/krea2_edit_saga/train_1024.toml \
  --samples /workspace/outputs/sampling_inputs/samples_1024.json \
  --sample-every 500 --min-free-gb 25 --resume"
```

`--resume` é obrigatória. Mudar `max_steps` ou `--sample-every` exige
relançar o supervisor: ele lê os dois na largada.

Para o dataset 2 (do zero, sem `--resume`), trocar a config para
`examples/krea2_edit_sfw/train_sfw.toml` e os samples para `samples_sfw.json`.
O `--extra` do uploader deve apontar para as cópias redigidas em
`/workspace/outputs/hf_upload_sfw/`, nunca para os `.toml` de `examples/`.

**A árvore de dados não sobrevive à instância** — só o que está no GitHub e no
HF persiste.

---

## 9. O que eu levaria para o próximo treino

1. **Conferir o LR no log, não na config**, sempre que houver resume.
2. **Nomear pasta de dataset pela origem**, nunca por ordem de download.
3. **Medir o balanço de conteúdo antes de treinar.** O viés de 60% NSFW era
   previsível por contagem de fonte e só foi percebido pelas gerações.
4. **Serviço no supervisord, não sessão tmux**, para qualquer coisa que
   precise sobreviver a reboot.
5. **Cache de texto indexado por aspect ratio** é o que torna barato subir de
   resolução. Vale desenhar para isso desde o começo.
6. **Divergência entre trainer e node de inferência é numérica antes de ser
   arquitetural.** A base fp8 sem escala explicou sozinha a maior parte do
   problema, e nenhuma inspeção de arquitetura teria achado.
