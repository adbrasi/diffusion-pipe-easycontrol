# Handoff para o agente da GPU — Anima "próxima cena" (2026-09-30)

Você está numa **RTX 5090 (32 GB) alugada na Vast.ai a ~US$0,62/h**, com disco de
~180 GB e sem volume persistente. O usuário (adbrasi / AdwolfCzar) vai deixar você
trabalhando por horas com bastante autonomia. Ele é do Brasil e cada hora pesa no
bolso. **Eficiência de GPU é requisito, não detalhe.**

Uma sessão anterior (sem GPU) fez a pesquisa completa e implementou o método. **A sua
parte é a execução:** preparar o ambiente, validar o método com experimentos curtos e
baratos, escolher com evidência e rodar o treino que resolve o problema. Você tem
liberdade total para corrigir código, mudar hiperparâmetros e desviar do plano quando
a evidência mandar. Documente o porquê.

---

## 0. O objetivo (o que o usuário quer de verdade)

Um adapter para o **Anima** (DiT anime de 2B, derivado do Cosmos-Predict2) que recebe
uma imagem de referência (cena 1) e um prompt e gera a **próxima cena**, preservando
personagem, estilo e ambiente. Exemplo: "um gato sentado no sofá" → "o mesmo gato em pé
no mesmo sofá". Hoje isso é quase impossível no Anima. No Krea 2 o usuário chegou a
algo usável com o mesmo dataset, então o dataset não é o problema. O problema era o
método.

As duas falhas que se repetiram por meses:
1. **"Devolve a mesma imagem"** (atrator de cópia): o adapter reproduz a referência e
   ignora o prompt.
2. **"Bonito, mas sem relação com a ref"** (atalho de legenda): a legenda descreve tudo,
   então a ref é ignorada.

## 1. Leia primeiro (nesta ordem)

1. **`docs/ANIMA_NEXTSCENE_PESQUISA_2026-09.md`** — o diagnóstico, as evidências e o
   método. Obrigatório, inteiro.
2. `models/anima_nextscene.py` — a implementação (tipo `anima_nextscene`).
3. `examples/anima_nextscene/*.toml` — os configs (A/B de geometria + run longo).
4. `tools/nextscene_pairs.py`, `tools/nextscene_eval.py`, e `--mode nextscene` em
   `infer_easycontrol.py`.
5. Leitura rápida para contexto histórico: `docs/AUDITORIA_2026-08-06.md`,
   `docs/ANIMA_FUTURO_TREINO.md`, `docs/RODADA2_ARM_D_VENCEDOR.md`, `docs/KREA2_EDIT_SAGA.md`
   §6 (a lista de armadilhas operacionais é ouro).

**Atenção:** o `CLAUDE.md` da raiz está desatualizado. Ele fala do IC-LoRA antigo e da
`anima-preview2`. Ignore as instruções de treino dele; valem este arquivo e o doc de
pesquisa.

Resumo do método, se você só tiver 30 segundos:
- O Anima é só T2I (x_embedder de 68 colunas). A ref colocada em `(t=1, h, w)`,
  no grid do alvo, fica posicionalmente igual ao pixel vizinho do alvo (núcleo RoPE
  0,987 contra 0,988). Isso cria o atrator de cópia.
- `anima_nextscene` faz o seguinte:
  - ref limpa em t=0 por frame;
  - **RoPE disjunto** (`disjoint_w`: a ref deslocada um frame inteiro na largura);
  - LoRA global em self_attn + mlp + cross_attn (nunca adaln nem llm_adapter);
  - `ref_dropout` 0,1;
  - loss ponderada pela diferença A↔B;
  - 20% dos steps em σ∈[0,8; 1].
- **O código nunca rodou em GPU.** Só passou em 13 testes de CPU com um DiT minúsculo.
  Espere bugs de integração e conserte.

## 2. Setup (faça, verifique, siga)

```bash
cd /workspace
git clone https://github.com/adbrasi/diffusion-pipe-easycontrol.git
cd diffusion-pipe-easycontrol
git checkout claude/elegant-ptolemy-8kywqy
git submodule update --init --recursive      # comfy é importado pelo train.py
pip install -r requirements.txt
pip install torchaudio dghs-imgutils          # torchaudio: comfy importa; imgutils: métrica CCIP
python -m pytest -q test/test_anima_nextscene.py test/test_anima_inference_contract.py
```
Armadilhas conhecidas de ambiente (sagas anteriores):
- O índice cu128 do PyTorch já deu timeout. O torch do PyPI padrão funciona na
  sm_120; confira com `get_device_capability()==(12,0)`.
- O `bitsandbytes` 0.50 quebrou o AdamW8bitKahan (já corrigido no fork).
- Se o venv for de root, `chown` antes de instalar.

Modelos (Anima **Base v1.0**, não preview2; as chaves com prefixo `net.` já são tratadas):
```bash
mkdir -p /workspace/models_anima && cd /workspace/models_anima
for f in diffusion_models/anima-base-v1.0.safetensors vae/qwen_image_vae.safetensors text_encoders/qwen_3_06b_base.safetensors; do
  huggingface-cli download circlestone-labs/Anima split_files/$f --local-dir .
done
```

## 3. Dataset

`https://huggingface.co/datasets/AdwolfCzar/proxima_cena_grounded_original_dataset`
(6,1 GB, 12.455 pares). Cada subset `dsN_*/images_A` (ref) + `images_B` (alvo +
`<stem>.txt`), casados por stem:

| subset | pares | nota |
|---|---|---|
| ds1_recortados | 2.900 | muito conteúdo NSFW |
| ds2_poxima_v2 | 5.400 | o principal |
| ds3_comikontext | 2.900 | quadrinhos; boa parte NSFW |
| ds4_contexto_curado | 1.255 | curadoria manual do usuário (usar `num_repeats` 2–3) |

Passos:
1. Baixar (`huggingface-cli download --repo-type dataset ... --local-dir /workspace/ds`).
2. **Held-out ANTES de tudo:** separe 24 pares (6 por subset, seed fixa) em
   `/workspace/heldout/{target,control}` e **tire-os do treino**. Inclua de propósito
   alguns casos "mesmo cenário, pose nova" e "câmera nova". Nunca avalie em pares de
   treino (a rodada 2 mostrou que eles são decorados).
3. Auditoria por subset: `python tools/nextscene_pairs.py audit --target <images_B> --control <images_A> --out <sub>.csv --dino`.
   Olhe as distribuições (`dhash_ham`, `pix_sim`, `dino_cos`, `words`), calibre os
   limiares olhando ~20 pares na fronteira e faça o `build` para
   `/workspace/ns/<sub>/{target,control}`. `--reverse` só serve se as refs tiverem
   legenda própria; aqui provavelmente não têm.
4. **Legendas: aqui mora o atalho de legenda.** As legendas atuais são descrições ricas
   de B com 60–110 palavras, incluindo aparência. O Krea 2 aguentava isso porque o
   texto dele via a imagem; **o texto do Anima não vê**. Monte, por pasta de alvo, um
   `captions.json` com o nome do arquivo de imagem como chave e uma lista de legendas
   como valor. Cada legenda vira uma amostra cacheada. Formato:
   `{"000123.png": ["<legenda completa>", "<legenda curta>"]}`.
   - A curta mantém enquadramento, ação/pose e o que é novo, e tira a aparência do que
     já está na ref. Pode ser a 1ª frase + a oração de ação, ou um LLM barato se houver
     chave.
   - Isso ensina "o que não está escrito vem da ref".
   - Registre a mediana de palavras de cada tier. Ver §6 do doc de pesquisa e
     `docs/PROPOSTA_CAPTIONS_v3.md`.
5. Ajuste `examples/anima_nextscene/dataset.toml` (um `[[directory]]` por subset; ds4
   com repeats). Meça e registre a proporção NSFW/SFW. No Krea 2, 60% de NSFW virou
   viés no adapter.

## 4. Disciplina de GPU (o usuário insiste nisto)

- **Smoke primeiro, sempre:** 10 steps, conferindo:
  - o log `RoPE layout=...`;
  - o número de alvos do LoRA (cross_attn incluído, adaln/llm_adapter fora);
  - o audit do adapter no save;
  - VRAM e **s/step medidos**;
  - resume funcionando.
  Depois, **uma** geração de smoke com `infer_easycontrol.py --mode nextscene`
  antes de qualquer batch de avaliação.
- **Hipóteses se validam com treinos curtos:** 250–1.000 steps, com grids e métricas
  em cada save. Treino longo só depois que um braço provar que usa a ref sem copiar.
- **Batch:** `gradient_accumulation_steps` não dá aprendizado de graça; só muda o que
  conta como um step. Compare braços por **amostras vistas**, não por steps. Se couber,
  prefira `micro_batch_size_per_gpu` maior (2–4 a 512px; todas as amostras de um batch
  precisam cair no mesmo bucket) a acumular. Meça o throughput em amostras/s e escolha.
- **Resolução:** 512 para os probes (o task independe de resolução). 768/1024 só no
  acabamento do run final.
- **Avaliação barata:** `tools/nextscene_eval.py --limit 12 --steps 20` nos probes.
  O conjunto completo (24 pares, 30 steps) fica para os candidatos finais.

## 5. Plano experimental (ponto de partida — adapte com evidência)

**E0 (grátis, opcional, sem treino):** se houver algum adapter IC-LoRA antigo no HF do
usuário (ex.: `AdwolfCzar/groundedsecret` → `anima_probes/`), gere com
`--mode nextscene --rope_layout aligned --ref_temporal_index 1` variando
`--ref_renoise` 0 / 0,15 / 0,3. Se a cópia cair com o re-ruído, o mecanismo de cópia
está confirmado. Só vale se for rápido; não bloqueie o resto por isso.

**E1 — o A/B de geometria (a pergunta principal):**
`probe_A_aligned.toml` contra `probe_B_disjoint_w.toml`. Os dois só diferem em
`rope_layout`. Reduza `max_steps` para caber no orçamento (por exemplo 1.000 steps com
batch efetivo 4), com saves a cada 250 e eval em cada save. Leia:
- `ref_gain` > 0: usa a ref de verdade (a saída muda ao trocar a ref, na direção do B real).
- `copy_rate` / `copy_gap`: o atrator de cópia. A hipótese central é que o
  `disjoint_w` fica bem menor.
- `ccip_true`: identidade de personagem contra B.
- **O grid** (A | B | ref certa | ref trocada | ref nula) — o veredito é visual; a métrica faz triagem.

**E2 — knobs, um de cada vez, sobre o vencedor do E1** (probes curtos, na ordem de
valor esperado):
1. Tiers de legenda (com e sem o tier curto), se o E1 mostrar `ref_gain` ≈ 0.
2. `ref_temporal_index` (1 contra 4) — a distância temporal funciona como dial de
   "quanta mudança" (FramePack-1f).
3. `ref_hflip_prob` 0,3 ou `ref_noise_prob` 0,5 — se a cópia persistir.
4. `rank` 128 ou `lr` 1e-4 — se aprender devagar demais.
5. `disjoint_h` contra `disjoint_w` — só se sobrar tempo.

**E3 — run final** (`full_run.toml`) com a melhor combinação, sobre o dataset filtrado
inteiro:
- avaliação a cada ~1.000 steps;
- escolha do checkpoint pelas métricas mais o olho, não pelo último;
- acabamento de 10–20% dos steps em 768/1024.

O AnimaRefLora viu a fase de cópia subir e depois recuar com o anti-cópia ligado. Olhe a
**tendência** da `copy_rate` antes de abortar um run.

Critério de "resolvido" para mostrar ao usuário: nos held-out, com `lora 1.0` e
`ref_cfg 1.0`, a saída segue o prompt (pose/ação/câmera nova), mantém personagem,
estilo e cenário da ref, e **não** é uma cópia de A. `ref_cfg` alto é dial, não muleta
(regra do próprio usuário).

## 6. Hugging Face e registro (padrões do usuário)

- Crie um repo **privado** no HF (ex.: `AdwolfCzar/anima-nextscene-runs`; o token já
  está no ambiente). Suba, por experimento: checkpoints candidatos
  (`adapter_model.safetensors` + `adapter_config.json` + `nextscene_contract.json`),
  grids, `metrics.json`/`summary.csv` e o `.toml` usado. Faça upload **contínuo**
  durante o treino; a instância pode morrer.
- O model card é vitrine do usuário: **curto**, só o essencial. Nada de encher de
  detalhe de método (ele já reclamou disso).
- **Documente tudo, como nas sagas anteriores** — esses registros foram o que permitiu
  diagnosticar o problema depois. Crie e mantenha `docs/NEXTSCENE_RUN_LOG.md` na branch
  `claude/elegant-ptolemy-8kywqy`, com commit e push a cada marco. Registre:
  - por experimento: hipótese, diff de config, amostras vistas, s/step, VRAM, métricas
    por checkpoint, link do grid no HF, veredito e o que muda depois;
  - erros e causas-raiz, incluindo os seus próprios erros de julgamento;
  - o estado atual e como retomar (comandos exatos).
- Escreva em português, direto. Nada de conclusão com n=1 sem avisar que é n=1.

## 7. Operação (lições pagas caro)

- **Parar treino:** `touch <run_dir>/save_quit` (salva o estado completo e sai).
  **Nunca SIGTERM** — já se perderam 500 steps assim.
- **Sobreviver a reboot:** processo longo como serviço do supervisord (o container do
  Krea 2 reiniciou duas vezes); no mínimo `nohup` com log em arquivo. Monitores do
  agente morrem junto com o reboot.
- **Disco (~180 GB):** cada `global_step*` do DeepSpeed tem alguns GB. Mantenha 2–3
  estados de resume, apague checkpoints já enviados ao HF, e apague o bruto do dataset
  depois do build.
- **Resume:** o LR que vale é o do log, não o da config; o DeepSpeed restaura o antigo.
- **Contrato treino↔inferência:** o adapter grava a geometria no header e o runner lê
  sozinho. Se alguma geração sair ruído puro, primeiro suspeite do contrato/modo, depois
  do método.
- `pkill`/`pgrep` com padrão que casa a própria linha de comando mata a própria shell.
- O CFG padrão do runner (`--uncond_ref keep`) é só de texto, com a ref nos dois ramos.
  `--ref_cfg` > 1 usa o nulo treinado pelo `ref_dropout`.

## 8. Liberdade e limites

- Pode e deve corrigir bugs, refatorar e adicionar knobs. Rode os testes de CPU depois
  e faça commit com mensagem clara.
- Se a evidência contradisser o doc de pesquisa, **confie na evidência** e escreva por
  quê no log. O doc é um ponto de partida bem fundamentado, não um dogma.
- Ideias de baixa prioridade (não foque nelas sem motivo):
  - treinar por cima do AnimaRefLora (LoKr LyCORIS, precisaria ser fundido no base;
    geometria compatível via `ref_temporal_index=-1` + `disjoint_w`);
  - node ComfyUI para o layout disjunto — o sampler nativo do ComfyUI limita as posições
    do RoPE, então o node precisa instalar `install_nextscene_rope` e ampliar esse
    limite. Faça só depois que houver um vencedor.
- Quando o usuário voltar, ele vai querer em poucas linhas:
  - o que rodou;
  - o que funcionou e o que não funcionou;
  - o melhor checkpoint, com link do HF e do grid;
  - quanto de GPU foi gasto;
  - o próximo passo recomendado.
