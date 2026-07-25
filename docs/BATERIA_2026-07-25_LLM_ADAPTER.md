# Bateria 2026-07-25 — resolvendo a tensão do llm_adapter (Anima)

**Objetivo:** decidir com evidência (não só arqueologia) se o `llm_adapter` deve
ser treinado ou congelado no Anima, depois do achado de que os probes de 500
steps que decidiram "v3 vencedor" em julho tinham `llm_adapter_lr=0` por bug
(ver `docs/ANIMA_V3_VENCEDOR.md`). Ver também a tensão com a recomendação
oficial do criador do Anima (NUNCA treinar o llm_adapter) documentada em
`/workspace/research/ANIMA_TRAINING_GUIDE.md` (fora do git, pesquisa de abril).

## Dataset

`dataset_pares_contexto_captioned.zip` (o `contexto_rush` da saga, 1255
pares) — baixado do mega.nz e extraído em
`/workspace/dataset_raw/extracted/{input_A,input_B}`. Verificado nesta sessão:
stems batem 1:1, gaps temporais reais entre ref e alvo (não é atalho de
reconstrução tipo Ideogram4), AR predominante 16:9 com alguns 2.4 widescreen.

## Método testado (Arm 1)

`models/ic_lora_dual.py::ICLoraV3Pipeline` (`type = 'ic_lora_v3'`) — escopo
LARGO de LoRA (self_attn + mlp + cross_attn + llm_adapter), target-first
`[alvo T=0 | ref T=1]`, adaln FORA, rank 32, batch efetivo 8, lr 1e-4,
sigmoid_scale 1.0, condition_dropout 0.0. Config:
`examples/battery_2026-07-25/arm1_broad_llm_frozen.toml`.

**Arm 1 especificamente:** `llm_adapter_lr = 0` (congelado) — a réplica
intencional do cenário oficial/conservador, como controle. 1000 steps,
checkpoints em 250/500/750/1000.

Confirmado no log de startup (a salvaguarda contra o bug de julho):
`BROAD LoRA targets: 340 linears (136 cross_attn, 60 llm_adapter, 0 adaln)`,
`llm_adapter_lr=0`, `Num llm_adapter params: 120`.

## Infra criada nesta sessão

- `examples/battery_2026-07-25/*.toml` — configs dos braços (arm1 congelado,
  arm2 lr baixo *[cortado por decisão do usuário, ver abaixo]*, arm3 lr cheio).
- `tools/battery_eval.sh` + `tools/battery_grid_assemble.py` — geram, por
  checkpoint, um grid de 3 pares fixos × [referência, alvo real, sem ref,
  ref força 1.0, ref força 1.5], mesma seed em tudo.
- `tools/battery_orchestrate.sh` — encadeia treino→avaliação→próximo braço
  sozinho, sem precisar de intervenção entre braços.
- `utils/reduction.py` — fix de compatibilidade com PyTorch ≥2.13 (removeu
  `torch._namedtensor_internals`; reimplementado inline). Necessário rodar
  neste ambiente (torch 2.13.0+cu130). Submódulos `ComfyUI` e `HunyuanVideo`
  precisaram ser inicializados (`git submodule update --init`) porque o
  backend de cache de dataset agora depende do pacote `comfy`.

**Exemplos de avaliação (v2, corrigidos):** o primeiro conjunto (hóquei
3D/CGI, demônio vermelho, ouriço) foi trocado a pedido do usuário — Anima é
treinado exclusivamente em anime, referência realista/CGI é ruim pra avaliar.
Conjunto atual: `imagem000180`, `imagem001129`, `imagem001549` (frames de
anime nítidos, estilo Demon Slayer, personagens bem definidos).

## Resultado em s1000 (Arm 1, llm_adapter CONGELADO)

Qualidade de geração muito boa — ex1 (espadachim) e ex3 (três homens em
trajes tradicionais) ficam muito próximos do alvo real em composição,
personagem e cores, já em 1000 steps.

**Mas**: a coluna "sem ref" está praticamente idêntica à "com ref" nos 3
exemplos, em todos os checkpoints (250/500/750/1000). Isso é sinal de
**atalho de caption** (mesma classe de falha documentada no debate do
Ideogram4, `docs/IDEOGRAM4_DEBATE_METODOS.md`), não prova de uso genuíno da
referência: os 3 exemplos de avaliação fazem parte do próprio dataset de
treino (não são held-out), e a 1000 steps × batch 8 o modelo já viu cada
par ~6,4 vezes — plausível memorizar caption→imagem diretamente.

**Conclusão parcial:** o método (ic_lora_v3, escopo largo, target-first)
produz geração de alta qualidade e fiel ao estilo anime do dataset — isso
está provado. Ainda NÃO está provado que a referência (vs. o caption sozinho)
é o que está guiando a geração. Precisa de teste held-out (referência fora
do dataset de treino) antes de declarar vitória do método ou decidir o eixo
do llm_adapter.

## Em andamento

Arm 3 (`llm_adapter_lr` = lr base, sem o bug — o teste real da "eureka" de
julho) treinando agora. Falta: teste held-out para separar memorização de
generalização, antes de comparar Arm 1 vs Arm 3 com confiança.
