# A saga Anima completa — do "meu de abril era melhor" ao raiz_iclora

**Período:** 2026-07-17 a 2026-07-18 · **Branch:** `ic-lora`
**Docs irmãos:** `ANIMA_V3_VENCEDOR.md` (receita v3) · `OMINI_CONTROL_KREA2.md` / `OMINI_GROUNDED_SEGREDO.md` (lado Krea 2)

---

## Linha do tempo das descobertas

1. **Braços modernos (jul)**: ic_lora_v2 / ic_lora_routed / omini_subject treinados
   250→5000 steps no contexto_rush. Resultados medianos; usuário: "o de abril era
   MUITO melhor".
2. **Audit da inferência**: dois bugs reais no runner (referência esticada e em
   [0,1] no VAE; guidance de 2 branches acoplando a força da ref ao CFG do texto
   ×4). Fixes: preprocessing == treino (testes numéricos) + guidance de 3
   branches (`--ref_cfg` / Dual Guider no node). Grids antigos estavam
   contaminados; reavaliação mostrou os 3 braços saudáveis, mas ainda aquém da
   memória de abril.
3. **Bateria controlada** (rank 32, lr 5e-5, sampler puro): routed_tf vs
   routed_rf vs global_tf → os três próximos entre si; confirmou que a ORDEM
   temporal importa pouco (RoPE relativo, sem embedding absoluto).
4. **Arqueologia de abril**: commits 5–14/abr + configs no HF revelaram que os
   adapters bons (`next_scene_*_test1_1875steps_8batch`, `v2_s1950_r64`) tinham
   **escopo largo de LoRA** — cross_attn e/ou llm_adapter e/ou adaln treinados —
   que o fix de maio (`97fc94c`/`f85e808`) removeu por precaução.
5. **EUREKA — llm_adapter**: A/B com/sem llm_adapter na mesma seed: com ele o
   modelo ADOTA estilo e continuidade da referência; sem, anime chapado.
   **Armadilha fatal**: `llm_adapter_lr = 0` nas configs congela silenciosamente
   até o LoRA do llm_adapter (requires_grad False → nem salva). Abril não tinha
   essa linha. NUNCA usar com escopo largo.
6. **Braços broad**: `ic_lora_v3` (target-first, tudo global, dropout 0) ⭐
   veredito do usuário; `ominicontrol_broad` (= v3 com dropout 0.1, também
   aprovado); `ic_lora_dual` (aparência routada + semântica global, estilo do
   base preservado). Receitas em ANIMA_V3_VENCEDOR.md.
7. **Dataset de abril decifrado** (scene_rush_test1_frame_pairs + capp, 21.181
   pares após conversão): grupos A/B/C com TRÊS verbos de caption —
   "create the next scene, same character" (B), "create a different scene,
   different characters" (C = treino contrastivo do CORTE) e, no parents,
   "Change the scene to..." — o modelo de abril aprendeu operações, não só
   descrições. Trigger de inferência era esse prefixo.
8. **Dataset bilíngue atual**: contexto_rush (1.255 pares, linguagem natural,
   A→B) + `ctx_part_2` (mesmos pares INVERTIDOS B→A com tags WD14) = 2.510
   pares, duas modalidades de prompt. Inversão é válida porque next-scene não
   tem direção privilegiada.
9. **adaln decifrado**: o v2 s1950 (o melhor de todos segundo o usuário) TREINOU
   adaln (bug do filtro de exclusão) mas o workflow de inferência usava
   **skip_adaln ativo** → adaln funciona como ABSORVEDOR DE ERRO no treino
   (o gradiente de ajuste global escoa para ele, deixando self_attn/mlp limpos)
   e é descartado ao gerar. Não é veneno nem tempero: é para-raios.
10. **Ambiguidade final**: o workflow de abril tinha `ref_first` DESLIGADO no
    node — ou o adapter foi treinado target-first (config pessoal), ou inferia
    descasado e mesmo assim era bom (plausível pelo RoPE relativo). Sem o
    arquivo original (`anima_ic_lora_v2_next_scene_v2_s1950_r64.safetensors`,
    só no PC local do usuário), não dá para cravar.

## O tipo `raiz_iclora` (registrado e treinável)

`models/ic_lora_dual.py::RaizICLoraPipeline` · type `raiz_iclora` ·
config `examples/raiz_iclora_2000.toml`:

- ref_first [ref T=0 | alvo T=1] (toggle via `[ic_lora_full] ref_first`)
- LoRA rank 64 em self_attn + mlp + **adaln** + llm_adapter (SEM cross_attn —
  o filtro remove cross_attn inclusive dentro do llm_adapter, como abril)
- shifted logit-normal (LTX-2), sigmoid_scale 1.3, condition_dropout 0.1
- batch efetivo 8, lr 1e-4
- **INFERÊNCIA: adaln SEMPRE descartado** — runner `--skip_adaln`, node ComfyUI
  descarta automaticamente (log "skip_adaln: N linears descartados")

Probe treinado no dataset bilíngue até step 750 (cancelado pelo usuário para
outros trabalhos): checkpoints locais em `/workspace/checkpoints/anima_raiz_iclora/`
e no HF `groundedsecret/anima_raiz/raiz_step{250,500,750}`; no ComfyUI como
`ctxrush_anima_raiz_s{500,750}` (mode auto → ic_lora_v2; para simular o
workflow de abril com ref_first OFF, usar mode `omini_subject`).

## Estado do ComfyUI (node `CtxRush - Anima Next-Scene`)

- 6 modos (auto resolve pelo nome do arquivo) + skip_adaln implícito
- Guidance 3-branch (`ref_cfg`/`expected_cfg` ou Dual Guider)
- Dials: `appearance_strength`, `cross_attn_strength`, `llm_adapter_strength`,
  `block_range`
- llm_adapter aplicado via object patch no `preprocess_text_embeds` (roda antes
  do forward — sem isso o canal do eureka não age)

## Experimentos em aberto (quando voltar a isso)

1. Gêmeo target-first do raiz (`ref_first = false`) — replica o workflow literal.
2. Testar o mesmo adapter raiz nas duas ordens de inferência (barato, responde
   a ambiguidade nº 10 na prática).
3. Conseguir o `v2_s1950_r64.safetensors` original do PC do usuário → forense
   de keys + comparação pixel a pixel encerra a saga de vez.
4. Treino longo do braço campeão (raiz ou v3) com prefixos de verbo no dataset
   ("create the next scene, ..."), batch 8, 1–2 épocas.

## Armadilhas catalogadas (não repetir)

- `llm_adapter_lr = 0` congela o LoRA do llm_adapter silenciosamente.
- Merge de LoRA sobre base fp8-scaled afoga deltas jovens (Krea 2).
- CFG de 2 branches com ref só no positivo multiplica a ref pelo CFG do texto.
- Runner: referência precisa de crop-fit + [-1,1] (igual treino).
- pgrep/pkill com padrão que casa o próprio comando mata a própria shell.
- Smoke de 1 imagem SEMPRE antes de batch de gerações.
- Datasets baixados são regeneráveis do HF; caches idem (--regenerate_cache).
