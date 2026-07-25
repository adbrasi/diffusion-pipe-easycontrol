# Anima — método campeão até 2026-07-25 (bateria pós-bug do llm_adapter)

**Veredito do usuário:** "incrivelmente bom... em 750 steps já tá fazendo mágica."
**Status:** líder provisório da Rodada 1 (llm_adapter congelado vs treinável).
Ainda falta comparar com Arm 3 antes de declarar vencedor definitivo — ver
`docs/BATERIA_2026-07-25_LLM_ADAPTER.md` para o contexto completo da bateria.

---

## 1. O método exato

`models/ic_lora_dual.py::ICLoraV3Pipeline` (`type = 'ic_lora_v3'`) —
IC-LoRA de escopo LARGO: LoRA em self_attn + mlp + cross_attn + llm_adapter,
adaln FORA (double-LoRA instável, ver `docs/BATERIA_...md` / memória do
projeto para a prova matemática). Pacote target-first
`[alvo T=0 | ref T=1]`.

Config completa: `examples/battery_2026-07-25/arm1_broad_llm_frozen.toml`.

```toml
[model]
type = 'ic_lora_v3'
transformer_path = '.../anima-base-v1.0.safetensors'
vae_path = '.../qwen_image_vae.safetensors'
llm_path = '.../qwen_3_06b_base.safetensors'
dtype = 'bfloat16'
sigmoid_scale = 1.0
llm_adapter_lr = 0        # <- ESTE braço: congelado

[ic_lora_full]
ref_first = false          # target-first
condition_dropout = 0.0
condition_timestep = 0.0
shifted_logit_normal = false
include_adaln = false

[adapter]
type = 'lora'
rank = 32

[optimizer]
type = 'adamw_optimi'
lr = 1e-4
```

Batch efetivo 8 (micro 1 × accum 8), 1000 steps totais, checkpoints a cada
250. Dataset: `contexto_rush` (1255 pares, ver bateria).

## 2. Protocolo de avaliação (o que prova que não é só caption decorado)

Grid por checkpoint: 3 exemplos fixos (1 do dataset — Demon Slayer — + 2
held-out curados manualmente pelo usuário, **genuinamente fora do
treino**) × 6 colunas: referência, alvo real, sem ref, ref força 1.0, ref
força 1.5, **ref EMBARALHADA** (mesmo caption, referência de outro
exemplo). Resolução 912×512 / 784×592, thumbnail 460×258 no grid — a v1
(688×384, thumbnail 260×146) era pequena demais pra julgar rosto/detalhe.

**A coluna decisiva é "ref embaralhada".** Se a saída não muda ao trocar a
referência por uma errada, é atalho de caption (memorização). Se muda —
refletindo a referência ERRADA —, é uso genuíno da referência.

## 3. Resultado observado (checkpoints 250/500/750)

- **Exemplo do dataset (Demon Slayer)**: saída praticamente invariante à
  troca de referência em todos os checkpoints — decorado (o modelo já viu
  esse par de treino ~4-6× a esses steps). Esperado, não é o teste que
  importa.
- **Exemplos held-out (elf/dungeon; floresta noturna)**: a referência
  correta bate elementos específicos que "sem ref" não captura (a lua
  crescente aparecendo em ex3 no step500; a pose de abraço mais fiel em
  ex2). **A referência EMBARALHADA muda visivelmente cor/luz/atmosfera**
  da geração em todos os checkpoints testados, puxando pra paleta da
  referência errada — evidência real, não suposta, de que a referência
  está sendo usada, mesmo em exemplos nunca vistos no treino.
- Até agora a referência parece influenciar majoritariamente **cor/luz/
  atmosfera/alguns props**; ainda não está claro se também transfere
  identidade específica de personagem com a mesma força — acompanhar nos
  checkpoints finais.
- **Observação a monitorar**: no step750, o exemplo held-out ex2 ("sem
  ref") mostrou um artefato de painel duplicado/dividido. Pode ser ruído
  de uma seed específica; não descartar reavaliar com outra seed se
  persistir em steps futuros ou em treinos mais longos.

## 4. Por que isso importa (a lição por trás)

Isso resolve, com evidência direta (não só arqueologia), a tensão
documentada entre a recomendação oficial do Anima ("nunca treine o
llm_adapter", `/workspace/research/ANIMA_TRAINING_GUIDE.md`) e a "eureka"
de julho (treinar o llm_adapter é o segredo). Este braço (Arm 1,
llm_adapter CONGELADO) já mostra uso genuíno de referência em exemplos
held-out — ou seja, **congelar o llm_adapter não impede o método de
funcionar**, ao menos até 750 steps. Falta comparar diretamente com o Arm 3
(llm_adapter treinável) no MESMO protocolo antes de decidir se treinar o
llm_adapter ajuda, atrapalha, ou é indiferente.

## 5. Atualização — Rodada 1 fechada: ARM 1 (llm_adapter CONGELADO) venceu

Arm 3 (llm_adapter treinável, mesmo protocolo, mesmos 1000 steps) terminou.
**Veredito do usuário: Arm 1 é muito melhor.** `llm_adapter_lr = 0`
(congelado) é a receita vencedora da Rodada 1 — bate com a recomendação
oficial do criador do Anima (nunca treinar o llm_adapter), e a "eureka" de
julho (que dizia o contrário) não se sustentou quando testada sem o bug.

Sinal secundário observado em ambos os braços: o efeito da ref embaralhada
(diferença entre referência correta e trocada) **diminui mas não some**
conforme o treino avança — mais forte em 250-500 steps, mais fraco (porém
ainda presente) em 750-1000, e mais pronunciado em alguns exemplos (ex2)
que em outros (ex3). Isso sugere que parte do "atalho de caption" é real e
seria atacado por `condition_dropout > 0` (hoje em 0.0 nesta receita), mas
não invalida o método — o Arm 1 venceu de forma clara na avaliação humana.

## 6. Próximos passos

1. ~~Fechar o checkpoint 1000 do Arm 1.~~ Feito.
2. ~~Rodar o mesmo protocolo no Arm 3.~~ Feito — Arm 1 venceu.
3. Investigar o artefato de painel duplicado (apareceu em ex2 nos dois
   braços, mais cedo no Arm 3) se reaparecer em treinos mais longos.
4. Rodada 2 (a definir com o usuário): candidatos — `condition_dropout > 0`
   (ataca o atalho de caption residual), adaln dentro/fora (réplica exata
   de abril via `raiz_iclora`), ref_first vs target-first, routing
   condition-only (`ic_lora_dual`) vs global — mantendo `llm_adapter_lr=0`
   fixo como a receita vencedora desta rodada.
