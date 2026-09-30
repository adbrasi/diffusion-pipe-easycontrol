# E2 — uma época do dataset completo em cada layout

Dois treinos novos, a partir do Anima base: `E2_A.toml` (`aligned`) e
`E2_B.toml` (`disjoint_w`). Nenhum adapter ou optimizer de E1 é retomado.
Os TOMLs diferem somente em `output_dir` e `nextscene.rope_layout`.

Receita compartilhada:

- `diff_weight=false`, LR `1e-4`, rank64, warmup100 e seed42.
- `ref_dropout=0.1`, `high_noise_prob=0.2`, cross-attention treinável.
- micro2×accum2, activation checkpointing habilitado.
- 512pixelbudget,7buckets de aspecto, uma época sem `max_steps`.
- Caption `[full, short, short]` quando há curta; `[full]` nos demais.
- Curtas heurísticas com `The same <subject>`; não são anotações de um VLM.

Dados:8.761pares-base filtrados, ds4repetido2,9.769exposições de captions full
+12.988de short=22.757amostras antes do arredondamento de buckets. A proporção
global de shorts passa de~40% (E1 full/short) a~57%; dentro de cada par com
short, são2/3. O loader informa o número exato de steps por época ao iniciar.

**Diferença importante:** E1 treinou um probe de2.048pares com4.183amostras
por época. Seus1.000steps correspondem a quase uma época desse recorte.
E2 usa o corpus completo; não confundir épocas do probe com épocas do corpus.
A comparação E1→E2 muda cobertura e receita em conjunto.

Smokes:28pares cobrindo os7buckets; os dois layouts passaram14steps de uma
época com checkpointing e adapters finitos/contrato válido. Micro2×accum2 sem
checkpointing deuOOM nos dois, portanto não é usado. Sampling comtargetAR
passou antes de iniciar a campanha completa.

Execução supervisionada: `tools/nextscene_gpu_ops/e2_campaign.py` faz A→B
sem sampling entre os treinos. Salva adapters a cada1.000steps e no fim
(`epoch1`), com estados DeepSpeed de retomada a cada10min. Interrompe as
etapas dependentes se uma etapa falhar; em recuperação retoma somente E2.

Depois dos dois treinos, avalia1000/2000/3000/5000/epoch1 em24heldouts,
seeds76/142,20passos de sampling,512pixelbudget, `--match-target-ar`,
CFG4/ref_cfg1/LoRA1, com referência correta, trocada e zero.

Logs, métricas, grids e outputs individuais:
`/workspace/nextscene_artifacts/E2/`, com sync contínuo para o HF privado.
`campaign_state.json` registra fases e tempos; `summary_all_seeds.csv`
reúne as avaliações. O run log recebe milestones/custo estimado e push.

O critério continua visual: herança de personagem, paleta e ambiente,
composição nova, e efeito do shuffle na direção da referência trocada.
Métricas não escolhem o vencedor automaticamente. E1 usa12pares/cropquadrado;
E2 usa24pares/targetAR, portanto médias cruas entre protocolos não são
uma ablação isolada da receita.
