# BETA1-R2: BETA1-R + anti-cópia

O BETA1-R (`docs/BETA1R_RECEITA.md`) ficou bom até o step 750 e passou a devolver a referência a partir do step 1000
(4/48 held-out, 10/18 exemplos do beta1; `ref_bias` não resolve). O R2 mantém **tudo** igual (método, arquitetura,
tokens, RoPE, routing, dataset no disco, node do ComfyUI, inferência) e muda só o treino:

| | BETA1-R | BETA1-R2 | por quê |
|---|---|---|---|
| peso da loss por par | 1 | idêntico 0 (783), quase igual 0.3 (607), parecido 0.7 (1685), resto 1 | nos pares idênticos copiar a referência é o ótimo exato em todo sigma; são eles que ensinam "copie e edite um atributo" |
| referência | latente intacto | zoom-crop aleatório em 50 % das amostras (zoom 1–1.25×, posição uniforme, volta ao mesmo grid por bilinear; sem flip/cor) | copiar pixel a pixel deixa de baixar a loss; o modelo precisa transferir quem/estilo/paleta |
| LR | 1e-4 constante | 1e-4 até o step 500, linear até 2e-5 no step 2000, constante depois | para de aumentar o ganho do laço depois que o modelo já aprendeu |
| eval de cópia | seed 76 | 11 casos × 8 seeds (76, 1–7) a cada 250 steps | cópia depende de caso e seed |

## Implementação

- Pesos: `data/beta1r2/pair_weights.json` (só pesos < 1; chave = nome do arquivo de referência). Gerado de
  `data/beta1r2/pairsim.json` (dHash 8×8 + MSE 64px de todos os 20.112 pares, `tools/beta1r2_pairsim.py`) por
  `tools/build_beta1r2_pair_weights.py`. Faixas: T0 mse<5e-4 ou dHash≤2; T1 dHash 3–5 ou mse<2e-3; T2 dHash 6–10.
  Peso médio 0.915 (sem renormalizar).
- `models/krea2_multiref.py::prepare_inputs`: o peso vira a máscara da loss `(B,1,1,1,1)` (`base.loss_fn` faz
  `loss *= mask`), segue o `split_batch` por amostra. O zoom-crop é feito no latente cacheado (o cache do VAE não muda).
  Nenhum dos dois roda no eval (`timestep_quantile` definido). Config: `[krea2_multiref_grounded] pair_weights_path`,
  `ref_aug_prob`, `ref_aug_max_zoom`.
- O Qwen3-VL continua vendo a referência inteira (semântica, não copia pixel).
- LR: `lr_scheduler = 'hold_linear'` (`utils/lr_schedule.py`, `lr_decay_start/lr_decay_end/lr_end_factor`).
- Testes CPU: `test/test_beta1r2_anticopy.py` (peso 0 → gradiente exatamente zero depois do split_batch; sem pesos =
  BETA1-R; eval não pondera nem aumenta; recorte preserva shape/dtype e não espelha; LR 1e-4 → 7.33e-5 no step 1000 →
  2e-5 no 2000).

## Como rodar

```bash
cd /workspace/dpe-beta1r && setsid nohup /workspace/dpe-venv/bin/python -u tools/beta1r_supervisor.py \
  --config examples/beta1r2/train.toml --cases /workspace/runs/beta1r_samples_inputs/cases.json \
  --name beta1r2 --hf-repo AdwolfCzar/krea2-beta1r2 --copy-seeds 1 2 3 4 5 6 7 \
  > /workspace/runs/logs/beta1r2_supervisor.out 2>&1 &
# parar limpo
touch /workspace/runs/beta1r2_supervisor.stop
```

Saída em `/workspace/runs/beta1r2`, samples em `/workspace/runs/beta1r2_samples/stepN` (`metrics.json` → `multi_seed`),
HF público `AdwolfCzar/krea2-beta1r2` sem README. Retomar: mesmo comando com `--resume`.

## Critério de sucesso

- 0/88 cópias nos steps 1000 e 1250; ≤ 1/88 no 1500.
- Consistência (personagens, estilo, paleta) no nível do BETA1-R step 750 — avaliação visual do usuário.
- Se ainda copiar em 1250: segunda rodada com estados de ruído alto montados a partir da referência ("ref-start", ver a
  proposta anti-cópia). Se a consistência cair: zoom máx. 1.15 ou p 0.3, mantendo os pesos.

## Adendo: correções de mecanismo (pedido do usuário, 02/11/2026)

Além de a–c, o R2 ataca o mecanismo da cópia, como na literatura:

| | técnica | origem | config |
|---|---|---|---|
| d | referência com ruído: `ref = (1-t)·ref + t·ε`, e os tokens da referência recebem `t` no timestep embedding | Captain Cinema, Long Context Tuning (referência limpa "amplifica"; t 1–500/1000 mantém a identidade) | `ref_noise_prob = 0.3`, `t ~ U[0.05, 0.35]` |
| e | descarte de tokens da referência (keys inválidas) | DreamO (descarta 95 % dos tokens no refino para parar a cópia) | `ref_token_drop_prob = 0.2`, mantém `U[5 %, 50 %]` |

As duas técnicas valem só numa fração das amostras. Nas outras, a referência fica limpa (t=0) e inteira, que é
exatamente o que o node do ComfyUI envia. A inferência não muda. Contrato testado em
`test/test_beta1r2_mechanism.py`: com t=0 e todos os tokens, o resultado é bit-idêntico ao caminho de inferência; com
todos os tokens descartados, é idêntico ao reference_dropout. O sorteio é feito em `prepare_inputs`, não na camada,
porque o activation checkpointing recomputa a camada.

Avaliação: 2 seeds extras nos steps 250 e 500, e 8 seeds (11 casos × 8) a partir do 750.

## Resultado: o zoom-crop ACELERA a cópia (ablação, 02/11/2026)

Cópia (dHash(saída, referência) ≤ 10) no step 250, seeds 76/1/2, 8 casos fora do treino + 3 exemplos do beta1:

| run | mudanças sobre o BETA1-R | fora do treino | exemplos beta1 |
|---|---|---|---|
| BETA1-R | — | 0/24 | 0/9 |
| R2 | pesos + zoom-crop + LR + ruído na ref + descarte de tokens | 1/24 | 6/9 |
| R2a | pesos + zoom-crop + LR | 2/24 | 5/9 |
| R2b | só pesos | **0/24** | **0/9** |

- Culpado: o **zoom-crop da referência** (p=0.5, até 1.25×, no latente). Com ele, o R2 copia no step 250 como o BETA1-R
  copiava no step 1000 (saída a 4–10 da referência e 23–30 do alvo). Hipótese: desalinhar a referência ensina a
  transportar o conteúdo dela por semelhança, independente de posição, e é essa habilidade que, forte, vira cópia.
- O ruído na referência e o descarte de tokens não pioram (R2 ≈ R2a); o efeito deles contra a cópia ainda não foi medido
  isolado.
- Referência com ruído também na inferência (t 0.2/0.3, `REF_INFER_NOISE`) não salva o R2 no step 250 (5–6/9).
- R2b segue até 1500 para o teste "só dados": se ficar limpo em 1000/1250, os pares idênticos explicam a cópia do BETA1-R.
