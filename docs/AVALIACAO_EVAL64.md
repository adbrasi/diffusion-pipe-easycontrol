# Avaliação ampliada anti-cópia (eval64)

Recomendação da revisão independente (docs/REVISAO_INDEPENDENTE_2026-11-02.md §2): a unidade estatística é o
**caso**, não a geração. São necessários mais casos, sem contaminação do treino, e métricas contínuas.

## Pool: `/workspace/runs/beta1r_samples_inputs/cases_eval64.json` (imagens em `eval64/`)
- 64 casos com prompt, referência e alvo verdadeiro: os 8 held-out ds2 antigos mais 56 novos.
- Os novos vêm de pares **não usados** no treino do BETA1-R de `AdwolfCzar/proxima_cena_grounded_original_dataset`:
  ds2_poxima_v2 (sobravam 471) e ds4_contexto_curado (sobravam 390). O ds3 foi todo usado.
- Contaminação: os pares não usados de ds2 são, muitas vezes, frames vizinhos dos usados no treino. Ficaram fora os
  pares em que A ou B têm dHash ≤ 8 contra qualquer uma das 40.224 imagens do treino (alvos e refs). Isso derrubou
  ~1/3 dos candidatos. Os 8 held-out antigos ficam com distância mínima de 7–15. `beta1_example1` está no treino
  (`recortados_udvuxwn25f_image_0002`) e não entra.
- Estratos pela similaridade DINOv2 entre referência e alvo: **compatible** (cos ≥ 0.6, mesmo personagem ou cenário,
  propenso à cópia, como o h000263) com 32 casos (24 ds2 + 8 ds4); **divergent** (cos < 0.6) com 32 casos (18 ds2 + 14 ds4).
  Ficam fora os pares quase idênticos (dHash ≤ 10 ou cos ≥ 0.95), em que copiar seria a resposta certa.
- Prompt = legenda de B (`images_B/*.txt`, a mesma usada no treino). Tamanhos: 688x384 (55), 784x336 (7), 576x448 (2).

## Métricas: `tools/beta1r_eval_metrics.py`
- Por geração: dHash e DINOv2 (`facebook/dinov2-small`) contra a referência e contra o alvo, mais a distância de
  paleta Lab contra a referência.
- Cópia: dHash(gen, ref) ≤ 10 **ou** DINO(gen, ref) ≥ 0.95, cada critério reportado também em separado.
- Por caso: o caso copia se a maioria dos seeds copia; reporta também "qualquer seed". IC 95 % por bootstrap sobre
  os casos, no total e por estrato.
- Validação (8 held-out, `cases.json`): step750 dá 0/8 casos; step1000 dá 1/8 (h000263, DINO 0.981, dHash 2);
  step1000 sem ref no 1º passo dá 0/8. Nas gerações do skip/r1000_k0, as que copiam têm DINO ≥ 0.957 e as que não
  copiam, ≤ 0.768: o limiar 0.95 separa bem.

## Uso
```bash
# avulso
python tools/beta1r_eval_metrics.py --samples <dir de samples> --cases /workspace/runs/beta1r_samples_inputs/cases_eval64.json
# no supervisor: depois dos samples normais, gera eval64 com 2 seeds e calcula as métricas
python -u tools/beta1r_supervisor.py ... --eval-cases /workspace/runs/beta1r_samples_inputs/cases_eval64.json
```
Saída em `<samples>/stepN/eval64/` (`metrics_eval.json`) e uma linha `eval stepN: ...` no log do supervisor.
Sobe ao HF junto com os samples. Custo estimado: ~12 min de GPU por checkpoint (128 gerações).
