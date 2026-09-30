# Anima nextscene — revisão de 250, 500 e 750 steps

Este pacote contém as imagens necessárias para uma revisão **somente pelo GitHub**.
Não exige acesso à instância GPU nem ao HF privado. São **dois treinos**, `aligned`
e `disjoint_w`, com checkpoints de 250, 500 e 750 steps. Cada checkpoint posterior
continua o mesmo treino; não são seis treinos diferentes.

## Abra estes seis grids

Todos têm **três colunas: A usada | B real | Resultado**. As linhas vêm em pares:
referência original, depois referência trocada (**shuffle**). No shuffle, a coluna
A mostra a imagem realmente fornecida ao modelo. O nome acima de cada linha inclui
layout, step, condição, par e nome da referência utilizada.

| Checkpoint | aligned — treino A | disjoint_w — treino B |
|---|---|---|
| 250 steps | [aligned_step0250.png](aligned_step0250.png) | [disjoint_w_step0250.png](disjoint_w_step0250.png) |
| 500 steps | [aligned_step0500.png](aligned_step0500.png) | [disjoint_w_step0500.png](disjoint_w_step0500.png) |
| 750 steps | [aligned_step0750.png](aligned_step0750.png) | [disjoint_w_step0750.png](disjoint_w_step0750.png) |

Há seis pares visuais, os mesmos nos seis grids, e 72 resultados salvos ao todo.
As imagens nos grids foram reduzidas para 320px por célula; nenhuma geração nova
foi feita para este pacote. As imagens originais da avaliação têm 512×512.

## Quais pares estão aqui

Esta é uma seleção ilustrativa, **não um ranking calculado sobre estes seis casos**.
Inclui os quatro subsets, situações de progresso e falhas claras de identidade:

| Par / stem | O que inspecionar |
|---|---|
| `00_ds1_bj579mxzos_image_0006` | Close de olho → perfil: conserva as características visíveis de A ou produz um perfil genérico? O chapéu de B não está visível em A. |
| `02_ds3_006895` | Página de comic com armadura e monstro: mudança de ação, estilo e organização dos painéis. |
| `03_ds4_imagem000268` | Grupo atrás da grade: figurino, luz noturna, ambiente e alteração de framing. |
| `05_ds2_004723` | Sala de aula 3D e roupa listrada: há preservação de estilo/roupa/ambiente com nova pose? |
| `06_ds3_005003` | Página de manga: personagens, corredor, interação e legibilidade estrutural. |
| `09_ds2_000248` | Mecha vermelho: falha recorrente de identidade/paleta, frequentemente vira azul. |

O shuffle usa o próximo par **na lista completa de 12**, não o próximo entre estes
seis selecionados. Por isso algumas referências trocadas vêm de pares que não
possuem uma linha original neste pacote. O nome da referência está impresso na linha.

## Configuração e dados

Os dois braços têm o mesmo dataset de probe, seed42 de treino, rank64, lr5e-5,
batch efetivo4, warmup100, ref_dropout0,1, cross-attention treinável,
diff-weighted loss e high_noise_prob0,2. A variável comparada é o layout RoPE:
`aligned` versus `disjoint_w`. Configurações completas:
[examples/anima_nextscene/gpu_20260930](../../../examples/anima_nextscene/gpu_20260930/).

Avaliação: 12 pares heldout, prompts curtos, seed76, 512×512, 20 passos de sampling,
CFG4, ref_cfg1, LoRA strength1. **Steps de treino e passos de sampling são coisas
separadas.** A/B são recortadas como na avaliação quadrada; isso corta os frames
largos e precisa ser considerado ao julgar composição e páginas de manga.

Probe: 512 pares-base por subset (2.048 ao todo), ds4 repetido2, com captions
completas e curtas quando o extrator conservador consegue produzir uma instrução.
Heldout excluído antes do cache; títulos/near-duplicates/alinhamento ruim filtrados.
Para origem, limitações de split e detalhes, leia o run log.

## Métricas — todos os 12 pares, não apenas os seis dos grids

| Checkpoint | GT_true ↑ | ref_gain ↑ | null_gain ↑ | copy_gap | copy_rate | CCIP |
|---|---:|---:|---:|---:|---:|---:|
| aligned250 | 0,5099 | 0,0263 | 0,0265 | −0,1497 | 0 | 0,4167 |
| aligned500 | 0,5678 | 0,0677 | 0,0798 | −0,1148 | 0 | 0,4167 |
| aligned750 | 0,5832 | 0,0865 | 0,0982 | −0,0902 | 0 | 0,4167 |
| disjoint250 | 0,5191 | 0,0543 | 0,0428 | −0,1235 | 0 | 0,4167 |
| disjoint500 | 0,5331 | 0,0673 | 0,0216 | −0,0944 | 0 | 0,2500 |
| disjoint750 | 0,5446 | 0,0386 | 0,0394 | −0,1020 | 0 | 0,4167 |

- `GT_true`: similaridade DINOv2 do resultado com B.
- `ref_gain`: similaridade com B usando A correta menos usando A trocada.
- `null_gain`: similaridade com B usando A correta menos usando latent de referência zero.
- `copy_gap`: similaridade resultado↔A menos similaridade B↔A; não é uma meta de maximização.
- `copy_rate`: proporção de resultados próximos de A segundo dHash≤6.
- `CCIP`: proxy de identidade em anime. Páginas com vários personagens e estilo3D
  limitam sua interpretação. Não prova identidade sozinho.

Os dados completos, inclusive métricas por par e do controle null, estão em
[metrics_summary.csv](metrics_summary.csv) e [metadata/](metadata/).
[Prompts exatos](prompts.md). [Manifest de correspondências](manifest.json),
com nomes dos outputs fonte e SHA256. Caminhos `/workspace/` nos metadados são
proveniência; para ver as imagens aqui use os seis PNGs deste diretório.

## Estado e perguntas para revisão

Ainda não há sucesso robusto no conjunto. A qualidade/framing e parte do uso de
referência melhoram, mas personagens frequentemente continuam genéricos. O caso3D
da sala de aula mostra uso de referência mais claro; perfil e mecha continuam
falhando. Uma seed e poucos exemplos não estabelecem um vencedor definitivo.
A continuação dos dois treinos até1.000steps está em andamento.

Para revisar o método, comece por:

1. [HANDOFF_AGENTE_GPU.md](../../HANDOFF_AGENTE_GPU.md).
2. [ANIMA_NEXTSCENE_PESQUISA_2026-09.md](../../ANIMA_NEXTSCENE_PESQUISA_2026-09.md).
3. [NEXTSCENE_RUN_LOG.md](../../NEXTSCENE_RUN_LOG.md).
4. Este pacote, o evaluator e as configurações de treino.

Avalie preservação de personagem/estilo/ambiente, mudança de cena solicitada,
sinais de cópia e diferença original↔shuffle. Separe constatações visuais,
evidência quantitativa e hipóteses; proponha um próximo teste curto com uma variável.

Reprodução da exportação nesta instância:

```bash
python tools/nextscene_export_review.py
```
