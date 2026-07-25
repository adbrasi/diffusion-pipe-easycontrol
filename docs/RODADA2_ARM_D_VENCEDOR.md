# Rodada 2 — Arm D VENCE: dropout 0.1 + treino longo (2026-07-25)

**Isto inverte a conclusão anterior da bateria.** Até aqui o arm1 (sem
dropout, ~500 steps) era o campeão e o dropout parecia apenas "proteger
contra degradação". Com o treino longo completo, o quadro mudou.

## O resultado

Métricas no checkpoint final de cada braço (`tools/battery_metrics.py`):

| braço | step | sensibilidade | fidelidade |
|---|---|---|---|
| **armD** (dropout 0.1, 2000 steps) | s2000 | **0.898** | **0.156** |
| armB (dropout 0.1, 1000 steps) | s1000 | 0.641 | 0.148 |
| armA (adaln dentro) | s1000 | 0.606 | 0.129 |
| arm3 (llm_adapter treinável) | s1000 | 0.597 | 0.107 |
| arm1 (campeão anterior) | s1000 | 0.372 | 0.078 |
| armC (routing) | s1000 | 0.342 | 0.041 |

Para comparação, o melhor checkpoint que o arm1 já produziu foi o s500, com
**0.778 / 0.246**. O armD em s2000 tem sensibilidade **maior que o pico
histórico do arm1** (0.898 vs 0.778), com fidelidade sólida.

Sensibilidade alta com fidelidade alta é o par que importa — sensibilidade
alta sozinha poderia ser só instabilidade, mas a fidelidade de 0.156 é a
maior entre todos os checkpoints finais, então não é ruído.

## Validação visual (o que a métrica não mostra)

Grid: `/workspace/outputs/armD_dropout_2000/GRID_s2000.png`

No exemplo held-out ex2 (elfa/calabouço, imagem que NÃO está no dataset):
- `lora 1.0` — elfa e garoto, composição correta
- `lora 1.0 + ref_cfg 1.75` — **a elfa aparece com o laço branco de cabelo**,
  o marcador de identidade específico da referência, e o figurino verde bate
- `ref EMBARALHADA` — cena completamente diferente (azulada, escura, outra
  composição)

Ou seja: fidelidade real quando a referência é a certa, divergência clara
quando é a errada. É exatamente o comportamento que a bateria inteira estava
tentando obter.

## Por que isso faz sentido

`condition_dropout = 0.1` zera os latentes da referência em 10% dos passos.
Consequência: o modelo vê a referência em apenas 90% do treino, então
**aprende mais devagar** — mas aprende uma solução em que a referência é
genuinamente necessária, porque ele nunca pôde assumir que ela estaria lá.

Sem dropout, o modelo converge rápido para a solução fácil (satisfazer a loss
pelo caption) e depois degrada, porque nada o impede de ir abandonando a
referência. Com dropout, essa rota está bloqueada por construção.

Daí a leitura anterior estar incompleta: a 1000 steps o armB ainda não tinha
convergido, e por isso parecia inferior ao arm1 no pico. A 2000 steps ele
passa.

## Receita vencedora atualizada

`examples/round2_2026-07-25/armD_dropout_2000.toml`

```toml
[model]
type = 'ic_lora_v3'          # escopo largo: self_attn+mlp+cross_attn+llm_adapter
llm_adapter_lr = 0           # congelado (Rodada 1)
sigmoid_scale = 1.0

[ic_lora_full]
ref_first = false            # target-first
condition_dropout = 0.1      # <-- a mudança que decide
condition_timestep = 0.0
shifted_logit_normal = false
include_adaln = false        # adaln fora (Arm A perdeu)

[adapter]
rank = 32

[optimizer]
type = 'adamw_optimi'
lr = 1e-4                    # batch efetivo 8
```

**Treinar 2000 steps, não 500.** Essa é a segunda metade do achado — a
receita com dropout precisa de mais steps para convergir, e parar cedo dá a
impressão errada de que ela é pior.

Inferência: `--mode ominicontrol_subject --lora_strength 1.0 --ref_cfg 1.0`
como padrão; `ref_cfg` 1.75 quando quiser puxar mais identidade.

## Ressalvas

1. **n=1.** Um único treino de 2000 steps, uma seed, três exemplos de
   avaliação. O salto é grande (0.372 → 0.898) e é consistente com a teoria,
   mas não foi replicado.
2. **Não sabemos onde é o teto.** 2000 steps foi o limite testado; pode ser
   que 3000 seja melhor, ou que 1500 já bastasse. A curva entre 1000 e 2000
   não foi medida (os checkpoints intermediários existem em
   `/workspace/checkpoints/round2_2026-07-25/armD_dropout_2000/`, então dá
   para levantar essa curva sem treinar de novo).
3. **Este veredito é meu, não do usuário.** Ele julgou o arm1 como "incrível"
   olhando os grids; ainda não viu o armD s2000. A avaliação humana é a que
   vale.

## Próximo passo natural

Levantar a curva do armD nos checkpoints que já existem (500/1000/1500/2000)
para achar onde ele ultrapassa o arm1 e se já saturou — custo zero de GPU
para treinar, só inferência.
