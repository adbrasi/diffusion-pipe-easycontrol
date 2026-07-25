# O CFG de referência do Anima — o que o condition_dropout faz de fato

**2026-07-25.** Este doc CORRIGE uma explicação errada que eu (Claude)
commitei em `ed046e7`. O usuário questionou ("a nossa matemática tá mal
definida? o que precisamos corrigir?") e ao verificar a fórmula real
descobri que eu tinha conectado duas coisas verdadeiras que não têm
relação causal entre si.

## A fórmula real (infer_easycontrol.py:347-358)

```
pred = u + text_cfg*(t - u) + ref_cfg*(c - t)

u = f(texto NEGATIVO, ref ZERADA)
t = f(texto POSITIVO, ref ZERADA)    <- branch que fica OOD sem dropout
c = f(texto POSITIVO, ref REAL)
```

E no treino (`models/ic_lora_full.py:143-147`), `condition_dropout` faz
`control_latents[drop_mask] = 0.0` — zera os latents da referência.

**Treino e inferência usam a mesma representação para "sem referência":
latentes zerados.** Isso é fato e continua valendo.

## O que eu errei

Afirmei que isso explicava a diferença de SENSIBILIDADE medida (quanto a
saída muda ao trocar a referência correta pela embaralhada). Não explica.
Fazendo a conta, com mesma seed e mesmo caption:

```
pred(ref correta) - pred(ref embaralhada)
  = [u + text_cfg*(t-u) + ref_cfg*(c_correta - t)]
  - [u + text_cfg*(t-u) + ref_cfg*(c_embaralhada - t)]
  = ref_cfg * (c_correta - c_embaralhada)
```

**O termo `t` cancela.** A sensibilidade depende só de
`c_correta - c_embaralhada`, ou seja: de quanto o modelo realmente
processa o CONTEÚDO da referência no forward normal. Nada a ver com o
branch uncond estar dentro ou fora da distribuição.

## O que continua verdadeiro (e importa)

O branch `t` (ref zerada) **é** OOD para adapters treinados sem dropout, e
isso **é** um problema real — só que afeta outras coisas:

1. **Qualidade absoluta da geração.** `t` entra em `text_cfg*(t-u)` e em
   `-ref_cfg*t`. Se `t` é lixo, ele contamina a predição inteira, com peso
   proporcional a `text_cfg` e `ref_cfg`.
2. **A resposta ao dial `ref_cfg`.** Como `ref_cfg` multiplica `(c - t)`,
   um `t` mal definido torna o dial imprevisível — aumentar `ref_cfg` pode
   não fazer o que se espera.

Previsão falsificável daí: **variar `ref_cfg` deve ter efeito mais
monotônico/previsível no armB (dropout) do que no arm1 (sem dropout)**.
Teste em `tools/refcfg_sweep.sh`.

## Então por que o armB tem mais sensibilidade?

Explicação em aberto — HIPÓTESE, não provada:

Sem dropout a referência está SEMPRE presente, então parte da informação
dela pode ser absorvida como um viés médio/estatística geral do dataset,
em vez de ser lida como conteúdo específico daquela imagem. Com dropout, o
modelo precisa aprender uma função que se comporta diferente conforme a
referência esteja presente ou zerada — o que obriga os pesos a de fato
processar o conteúdo.

Isso precisa de teste próprio. Não afirmar como fato até lá.

## Duas correções possíveis (ambas testáveis)

1. **Treinar com `condition_dropout > 0`** — torna `t` parte da
   distribuição. É o Arm B.
2. **Para adapters já treinados sem dropout: não usar o branch `t`** na
   inferência (CFG de 2 vias, sem o termo `ref_cfg*(c-t)` dependente de
   `t` OOD). Corrige o problema 1 e 2 acima sem retreinar nada.

## Nota de método

Este episódio é um bom lembrete do princípio que a saga toda documenta:
*"o contrato real é o que o código executa, não o que a config/metadata/
intuição dizem"*. Eu tinha uma explicação elegante e coerente, mas não
tinha feito a subtração algébrica. O usuário perguntou, a conta foi feita,
a explicação caiu. Fazer a conta antes de afirmar.
