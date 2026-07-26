#!/bin/bash
# Guarda de disco para os treinos que cacheiam embeddings de texto.
#
# POR QUE EXISTE: o cache do canal grounded custa ~29 MB por amostra (12
# camadas x 2560 x 2 bytes por token, dobrado pelo caption_dropout, que
# cacheia também a versão incondicional). Um treino de 20k amostras projeta
# ~690 GB e enche o disco no meio do cache — já aconteceu neste projeto
# (incidente de 2026-07-19, commit f94067e) e quase aconteceu de novo aqui.
#
# Emite uma linha por checagem SÓ quando cruza um limiar, para não virar
# ruído. Em CRÍTICO, mata o treino — é melhor perder o cache do que o
# filesystem, porque disco cheio corrompe checkpoint em escrita.
set -uo pipefail

AVISO_GB="${AVISO_GB:-30}"
CRITICO_GB="${CRITICO_GB:-12}"
INTERVALO="${INTERVALO:-30}"
estado=ok

# NUNCA medir o cache com `du` aqui. Um cache de 110 GB tem milhoes de
# arquivos e o `du` leva minutos varrendo — o loop fica PRESO na medicao e
# nao chega a checar o `df`. Foi exatamente assim que esta guarda falhou em
# 2026-07-26: alertou em 16 GB e depois nunca mais rodou, enquanto o disco
# caia para 5 GB. `df` e O(1); `du` e O(arquivos). So `df` no caminho quente.
while true; do
  livre=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')

  if [ "$livre" -le "$CRITICO_GB" ]; then
    echo "CRITICO: ${livre}GB livres — matando o treino para nao corromper checkpoint"
    pkill -9 -f 'python.*train.py --local_rank'
    pkill -9 -f 'deepspeed --num_gpus'
    exit 1
  elif [ "$livre" -le "$AVISO_GB" ] && [ "$estado" = ok ]; then
    echo "ALERTA: disco em ${livre}GB livres. Critico em ${CRITICO_GB}GB."
    estado=avisado
  elif [ "$livre" -gt "$AVISO_GB" ] && [ "$estado" = avisado ]; then
    echo "disco normalizou: ${livre}GB livres"
    estado=ok
  fi

  # fim natural: o treino acabou
  if ! pgrep -f 'python.*train.py --local_rank' >/dev/null; then
    echo "treino terminou — guarda de disco encerrando (${livre}GB livres)"
    exit 0
  fi
  sleep "$INTERVALO"
done
