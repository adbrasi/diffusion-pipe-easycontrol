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

AVISO_GB="${AVISO_GB:-25}"
CRITICO_GB="${CRITICO_GB:-8}"
INTERVALO="${INTERVALO:-120}"
estado=ok

while true; do
  livre=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  cache=$(du -sm /workspace/datasets/*/target/cache 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')

  if [ "$livre" -le "$CRITICO_GB" ]; then
    echo "CRITICO: ${livre}GB livres (cache ${cache:-0}GB) — matando o treino para nao corromper checkpoint"
    pkill -f "train.py --local_rank"
    exit 1
  elif [ "$livre" -le "$AVISO_GB" ] && [ "$estado" = ok ]; then
    echo "ALERTA: disco em ${livre}GB livres (cache ${cache:-0}GB). Critico em ${CRITICO_GB}GB."
    estado=avisado
  elif [ "$livre" -gt "$AVISO_GB" ] && [ "$estado" = avisado ]; then
    echo "disco normalizou: ${livre}GB livres"
    estado=ok
  fi

  # fim natural: o treino acabou
  if ! pgrep -f "train.py --local_rank" >/dev/null; then
    echo "treino terminou — guarda de disco encerrando (${livre}GB livres)"
    exit 0
  fi
  sleep "$INTERVALO"
done
