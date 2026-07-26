#!/bin/bash
# Adiciona trilha de audio SILENCIOSA aos negativos.
#
# POR QUE: gerei os negativos com -an (sem audio), mas o loader do
# ltx2_cache_latents usa PyAV e acessa o stream de audio sem checar se existe
# -> "IndexError: tuple index out of range". Os positivos do ANIMATEKA todos
# tem stream de audio (parte deles e silencio digital, mas o STREAM existe),
# entao o caminho positivo passou e o negativo quebrou.
# Remuxa: copia o video como esta (sem re-encodar) e anexa aac silencioso.
v="$1"
tmp="${v%.mp4}_tmp.mp4"
ffmpeg -loglevel error -y -i "$v" -f lavfi -i anullsrc=r=44100:cl=stereo \
  -c:v copy -c:a aac -b:a 32k -shortest "$tmp" 2>/dev/null \
  && mv "$tmp" "$v" || rm -f "$tmp"
