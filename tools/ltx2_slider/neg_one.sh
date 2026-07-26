#!/bin/bash
# Converte UM vídeo para a versão negativa (frames segurados destruídos).
v="$1"; DST=/workspace/datasets/animateka_neg/videos; MIN=57
base=$(basename "$v"); out="$DST/$base"
[ -f "$out" ] && exit 0
n=$(ffprobe -v error -select_streams v -show_entries stream=nb_frames -of csv=p=0 "$v" 2>/dev/null)
[ -z "$n" ] && exit 0
[ "$n" -lt "$MIN" ] && exit 0
ffmpeg -loglevel error -y -threads 2 -i "$v" \
  -vf "mpdecimate,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir" \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -an "$out" 2>/dev/null \
  && cp "${v%.mp4}.txt" "${out%.mp4}.txt" 2>/dev/null || rm -f "$out"
