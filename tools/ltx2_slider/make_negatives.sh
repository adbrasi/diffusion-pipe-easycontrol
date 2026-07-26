#!/bin/bash
# Constrói o lado NEGATIVO do par para o slider de timing.
#
# A IDEIA: os vídeos do dataset ANIMATEKA já são exatamente o alvo — animação
# limitada de verdade (medido: ~44% de frames DUPLICADOS quase exatos, diff
# mediana entre vizinhos 0.00021). O negativo é o MESMO vídeo com a estrutura
# de frames segurados destruída:
#
#   mpdecimate    remove as duplicatas -> só os desenhos distintos sobram
#   minterpolate  reconstrói 24fps por interpolação com compensação de
#                 movimento -> todo frame passa a ser único e o movimento
#                 vira contínuo, que é o defeito do LTX
#
# MEDIDO em c_000: duplicatas 44.3% -> 0.0%, diff mediana 0.00021 -> 0.00256
# (12x). E o ALINHAMENTO temporal se preserva: orig[i] vs neg[i] difere 0.0003,
# MENOS que a diferença entre frames vizinhos do original (0.0022). Ou seja o
# par mostra o mesmo instante em cada índice e difere só na textura temporal —
# exatamente o que um slider precisa para aprender um eixo e não um conteúdo.
#
# A contagem de frames encolhe (mpdecimate corta o fim), então os dois datasets
# usam o MESMO target_frames com frame_extraction="head" — o cache trunca
# ambos para o mesmo comprimento e os shapes casam.
set -uo pipefail

SRC="${SRC:-/workspace/datasets/animateka/videos}"
DST="${DST:-/workspace/datasets/animateka_neg/videos}"
MIN_FRAMES="${MIN_FRAMES:-57}"   # 8k+1; descarta clipes curtos demais p/ o par
mkdir -p "$DST"

total=0; feitos=0; curtos=0; falhas=0
for v in "$SRC"/*.mp4; do
  total=$((total+1))
  base=$(basename "$v")
  out="$DST/$base"
  [ -f "$out" ] && { feitos=$((feitos+1)); continue; }

  n=$(ffprobe -v error -select_streams v -show_entries stream=nb_frames -of csv=p=0 "$v" 2>/dev/null)
  [ -z "$n" ] && n=0
  if [ "$n" -lt "$MIN_FRAMES" ]; then curtos=$((curtos+1)); continue; fi

  if ffmpeg -loglevel error -y -i "$v" \
      -vf "mpdecimate,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir" \
      -c:v libx264 -crf 16 -pix_fmt yuv420p -an "$out" 2>/dev/null; then
    # a caption é a MESMA: a direção vem das imagens, não do texto
    cp "${v%.mp4}.txt" "${out%.mp4}.txt" 2>/dev/null
    feitos=$((feitos+1))
  else
    falhas=$((falhas+1)); rm -f "$out"
  fi
  [ $((feitos % 25)) -eq 0 ] && echo "  $feitos processados..."
done

echo "PRONTO: $feitos negativos de $total | curtos descartados: $curtos | falhas: $falhas"
