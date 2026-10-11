# Upload every new stepN adapter of a run dir to HF (verified). Usage: ckpt_uploader.sh <output_dir> <hf_prefix>
source /workspace/venv/bin/activate; cd /workspace/dpk
OUT=$1; PREFIX=$2; DONE=/workspace/logs/uploaded_$(echo $PREFIX | tr / _).txt; touch $DONE
while true; do
  for d in $(ls -d $OUT/*/step* 2>/dev/null); do
    grep -qx "$d" $DONE && continue
    [ -f $d/adapter_model.safetensors ] || continue
    sleep 20  # let the save finish
    python tools/k2v3_upload.py --message "$(basename $d)" "$d:$PREFIX/$(basename $d)" >> /workspace/logs/uploader.log 2>&1 && echo "$d" >> $DONE
  done
  sleep 60
done
