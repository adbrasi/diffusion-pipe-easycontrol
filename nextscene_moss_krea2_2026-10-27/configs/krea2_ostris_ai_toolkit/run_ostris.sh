#!/bin/bash
until grep -q KREA_HF_OK /workspace/krea2/download2.log 2>/dev/null; do sleep 15; grep -q Traceback /workspace/krea2/download2.log && { echo DL_FAILED; exit 1; }; done
echo "[$(date +%H:%M)] START ostris"
source /workspace/venv/bin/activate; cd /workspace/ai-toolkit
python run.py /workspace/krea2/ostris_edit/krea2_ostris_edit_cap1000.yaml > /workspace/train_ostris.log 2>&1; echo "[$(date +%H:%M)] END ostris exit=$?"
python /workspace/scripts/grids.py krea2_ostris_edit_cap1000 | tail -1
python /workspace/scripts/upload_run.py krea2_ostris_edit_cap1000 krea2_ostris_edit_cap1000 /workspace/krea2/ostris_edit/krea2_ostris_edit_cap1000.yaml
echo OSTRIS_DONE
