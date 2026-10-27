#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/ai-toolkit
while pgrep -f "pipeline_v2.sh" >/dev/null; do sleep 20; done
echo "[$(date +%H:%M)] START moss_nextscene_v3_big"
python run.py /workspace/scripts/moss_nextscene_v3_big.yaml > /workspace/train_v3.log 2>&1; echo "[$(date +%H:%M)] END moss v3 exit=$?"
python /workspace/scripts/grids.py moss_nextscene_v3_big >/dev/null 2>&1
python /workspace/scripts/upload_run.py moss_nextscene_v3_big v3_big /workspace/scripts/moss_nextscene_v3_big.yaml || echo UPLOAD_FAIL_v3
echo "[$(date +%H:%M)] START krea2_ostris_edit_ns_v1"
python run.py /workspace/krea2/ostris_edit/krea2_ostris_edit_raw_768.yaml > /workspace/train_krea_a.log 2>&1; echo "[$(date +%H:%M)] END krea A exit=$?"
python /workspace/scripts/grids.py krea2_ostris_edit_ns_v1 >/dev/null 2>&1
python /workspace/scripts/upload_run.py krea2_ostris_edit_ns_v1 krea2_ostris_edit /workspace/krea2/ostris_edit/krea2_ostris_edit_raw_768.yaml || echo UPLOAD_FAIL_A
echo "[$(date +%H:%M)] START krea2edit_ns_v1"
python run.py /workspace/krea2/krea2edit/krea2edit_ns_raw_768.yaml > /workspace/train_krea_b.log 2>&1; echo "[$(date +%H:%M)] END krea B exit=$?"
python /workspace/scripts/upload_run.py krea2edit_ns_v1 krea2edit /workspace/krea2/krea2edit/krea2edit_ns_raw_768.yaml || echo UPLOAD_FAIL_B
echo QUEUE_DONE
