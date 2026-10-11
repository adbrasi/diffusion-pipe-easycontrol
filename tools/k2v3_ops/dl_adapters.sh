source /venv/main/bin/activate
cd /workspace/adapters
hf download AdwolfCzar/krea2-turbo-1024 --include "checkpoints/*/step500/*" "checkpoints/*/step1500/*" "checkpoints/*/step3000/*" "evidence/train_until_*.log" --local-dir t1024
hf download AdwolfCzar/krea2-probe512-turbo --include "checkpoints/*/step250/*" "checkpoints/*/step1000/*" "evidence/train_until_*.log" --local-dir p512
hf download AdwolfCzar/krea2-ab-runs --include "checkpoints/A_native_fp8_512_micro2_probe/20260930_20-41-00/step1000/*" --local-dir a1000
echo DONE_ADAPTERS
