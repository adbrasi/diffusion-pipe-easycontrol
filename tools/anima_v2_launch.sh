#!/bin/bash
# One-command launch of the Anima A/aligned 1024 v2 campaign.
#   tools/anima_v2_launch.sh <hf_dataset_repo> [backup_repo]
# Downloads the dataset (dsN_*/images_A + images_B/<stem>.txt), reserves held-out pairs,
# materializes training data, sets max_steps to 5 epochs, creates the public backup repo
# and starts the resumable campaign as supervisor service `anima_v2`.
# Stop with save: touch /workspace/nextscene_artifacts/anima_v2/stop_campaign
set -euo pipefail
DATASET=${1:?usage: anima_v2_launch.sh <hf_dataset_repo> [backup_repo]}
REPO=${2:-AdwolfCzar/anima-nextscene-a-aligned-1024-v2}
ROOT=/workspace/diffusion-pipe-easycontrol
CFG=$ROOT/examples/anima_nextscene/gpu_20261008_1024_v2/A_full.toml
ART=/workspace/nextscene_artifacts/anima_v2
PY=/venv/main/bin/python
mkdir -p "$ART"

$PY -I -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$DATASET', repo_type='dataset', local_dir='/workspace/ds_v2', max_workers=16))"

cd "$ROOT/tools"
$PY anima_v2_prepare.py --source /workspace/ds_v2 --destination /workspace/anima_v2_data \
    --heldout /workspace/heldout_v2 --report "$ART/data/dataset_report.json" --per-subset 6 --epochs 5

STEPS=$($PY -c "import json;print(json.load(open('$ART/data/dataset_summary.json'))['estimated_max_steps'])")
sed -i "s/^max_steps = .*/max_steps = $STEPS/" "$CFG"
echo "max_steps = $STEPS (5 epochs)"

$PY -I -c "
from huggingface_hub import HfApi
api = HfApi(); api.create_repo('$REPO', private=False, exist_ok=True)
assert not api.repo_info('$REPO').private"

sudo tee /opt/supervisor-scripts/anima_v2.sh > /dev/null <<EOF
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
cd $ROOT
exec $PY -u tools/anima1024_campaign.py --config $CFG --artifacts $ART --repo $REPO --interval 500 --heldout /workspace/heldout_v2
EOF
sudo chmod +x /opt/supervisor-scripts/anima_v2.sh
sudo tee /etc/supervisor/conf.d/anima_v2.conf > /dev/null <<EOF
[program:anima_v2]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/anima_v2.sh
autostart=true
autorestart=unexpected
startretries=3
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
stopwaitsecs=3600
EOF
sudo supervisorctl reread && sudo supervisorctl update
sudo supervisorctl status anima_v2
echo "Logs: /var/log/portal/anima_v2.log  |  state: $ART/campaign_state.json"
