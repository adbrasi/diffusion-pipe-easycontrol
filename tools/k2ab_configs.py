"""Create reviewable runtime smoke/probe configs from the checked-in recipes."""
import json
import math
import os
from pathlib import Path

from PIL import Image
import toml

ROOT = Path('/workspace/k2ab')
REPO = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'artifacts/data_manifest.json').read_text())
buckets = [2 ** (-1 + index / 3) for index in range(7)]
groups = {bucket: [] for bucket in buckets}
for row in manifest['pairs']:
    with Image.open(row['target']) as image:
        ar = image.width / image.height
    bucket = min(buckets, key=lambda value: abs(math.log(ar / value)))
    groups[bucket].append(row)
smoke = [row for rows in groups.values() for row in rows[:4]]
for name in ('target', 'control', 'refs'):
    (ROOT / 'smoke' / name).mkdir(parents=True, exist_ok=True)
captions = {}
for row in smoke:
    for field in ('target', 'control'):
        src = Path(row[field])
        dst = ROOT / 'smoke' / field / src.name
        if not dst.exists():
            os.link(src, dst)
    src = Path(row['control'])
    ref = ROOT / 'smoke/refs' / (src.stem + '_1' + src.suffix)
    if not ref.exists():
        ref.symlink_to(ROOT / 'smoke/control' / src.name)
    captions[Path(row['target']).name] = [row['caption']]
(ROOT / 'smoke/target/captions.json').write_text(json.dumps(captions, indent=2))
(ROOT / 'artifacts/smoke_manifest.json').write_text(json.dumps(smoke, indent=2))
config_dir = ROOT / 'artifacts/configs'
config_dir.mkdir(exist_ok=True)
for arm in ('A_native', 'B_beta1_fixed'):
    for mode in ('smoke', 'probe'):
        config = toml.load(REPO / f'examples/krea2_ab/{arm}.toml')
        dataset = toml.load(REPO / f'examples/krea2_ab/dataset_{arm}.toml')
        if mode == 'smoke':
            directory = dataset['directory'][0]
            directory['path'] = str(ROOT / 'smoke/target')
            directory['control_path'] = str(ROOT / ('smoke/control' if arm.startswith('A') else 'smoke/refs'))
            config.update(max_steps=10, save_every_n_steps=5, checkpoint_every_n_steps=5,
                          warmup_steps=5)
        config['output_dir'] = str(ROOT / 'checkpoints' / (arm + '_' + mode))
        dataset_path = config_dir / f'{arm}_{mode}_dataset.toml'
        toml.dump(dataset, dataset_path.open('w'))
        config['dataset'] = str(dataset_path)
        toml.dump(config, (config_dir / f'{arm}_{mode}.toml').open('w'))
print('Smoke pairs:', len(smoke), 'buckets:', {str(k):len(v) for k,v in groups.items()})
