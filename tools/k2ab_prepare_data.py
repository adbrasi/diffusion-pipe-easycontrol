"""Build the identical, seeded SFW pair subset for the two GPU probes."""
import collections
import functools
import json
import os
from pathlib import Path
import random
import re

os.environ['ONNX_MODE'] = 'cpu'
import onnxruntime as ort
import imgutils.utils.onnxruntime as img_ort


@functools.lru_cache()
def cpu_model(checkpoint, provider, use_cpu=True):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(checkpoint, options, providers=['CPUExecutionProvider'])


img_ort._open_onnx_model = cpu_model
from imgutils.validate import anime_rating

ROOT = Path('/workspace/k2ab')
DENY = re.compile(r'\b(nude|naked|nudity|topless|bare breasts?|nipples?|genitals?|penis|vagina|sexual|sex|intercourse|penetration|masturbat\w*|orgasm\w*|cum|ejaculat\w*|lingerie|underwear|panties|pussy|porn\w*)\b', re.I)


def safe(path):
    label, confidence = anime_rating(str(path))
    return label == 'safe' and confidence >= .7


def main():
    rng = random.Random(42)
    candidates = []
    for subset in sorted(Path('/workspace/ns_E2').glob('ds*')):
        captions = json.loads((subset / 'target/captions.json').read_text())
        for target in sorted((subset / 'target').iterdir()):
            if target.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                continue
            texts = captions.get(target.name, [])
            if isinstance(texts, str):
                texts = [texts]
            if not texts or any(DENY.search(text) for text in texts):
                continue
            control = subset / 'control' / target.name
            candidates.append((subset.name, target, control, texts))
    rng.shuffle(candidates)
    accepted = []
    for index, (subset, target, control, texts) in enumerate(candidates):
        if safe(target) and safe(control):
            accepted.append(dict(subset=subset, stem=target.stem, target=str(target),
                                 control=str(control), caption=rng.choice(texts)))
        if index % 100 == 0:
            print('Rated', index, 'accepted', len(accepted), flush=True)
        if len(accepted) == 1500:
            break
    if len(accepted) < 1500:
        raise RuntimeError(f'Only {len(accepted)} SFW pairs available')
    for name in ('target', 'control', 'refs'):
        (ROOT / 'data' / name).mkdir(parents=True, exist_ok=True)
    captions = {}
    for row in accepted:
        for field in ('target', 'control'):
            src = Path(row[field])
            dst = ROOT / 'data' / field / src.name
            if not dst.exists():
                os.link(src, dst)
        src = Path(row['control'])
        ref = ROOT / 'data/refs' / (src.stem + '_1' + src.suffix)
        if not ref.exists():
            ref.symlink_to(ROOT / 'data/control' / src.name)
        captions[Path(row['target']).name] = [row['caption']]
    (ROOT / 'data/target/captions.json').write_text(json.dumps(captions, indent=2))
    manifest = dict(seed=42, count=len(accepted), sfw=len(accepted), r18=0,
                    subsets=dict(collections.Counter(r['subset'] for r in accepted)), pairs=accepted)
    (ROOT / 'artifacts/data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DONE', manifest['subsets'], flush=True)


if __name__ == '__main__':
    main()
