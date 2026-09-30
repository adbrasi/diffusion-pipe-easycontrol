#!/usr/bin/env python3
"""Export a small, self-contained visual review into the Git repository."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

from nextscene_review_grid import find_image, render, save_png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluations', type=Path, default=Path('/workspace/nextscene_artifacts/E1'))
    ap.add_argument('--out', type=Path, default=Path('docs/nextscene_results/2026-09-30'))
    ap.add_argument('--steps', nargs='+', type=int, default=[250, 500, 750])
    ap.add_argument('--indices', nargs='+', type=int, default=[0, 2, 3, 5, 6, 9])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.out / 'metadata'
    metadata_dir.mkdir(exist_ok=True)
    summaries, manifest, captions, stems = [], [], {}, []
    for step in args.steps:
        for arm, layout in [('A', 'aligned'), ('B', 'disjoint_w')]:
            name = f'{arm}{step}'
            paths = list((args.evaluations / name).glob('*/metrics.json'))
            if len(paths) != 1:
                raise ValueError(f'{name}: need exactly one completed evaluation')
            path = paths[0]
            metrics = json.loads(path.read_text())
            config = json.loads((path.parent / 'eval_config.json').read_text())
            prompts = json.loads((path.parent / 'prompts.json').read_text())
            pairs = metrics['pairs']
            order = [r['stem'] for r in pairs]
            if stems and order != stems:
                raise ValueError(f'{name}: inconsistent held-out order')
            stems = order
            file_stem = f'{layout}_step{step:04d}'
            rows = []
            for i in args.indices:
                stem = pairs[i]['stem']
                if stem in captions and captions[stem] != prompts[stem]:
                    raise ValueError(f'{name}: prompt changed for {stem}')
                captions[stem] = prompts[stem]
                for condition in ['true', 'shuffled']:
                    reference_stem = stem if condition == 'true' else pairs[(i + 1) % len(pairs)]['stem']
                    output = path.parent / f'{stem}_{condition}.png'
                    row = dict(run=name, step=step, layout=layout, stem=stem,
                               pair_number=i + 1, condition=condition, reference_stem=reference_stem,
                               reference=str(find_image(Path(config['pairs']) / 'control', reference_stem)),
                               target=str(find_image(Path(config['pairs']) / 'target', stem)),
                               output=str(output), width=pairs[i].get('width', config['width']),
                               height=pairs[i].get('height', config['height']), seed=config['seed'])
                    rows.append(row)
                    manifest.append(dict(row, grid=file_stem + '.png', row_in_grid=len(rows),
                                         output_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
            save_png(render(rows, 320, f'{layout} | step {step} | A - B - Resultado | original + shuffle'),
                     args.out / (file_stem + '.png'))
            with safe_open(metrics['summary']['adapter'], framework='pt') as sf:
                contract = json.loads(sf.metadata()['nextscene_contract'])
            metadata = dict(layout=layout, step=step, evaluation_config=config,
                            checkpoint_contract=contract, metrics=metrics, prompts=prompts)
            (metadata_dir / (file_stem + '.json')).write_text(json.dumps(metadata, indent=2))
            summaries.append(dict(layout=layout, step=step, **metrics['summary']))
    with (args.out / 'metrics_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    text = '# Prompts efetivamente usados\n\nIguais em todos os checkpoints e condições.\n'
    for stem, caption in captions.items():
        text += f'\n## {stem}\n\n{caption}\n'
    (args.out / 'prompts.md').write_text(text)
    print(f'Exported {len(summaries)} checkpoint grids; {len(manifest)} saved outputs; {len(captions)} pairs.')


if __name__ == '__main__':
    main()
