#!/usr/bin/env python3
"""Three-column comparison of only the final E2 checkpoints, saved outputs.

The main PNG compares true-reference generations of both layouts at seed76.
The complete PDF groups both seeds and true/shuffled/null controls by pair.
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from nextscene_review_grid import find_image, render, save_png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, default=Path('/workspace/nextscene_artifacts/E2'))
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    out = args.out or args.artifacts / 'comparacoes_ultimo_checkpoint'
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((args.artifacts / 'campaign_state.json').read_text())
    evaluations, summaries = [], []
    stems = None
    for seed in [76, 142]:
        for arm, layout in [('A', 'aligned'), ('B', 'disjoint_w')]:
            paths = list((args.artifacts / f'eval_seed{seed}' / f'{arm}epoch1').glob('*/metrics.json'))
            if len(paths) != 1:
                raise ValueError(f'{arm}/seed{seed}: expected one completed final evaluation')
            path = paths[0]
            metrics = json.loads(path.read_text())
            config = json.loads((path.parent / 'eval_config.json').read_text())
            assert Path(metrics['summary']['adapter']).parent.name == 'epoch1'
            assert config['seed'] == seed
            order = [r['stem'] for r in metrics['pairs']]
            assert len(order) == metrics['summary']['n'] == 24
            if stems is not None:
                assert order == stems, 'Pair order differs between evaluations'
            stems = order
            step = state['stages'][f'train_{arm}_epoch1']['final_step']
            evaluations.append((arm, layout, seed, step, path.parent, metrics, config))
            summaries.append(dict(layout=layout, step=step, seed=seed, **metrics['summary']))
    rows = []
    for i, stem in enumerate(stems):
        for arm, layout, seed, step, directory, metrics, config in evaluations:
            pair = metrics['pairs'][i]
            root = Path(config['pairs'])
            for condition in ['true', 'shuffled', 'null']:
                reference_stem = (stem if condition == 'true' else
                                  stems[(i + 1) % len(stems)] if condition == 'shuffled' else
                                  'SEM REFERENCIA / latent zero')
                reference = (str(find_image(root / 'control', reference_stem))
                             if condition != 'null' else None)
                output = directory / f'{stem}_{condition}.png'
                with Image.open(output) as im:
                    assert im.size == (pair['width'], pair['height']), output
                rows.append(dict(run=f'E2_{arm}_FINAL', step=step, layout=layout,
                                 stem=stem, pair_number=i + 1, condition=condition,
                                 reference_stem=reference_stem, reference=reference,
                                 target=str(find_image(root / 'target', stem)), output=str(output),
                                 width=pair['width'], height=pair['height'], seed=seed))
    assert len(rows) == 288
    assert len({(r['layout'], r['seed'], r['stem'], r['condition']) for r in rows}) == 288
    main_rows = [r for r in rows if r['seed'] == 76 and r['condition'] == 'true']
    save_png(render(main_rows, 320, 'E2 final | 5685 steps | aligned vs disjoint_w | seed76'),
             out / 'E2_FINAL_A_B_RESULTADO.png')
    # One pair per page:12 results (2layouts x2seeds x3reference conditions).
    pages = [render([r for r in rows if r['stem'] == stem], 512,
                    f'E2 final | par{i + 1:02d}/24 | seeds76/142 | original + shuffle + null')
             for i, stem in enumerate(stems)]
    pdf = out / 'E2_FINAL_TODOS_3_COLUNAS.pdf'
    tmp = pdf.with_suffix('.tmp')
    pages[0].save(tmp, format='PDF', save_all=True, append_images=pages[1:],
                  resolution=150, quality=95, title='E2 final: ambos layouts, seeds e controles')
    tmp.replace(pdf)
    save_png(pages[0], out / 'E2_FINAL_PAGINA_01.png')
    (out / 'manifest.json').write_text(json.dumps(rows, indent=2))
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (out / 'README.md').write_text(
        '# E2 — somente os últimos checkpoints\n\n'
        'Os dois treinos completaram uma época:5685steps cada. Aqui são usados somente '
        '`epoch1` de E2_A/aligned e E2_B/disjoint_w.\n\n'
        '[Grid principal](E2_FINAL_A_B_RESULTADO.png):A|B|resultado,24pares, ambos os layouts, '
        'seed76 e referência original (48resultados).\n\n'
        '[PDF completo](E2_FINAL_TODOS_3_COLUNAS.pdf):24páginas,1par por página,12resultados '
        'por par:2layouts ×2seeds(76/142) ×3condições(original/shuffle/null),288resultados.\n\n'
        'Nomes acima das linhas indicam experimento/layout/step/seed/condição. No shuffle, '
        'A é a referência efetivamente trocada. No null, A aparece cinza apenas como '
        'marcador de ausência de imagem; o modelo recebeu latent zero. B é sempre o '
        'próximo frame real daquele par. A proporção das imagens é preservada.\n\n'
        '[Manifest](manifest.json) · [Métricas dos finais](metrics.csv). '
        'Nenhum sampling novo foi executado para esta montagem.\n')
    print(f'Prepared48 main rows and288 complete rows in24PDF pages: {out}', flush=True)


if __name__ == '__main__':
    main()
