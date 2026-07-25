#!/usr/bin/env python3
"""Converte o Macro-Dataset para o layout que o loader do fork entende.

    target/<idx>.jpg      alvo
    target/<idx>.txt      prompt
    refs/<idx>_1.jpg      referência 1   <- a ORDEM vem do manifesto,
    refs/<idx>_2.jpg      referência 2      nunca do filesystem
    …

O sufixo numérico é o contrato: `utils/dataset.py` (ramo `multi_ref`) ordena
por ele. `Path.glob` devolve ordem de `os.scandir`, arbitrária e instável
entre máquinas — se a ordem dos slots viesse dali, o binding `<image N>` ->
slot N seria aprendido embaralhado e o treino inteiro seria inválido.

Uso:
    prepare_macro.py <dir_extraido> <dir_saida> [--max-refs 3] [--limit N]
"""
import argparse
import json
import shutil
from pathlib import Path


def iter_samples(root: Path):
    """Cada amostra tem um JSON em `<bracket>/json/` e as imagens em
    `<bracket>/data/<idx>/`. Os paths do manifesto são relativos à raiz da
    extração."""
    for json_dir in sorted(root.rglob('json')):
        if not json_dir.is_dir():
            continue
        for name in sorted(p.name for p in json_dir.iterdir() if p.suffix == '.json'):
            meta = json_dir / name
            try:
                with open(meta) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            if 'input_images' not in data or 'output_image' not in data:
                continue
            yield meta, data


def caption_of(data: dict) -> str | None:
    for key in ('instruction', 'prompt'):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def passes_quality(data: dict, min_following: float, min_consistency: float) -> bool:
    """O Macro traz notas de um juiz. A curadoria fraca foi o que estragou o
    dataset anterior do projeto, então filtramos aqui em vez de descobrir
    depois no grid."""
    following = data.get('following_score')
    if following is not None and float(following) < min_following:
        return False
    scores = data.get('consistency_scores') or []
    if scores and min(float(s) for s in scores) < min_consistency:
        return False
    return True


def resolve(root: Path, meta: Path, rel: str) -> Path | None:
    """Os paths do manifesto são relativos à raiz da extração; alguns dumps
    guardam só o basename. Tenta os dois, nessa ordem."""
    for candidate in (root / rel, meta.parent / Path(rel).name):
        if candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--max-refs', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-following', type=float, default=9.0)
    ap.add_argument('--min-consistency', type=float, default=9.0)
    args = ap.parse_args()

    target_dir = args.dst / 'target'
    refs_dir = args.dst / 'refs'
    target_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped_missing = skipped_toomany = skipped_quality = skipped_caption = 0
    for meta, data in iter_samples(args.src):
        refs = data['input_images']
        if len(refs) > args.max_refs:
            skipped_toomany += 1
            continue
        if not passes_quality(data, args.min_following, args.min_consistency):
            skipped_quality += 1
            continue
        caption = caption_of(data)
        if caption is None:
            skipped_caption += 1
            continue
        out = resolve(args.src, meta, data['output_image'])
        ref_paths = [resolve(args.src, meta, r) for r in refs]
        if out is None or any(p is None for p in ref_paths):
            skipped_missing += 1
            continue

        idx = f'{kept:07d}'
        shutil.copyfile(out, target_dir / f'{idx}{out.suffix}')
        (target_dir / f'{idx}.txt').write_text(caption + '\n')
        for slot, path in enumerate(ref_paths, start=1):
            shutil.copyfile(path, refs_dir / f'{idx}_{slot}{path.suffix}')
        kept += 1
        if kept % 2000 == 0:
            print(f'  {kept} amostras…', flush=True)
        if args.limit and kept >= args.limit:
            break

    print(f'PRONTO: {kept} amostras -> {args.dst}')
    print(f'  descartadas por >{args.max_refs} refs:   {skipped_toomany}')
    print(f'  descartadas por qualidade (juiz):       {skipped_quality}')
    print(f'  descartadas sem caption:                {skipped_caption}')
    print(f'  descartadas por arquivo faltando:       {skipped_missing}')


if __name__ == '__main__':
    main()
