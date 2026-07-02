#!/usr/bin/env python3
"""Audit Anima control LoRA checkpoints for module contamination (F1.2).

Reads ONLY the safetensors JSON header (no tensor data, no torch needed) and
fails if any key touches a module that control adapters must never train:

    adaln_modulation  - Anima has an internal adaln_lora path; PEFT LoRA on top
                        causes double-LoRA amplification (blurry outputs).
    cross_attn        - text conditioning; contaminated April checkpoints.
    llm_adapter       - text conditioning; same problem.

Usage:
    python tools/audit_lora_header.py path/to/adapter_model.safetensors
    python tools/audit_lora_header.py output_dir/            # scans recursively
    python tools/audit_lora_header.py a.safetensors b.safetensors --quiet

Exit code 0 = all clean, 1 = contamination found, 2 = usage/read error.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = ('adaln_modulation', 'cross_attn', 'llm_adapter')


def read_safetensors_header(path):
    """Return (header_dict, metadata_dict) reading only the JSON header."""
    with open(path, 'rb') as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError('file too small to be a safetensors file')
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        if header_len > 100 * 1024 * 1024:
            raise ValueError(f'implausible header length {header_len}')
        header = json.loads(f.read(header_len))
    metadata = header.pop('__metadata__', {})
    return header, metadata


def classify_key(key):
    # adaln_modulation_cross_attn must count as adaln, so check in this order.
    for pattern in ('adaln_modulation', 'llm_adapter', 'cross_attn'):
        if pattern in key:
            return pattern
    if 'self_attn' in key:
        return 'self_attn'
    if '.mlp' in key or 'mlp.' in key:
        return 'mlp'
    return 'other'


def audit_file(path, quiet=False):
    """Print a report for one file. Returns True if clean, False if contaminated."""
    header, metadata = read_safetensors_header(path)
    counts = {}
    contaminated = []
    for key in header:
        category = classify_key(key)
        counts[category] = counts.get(category, 0) + 1
        if category in FORBIDDEN_PATTERNS:
            contaminated.append(key)

    clean = not contaminated
    status = 'CLEAN' if clean else 'CONTAMINATED'
    print(f'{status}  {path}')
    if not quiet:
        print(f'  keys: {len(header)}  breakdown: {counts}')
        interesting = {k: v for k, v in metadata.items()
                       if k in ('diffusion_pipe_commit', 'model_type', 'rank',
                                'network_alpha', 'cond_size', 'condition_dropout')}
        if interesting:
            print(f'  metadata: {interesting}')
        if contaminated:
            print(f'  {len(contaminated)} forbidden keys, first 10:')
            for key in contaminated[:10]:
                print(f'    {key}')
    return clean


def collect_files(paths):
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob('*.safetensors'))
            if not found:
                print(f'warning: no .safetensors files under {p}', file=sys.stderr)
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(raw)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='+', help='.safetensors files or directories to scan')
    parser.add_argument('--quiet', action='store_true', help='one status line per file')
    args = parser.parse_args()

    try:
        files = collect_files(args.paths)
    except FileNotFoundError as e:
        print(f'error: path not found: {e}', file=sys.stderr)
        return 2
    if not files:
        print('error: nothing to audit', file=sys.stderr)
        return 2

    all_clean = True
    for path in files:
        try:
            if not audit_file(path, quiet=args.quiet):
                all_clean = False
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f'ERROR  {path}: {e}', file=sys.stderr)
            return 2

    if not all_clean:
        print('\nContaminated checkpoints reproduce the blur problem at inference.'
              '\nRetrain on a fixed commit; skip_adaln-style stripping does NOT salvage them.')
    return 0 if all_clean else 1


if __name__ == '__main__':
    sys.exit(main())
