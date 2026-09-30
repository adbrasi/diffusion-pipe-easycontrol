#!/usr/bin/env python3
"""Build full/action-only cached caption tiers without a captioning API.

Conservative fallback: keep the original caption for every image; add a shorter
variant only when an explicit action and a readable subject occur in the text.
This extracts text, it does not infer identity or continuity from image pixels.
Inspect the emitted samples before training; it is not a substitute for VLM
annotations of new vs inherited subjects.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

ACTION = re.compile(
    r'\b(?:standing|sitting|seated|holding|looking|turning|leaning|reaching|walking|'
    r'running|jumping|facing|smiling|talking|gesturing|raising|pointing|kneeling|'
    r'crouching|lying|resting|opening|closing|gripping|clutching|extending|crossing|'
    r'grinning|laughing|shouting|crying|gazing|glancing|staring|screaming|frowning|'
    r'covering|pushing|pulling|lifting|stretching|bending|embracing|hugging|fighting|'
    r'falling|floating|waving|drawing|drinking|eating|carrying|reading|writing|'
    r'bowing|reclining|looming|posing|slumping|climbing|swinging|striking|kicking|'
    r'folding|peering|tilting|striding|stepping|balancing|riding|driving|flying)\b',
    re.I,
)
SUBJECT = re.compile(r'\b(?:girl|boy|man|woman|men|women|person|people|character|'
                     r'figure|child|children|player|referee|soldier|warrior|knight|'
                     r'cat|dog|bird|dragon|robot)\b', re.I)
APPEARANCE = re.compile(r'\b(?:wearing|dressed|clad|outfit|hair|haired|skin|eyes|'
                        r'lighting|palette|shading|rendered|style|background)\b', re.I)


def action_caption(caption, same_subject=False):
    text = re.split(r'\b(?:Character|Background|Style) continuity:', caption, maxsplit=1)[0].strip()
    subject = SUBJECT.search(text)
    if not subject:
        return None
    actions = []
    last_end = -1
    for m in ACTION.finditer(text):
        if m.start() < subject.end() or m.start() < last_end:
            continue
        end = re.search(r'[,.;]', text[m.start():])
        end = m.start() + end.start() if end else len(text)
        chunk = text[m.start():end].strip()
        if re.match(r'(?:floating|standing|hanging)\s+(?:white\s+)?(?:text|credits|letters|trees|clouds|particles|objects)\b', chunk, re.I):
            continue
        if re.search(r'\bis (?:a|an|the)\b', chunk, re.I):
            continue
        appearance = APPEARANCE.search(chunk)
        if appearance:
            chunk = chunk[:appearance.start()].strip(' ,')
        # Cutting appearance must not leave broken articles/prepositions.
        chunk = re.sub(r'\b(with a (?:surprised|angry|nervous|shocked|worried|sad|happy))$', r'\1 expression', chunk, flags=re.I)
        chunk = re.sub(r'\s+(?:(?:of|with|in|and|on|at)\s+)?(?:the|a|an)$', '', chunk, flags=re.I)
        chunk = re.sub(r'\s+(?:of|with|in|and|on|at)$', '', chunk, flags=re.I)
        if len(chunk.split()) < 2:
            continue
        # Avoid unbounded clauses; keep a whole extracted phrase or skip it.
        if len(chunk.split()) > 24:
            continue
        actions.append(chunk)
        last_end = end
        if len(actions) == 2:
            break
    if not actions:
        return None
    framing = []
    for chunk in re.split(r'[,.;]', text)[:3]:
        if SUBJECT.search(chunk):
            break
        if re.search(r'\b(?:shot|view|angle|close-up|closeup|eye-level)\b', chunk, re.I) and len(chunk.split()) <= 12:
            chunk = re.split(r'\b(?:in|inside|during)\b', chunk, maxsplit=1)[0].strip()
            framing.append(chunk.strip())
    noun = subject.group().lower()
    noun = noun if same_subject or noun in ['cat', 'dog', 'bird', 'dragon', 'robot'] else 'character'
    # Deictic "the character" omits appearance without asserting SAME identity.
    prefix = 'The same ' if same_subject else 'The '
    auxiliary = ' are ' if same_subject and noun in ['men', 'women', 'people', 'children'] else ' is '
    short = (', '.join(framing) + '. ' if framing else '') + prefix + noun + auxiliary + ', '.join(actions) + '.'
    return short if len(short.split()) < len(caption.split()) else None


def build_tiers(target, short_repeats=1, same_subject=False):
    if short_repeats < 1:
        raise ValueError('short_repeats must be positive')
    captions, full_words, short_words = {}, [], []
    samples = []
    for p in sorted(Path(target).iterdir()):
        if p.suffix.lower() not in {'.jpg', '.png', '.jpeg', '.webp', '.bmp'}:
            continue
        cap = p.with_suffix('.txt')
        if not cap.exists():
            continue
        full = cap.read_text().strip()
        if not full:
            continue
        short = action_caption(full, same_subject=same_subject)
        captions[p.name] = [full] + ([short] * short_repeats if short else [])
        full_words.append(len(full.split()))
        if short:
            short_words.append(len(short.split()))
            if len(samples) < 30:
                samples.append({'image': p.name, 'full': full, 'short': short})
    (Path(target) / 'captions.json').write_text(json.dumps(captions, indent=2, ensure_ascii=False))
    return {'pairs': len(captions), 'short_tier_pairs': len(short_words),
            'cached_caption_samples': len(full_words) + short_repeats * len(short_words),
            'short_repeats': short_repeats, 'same_subject': same_subject,
            'median_full_words': statistics.median(full_words) if full_words else None,
            'median_short_words': statistics.median(short_words) if short_words else None,
            'samples': samples}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--short-repeats', type=int, default=1)
    ap.add_argument('--same-subject', action='store_true', help='use "The same <subject>" for continuity training')
    a = ap.parse_args()
    report = build_tiers(a.target, short_repeats=a.short_repeats, same_subject=a.same_subject)
    Path(a.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}))
