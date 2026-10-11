#!/usr/bin/env python3
"""Short 'instruction' caption tier for the v11 natural captions (structural, no LLM).

v11 captions follow: <framing sentence>. <subject + position + action sentence>. [appearance /
clothing sentences] [The same background/place ...] [text / watermark / subtitle sentences].
Short tier = framing + the subject/action sentence with appearance clauses removed (+ "The same
background." when present). Appearance, clothing, text and watermark sentences are dropped, so the
model must take identity, outfit, style and watermarks from the reference A instead of the text.

Writes <target_dir>/captions.json = {stem.ext: [full, short]} (each entry is one sample per epoch).
"""
import argparse
import json
from pathlib import Path
import re

APPEARANCE_SENT = re.compile(r'^(?:she|he|they|it|her|his|their)\s+(?:wears|has|is wearing|is dressed|are wearing|have)\b|'
                             r'\b(?:wears|wearing|dressed in)\b.*\b(?:dress|shirt|jacket|uniform|outfit|top|skirt|coat|suit)\b', re.I)
TEXT_SENT = re.compile(r'\b(?:watermark|subtitle|caption|text|reads|logo|credit|speech bubble|lettering|signature)\b', re.I)
SAME_BG = re.compile(r'^the same (?:background|place|room|setting|location|scene)', re.I)
# appearance clauses inside the action sentence: "with long red hair", "wearing a ...", "in a black dress"
CLAUSE = re.compile(r',?\s+(?:with (?:long|short|messy|spiky|wavy|curly|straight|shoulder-length|chin-length|pale|dark|light|tan|blonde|black|brown|red|white|silver|pink|blue|green|purple|orange|grey|gray|golden)[^,.;]*?(?:hair|skin|eyes|ears|horns|tail)[^,.;]*|wearing [^,.;]*|dressed in [^,.;]*|in (?:a|an|her|his|their) [^,.;]*?(?:dress|shirt|jacket|uniform|outfit|top|skirt|coat|suit|bikini|hoodie)[^,.;]*)', re.I)


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text.strip()) if s.strip()]


def short_caption(caption):
    sents = sentences(caption)
    if len(sents) < 2:
        return None
    out = [sents[0]]
    action = next((s for s in sents[1:3] if re.match(r'^(?:the same|a new|two|three|both|an?|in)\b', s, re.I)
                   and not TEXT_SENT.search(s) and not SAME_BG.match(s)), None)
    if action is None:
        return None
    # New subjects are not in A: their description must stay, and no other new subject may be dropped.
    rest = ' '.join(x for x in sents[1:] if x is not action)
    if re.search(r'\b(?:a|an|two|three|several) new\b', rest, re.I):
        return None
    if not re.match(r'^(?:a|an) new\b', action, re.I):
        action = CLAUSE.sub('', action)
    words = action.split()
    if len(words) > 40:  # keep the head of long action sentences, cut at a clause boundary
        cut = ' '.join(words[:40])
        action = cut[:cut.rfind(',')] + '.' if ',' in cut else cut + '.'
    out.append(action)
    if any(SAME_BG.match(s) for s in sents):
        out.append('The same background.')
    short = ' '.join(out)
    return short if len(short.split()) >= 6 and short != caption else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target-dir', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    caps, n_short, lens = {}, 0, []
    for txt in sorted(args.target_dir.glob('*.txt')):
        img = next(p for p in txt.parent.glob(txt.stem + '.*') if p.suffix.lower() != '.txt')
        full = txt.read_text().strip()
        short = short_caption(full)
        caps[img.name] = [full] + ([short] if short else [full])
        if short:
            n_short += 1
            lens.append(len(short.split()))
    print(f'{n_short}/{len(caps)} pairs got a short caption; median {sorted(lens)[len(lens) // 2]} words '
          f'(pairs without one repeat the full caption so every pair keeps 2 samples)')
    for k in (list(caps)[:: max(1, len(caps) // args.samples)][:args.samples] if args.samples else []):
        print('FULL :', caps[k][0][:220]); print('SHORT:', caps[k][1]); print()
    if not args.dry_run:
        (args.target_dir / 'captions.json').write_text(json.dumps(caps, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
