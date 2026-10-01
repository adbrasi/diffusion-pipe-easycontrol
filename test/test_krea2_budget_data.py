import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.krea2_prepare_budget_data import inventory, materialize, select_pairs


class BudgetDataTests(unittest.TestCase):
    def test_selection_only_uses_global_budget_and_seed(self):
        rows = [{'subset': f'ds{i % 4}', 'id': i} for i in range(2000)]
        selected, capacity = select_pairs(rows, 20_000, 1000, 10, 1, 42)
        self.assertEqual(capacity, 1900)
        self.assertEqual(len(selected), 1900)  # Regression: no 1500 cap.
        self.assertEqual(selected, select_pairs(rows, 20_000, 1000, 10, 1, 42)[0])
        self.assertEqual(len({p['subset'] for p in selected}), 4)
        self.assertEqual(len(select_pairs(rows, 100, 200, 10, 1, 42)[0]), 0)

    def test_integrity_holdout_and_verbatim_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            caption = '  Original caption, no editing.\nKeep all punctuation!  '
            for subset_name in ('ds1', 'ds2'):
                subset = source / subset_name
                for field in ('images_A', 'images_B'):
                    (subset / field).mkdir(parents=True)
                    for name in ('same.png', 'held.png', 'missing_caption.png'):
                        Image.new('RGB', (16, 16)).save(subset / field / name)
                rows = [{'video': 'images_B/same.png', 'caption': caption},
                        {'video': 'images_B/held.png', 'caption': 'Held out'}]
                (subset / 'captions.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            published_caption = 'Published successful retry: preserve this exact caption.'
            (source / 'ds2/images_B/same.txt').write_text(published_caption)
            candidates, excluded, _ = inventory(source, {('ds1', 'held.png'), ('ds2', 'held.png')})
            self.assertEqual(len(candidates), 2)
            self.assertEqual(excluded['heldout_pairs'], 2)
            self.assertEqual(excluded['target_without_caption'], 2)
            self.assertEqual(len({p['output_filename'] for p in candidates}), 2)
            self.assertEqual([p['caption'] for p in candidates], [caption, published_caption])
            destination = source / 'fresh_dataset'
            materialize(candidates, destination)
            captions = json.loads((destination / 'target/captions.json').read_text())
            self.assertEqual(list(captions.values()), [[caption], [published_caption]])
            for row in candidates:
                self.assertEqual(Path(row['target']).stat().st_ino,
                                 (destination / 'target' / row['output_filename']).stat().st_ino)
            with self.assertRaises(FileExistsError):
                materialize(candidates, destination)


if __name__ == '__main__':
    unittest.main()
