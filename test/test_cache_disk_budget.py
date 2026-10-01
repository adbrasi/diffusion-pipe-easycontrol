import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch

from utils.cache import Cache


class CacheDiskBudgetTests(unittest.TestCase):
    def test_budget_stop_commits_previously_written_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Cache(directory, 'test')
            cache.add({'tensor': torch.arange(8)})
            with patch.dict(os.environ, {'KREA2_CACHE_MIN_FREE_BYTES': str(2 ** 63)}):
                with self.assertRaisesRegex(OSError, 'Cache disk budget reached'):
                    cache.add({'tensor': torch.arange(8)})
            with sqlite3.connect(str(Path(directory) / 'metadata.db')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM items').fetchone()[0], 1)
            reopened = Cache(directory, 'test')
            self.assertEqual(len(reopened), 1)
            self.assertTrue(torch.equal(reopened[0]['tensor'], torch.arange(8)))
            for stream in reopened.open_files.values():
                stream.close()
            cache.con.close()
            reopened.con.close()


if __name__ == '__main__':
    unittest.main()
