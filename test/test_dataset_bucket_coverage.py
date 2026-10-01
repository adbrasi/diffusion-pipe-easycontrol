"""The all-pairs recipe must retain small buckets and the last partial batch."""
from types import SimpleNamespace

import numpy as np
import pytest

from utils.dataset import ConcatenatedBatchedDataset


@pytest.mark.parametrize('count', [1, 2, 3, 4, 5, 11])
@pytest.mark.parametrize('batch', [1, 2, 4])
def test_padding_retains_every_example(count, batch):
    bucket = ConcatenatedBatchedDataset([], pad_last_batch=True)
    original = np.array([(0, i) for i in range(count)])
    bucket.iteration_order = original.copy()
    bucket._make_divisible_by(batch)
    assert len(bucket.iteration_order) % batch == 0
    assert len(bucket.iteration_order) == ((count + batch - 1) // batch) * batch
    assert np.array_equal(bucket.iteration_order[:count], original)
    assert set(bucket.iteration_order[:, 1]) == set(range(count))


def test_default_retains_existing_drop_behavior():
    bucket = ConcatenatedBatchedDataset([SimpleNamespace(size_bucket=(1024, 1024, 1))])
    bucket.iteration_order = np.array([(0, i) for i in range(5)])
    bucket._make_divisible_by(4)
    assert len(bucket.iteration_order) == 4
