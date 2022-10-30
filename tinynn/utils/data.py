import collections
import random
from typing import Any, List

import numpy as np

import tinynn


def default_collate(batch: List[Any]):
    assert batch, "cannot collate empty batch"
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, tinynn.Tensor):
        out = tinynn.stack(batch, dim=0)
    elif isinstance(elem, np.ndarray):
        tensors = [tinynn.tensor(x) for x in batch]
        out = tinynn.stack(tensors, dim=0)
    elif isinstance(elem, (int, float)):
        out = tinynn.tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        out = elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
    elif isinstance(elem, collections.abc.Sequence):
        out = elem_type(default_collate(samples) for samples in zip(*batch))
    else:
        raise TypeError(f"Unknown batch element type {elem_type}")
    return out


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.indexes = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indexes)

        batch = []
        for index in self.indexes:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield default_collate(batch)
                batch = []

        if batch and not self.drop_last:
            yield default_collate(batch)

    def __len__(self):
        if not self.drop_last:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size
