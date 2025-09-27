# ---------------------------------------------------------------
# 2.  Helper that builds one DataLoader whose batches are homogeneous
# ---------------------------------------------------------------
from torch.utils.data import Sampler, DataLoader, RandomSampler, BatchSampler
from itertools import chain
from collections import defaultdict
import math
import numpy as np

def make_size_bucketing_loader(dataset, batch_size, shuffle=True):
    """
    Groups indices by grid size N so that every mini-batch is size-homogeneous.
    """
    # -------- group indices by N ---------------------------------
    buckets = defaultdict(list)          # N → list[idx]
    for idx in range(len(dataset)):
        N = dataset[idx]["N"]
        buckets[N].append(idx)

    # -------- build one BatchSampler per bucket ------------------
    samplers = []
    for N, idxs in buckets.items():
        if shuffle:
            np.random.shuffle(idxs)
        # drop last incomplete batch to keep shapes equal
        n_batches = len(idxs) // batch_size
        idxs = idxs[: n_batches * batch_size]
        if not idxs:        # bucket too small for one batch
            continue
        bsampler = BatchSampler(
            sampler=iter(idxs),          # just an iterator over indices
            batch_size=batch_size,
            drop_last=True
        )
        samplers.append(bsampler)

    # -------- chain all bucket batch-samplers --------------------
    chained = chain(*samplers)           # yields lists[batch_size] of indices
    return DataLoader(dataset,
                      batch_sampler=chained,
                      num_workers=0)     # add workers as you like

class MultiBucketBatchSampler(Sampler):
    """
    Groups dataset indices by grid size N and yields fixed-size
    batches that are homogeneous in N.  Works for every epoch.
    """
    def __init__(self, sizes, batch_size, shuffle=True, drop_last=True):
        """
        sizes       iterable of int, length = len(dataset)   (grid-size per sample)
        batch_size  how many samples per batch
        shuffle     reshuffle indices inside every bucket each epoch
        drop_last   drop incomplete batches
        """
        self.sizes      = np.asarray(sizes)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

        # build static buckets: N ➝ [indices, …]
        self.buckets = defaultdict(list)
        for idx, N in enumerate(self.sizes):
            self.buckets[int(N)].append(idx)

    # ----------------------------------------------------------
    # PyTorch API
    # ----------------------------------------------------------
    def __iter__(self):
        batches = []

        for idxs in self.buckets.values():        # one bucket per size
            idxs = idxs.copy()                    # work on copy
            if self.shuffle:
                np.random.shuffle(idxs)

            n_full = len(idxs) // self.batch_size
            if n_full == 0:                       # bucket too small
                continue

            cut = n_full * self.batch_size if self.drop_last else len(idxs)
            idxs = idxs[:cut]

            # split into consecutive batches
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i : i + self.batch_size])

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        """total number of batches per epoch"""
        n = 0
        for idxs in self.buckets.values():
            n_full = len(idxs) // self.batch_size
            n += n_full
        return n