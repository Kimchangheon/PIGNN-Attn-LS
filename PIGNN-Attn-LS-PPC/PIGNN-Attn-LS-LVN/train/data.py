from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from data_loading.samplers import MultiBucketBatchSampler


@dataclass(frozen=True)
class DataSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_train: int
    n_val: int
    n_test: int


def build_dataloaders(
    *,
    parquet_paths,
    per_unit: bool,
    device: torch.device,
    batch_size: int,
    block_diag: bool,
    seed: int,
    split_mode: str,
    train_ratio: float,
    valid_ratio: float,
    train_subset_frac: float | None = None,
    train_subset_min_n: int = 1,
) -> DataSplits:
    full_ds = ChanghunDataset(parquet_paths, per_unit=per_unit, device=device)

    n_total = len(full_ds)
    if split_mode == "equal3":
        base, rem = divmod(n_total, 3)
        # Distribute remainder to train then valid (difference between splits <= 1)
        n_train = base + (1 if rem > 0 else 0)
        n_val = base + (1 if rem > 1 else 0)
        n_test = n_total - n_train - n_val
    else:
        n_train = int(train_ratio * n_total)
        n_val = int(valid_ratio * n_total)
        n_test = n_total - n_train - n_val

    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError(
            f"Invalid split sizes: n_total={n_total}, n_train={n_train}, n_val={n_val}, n_test={n_test}. "
            f"Check split_mode/train_ratio/valid_ratio."
        )

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    # Optional: subsample ONLY the training split for few-shot / target-budget experiments.
    # This keeps val/test fixed across budgets (assuming the same split seed).
    if train_subset_frac is not None:
        frac = float(train_subset_frac)
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"train_subset_frac must be in [0,1], got: {train_subset_frac}")

        if frac < 1.0:
            # random_split returns torch.utils.data.Subset
            if not isinstance(train_ds, Subset):
                raise TypeError("Expected train_ds to be a Subset after random_split")

            base_indices = list(train_ds.indices)
            n_base = len(base_indices)
            if n_base == 0:
                # nothing to do
                pass
            else:
                # Deterministic permutation based on the same seed.
                g = torch.Generator().manual_seed(int(seed))
                perm = torch.randperm(n_base, generator=g).tolist()
                k = int(round(frac * n_base))
                k = max(int(train_subset_min_n), k) if frac > 0.0 else 0
                k = min(k, n_base)
                chosen = [base_indices[i] for i in perm[:k]]
                train_ds = Subset(full_ds, chosen)
                n_train = len(train_ds)

    # If a few-shot budget forces an empty training split (e.g. budget=0 for zero-shot transfer),
    # avoid DataLoader(shuffle=True) which uses RandomSampler and crashes on num_samples=0.
    train_is_empty = len(train_ds) == 0

    # Optional DataLoader worker count, env-var gated. Default 0 preserves the
    # original single-thread loader (HVN behaviour). Set GNN_NUM_WORKERS=4 (or
    # similar) for LVN training, where per-row sparse->dense reconstruction in
    # the lazy dataset is ~30-40 ms and benefits from parallelism.
    import os as _os
    _nw = int(_os.environ.get("GNN_NUM_WORKERS", "0"))
    _persistent = _nw > 0
    _loader_kw = dict(num_workers=_nw, persistent_workers=_persistent)

    if block_diag:
        # Important: collate_blockdiag provides the 'sizes' key expected by the
        # training/eval loop. Use it even for batch_size=1.
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(not train_is_empty),
            collate_fn=collate_blockdiag,
            **_loader_kw,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_blockdiag,
            **_loader_kw,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_blockdiag,
            **_loader_kw,
        )
    elif batch_size == 1:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(not train_is_empty))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        if train_is_empty:
            # Fallback to a simple loader; bucket samplers expect at least one sample.
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

            sizes = [full_ds[i]["N"] for i in range(len(full_ds))]
            val_sampler = MultiBucketBatchSampler(
                sizes=np.take(sizes, val_ds.indices),
                batch_size=batch_size,
                shuffle=False,
            )
            test_sampler = MultiBucketBatchSampler(
                sizes=np.take(sizes, test_ds.indices),
                batch_size=batch_size,
                shuffle=False,
            )
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
            test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

            return DataSplits(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
            )

        sizes = [full_ds[i]["N"] for i in range(len(full_ds))]
        train_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, train_ds.indices),
            batch_size=batch_size,
            shuffle=True,
        )
        val_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, val_ds.indices),
            batch_size=batch_size,
            shuffle=False,
        )
        test_sampler = MultiBucketBatchSampler(
            sizes=np.take(sizes, test_ds.indices),
            batch_size=batch_size,
            shuffle=False,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

    return DataSplits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
