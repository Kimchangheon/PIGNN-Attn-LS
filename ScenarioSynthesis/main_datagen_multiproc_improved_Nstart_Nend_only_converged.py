#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-throughput parallel data generator with a custom Parquet writer (no database module).

This version generates **one converged sample per bus size** in the inclusive range
[N_start .. N_end]. Each worker task handles a single bus size and repeats:
    case_generation -> newtonrapson
until it gets a connected topology and a non-zero (converged) NR solution,
or until --max-attempts is exceeded.

Key features
- CPU-saturating multiprocessing (one task per bus size)
- Single-writer, append-by-row-group Parquet output (fast + safe)
- Arrays (including complex) stored as compact .npy bytes in Parquet binary columns
- BLAS thread limiting to avoid over-subscription across processes

Usage example:
  python generate.py --gridtype MVN --K 40 --n-start 4 --n-end 512 \
                     --workers 0 --save_steps 100 --max-attempts 2000 \
                     --tol-zero 1e-12 --overwrite
"""

# ---- Prevent BLAS/numexpr oversubscription (set BEFORE numpy import) ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp
from typing import Any, Dict, List, Optional

# Project functions you already have
from case_generator_improved import case_generation
from newton_raphson_improved import newtonrapson
# from newton_raphson_improved_armijo import newtonrapson

# Parquet (no custom database module)
import pyarrow as pa
import pyarrow.parquet as pq


# ==========================
# Helpers: .npy <-> bytes
# ==========================
def ndarray_to_npy_bytes(x: Any) -> bytes:
    """
    Convert scalars/lists/ndarrays (including complex) to .npy bytes (no pickle).
    This makes Parquet schema stable (binary) and fast to write/read.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    buf = io.BytesIO()
    # allow_pickle=False keeps things safer and schema-stable
    np.save(buf, x, allow_pickle=False)
    return buf.getvalue()


def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    """Reverse of ndarray_to_npy_bytes (for optional reading utilities)."""
    buf = io.BytesIO(b)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)


# ==================================
# Custom Parquet appendable writer
# ==================================
class ParquetAppendWriter:
    """
    Single-file Parquet writer that appends row groups as you flush buffers.
    We define a fixed Arrow schema up front so each append is O(1) open/close.
    """

    _schema = pa.schema([
        pa.field('bus_number', pa.int32()),
        pa.field('gridtype', pa.string()),
        pa.field('U_base', pa.float64()),
        pa.field('S_base', pa.float64()),
        # All large, structured values encoded as .npy bytes (binary)
        pa.field('bus_typ', pa.binary()),
        pa.field('Y_Lines', pa.binary()),
        pa.field('Y_C_Lines', pa.binary()),
        pa.field('Lines_connected', pa.binary()),
        pa.field('Y_matrix', pa.binary()),
        pa.field('u_start', pa.binary()),
        pa.field('u_newton', pa.binary()),
        pa.field('S_start', pa.binary()),
        pa.field('S_newton', pa.binary()),
        pa.field('I_newton', pa.binary()),
    ])

    def __init__(self, path: str, compression: str = "zstd", compression_level: int = 3, overwrite: bool = True):
        self.path = path
        # Ensure parent directory exists
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if overwrite and os.path.exists(path):
            os.remove(path)

        self._writer = pq.ParquetWriter(
            where=path,
            schema=self._schema,
            compression=compression,
            use_dictionary=True
        )
        self._compression_level = compression_level  # not directly used by ParquetWriter; kept for API symmetry

    def write_records(self, records: List[Dict[str, Any]]):
        """Convert a list of dicts into a single Arrow table and append it."""
        if not records:
            return

        # Prebuild column lists (fast)
        bus_number = []
        gridtype = []
        U_base = []
        S_base = []

        bus_typ = []
        Y_Lines = []
        Y_C_Lines = []
        Lines_connected = []
        Y_matrix = []
        u_start = []
        u_newton = []
        S_start = []
        S_newton = []
        I_newton = []

        # Tight loop for speed
        for r in records:
            bus_number.append(int(r['bus_number']))
            gridtype.append(str(r['gridtype']))

            # Attempt float conversion; if it fails, store NaN
            try:
                U_base.append(float(r['U_base']))
            except Exception:
                U_base.append(float('nan'))
            try:
                S_base.append(float(r['S_base']))
            except Exception:
                S_base.append(float('nan'))

            bus_typ.append(r['bus_typ'])
            Y_Lines.append(r['Y_Lines'])
            Y_C_Lines.append(r['Y_C_Lines'])
            Lines_connected.append(r['Lines_connected'])
            Y_matrix.append(r['Y_matrix'])
            u_start.append(r['u_start'])
            u_newton.append(r['u_newton'])
            S_start.append(r['S_start'])
            S_newton.append(r['S_newton'])
            I_newton.append(r['I_newton'])

        arrays = [
            pa.array(bus_number, type=pa.int32()),
            pa.array(gridtype, type=pa.string()),
            pa.array(U_base, type=pa.float64()),
            pa.array(S_base, type=pa.float64()),
            pa.array(bus_typ, type=pa.binary()),
            pa.array(Y_Lines, type=pa.binary()),
            pa.array(Y_C_Lines, type=pa.binary()),
            pa.array(Lines_connected, type=pa.binary()),
            pa.array(Y_matrix, type=pa.binary()),
            pa.array(u_start, type=pa.binary()),
            pa.array(u_newton, type=pa.binary()),
            pa.array(S_start, type=pa.binary()),
            pa.array(S_newton, type=pa.binary()),
            pa.array(I_newton, type=pa.binary()),
        ]

        table = pa.Table.from_arrays(arrays, schema=self._schema)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# Optional helpers to inspect results (no external imports)
def get_parquet_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def read_first_row_parquet(path: str) -> Dict[str, Any]:
    """
    Lightweight peek at the first row. For binary columns we load back to ndarrays.
    Intended only for debugging small samples.
    """
    pf = pq.ParquetFile(path)
    tb = pf.read_row_group(0)
    # extract row 0 as Python objects
    row = {name: tb.column(i)[0].as_py() for i, name in enumerate(tb.schema.names)}
    # decode binary -> ndarray for convenience
    out = {
        'bus_number': row['bus_number'],
        'gridtype': row['gridtype'],
        'U_base': row['U_base'],
        'S_base': row['S_base'],
        'bus_typ': npy_bytes_to_ndarray(row['bus_typ']),
        'Y_Lines': npy_bytes_to_ndarray(row['Y_Lines']),
        'Y_C_Lines': npy_bytes_to_ndarray(row['Y_C_Lines']),
        'Lines_connected': npy_bytes_to_ndarray(row['Lines_connected']),
        'Y_matrix': npy_bytes_to_ndarray(row['Y_matrix']),
        'u_start': npy_bytes_to_ndarray(row['u_start']),
        'u_newton': npy_bytes_to_ndarray(row['u_newton']),
        'S_start': npy_bytes_to_ndarray(row['S_start']),
        'S_newton': npy_bytes_to_ndarray(row['S_newton']),
        'I_newton': npy_bytes_to_ndarray(row['I_newton']),
    }
    return out


# ==================================
# Multiprocessing worker machinery
# ==================================
_CFG: Dict[str, Any] = {}

# global per-process RNG (store it so your worker functions can use it)
_RNG = None

def _init_worker(cfg: dict, seed_base: int):
    global _CFG, _RNG
    _CFG = cfg
    pid = os.getpid()
    ss = np.random.SeedSequence([seed_base & 0xFFFFFFFF, pid & 0xFFFFFFFF])
    _RNG = np.random.default_rng(ss)
    # If you still have legacy code using np.random.*, you can ALSO set:
    np.random.seed(int(ss.generate_state(1, dtype=np.uint32)[0]))


def _generate_one_record_for_bus(bus_number: int) -> Optional[Dict[str, Any]]:
    """
    Generate exactly one converged record for a fixed bus_number.
    Repeats (case_generation -> newtonrapson) until:
      - topology is connected, AND
      - newtonrapson returns a non-zero u_newton (|.| > tol_zero),
    or until max_attempts is reached (returns None in that case).
    """
    K = _CFG['K']
    gridtype = _CFG['gridtype']
    fixed = _CFG['fixed']
    max_attempts = _CFG['max_attempts']
    tol_zero = _CFG['tol_zero']

    debugging = False
    pic = False

    attempt = 0
    while attempt < max_attempts:
        attempt += 1

        (gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
         Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base) = case_generation(
            gridtype, int(bus_number), fixed, debugging, pic
        )

        if not is_connected:
            continue

        u_try, I_try, S_try = newtonrapson(
            bus_typ, Y_matrix.copy(), s_multi.copy(), u_start.copy(), K=K
        )

        if np.allclose(u_try, 0, atol=tol_zero):
            # treat zero solution as non-converged
            continue

        # success
        Y_matrix_flatten = Y_matrix.copy().flatten().astype(complex)

        rec = {
            'bus_number': int(bus_number),
            'gridtype': gridtype_out,
            'U_base': float(U_base) if np.isscalar(U_base) else float(np.asarray(U_base).ravel()[0]),
            'S_base': float(S_base) if np.isscalar(S_base) else float(np.asarray(S_base).ravel()[0]),

            'bus_typ': ndarray_to_npy_bytes(bus_typ),
            'Y_Lines': ndarray_to_npy_bytes(Y_Lines),
            'Y_C_Lines': ndarray_to_npy_bytes(Y_C_Lines),
            'Lines_connected': ndarray_to_npy_bytes(Lines_connected),
            'Y_matrix': ndarray_to_npy_bytes(Y_matrix_flatten),
            'u_start': ndarray_to_npy_bytes(u_start),
            'u_newton': ndarray_to_npy_bytes(u_try),
            'S_start': ndarray_to_npy_bytes(s_multi),
            'S_newton': ndarray_to_npy_bytes(S_try),
            'I_newton': ndarray_to_npy_bytes(I_try),
        }
        return rec

    # give up for this bus_number
    return None


# ==========================
# CLI and main loop
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel dataset generation: one converged sample per bus size (N_start..N_end), multiprocessing per bus."
    )
    parser.add_argument('--K', type=int, default=40, help='newton_raphson iterations/steps')
    parser.add_argument('--gridtype', type=str, default="MVN", help='Grid type (e.g., HVN, MVN).')
    parser.add_argument("--fixed", action="store_true", help="Pass through to case_generation if it has special fixed behavior.")
    parser.add_argument('--n_start', type=int, required=True, help='Inclusive start of bus count range.')
    parser.add_argument('--n_end', type=int, required=True, help='Inclusive end of bus count range.')
    parser.add_argument('--save_steps', type=int, default=100, help='Flush this many rows per Parquet append.')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker processes. 0 = all logical CPUs.')
    parser.add_argument("--save_path", type=str, default="", help="Output directory.")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if it already exists.')
    parser.add_argument('--max_attempts', type=int, default=2000,
                        help='Max case_generation/NR attempts per bus before skipping.')
    parser.add_argument('--tol-zero', type=float, default=1e-12,
                        help='Treat |u_newton|<=tol as non-converged.')
    return parser.parse_args()


def main():
    args = parse_args()

    K = int(args.K)
    gridtype = str(args.gridtype)
    fixed = bool(args.fixed)
    n_start = int(args.n_start)
    n_end = int(args.n_end)
    save_steps = max(int(args.save_steps), 1)
    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    max_attempts = int(args.max_attempts)
    tol_zero = float(args.tol_zero)

    if n_end < n_start:
        raise ValueError("n_end must be >= n_start (inclusive).")

    bus_numbers = list(range(n_start, n_end + 1))  # inclusive
    runs = len(bus_numbers)

    # Output path: single Parquet file
    filename = os.path.join(
        args.save_path,
        f"{gridtype}_per_bus_{n_start}_to_{n_end}_one_each_max_att_{max_attempts}.parquet"
    )
    print(f"[INFO] Output file: {filename}")

    # Build config for workers
    cfg = dict(
        K=K,
        gridtype=gridtype,
        fixed=fixed,
        max_attempts=max_attempts,
        tol_zero=tol_zero,
    )

    # Start method
    if sys.platform.startswith("win"):
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")  # fastest on UNIX-like

    total_written = 0
    failed = 0
    start = time.time()

    # Create writer up-front (single file)
    writer = ParquetAppendWriter(filename, compression="zstd", compression_level=3, overwrite=args.overwrite)

    print(f"[INFO] Starting generation with {workers} workers over buses [{n_start}..{n_end}] (inclusive) -> {runs} target rows, save_steps={save_steps}")

    buffer: List[Dict[str, Any]] = []
    try:
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(cfg, int(time.time())),
            maxtasksperchild=1000
        ) as pool:

            # Stream each bus_number as a separate task
            for rec in pool.imap_unordered(_generate_one_record_for_bus, bus_numbers, chunksize=1):
                if rec is None:
                    failed += 1
                else:
                    buffer.append(rec)

                # Flush periodically
                if len(buffer) >= save_steps:
                    writer.write_records(buffer)
                    total_written += len(buffer)
                    buffer.clear()

                    elapsed = time.time() - start
                    rate = total_written / max(elapsed, 1e-6)
                    print(f"[INFO] {total_written:,}/{runs:,} rows written ({rate:,.0f} rows/s), failures={failed}")

            # Final flush
            if buffer:
                writer.write_records(buffer)
                total_written += len(buffer)
                buffer.clear()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Flushing remaining buffer...")
        if buffer:
            writer.write_records(buffer)
            total_written += len(buffer)
    finally:
        writer.close()

    elapsed = time.time() - start
    rate = total_written / max(elapsed, 1e-6)
    size_gb = get_parquet_file_size(filename) / 1e9
    print(f"[DONE] Wrote {total_written:,}/{runs:,} rows in {elapsed:,.1f}s ({rate:,.0f} rows/s), failures={failed}")
    print(f"[INFO] File size: {size_gb:.3f} GB -> {filename}")

    # Optional quick peek (uncomment for debugging; avoid on huge files)
    # print(read_first_row_parquet(filename))


if __name__ == "__main__":
    main()
