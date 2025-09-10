#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-throughput parallel data generator with a custom Parquet writer (no database module).

Key features
- CPU-saturating multiprocessing with batched tasks
- Single-writer, append-by-row-group Parquet output (fast + safe)
- Arrays (including complex) stored as compact .npy bytes in Parquet binary columns
- BLAS thread limiting to avoid over-subscription across processes

Usage example:
  python generate.py --runs 150000 --save_steps 10000 --gridtype HVN \
                     --min_number_of_buses 4 --max_number_of_buses 32
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
import math
import time
import argparse
import numpy as np
import multiprocessing as mp
from typing import Any, Dict, List

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
        self._compression_level = compression_level

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


def _generate_one_record_serialized() -> Dict[str, Any]:
    """
    Generate a single row and serialize all arrays to .npy bytes *in the worker*.
    This reduces pickling overhead between processes.
    """
    K = _CFG['K']
    gridtype = _CFG['gridtype']
    fixed = _CFG['fixed']
    min_buses = _CFG['min_buses']
    max_buses = _CFG['max_buses']
    debugging = False
    pic = False

    if fixed:
        # powers of two in [4, 32] when exponents in [2..5]
        exponents = np.array([2, 3, 4, 5])
        bus_number = int(2 ** np.random.choice(exponents))
    else:
        bus_number = int(np.random.randint(min_buses, max_buses + 1))

    (gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base) = case_generation(
        gridtype, bus_number, fixed, debugging, pic
    )

    Y_matrix_flatten = Y_matrix.copy().flatten().astype(complex)

    if not is_connected:
        # If topology is disconnected, keep schema but zero-out solution arrays
        u_newton = np.zeros_like(u_start, dtype=complex)
        I_newton = np.zeros_like(u_start, dtype=complex)
        S_newton = np.zeros_like(u_start, dtype=complex)
    else:
        u_newton, I_newton, S_newton = newtonrapson(
            bus_typ, Y_matrix.copy(), s_multi.copy(), u_start.copy(),
            K=K
        )

    rec = {
        'bus_number': bus_number,
        'gridtype': gridtype_out,
        'U_base': float(U_base) if np.isscalar(U_base) else float(np.asarray(U_base).ravel()[0]),
        'S_base': float(S_base) if np.isscalar(S_base) else float(np.asarray(S_base).ravel()[0]),

        # arrays serialized to bytes
        'bus_typ': ndarray_to_npy_bytes(bus_typ),
        'Y_Lines': ndarray_to_npy_bytes(Y_Lines),
        'Y_C_Lines': ndarray_to_npy_bytes(Y_C_Lines),
        'Lines_connected': ndarray_to_npy_bytes(Lines_connected),
        'Y_matrix': ndarray_to_npy_bytes(Y_matrix_flatten),
        'u_start': ndarray_to_npy_bytes(u_start),
        'u_newton': ndarray_to_npy_bytes(u_newton),
        'S_start': ndarray_to_npy_bytes(s_multi),
        'S_newton': ndarray_to_npy_bytes(S_newton),
        'I_newton': ndarray_to_npy_bytes(I_newton),
    }
    return rec


def _generate_batch(n_rows: int) -> List[Dict[str, Any]]:
    """Worker task: generate n_rows serialized records."""
    gen = _generate_one_record_serialized
    out: List[Dict[str, Any]] = []
    for _ in range(n_rows):
        try:
            out.append(gen())
        except Exception:
            # Swallow per-sample errors to keep the pipeline running
            continue
    return out


# ==========================
# CLI and main loop
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Parallel dataset generation with multiprocessing (no database module)")
    parser.add_argument('--K', type=int, default=40, help='newton_raphson steps')
    parser.add_argument('--runs', type=int, default=300000, help='Total number of samples to generate.')
    parser.add_argument('--save_steps', type=int, default=20000, help='Flush this many rows per Parquet append.')
    parser.add_argument("--fixed", action="store_true", help="Use fixed topology (powers of two bus counts).")
    parser.add_argument('--min_number_of_buses', type=int, default=4, help='Minimum number of buses.')
    parser.add_argument('--max_number_of_buses', type=int, default=512, help='Maximum number of buses.')
    parser.add_argument('--gridtype', type=str, default="MVN", help='Grid type (e.g., HVN, MVN).')
    parser.add_argument("--save_path", type=str, default="", help="Output directory.")
    parser.add_argument('--rows_per_task', type=int, default=4000,
                        help='Rows computed per worker task (batch). Larger reduces scheduling overhead.')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes. 0 = use all logical CPUs.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if it already exists.')
    return parser.parse_args()


def main():
    args = parse_args()

    K = int(args.K)
    runs = int(args.runs)
    save_steps = max(int(args.save_steps), 1)
    fixed = bool(args.fixed)
    gridtype = str(args.gridtype)
    min_buses = int(args.min_number_of_buses)
    max_buses = int(args.max_number_of_buses)
    rows_per_task = max(int(args.rows_per_task), 1)
    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)

    # Output path: single Parquet file
    filename = os.path.join(
        args.save_path,
        f"{gridtype}_{runs}_NR_plain_{min_buses}_to_{max_buses}_buses.parquet"
    )
    print(f"[INFO] Output file: {filename}")

    # Build config for workers
    cfg = dict(
        K = K,
        gridtype=gridtype,
        fixed=fixed,
        min_buses=min_buses,
        max_buses=max_buses,
    )

    # Tasking plan
    num_tasks = math.ceil(runs / rows_per_task)
    task_sizes = [rows_per_task] * (num_tasks - 1) + [runs - rows_per_task * (num_tasks - 1)]

    # Start method
    if sys.platform.startswith("win"):
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")  # fastest on UNIX-like

    total_written = 0
    start = time.time()

    # Create writer up-front (single file)
    writer = ParquetAppendWriter(filename, compression="zstd", compression_level=3, overwrite=args.overwrite)

    print(f"[INFO] Starting generation with {workers} workers, {num_tasks} tasks, rows_per_task={rows_per_task}, save_steps={save_steps}")

    buffer: List[Dict[str, Any]] = []
    try:
        with ctx.Pool(processes=workers,
                      initializer=_init_worker,
                      initargs=(cfg, int(time.time())),
                      maxtasksperchild=1000) as pool:

            # Stream batches as they complete
            for i, batch in enumerate(pool.imap_unordered(_generate_batch, task_sizes, chunksize=1), 1):
                if batch:
                    buffer.extend(batch)

                # Flush when buffer hits threshold
                if len(buffer) >= save_steps:
                    writer.write_records(buffer)
                    total_written += len(buffer)
                    buffer.clear()

                    elapsed = time.time() - start
                    rate = total_written / max(elapsed, 1e-6)
                    print(f"[INFO] {total_written:,}/{runs:,} rows written ({rate:,.0f} rows/s)")

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
    print(f"[DONE] Wrote {total_written:,} rows in {elapsed:,.1f}s ({rate:,.0f} rows/s)")
    print(f"[INFO] File size: {size_gb:.3f} GB -> {filename}")

    # Optional quick peek (uncomment for debugging; avoid on huge files)
    # print(read_first_row_parquet(filename))


if __name__ == "__main__":
    main()
