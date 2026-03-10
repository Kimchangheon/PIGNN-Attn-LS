#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from case_generator_improved import case_generation, case_generation_ieee14, case_generation_ieee14_pandapower
from newton_raphson_improved import newtonrapson

import pyarrow as pa
import pyarrow.parquet as pq

# ==========================
# Helpers: .npy <-> bytes
# ==========================
def ndarray_to_npy_bytes(x: Any) -> bytes:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    buf = io.BytesIO()
    np.save(buf, x, allow_pickle=False)
    return buf.getvalue()

def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    buf = io.BytesIO(); buf.seek(0)
    return np.load(io.BytesIO(b), allow_pickle=False)

# ==================================
# Parquet appendable writer (updated)
# ==================================
class ParquetAppendWriter:
    """
    Writer with optional Y_matrix and new transformer/multilevel metadata columns.
    """
    def __init__(
        self,
        path: str,
        compression: str = "zstd",
        compression_level: int = 3,
        overwrite: bool = True,
        save_y_matrix: bool = True,
    ):
        self.path = path
        self._compression_level = compression_level
        self.save_y_matrix = bool(save_y_matrix)

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if overwrite and os.path.exists(path):
            os.remove(path)

        fields = [
            pa.field('bus_number', pa.int32()),
            pa.field('gridtype', pa.string()),
            pa.field('U_base', pa.float64()),
            pa.field('S_base', pa.float64()),

            # legacy arrays (compatible with your old code)
            pa.field('bus_typ', pa.binary()),
            pa.field('Y_Lines', pa.binary()),        # upper-tri series admittance for line edges (S); 0 for others
            pa.field('Y_C_Lines', pa.binary()),      # upper-tri per-end shunt for lines (S); 0 otherwise
            pa.field('Lines_connected', pa.binary()),

            # NEW: multi-level & transformer metadata
            pa.field('vn_kv', pa.binary()),          # per-bus nominal kV
            pa.field('Is_trafo', pa.binary()),       # upper-tri 0/1
            pa.field('Trafo_tau', pa.binary()),      # upper-tri off-nominal magnitude
            pa.field('Trafo_shift_deg', pa.binary()),# upper-tri degrees
            pa.field('Trafo_y_series', pa.binary()), # upper-tri series admittance (S) for trafos; 0 otherwise
            pa.field('Trafo_b_total', pa.binary()),  # upper-tri total magnetizing susceptance (S); 0 else

            # 🔥 NEW: per-bus shunt admittance in SI [S]
            pa.field('Y_shunt_bus', pa.binary()),
        ]
        if self.save_y_matrix:
            fields.append(pa.field('Y_matrix', pa.binary()))

        fields.extend([
            pa.field('u_start', pa.binary()),
            pa.field('u_newton', pa.binary()),
            pa.field('S_start', pa.binary()),
            pa.field('S_newton', pa.binary()),
        ])
        self._schema = pa.schema(fields)

        self._writer = pq.ParquetWriter(
            where=path,
            schema=self._schema,
            compression=compression,
            use_dictionary=True
        )

    def write_records(self, records: List[Dict[str, Any]]):
        if not records:
            return

        cols = {name: [] for name in self._schema.names}
        for r in records:
            for k in cols.keys():
                cols[k].append(r[k])

        arrays = []
        for name, field in zip(self._schema.names, self._schema):
            if field.type == pa.binary():
                arrays.append(pa.array(cols[name], type=pa.binary()))
            elif field.type == pa.int32():
                arrays.append(pa.array(cols[name], type=pa.int32()))
            elif field.type == pa.float64():
                arrays.append(pa.array(cols[name], type=pa.float64()))
            elif pa.types.is_string(field.type):
                arrays.append(pa.array(cols[name], type=pa.string()))
            else:
                arrays.append(pa.array(cols[name]))
        table = pa.Table.from_arrays(arrays, schema=self._schema)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

# ==========================
# Worker state
# ==========================
_CFG: Dict[str, Any] = {}
_RNG = None

def _init_worker(cfg: dict, seed_base: int):
    global _CFG, _RNG
    _CFG = cfg
    pid = os.getpid()
    ss = np.random.SeedSequence([seed_base & 0xFFFFFFFF, pid & 0xFFFFFFFF])
    _RNG = np.random.default_rng(ss)
    np.random.seed(int(ss.generate_state(1, dtype=np.uint32)[0]))

def _generate_one_record_serialized() -> Dict[str, Any]:
    K = _CFG['K']
    gridtype = _CFG['gridtype']
    fixed = _CFG['fixed']
    min_buses = _CFG['min_buses']
    max_buses = _CFG['max_buses']
    save_y_matrix = _CFG['save_y_matrix']
    preset = _CFG.get('preset', "")

    debugging = False
    pic = False

    if preset == "ieee14":
        # Force exact IEEE-14
        (gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
         Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
         vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total) = case_generation_ieee14()
        bus_number = 14
    elif preset =="ieee14_pandapower":
        (gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
         Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
         vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total, Y_shunt_bus) = case_generation_ieee14_pandapower(
            jitter_load=0.10,          # ±10% (Gaussian) on each load p/q
            jitter_gen=0.10,           # ±10% on each gen P
            pv_vset_range=(0.98, 1.04),# PV |V| in [0.98, 1.04] pu
            rand_u_start=True,         # randomize start |U|/angles
            angle_jitter_deg=5.0,      # ±5° on non-slack angles
            mag_jitter_pq=0.02,        # ±2% on PQ start magnitudes
            seed=None                  # or a fixed int for reproducibility
        )
        bus_number = 14
    else:
        # Original random generator path
        if fixed:
            exponents = np.array([2, 3, 4, 5])
            bus_number = int(2 ** np.random.choice(exponents))
        else:
            bus_number = int(np.random.randint(min_buses, max_buses + 1))

        (gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
         Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
         vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total) = case_generation(
            gridtype, bus_number, fixed, debugging, pic
        )

    ymat_bytes = ndarray_to_npy_bytes(Y_matrix.copy().flatten().astype(complex)) if save_y_matrix else None

    if not is_connected:
        u_newton = np.zeros_like(u_start, dtype=complex)
        S_newton = np.zeros_like(u_start, dtype=complex)
    else:
        u_newton, _I_unused, S_newton = newtonrapson(
            bus_typ, Y_matrix.copy(), s_multi.copy(), u_start.copy(), K=K
        )

    rec = {
        'bus_number': bus_number,
        'gridtype': gridtype_out,
        'U_base': float(U_base),
        'S_base': float(S_base),

        'bus_typ': ndarray_to_npy_bytes(bus_typ),
        'Y_Lines': ndarray_to_npy_bytes(Y_Lines),
        'Y_C_Lines': ndarray_to_npy_bytes(Y_C_Lines), #  Y_C_Lines = 0.5 * B_total_S
        'Lines_connected': ndarray_to_npy_bytes(Lines_connected.astype(np.int8)),

        'vn_kv': ndarray_to_npy_bytes(vn_kv), # length N (per-bus)
        'Is_trafo': ndarray_to_npy_bytes(Is_trafo.astype(np.int8)), # length C = N(N−1)/2 (0/1, len C): 1 if the pair (i,j) is a transformer branch, else 0.
        'Trafo_tau': ndarray_to_npy_bytes(Trafo_tau), # (float, len C): off-nominal tap magnitude τ for that pair. Non-transformer pairs keep 1.0.
        'Trafo_shift_deg': ndarray_to_npy_bytes(Trafo_shift_deg), # (float, len C): phase-shift angle (degrees) for that pair. IEEE-14 has 0 for all.
        'Trafo_y_series': ndarray_to_npy_bytes(Trafo_y_series), # (complex Siemens, len C): transformer series admittance y for that pair (0 for non-trafos).
        'Trafo_b_total': ndarray_to_npy_bytes(Trafo_b_total), # (float Siemens, len C): transformer total magnetizing susceptance b_{\text{total}} (0 in IEEE-14).

        # 🔥 NEW: per-bus shunt admittance in SI [S], shape (N,)
        'Y_shunt_bus': ndarray_to_npy_bytes(Y_shunt_bus),

        # transformer total magnetizing susceptance (not per-end).
        'u_start': ndarray_to_npy_bytes(u_start),
        'u_newton': ndarray_to_npy_bytes(u_newton),
        'S_start': ndarray_to_npy_bytes(s_multi),
        'S_newton': ndarray_to_npy_bytes(S_newton),
    }
    if save_y_matrix:
        rec['Y_matrix'] = ymat_bytes
    return rec

# in main_datagen_multiproc_improved.py
# near imports
import traceback, os
DBG = os.environ.get("GEN_DEBUG", "0") == "1"

def _generate_batch(n_rows: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    gen = _generate_one_record_serialized
    err_shown = 0
    for _ in range(n_rows):
        try:
            out.append(gen())
        except Exception as e:
            if DBG and err_shown < 3:
                print("[WORKER ERROR]", repr(e))
                traceback.print_exc()
                err_shown += 1
            # continue
            continue
    return out


# ==========================
# CLI and main loop
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Parallel dataset generation with multiprocessing (transformers + multi-voltage)")
    parser.add_argument('--preset', type=str, default="", choices=["", "ieee14","ieee14_pandapower"],
                        help='Optional fixed network preset (e.g., ieee14).')
    parser.add_argument('--K', type=int, default=40, help='newton_raphson steps')
    parser.add_argument('--runs', type=int, default=300000, help='Total samples.')
    parser.add_argument('--save_steps', type=int, default=20000, help='Rows per Parquet append.')
    parser.add_argument("--fixed", action="store_true", help="Use fixed topology (bus counts as powers of two).")
    parser.add_argument('--min_number_of_buses', type=int, default=4, help='Minimum buses.')
    parser.add_argument('--max_number_of_buses', type=int, default=512, help='Maximum buses.')
    parser.add_argument('--gridtype', type=str, default="MVN", help='HVN/MVN/LVN/HMVN.')
    parser.add_argument("--save_path", type=str, default="", help="Output directory.")
    parser.add_argument('--rows_per_task', type=int, default=4000, help='Rows per worker batch.')
    parser.add_argument('--workers', type=int, default=0, help='0 = all logical CPUs.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if exists.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save_y_matrix', dest='save_y_matrix', action='store_true',
                       help='Save Y_matrix bytes (default).')
    group.add_argument('--no_save_y_matrix', dest='save_y_matrix', action='store_false',
                       help='Do not save Y_matrix.')
    parser.set_defaults(save_y_matrix=True)
    return parser.parse_args()

def build_output_filename(args) -> str:
    """
    If a preset is chosen, use a compact name like:
        IEEE14_<runs>_NR_plain.parquet
    Else keep the original pattern:
        <gridtype>_<runs>_NR_plain_<min>_to_<max>_buses.parquet
    """
    if "ieee" in  str(args.preset).lower().strip():
        return os.path.join(args.save_path, f"{str(args.preset).lower().strip()}_{args.runs}_NR_plain.parquet")
    return os.path.join(
        args.save_path,
        f"{args.gridtype}_{args.runs}_NR_plain_{args.min_number_of_buses}_to_{args.max_number_of_buses}_buses.parquet"
    )

def get_parquet_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0

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
    save_y_matrix = bool(args.save_y_matrix)

    filename = build_output_filename(args)
    print(f"[INFO] Output file: {filename}")

    cfg = dict(
        K=K,
        gridtype=gridtype,
        fixed=fixed,
        min_buses=min_buses,
        max_buses=max_buses,
        save_y_matrix=save_y_matrix,
        preset=str(args.preset).lower().strip(),

    )

    num_tasks = math.ceil(runs / rows_per_task)
    task_sizes = [rows_per_task] * (num_tasks - 1) + [runs - rows_per_task * (num_tasks - 1)]

    if sys.platform.startswith("win"):
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")

    total_written = 0
    start = time.time()

    writer = ParquetAppendWriter(
        filename,
        compression="zstd",
        compression_level=3,
        overwrite=args.overwrite,
        save_y_matrix=save_y_matrix,
    )

    print(f"[INFO] Starting generation with {workers} workers, {num_tasks} tasks, rows_per_task={rows_per_task}, save_steps={save_steps}, save_y_matrix={save_y_matrix}")

    buffer: List[Dict[str, Any]] = []
    try:
        with ctx.Pool(processes=workers,
                      initializer=_init_worker,
                      initargs=(cfg, int(time.time())),
                      maxtasksperchild=1000) as pool:

            for i, batch in enumerate(pool.imap_unordered(_generate_batch, task_sizes, chunksize=1), 1):
                if batch:
                    buffer.extend(batch)

                if len(buffer) >= save_steps:
                    writer.write_records(buffer)
                    total_written += len(buffer)
                    buffer.clear()

                    elapsed = time.time() - start
                    rate = total_written / max(elapsed, 1e-6)
                    print(f"[INFO] {total_written:,}/{runs:,} rows written ({rate:,.0f} rows/s)")

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

if __name__ == "__main__":
    main()

# python main_datagen_multiproc_improved.py \
#   --preset ieee14 \
#   --runs 1 \
#   --workers 1 \
#   --rows_per_task 1 \
#   --save_steps 1 \
#   --overwrite

# python main_datagen_multiproc_improved.py --preset ieee14_pandapower --runs 1 --workers 1 --rows_per_task 1 --save_steps 1 --overwrite
