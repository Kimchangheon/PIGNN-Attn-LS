"""Convert LVN branch-list parquet to HVN-compatible upper-triangular-pair schema.

The LVN dataset (e.g. LVN_snapshot_envelope_..._branchrows_directSI.parquet)
stores admittances per-branch with Branch_f_bus / Branch_t_bus endpoints,
and bus voltages span multiple voltage levels (110kV/20kV/6kV in the same
row). The existing ChanghunDataset pipeline expects:
  - Lines_connected: bool mask over the N*(N-1)/2 canonical upper-triangular
    pairs (j<i).
  - Y_Lines, Y_C_Lines: complex arrays at those canonical positions.
  - bus_typ in {0: PQ, 1: slack, 2: PV} (PandaPower → HVN remap).
  - Per-unit voltages that come out near 1.0 (clamp range is [0.75, 1.20]).

This converter:
  1. Decodes binary columns.
  2. Builds Lines_connected from active branches (status==1), summing
     admittances of parallel branches sharing the same (min, max) endpoints.
  3. Reorders Y_Lines (complex per-unit), Y_C_Lines (treats float Y_C as
     imaginary; real=0) into canonical-pair positions.
  4. Per-unit normalises voltages by per-bus vn_kv (NOT global U_base) so
     buses at any voltage level come out near 1.0 p.u.
  5. Per-units series + shunt admittances using the from-bus's base
     (V_base^2 / S_base).
  6. Remaps bus_typ {1→0 PQ, 2→2 PV, 3→1 slack}.
  7. Re-encodes columns as .npy bytes matching HVN format.

Output: a parquet file readable by ChanghunDataset without modification.
"""

from __future__ import annotations

import argparse
import gc
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_loading.npy_decode import decode_columns_mp_columnwise


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


# LVN PandaPower → HVN convention.
# The LVN snapshot uses INVERTED slack/PQ encoding (verified empirically: each
# 722-bus grid has 1 type-1, 4 type-2, 717 type-3 buses — the type-3 majority
# can only be PQ loads, the singleton type-1 can only be slack).
# Convention: LVN 1=slack, 2=PV, 3=PQ → HVN 0=PQ, 1=slack, 2=PV
LVN_TO_HVN_BUS_TYPE = {1: 1, 2: 2, 3: 0}  # 1=slack→1, 2=PV→2, 3=PQ→0

# System per-unit base. Source LVN uses S_base = 1 MVA, which makes S_pu
# values O(1000-500,000) and KCL residuals huge in absolute terms (~370k).
# Re-base to 100 MVA (matches HVN convention) so phys-loss readouts are
# comparable across datasets. Y is scaled by (source_S_base / TARGET_S_BASE)
# so the power-flow equation S = V·conj(Y·V) is preserved — V_newton stays
# valid, only the units change.
TARGET_S_BASE = 100e6  # 100 MVA (standard system base)


def convert_row(row: pd.Series) -> dict:
    N = int(row.bus_number)
    vn_kv = np.asarray(row.vn_kv, dtype=np.float64)
    bus_typ_lvn = np.asarray(row.bus_typ, dtype=np.int32)
    f = np.asarray(row.Branch_f_bus, dtype=np.int64)
    t = np.asarray(row.Branch_t_bus, dtype=np.int64)
    status = np.asarray(row.Branch_status, dtype=np.int8)
    Y_series = np.asarray(row.Y_Lines, dtype=np.complex128)
    Y_shunt = np.asarray(row.Y_C_Lines, dtype=np.float64)  # imaginary part of Yc
    u_start = np.asarray(row.u_start, dtype=np.complex128)
    u_newton = np.asarray(row.u_newton, dtype=np.complex128)
    S_start = np.asarray(row.S_start, dtype=np.complex128)
    S_newton = np.asarray(row['S_newton'], dtype=np.complex128)
    S_base = float(row.S_base)

    # --- Per-unit normalisation ---
    # Voltages: per-bus by vn_kv (LVN is multi-voltage; a global U_base
    # produces voltages spanning many orders of magnitude).
    # S: re-based to TARGET_S_BASE (100 MVA) so per-unit values are O(1-100)
    #    instead of O(1000-500,000) — same scale as HVN.
    # Y_Lines: rescaled by (source S_base / target S_base) so the power-flow
    #    equation S = V·conj(Y·V) is preserved. (The source Y is in PU on the
    #    source S_base; under a base change Y_pu_new = Y_pu_old * S_old/S_new.)
    V_base = vn_kv * 1000.0
    s_scale = S_base / TARGET_S_BASE  # e.g. 1e6 / 1e8 = 0.01 for LVN
    u_start_pu = (u_start / V_base).astype(np.complex64)
    u_newton_pu = (u_newton / V_base).astype(np.complex64)
    S_start_pu  = (S_start  / TARGET_S_BASE).astype(np.complex64)
    S_newton_pu = (S_newton / TARGET_S_BASE).astype(np.complex64)
    Y_series_pu = (Y_series * s_scale).astype(np.complex64)
    Y_shunt_pu = (Y_shunt * s_scale).astype(np.float64)  # PU susceptance (treated as imaginary in encoder)

    # --- Remap bus_typ ---
    bus_typ_hvn = np.zeros_like(bus_typ_lvn, dtype=np.int32)
    bus_typ_hvn[bus_typ_lvn == 1] = LVN_TO_HVN_BUS_TYPE[1]
    bus_typ_hvn[bus_typ_lvn == 2] = LVN_TO_HVN_BUS_TYPE[2]
    bus_typ_hvn[bus_typ_lvn == 3] = LVN_TO_HVN_BUS_TYPE[3]

    # --- Build SPARSE active-edge triplets (canonical pair idx + Y values) ---
    # Dense storage of Y_Lines / Y_C_Lines / Lines_connected as 260,281-element
    # arrays expanded to ~160 GB in RAM when pd.read_parquet decompresses the
    # full 36k-row file (each row is 4.5 MB of bytes; snappy gets us 500x on
    # disk but pandas materializes uncompressed Python bytes objects in RAM).
    # Sparse triplets: per row ~796 active edges -> ~16 KB per row, 36k rows
    # -> ~600 MB in RAM. Dataset class reconstructs the dense arrays in
    # __getitem__ as temporary tensors (per-batch peak only).

    active = (status == 1) & (f != t)
    f_act = f[active]
    t_act = t[active]
    Y_s_act = Y_series_pu[active].astype(np.complex64)
    Y_c_act = (1j * Y_shunt_pu[active]).astype(np.complex64)
    jj = np.minimum(f_act, t_act)
    ii = np.maximum(f_act, t_act)
    # Canonical pair index for triu(N, k=1) row-major ordering:
    # k = j*N - j*(j+1)//2 + (i - j - 1)
    pair_idx_per_branch = (jj * N - jj * (jj + 1) // 2 + (ii - jj - 1)).astype(np.int64)

    # Sum parallels: group-by canonical pair via np.add.at on a scratch dict.
    # Use unique pair indices + sum (handles duplicates from parallel branches).
    uniq_idx, inverse = np.unique(pair_idx_per_branch, return_inverse=True)
    M = uniq_idx.shape[0]
    Y_s_sum = np.zeros(M, dtype=np.complex64)
    Y_c_sum = np.zeros(M, dtype=np.complex64)
    np.add.at(Y_s_sum, inverse, Y_s_act)
    np.add.at(Y_c_sum, inverse, Y_c_act)
    active_pair_idx = uniq_idx.astype(np.int32)  # int32 sufficient for N<=46340

    # vn_log: per-bus voltage class as log10(vn_kv) — a scalar in roughly
    # [0.5, 2.6] across 3-380 kV. Provides the model with explicit
    # voltage-class info that's otherwise lost by per-bus per-unit
    # normalisation (every bus looks like ~1.0 p.u. regardless of class).
    vn_log = np.log10(vn_kv).astype(np.float32)

    return {
        'bus_number': N,
        'gridtype': 'LVN_converted_to_HVN_schema_sparse',
        'U_base': 1.0,
        'S_base': 1.0,
        'bus_typ': npy_bytes(bus_typ_hvn),
        # Sparse triplets (active edges only).
        'active_pair_idx': npy_bytes(active_pair_idx),
        'active_Y_series': npy_bytes(Y_s_sum),
        'active_Y_shunt': npy_bytes(Y_c_sum),
        'u_start': npy_bytes(u_start_pu),
        'u_newton': npy_bytes(u_newton_pu),
        'S_start':  npy_bytes(S_start_pu),
        'S_newton': npy_bytes(S_newton_pu),   # real NR solution power (rebased to 100 MVA)
        'I_newton': npy_bytes(np.zeros_like(S_start_pu)),
        'vn_log': npy_bytes(vn_log),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-parquet', required=True)
    ap.add_argument('--out-parquet', required=True)
    ap.add_argument('--limit', type=int, default=None,
                    help='Number of rows to convert (default: all).')
    ap.add_argument('--batch-size', type=int, default=200,
                    help='Per-batch decode/convert size (memory tradeoff). Each '
                         'in-flight row holds ~4.5MB of dense Y arrays before '
                         'compression, so peak RAM ~ batch_size * 4.5MB.')
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    out_path = Path(args.out_parquet)

    print(f'Loading {in_path} ...', flush=True)
    cols = ['bus_number', 'branch_number', 'gridtype', 'U_base', 'S_base',
            'vn_kv', 'bus_typ', 'Branch_f_bus', 'Branch_t_bus', 'Branch_status',
            'Y_Lines', 'Y_C_Lines', 'u_start', 'u_newton', 'S_start', 'S_newton']
    df_full = pd.read_parquet(in_path, columns=cols)
    if args.limit is not None:
        df_full = df_full.iloc[:args.limit].reset_index(drop=True)
    n = len(df_full)
    print(f'  {n} rows', flush=True)

    bcols = ['vn_kv', 'bus_typ', 'Branch_f_bus', 'Branch_t_bus',
             'Branch_status', 'Y_Lines', 'Y_C_Lines',
             'u_start', 'u_newton', 'S_start', 'S_newton']

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # STREAMING output: ParquetWriter holds at most one row group in memory.
    # Each converted row carries ~4.5 MB of dense Y_Lines/Y_C_Lines arrays
    # (260,281 complex64 each, mostly zero). Accumulating all 36000 in a
    # Python list before writing peaks at ~160 GB RAM and OOM-kills the
    # process. Writing per-chunk keeps peak RAM bounded to ~ batch_size *
    # 4.5 MB.
    writer = None
    n_written = 0
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        chunk = df_full.iloc[start:end].copy()
        chunk = decode_columns_mp_columnwise(chunk, bcols)
        chunk_rows = [convert_row(row) for _, row in chunk.iterrows()]
        chunk_df = pd.DataFrame(chunk_rows)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression='snappy')
        writer.write_table(table)
        n_written += len(chunk_rows)
        # Drop refs and force GC so peak RAM doesn't drift up over many chunks.
        del chunk_rows, chunk_df, table, chunk
        gc.collect()
        elapsed = time.time() - t0
        eta = elapsed / (end / n) - elapsed if end < n else 0
        print(f'  [{end}/{n}] {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining', flush=True)

    if writer is not None:
        writer.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f'Wrote {n_written} rows to {out_path}, {size_mb:.1f} MB', flush=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
