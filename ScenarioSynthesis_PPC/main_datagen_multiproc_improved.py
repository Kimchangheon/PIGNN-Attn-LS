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
import time
import math
import argparse
import traceback
import multiprocessing as mp
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandapower.networks as pn

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import (
    case_generation_pandapower,
    case_generation_pandapower_stamped,
)
from newton_raphson_improved import newtonrapson


# ============================================================
# Supported pandapower presets
# ============================================================

CASES = [
    "case4gs",
    "case5",
    "case6ww",
    "case9",
    "case14",
    "case24_ieee_rts",
    "case30",
    "case_ieee30",
    "case33bw",
    "case39",
    "case57",
    "case89pegase",
    "case118",
    "case145",
    "case_illinois200",
    "case300",
    "case1354pegase",
    "case1888rte",
    "case2848rte",
    "case2869pegase",
    "case3120sp",
    "case6470rte",
    "case6495rte",
    "case6515rte",
    "case9241pegase",
    "GBnetwork",
    "GBreducednetwork",
    "iceland",
]

# From your scan: these do not have net.trafo rows, so pfe_kw / i0_percent do nothing.
FORCE_SHUNT_CASES = {
    "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
}


# ============================================================
# Helpers: .npy <-> bytes
# ============================================================

def ndarray_to_npy_bytes(x: Any) -> bytes:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    buf = io.BytesIO()
    np.save(buf, x, allow_pickle=False)
    return buf.getvalue()


def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


# ============================================================
# Parquet writer for branch-row metadata
# ============================================================

class ParquetAppendWriter:
    """
    Branch-row schema:
      - one metadata row per PPC branch row (stored as arrays per sample)
      - no upper-triangular compression anymore
    """
    def __init__(
        self,
        path: str,
        compression: str = "zstd",
        overwrite: bool = True,
        save_y_matrix: bool = True,
    ):
        self.path = path
        self.save_y_matrix = bool(save_y_matrix)

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if overwrite and os.path.exists(path):
            os.remove(path)

        fields = [
            pa.field("bus_number", pa.int32()),
            pa.field("branch_number", pa.int32()),
            pa.field("gridtype", pa.string()),
            pa.field("U_base", pa.float64()),
            pa.field("S_base", pa.float64()),

            # per-bus arrays
            pa.field("bus_typ", pa.binary()),
            pa.field("vn_kv", pa.binary()),
            pa.field("Y_shunt_bus", pa.binary()),

            # branch-row arrays (length = nl)
            pa.field("Branch_f_bus", pa.binary()),
            pa.field("Branch_t_bus", pa.binary()),
            pa.field("Branch_status", pa.binary()),
            pa.field("Branch_tau", pa.binary()),
            pa.field("Branch_shift_deg", pa.binary()),
            pa.field("Branch_y_series_from", pa.binary()),
            pa.field("Branch_y_series_to", pa.binary()),
            pa.field("Branch_y_series_ft", pa.binary()),
            pa.field("Branch_y_shunt_from", pa.binary()),
            pa.field("Branch_y_shunt_to", pa.binary()),
            pa.field("Is_trafo", pa.binary()),
            pa.field("Branch_hv_is_f", pa.binary()),
            pa.field("Branch_n", pa.binary()),

            # informational line-only arrays (also length = nl)
            pa.field("Y_Lines", pa.binary()),
            pa.field("Y_C_Lines", pa.binary()),
        ]

        if self.save_y_matrix:
            fields.append(pa.field("Y_matrix", pa.binary()))

        fields.extend([
            pa.field("u_start", pa.binary()),
            pa.field("u_newton", pa.binary()),
            pa.field("S_start", pa.binary()),
            pa.field("S_newton", pa.binary()),
        ])

        self._schema = pa.schema(fields)
        self._writer = pq.ParquetWriter(
            where=path,
            schema=self._schema,
            compression=compression,
            use_dictionary=True,
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


# ============================================================
# Worker state
# ============================================================

_CFG: Dict[str, Any] = {}
_RNG = None
DBG = os.environ.get("GEN_DEBUG", "0") == "1"


def _init_worker(cfg: dict, seed_base: int):
    global _CFG, _RNG
    _CFG = cfg
    pid = os.getpid()
    ss = np.random.SeedSequence([seed_base & 0xFFFFFFFF, pid & 0xFFFFFFFF])
    _RNG = np.random.default_rng(ss)
    np.random.seed(int(ss.generate_state(1, dtype=np.uint32)[0]))


def _build_force_branch_shunt_from_cfg(preset: str, cfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Builds force_branch_shunt_pu dict only if requested.
    Recommended usage: only for cases with no net.trafo rows.
    """
    auto_force_if_no_trafo_case = bool(cfg.get("auto_force_if_no_trafo_case", False))
    g = float(cfg.get("force_branch_shunt_g_pu", 0.0))
    b = float(cfg.get("force_branch_shunt_b_pu", 0.0))
    g_asym = float(cfg.get("force_branch_shunt_g_asym_pu", 0.0))
    b_asym = float(cfg.get("force_branch_shunt_b_asym_pu", 0.0))

    anything_nonzero = any(abs(x) > 0 for x in [g, b, g_asym, b_asym])

    if anything_nonzero:
        return {"g": g, "b": b, "g_asym": g_asym, "b_asym": b_asym}

    if auto_force_if_no_trafo_case and preset in FORCE_SHUNT_CASES:
        # sensible default for “magnetizing-like” perturbation
        return {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

    return None


def _generate_one_record_serialized() -> Dict[str, Any]:
    global _CFG, _RNG

    K = int(_CFG["K"])
    preset = str(_CFG["preset"]).strip()
    ybus_mode = str(_CFG["ybus_mode"]).strip()
    save_y_matrix = bool(_CFG["save_y_matrix"])

    if not preset:
        raise ValueError("This adapted script expects a pandapower preset in --preset.")

    case_fn = getattr(pn, preset, None)
    if case_fn is None:
        raise ValueError(f"Unknown pandapower case preset: {preset}")

    force_branch_shunt_pu = _build_force_branch_shunt_from_cfg(preset, _CFG)

    sample_seed = int(_RNG.integers(0, 2**32 - 1, dtype=np.uint32))

    if ybus_mode.lower() == "stamped":
        out = case_generation_pandapower_stamped(
            case_fn=case_fn,
            case_kwargs={},
            seed=sample_seed,
            jitter_load=float(_CFG["jitter_load"]),
            jitter_gen=float(_CFG["jitter_gen"]),
            pv_vset_range=_CFG["pv_vset_range"],
            rand_u_start=bool(_CFG["rand_u_start"]),
            angle_jitter_deg=float(_CFG["angle_jitter_deg"]),
            mag_jitter_pq=float(_CFG["mag_jitter_pq"]),
            trafo_pfe_kw=_CFG["trafo_pfe_kw"],
            trafo_i0_percent=_CFG["trafo_i0_percent"],
            force_branch_shunt_pu=force_branch_shunt_pu,
        )
    else:
        out = case_generation_pandapower(
            case_fn=case_fn,
            case_kwargs={},
            ybus_mode="ppcY",
            seed=sample_seed,
            jitter_load=float(_CFG["jitter_load"]),
            jitter_gen=float(_CFG["jitter_gen"]),
            pv_vset_range=_CFG["pv_vset_range"],
            rand_u_start=bool(_CFG["rand_u_start"]),
            angle_jitter_deg=float(_CFG["angle_jitter_deg"]),
            mag_jitter_pq=float(_CFG["mag_jitter_pq"]),
            trafo_pfe_kw=_CFG["trafo_pfe_kw"],
            trafo_i0_percent=_CFG["trafo_i0_percent"],
            force_branch_shunt_pu=force_branch_shunt_pu,
        )

    (
        gridtype_out, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,
        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,
        Y_shunt_bus,
        Is_trafo, Branch_hv_is_f, Branch_n,
        Y_Lines, Y_C_Lines,
        U_base, S_base, vn_kv
    ) = out

    bus_number = int(len(bus_typ))
    branch_number = int(len(Branch_f_bus))

    ymat_bytes = ndarray_to_npy_bytes(Y_matrix.copy()) if save_y_matrix else None

    if not is_connected:
        u_newton = np.zeros_like(u_start, dtype=np.complex128)
        S_newton = np.zeros_like(s_multi, dtype=np.complex128)
    else:
        u_newton, _I_unused, S_newton = newtonrapson(
            bus_typ, Y_matrix.copy(), s_multi.copy(), u_start.copy(), K=K
        )

    rec = {
        "bus_number": bus_number,
        "branch_number": branch_number,
        "gridtype": gridtype_out,
        "U_base": float(U_base),
        "S_base": float(S_base),

        "bus_typ": ndarray_to_npy_bytes(bus_typ.astype(np.int32)),
        "vn_kv": ndarray_to_npy_bytes(vn_kv.astype(np.float64)),
        "Y_shunt_bus": ndarray_to_npy_bytes(Y_shunt_bus.astype(np.complex128)),

        "Branch_f_bus": ndarray_to_npy_bytes(Branch_f_bus.astype(np.int32)),
        "Branch_t_bus": ndarray_to_npy_bytes(Branch_t_bus.astype(np.int32)),
        "Branch_status": ndarray_to_npy_bytes(Branch_status.astype(np.int8)),
        "Branch_tau": ndarray_to_npy_bytes(Branch_tau.astype(np.float64)),
        "Branch_shift_deg": ndarray_to_npy_bytes(Branch_shift_deg.astype(np.float64)),
        "Branch_y_series_from": ndarray_to_npy_bytes(Branch_y_series_from.astype(np.complex128)),
        "Branch_y_series_to": ndarray_to_npy_bytes(Branch_y_series_to.astype(np.complex128)),
        "Branch_y_series_ft": ndarray_to_npy_bytes(Branch_y_series_ft.astype(np.complex128)),
        "Branch_y_shunt_from": ndarray_to_npy_bytes(Branch_y_shunt_from.astype(np.complex128)),
        "Branch_y_shunt_to": ndarray_to_npy_bytes(Branch_y_shunt_to.astype(np.complex128)),
        "Is_trafo": ndarray_to_npy_bytes(Is_trafo.astype(np.int8)),
        "Branch_hv_is_f": ndarray_to_npy_bytes(Branch_hv_is_f.astype(np.int8)),
        "Branch_n": ndarray_to_npy_bytes(Branch_n.astype(np.float64)),

        "Y_Lines": ndarray_to_npy_bytes(Y_Lines.astype(np.complex128)),
        "Y_C_Lines": ndarray_to_npy_bytes(Y_C_Lines.astype(np.float64)),

        "u_start": ndarray_to_npy_bytes(u_start.astype(np.complex128)),
        "u_newton": ndarray_to_npy_bytes(u_newton.astype(np.complex128)),
        "S_start": ndarray_to_npy_bytes(s_multi.astype(np.complex128)),
        "S_newton": ndarray_to_npy_bytes(S_newton.astype(np.complex128)),
    }

    if save_y_matrix:
        rec["Y_matrix"] = ymat_bytes

    return rec


def _generate_batch(n_rows: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    err_shown = 0

    for _ in range(n_rows):
        try:
            out.append(_generate_one_record_serialized())
        except Exception as e:
            if DBG and err_shown < 3:
                print("[WORKER ERROR]", repr(e))
                traceback.print_exc()
                err_shown += 1
            continue

    return out


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel dataset generation for pandapower test cases (branch-row direct-SI metadata)."
    )

    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        choices=CASES,
        help="Pandapower test case name.",
    )
    parser.add_argument(
        "--ybus_mode",
        type=str,
        default="ppcY",
        choices=["ppcY", "stamped"],
        help="Use pandapower ppcY directly or rebuild Y_matrix by stamping PPC.",
    )

    parser.add_argument("--K", type=int, default=40, help="newton_raphson steps")
    parser.add_argument("--runs", type=int, default=10000, help="Total samples")
    parser.add_argument("--save_steps", type=int, default=2000, help="Rows per Parquet append")
    parser.add_argument("--rows_per_task", type=int, default=1000, help="Rows per worker batch")
    parser.add_argument("--workers", type=int, default=0, help="0 = all logical CPUs")
    parser.add_argument("--save_path", type=str, default="", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save_y_matrix", dest="save_y_matrix", action="store_true", help="Save Y_matrix (default)")
    group.add_argument("--no_save_y_matrix", dest="save_y_matrix", action="store_false", help="Do not save Y_matrix")
    parser.set_defaults(save_y_matrix=True)

    # Optional perturbations
    parser.add_argument("--jitter_load", type=float, default=0.0)
    parser.add_argument("--jitter_gen", type=float, default=0.0)
    parser.add_argument("--pv_vset_lo", type=float, default=None)
    parser.add_argument("--pv_vset_hi", type=float, default=None)
    parser.add_argument("--rand_u_start", action="store_true")
    parser.add_argument("--angle_jitter_deg", type=float, default=5.0)
    parser.add_argument("--mag_jitter_pq", type=float, default=0.02)

    # Optional trafo magnetizing via net.trafo
    parser.add_argument("--trafo_pfe_kw", type=float, default=None)
    parser.add_argument("--trafo_i0_percent", type=float, default=None)

    # Optional PPC forcing (best for cases with no net.trafo)
    parser.add_argument("--auto_force_if_no_trafo_case", action="store_true",
                        help="For known no-trafo cases, apply a default forced branch shunt (b=0.2 pu).")
    parser.add_argument("--force_branch_shunt_g_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_b_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_g_asym_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_b_asym_pu", type=float, default=0.0)

    return parser.parse_args()


def build_output_filename(args) -> str:
    mode = str(args.ybus_mode).strip()
    name = f"{args.preset}_{mode}_{args.runs}_NR_branchrows_directSI.parquet"
    return os.path.join(args.save_path, name)


def get_parquet_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    pv_vset_range = None
    if args.pv_vset_lo is not None and args.pv_vset_hi is not None:
        pv_vset_range = (float(args.pv_vset_lo), float(args.pv_vset_hi))

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    runs = int(args.runs)
    rows_per_task = max(int(args.rows_per_task), 1)
    save_steps = max(int(args.save_steps), 1)
    save_y_matrix = bool(args.save_y_matrix)

    filename = build_output_filename(args)
    print(f"[INFO] Output file: {filename}")

    cfg = dict(
        preset=str(args.preset).strip(),
        ybus_mode=str(args.ybus_mode).strip(),
        K=int(args.K),
        save_y_matrix=save_y_matrix,

        jitter_load=float(args.jitter_load),
        jitter_gen=float(args.jitter_gen),
        pv_vset_range=pv_vset_range,
        rand_u_start=bool(args.rand_u_start),
        angle_jitter_deg=float(args.angle_jitter_deg),
        mag_jitter_pq=float(args.mag_jitter_pq),

        trafo_pfe_kw=args.trafo_pfe_kw,
        trafo_i0_percent=args.trafo_i0_percent,

        auto_force_if_no_trafo_case=bool(args.auto_force_if_no_trafo_case),
        force_branch_shunt_g_pu=float(args.force_branch_shunt_g_pu),
        force_branch_shunt_b_pu=float(args.force_branch_shunt_b_pu),
        force_branch_shunt_g_asym_pu=float(args.force_branch_shunt_g_asym_pu),
        force_branch_shunt_b_asym_pu=float(args.force_branch_shunt_b_asym_pu),
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
        overwrite=args.overwrite,
        save_y_matrix=save_y_matrix,
    )

    print(
        f"[INFO] Starting generation with {workers} workers, "
        f"{num_tasks} tasks, rows_per_task={rows_per_task}, "
        f"save_steps={save_steps}, save_y_matrix={save_y_matrix}"
    )

    buffer: List[Dict[str, Any]] = []

    try:
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(cfg, int(time.time())),
            maxtasksperchild=1000
        ) as pool:

            for batch in pool.imap_unordered(_generate_batch, task_sizes, chunksize=1):
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