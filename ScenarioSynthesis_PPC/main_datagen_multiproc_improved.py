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

FORCE_SHUNT_CASES = {
    "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
}

CASE_KWARGS = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case2848rte": {"ref_bus_idx": 271},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6495rte": {"ref_bus_idx": None},
    # "case6515rte": {"ref_bus_idx": 6171},
}

SCENARIO_PRESETS = {
    "easy": dict(
        jitter_load=0.02,
        jitter_gen=0.01,
        pv_vset_range=(0.995, 1.005),
        rand_u_start=False,
        angle_jitter_deg=0.5,
        mag_jitter_pq=0.002,
    ),
    "no_change": dict(
        jitter_load=0.0,
        jitter_gen=0.0,
        pv_vset_range=(1.0, 1.0),
        rand_u_start=False,
        angle_jitter_deg=0.0,
        mag_jitter_pq=0.0,
    ),
    "A": dict(
        jitter_load=0.05,
        jitter_gen=0.03,
        pv_vset_range=(0.99, 1.02),
        rand_u_start=True,
        angle_jitter_deg=3.0,
        mag_jitter_pq=0.01,
    ),
    "B": dict(
        jitter_load=0.10,
        jitter_gen=0.05,
        pv_vset_range=(0.98, 1.03),
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
    ),
    "C": dict(
        jitter_load=0.15,
        jitter_gen=0.08,
        pv_vset_range=(0.97, 1.04),
        rand_u_start=True,
        angle_jitter_deg=7.0,
        mag_jitter_pq=0.03,
    ),
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


# ============================================================
# PU/SI conversion helpers for NR
# ============================================================

def ybus_si_to_pu(Y_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float) -> np.ndarray:
    scale = np.outer(Vbase_bus, Vbase_bus) / float(S_base)
    return np.asarray(Y_si, dtype=np.complex128) * scale.astype(np.float64, copy=False)


def u_si_to_pu_per_bus(u_si: np.ndarray, Vbase_bus: np.ndarray) -> np.ndarray:
    return np.asarray(u_si, dtype=np.complex128) / np.asarray(Vbase_bus, dtype=np.float64)


def s_si_to_pu(S_si: np.ndarray, S_base: float) -> np.ndarray:
    return np.asarray(S_si, dtype=np.complex128) / float(S_base)


def u_pu_to_si_per_bus(u_pu: np.ndarray, Vbase_bus: np.ndarray) -> np.ndarray:
    return np.asarray(u_pu, dtype=np.complex128) * np.asarray(Vbase_bus, dtype=np.float64)


def s_pu_to_si(S_pu: np.ndarray, S_base: float) -> np.ndarray:
    return np.asarray(S_pu, dtype=np.complex128) * float(S_base)


def convert_nr_inputs_to_pu(
    Y_matrix_si: np.ndarray,
    s_multi_si: np.ndarray,
    u_start_si: np.ndarray,
    vn_kv: np.ndarray,
    S_base: float,
):
    Vbase_bus = np.asarray(vn_kv, dtype=np.float64) * 1e3

    Y_pu = ybus_si_to_pu(np.asarray(Y_matrix_si, dtype=np.complex128), Vbase_bus, S_base)
    s_pu = s_si_to_pu(np.asarray(s_multi_si, dtype=np.complex128), S_base)
    u_pu = u_si_to_pu_per_bus(np.asarray(u_start_si, dtype=np.complex128), Vbase_bus)

    return Y_pu, s_pu, u_pu, Vbase_bus


# ============================================================
# Parquet writer
# ============================================================

class ParquetAppendWriter:
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

            pa.field("bus_typ", pa.binary()),
            pa.field("vn_kv", pa.binary()),
            pa.field("Y_shunt_bus", pa.binary()),

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
    use_force_shunt_when_no_trafo = bool(cfg.get("use_force_shunt_when_no_trafo", False))
    g = float(cfg.get("force_branch_shunt_g_pu", 0.0))
    b = float(cfg.get("force_branch_shunt_b_pu", 0.0))
    g_asym = float(cfg.get("force_branch_shunt_g_asym_pu", 0.0))
    b_asym = float(cfg.get("force_branch_shunt_b_asym_pu", 0.0))

    anything_nonzero = any(abs(x) > 0 for x in [g, b, g_asym, b_asym])
    if anything_nonzero:
        return {"g": g, "b": b, "g_asym": g_asym, "b_asym": b_asym}

    if use_force_shunt_when_no_trafo and preset in FORCE_SHUNT_CASES:
        return {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

    return None


def _generate_one_record_serialized() -> Dict[str, Any]:
    global _CFG, _RNG

    K = int(_CFG["K"])
    preset = str(_CFG["preset"]).strip()
    ybus_mode = str(_CFG["ybus_mode"]).strip()
    save_y_matrix = bool(_CFG["save_y_matrix"])
    pu_nr = bool(_CFG["pu_nr"])
    start_mode = str(_CFG["start_mode"]).strip()

    if not preset:
        raise ValueError("This adapted script expects a pandapower preset in --preset.")

    case_fn = getattr(pn, preset, None)
    if case_fn is None:
        raise ValueError(f"Unknown pandapower case preset: {preset}")

    force_branch_shunt_pu = _build_force_branch_shunt_from_cfg(preset, _CFG)
    sample_seed = int(_RNG.integers(0, 2**32 - 1, dtype=np.uint32))

    gen_kwargs = dict(
        case_fn=case_fn,
        case_kwargs=CASE_KWARGS.get(preset, {}).copy(),
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
        start_mode=start_mode,
    )

    if ybus_mode.lower() == "stamped":
        out = case_generation_pandapower_stamped(**gen_kwargs)
    else:
        out = case_generation_pandapower(ybus_mode="ppcY", **gen_kwargs)

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

    ymat_bytes = ndarray_to_npy_bytes(np.asarray(Y_matrix, dtype=np.complex128).copy()) if save_y_matrix else None

    if not is_connected:
        u_newton_si = np.zeros_like(u_start, dtype=np.complex128)
        S_newton_si = np.zeros_like(s_multi, dtype=np.complex128)
    else:
        bus_typ_arr = np.asarray(bus_typ, dtype=np.int64).copy()

        if pu_nr:
            Y_for_nr, S_for_nr, U_for_nr, Vbase_bus = convert_nr_inputs_to_pu(
                Y_matrix_si=np.asarray(Y_matrix, dtype=np.complex128),
                s_multi_si=np.asarray(s_multi, dtype=np.complex128),
                u_start_si=np.asarray(u_start, dtype=np.complex128),
                vn_kv=np.asarray(vn_kv, dtype=np.float64),
                S_base=float(S_base),
            )
        else:
            Y_for_nr = np.asarray(Y_matrix, dtype=np.complex128).copy()
            S_for_nr = np.asarray(s_multi, dtype=np.complex128).copy()
            U_for_nr = np.asarray(u_start, dtype=np.complex128).copy()
            Vbase_bus = None

        nr_out = newtonrapson(
            bus_typ_arr,
            Y_for_nr,
            S_for_nr,
            U_for_nr,
            K=K,
            diagnose=bool(_CFG["diagnose_nr"]),
            print_misinf=bool(_CFG["print_misinf"]),
            return_diagnostics=True,
            near_misinf_tol=float(_CFG["near_misinf_tol"]),
            convergence_mode=str(_CFG["convergence_mode"]),
            step_tol=float(_CFG["step_tol"]),
            mismatch_tol=float(_CFG["mismatch_tol"]),
        )

        u_newton_raw, _I_unused, S_newton_raw, _nr_diag = nr_out

        if pu_nr:
            u_newton_arr = np.asarray(u_newton_raw)
            s_newton_arr = np.asarray(S_newton_raw)

            if u_newton_arr.size == 0:
                u_newton_si = np.asarray(u_newton_arr, dtype=np.complex128)
            else:
                u_newton_si = u_pu_to_si_per_bus(u_newton_arr, Vbase_bus)

            if s_newton_arr.size == 0:
                S_newton_si = np.asarray(s_newton_arr, dtype=np.complex128)
            else:
                S_newton_si = s_pu_to_si(s_newton_arr, float(S_base))
        else:
            u_newton_si = np.asarray(u_newton_raw, dtype=np.complex128)
            S_newton_si = np.asarray(S_newton_raw, dtype=np.complex128)

    rec = {
        "bus_number": bus_number,
        "branch_number": branch_number,
        "gridtype": gridtype_out,
        "U_base": float(U_base),
        "S_base": float(S_base),

        "bus_typ": ndarray_to_npy_bytes(np.asarray(bus_typ, dtype=np.int32)),
        "vn_kv": ndarray_to_npy_bytes(np.asarray(vn_kv, dtype=np.float64)),
        "Y_shunt_bus": ndarray_to_npy_bytes(np.asarray(Y_shunt_bus, dtype=np.complex128)),

        "Branch_f_bus": ndarray_to_npy_bytes(np.asarray(Branch_f_bus, dtype=np.int32)),
        "Branch_t_bus": ndarray_to_npy_bytes(np.asarray(Branch_t_bus, dtype=np.int32)),
        "Branch_status": ndarray_to_npy_bytes(np.asarray(Branch_status, dtype=np.int8)),
        "Branch_tau": ndarray_to_npy_bytes(np.asarray(Branch_tau, dtype=np.float64)),
        "Branch_shift_deg": ndarray_to_npy_bytes(np.asarray(Branch_shift_deg, dtype=np.float64)),
        "Branch_y_series_from": ndarray_to_npy_bytes(np.asarray(Branch_y_series_from, dtype=np.complex128)),
        "Branch_y_series_to": ndarray_to_npy_bytes(np.asarray(Branch_y_series_to, dtype=np.complex128)),
        "Branch_y_series_ft": ndarray_to_npy_bytes(np.asarray(Branch_y_series_ft, dtype=np.complex128)),
        "Branch_y_shunt_from": ndarray_to_npy_bytes(np.asarray(Branch_y_shunt_from, dtype=np.complex128)),
        "Branch_y_shunt_to": ndarray_to_npy_bytes(np.asarray(Branch_y_shunt_to, dtype=np.complex128)),
        "Is_trafo": ndarray_to_npy_bytes(np.asarray(Is_trafo, dtype=np.int8)),
        "Branch_hv_is_f": ndarray_to_npy_bytes(np.asarray(Branch_hv_is_f, dtype=np.int8)),
        "Branch_n": ndarray_to_npy_bytes(np.asarray(Branch_n, dtype=np.float64)),

        "Y_Lines": ndarray_to_npy_bytes(np.asarray(Y_Lines, dtype=np.complex128)),
        "Y_C_Lines": ndarray_to_npy_bytes(np.asarray(Y_C_Lines, dtype=np.float64)),

        "u_start": ndarray_to_npy_bytes(np.asarray(u_start, dtype=np.complex128)),
        "u_newton": ndarray_to_npy_bytes(np.asarray(u_newton_si, dtype=np.complex128)),
        "S_start": ndarray_to_npy_bytes(np.asarray(s_multi, dtype=np.complex128)),
        "S_newton": ndarray_to_npy_bytes(np.asarray(S_newton_si, dtype=np.complex128)),
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

    parser.add_argument("--K", type=int, default=40, help="Newton-Raphson max iterations")
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

    parser.add_argument(
        "--scenario_level",
        type=str,
        default="no_change",
        choices=list(SCENARIO_PRESETS.keys()),
        help="Scenario preset for perturbations.",
    )

    parser.add_argument(
        "--use_force_shunt_when_no_trafo",
        action="store_true",
        help="For known no-trafo cases, apply default forced branch shunt (b=0.2 pu).",
    )
    parser.add_argument("--force_branch_shunt_g_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_b_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_g_asym_pu", type=float, default=0.0)
    parser.add_argument("--force_branch_shunt_b_asym_pu", type=float, default=0.0)

    parser.add_argument("--trafo_pfe_kw", type=float, default=None)
    parser.add_argument("--trafo_i0_percent", type=float, default=None)

    parser.add_argument("--pu_nr", action="store_true", help="Solve NR in per-unit, save results back in SI.")
    parser.add_argument("--diagnose_nr", action="store_true", help="Enable NR diagnostics.")
    parser.add_argument("--print_misinf", action="store_true", help="Print NR mismatch per iteration.")
    parser.add_argument("--near_misinf_tol", type=float, default=1e-3)
    parser.add_argument(
        "--start_mode",
        type=str,
        default="auto",
        choices=["auto", "manual_flat", "ppc_v0", "dc_compile"],
        help="Voltage start mode passed to case_generation_pandapower.",
    )

    parser.add_argument(
        "--convergence_mode",
        type=str,
        default="misinf",
        choices=["two_step", "misinf"],
        help="NR convergence criterion.",
    )
    parser.add_argument("--step_tol", type=float, default=5e-4)
    parser.add_argument("--mismatch_tol", type=float, default=1e-8)

    return parser.parse_args()


def build_output_filename(args) -> str:
    mode = str(args.ybus_mode).strip()
    nr_unit = "puNR" if args.pu_nr else "siNR"
    name = (
        f"{args.preset}_{mode}_{args.scenario_level}_{args.start_mode}_"
        f"{nr_unit}_{args.runs}_NR_branchrows_directSI.parquet"
    )
    return os.path.join(args.save_path, name)


def get_parquet_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    scenario_cfg = SCENARIO_PRESETS[args.scenario_level]

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

        jitter_load=float(scenario_cfg["jitter_load"]),
        jitter_gen=float(scenario_cfg["jitter_gen"]),
        pv_vset_range=scenario_cfg["pv_vset_range"],
        rand_u_start=bool(scenario_cfg["rand_u_start"]),
        angle_jitter_deg=float(scenario_cfg["angle_jitter_deg"]),
        mag_jitter_pq=float(scenario_cfg["mag_jitter_pq"]),

        trafo_pfe_kw=args.trafo_pfe_kw,
        trafo_i0_percent=args.trafo_i0_percent,

        use_force_shunt_when_no_trafo=bool(args.use_force_shunt_when_no_trafo),
        force_branch_shunt_g_pu=float(args.force_branch_shunt_g_pu),
        force_branch_shunt_b_pu=float(args.force_branch_shunt_b_pu),
        force_branch_shunt_g_asym_pu=float(args.force_branch_shunt_g_asym_pu),
        force_branch_shunt_b_asym_pu=float(args.force_branch_shunt_b_asym_pu),

        pu_nr=bool(args.pu_nr),
        diagnose_nr=bool(args.diagnose_nr),
        print_misinf=bool(args.print_misinf),
        near_misinf_tol=float(args.near_misinf_tol),
        start_mode=str(args.start_mode).strip(),

        convergence_mode=str(args.convergence_mode).strip(),
        step_tol=float(args.step_tol),
        mismatch_tol=float(args.mismatch_tol),
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

    print("[INFO] Configuration")
    print(f"  preset                    = {args.preset}")
    print(f"  ybus_mode                 = {args.ybus_mode}")
    print(f"  scenario_level            = {args.scenario_level}")
    print(f"  scenario_cfg              = {scenario_cfg}")
    print(f"  K                         = {args.K}")
    print(f"  pu_nr                     = {args.pu_nr}")
    print(f"  start_mode                = {args.start_mode}")
    print(f"  use_force_shunt_when_no_trafo = {args.use_force_shunt_when_no_trafo}")
    print(f"  diagnose_nr               = {args.diagnose_nr}")
    print(f"  convergence_mode          = {args.convergence_mode}")
    print(f"  step_tol                  = {args.step_tol}")
    print(f"  mismatch_tol              = {args.mismatch_tol}")
    print(f"  workers                   = {workers}")
    print(f"  rows_per_task             = {rows_per_task}")
    print(f"  save_steps                = {save_steps}")
    print(f"  save_y_matrix             = {save_y_matrix}")

    if args.print_misinf and workers != 1:
        print("[WARN] print_misinf=True with workers>1 will produce interleaved logs.")

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