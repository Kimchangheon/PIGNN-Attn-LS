import io
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from read_npy_columns_optimized import decode_columns_mp_columnwise, npy_bytes_to_ndarray

FLOAT_DTYPE = np.float32
COMPLEX_DTYPE = np.complex64


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _as_np(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _all_zero_complex_fast(cell) -> bool:
    a = np.asarray(cell)
    if a.size == 0:
        return True
    if np.iscomplexobj(a):
        return bool(np.all((a.real == 0) & (a.imag == 0)))
    return bool(np.all(a == 0))


def reconstruct_Y_pandapower_branchrows_direct_SI_np(
    N: int,
    Branch_f_bus: np.ndarray,
    Branch_t_bus: np.ndarray,
    Branch_status: np.ndarray,
    Branch_tau: np.ndarray,
    Branch_shift_deg: np.ndarray,
    Branch_y_series_from: np.ndarray,
    Branch_y_series_to: np.ndarray,
    Branch_y_series_ft: np.ndarray,
    Branch_y_shunt_from: np.ndarray,
    Branch_y_shunt_to: np.ndarray,
    Y_shunt_bus: np.ndarray,
    Vbase_bus: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct dense Ybus in SI from one metadata row per PPC branch row.
    """
    Y = np.zeros((N, N), dtype=np.complex128)
    Y[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    nl = len(Branch_f_bus)
    for k in range(nl):
        if int(Branch_status[k]) == 0:
            continue

        f = int(Branch_f_bus[k])
        t = int(Branch_t_bus[k])

        tau = float(Branch_tau[k])
        theta = np.deg2rad(float(Branch_shift_deg[k]))
        a = tau * np.exp(1j * theta)

        y_from = complex(Branch_y_series_from[k])
        y_to   = complex(Branch_y_series_to[k])
        y_ft   = complex(Branch_y_series_ft[k])
        y_tf   = y_ft
        ysh_f  = complex(Branch_y_shunt_from[k])
        ysh_t  = complex(Branch_y_shunt_to[k])

        if Vbase_bus is not None:
            Vf = float(Vbase_bus[f])
            Vt = float(Vbase_bus[t])
            if Vf > 0.0 and Vt > 0.0:
                y_ft = y_from * (Vf / Vt)
                y_tf = y_to * (Vt / Vf)

        Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
        Ytt = (y_to   + ysh_t / 2.0)
        Yft = -y_ft / np.conj(a)
        Ytf = -y_tf / a

        Y[f, f] += Yff
        Y[t, t] += Ytt
        Y[f, t] += Yft
        Y[t, f] += Ytf

    return Y.astype(COMPLEX_DTYPE, copy=False)


def ybus_si_to_pu(Y_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float) -> np.ndarray:
    """
    Entrywise Ybus SI -> pu with local bus bases:
        Y_pu[i,j] = Y_SI[i,j] * V_i * V_j / S_base
    """
    scale = np.outer(Vbase_bus, Vbase_bus) / float(S_base)
    return (Y_si * scale).astype(COMPLEX_DTYPE, copy=False)


def ysh_bus_si_to_pu(Ysh_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float) -> np.ndarray:
    """
    Bus shunt SI -> pu:
        Ysh_pu[i] = Ysh_SI[i] * V_i^2 / S_base
    """
    scale = (Vbase_bus ** 2) / float(S_base)
    return (Ysh_si * scale).astype(COMPLEX_DTYPE, copy=False)


def u_si_to_pu_per_bus(u_si: np.ndarray, Vbase_bus: np.ndarray) -> np.ndarray:
    """
    Voltage SI -> pu per bus:
        U_pu[i] = U_SI[i] / V_i
    """
    return (u_si / Vbase_bus).astype(COMPLEX_DTYPE, copy=False)


def s_si_to_pu(S_si: np.ndarray, S_base: float) -> np.ndarray:
    """
    Power SI -> pu:
        S_pu[i] = S_SI[i] / S_base
    """
    return (S_si / float(S_base)).astype(COMPLEX_DTYPE, copy=False)


# --------------------------------------------------------------------------- #
# main Dataset
# --------------------------------------------------------------------------- #

class ChanghunDataset(Dataset):
    """
    Dataset for the branch-row parquet schema with CORRECT multi-voltage per-unit conversion.

    Expected per-row parquet fields:
      bus_number, branch_number, gridtype, U_base, S_base
      bus_typ, vn_kv, Y_shunt_bus
      Branch_f_bus, Branch_t_bus, Branch_status, Branch_tau, Branch_shift_deg
      Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft
      Branch_y_shunt_from, Branch_y_shunt_to
      Is_trafo, Branch_hv_is_f, Branch_n
      Y_Lines, Y_C_Lines
      u_start, u_newton, S_start, S_newton
      optional: Y_matrix

    If per_unit=True:
      - Ybus uses local (Vi,Vj) base conversion
      - branch-row admittances use correct local from/to bases
      - voltages use per-bus base
      - powers use S_base
    """
    __slots__ = (
        "rows",
        "_per_unit",
        "_device",
        "_no_cache_dense_ybus",
        "_lazy_row_groups",
        "_row_group_cache_size",
        "_binary_cols_master",
        "_read_columns",
        "_parquet_files",
        "_row_group_metas",
        "_row_group_ends",
        "_row_group_cache",
        "_num_rows",
    )

    def __init__(
        self,
        paths: Union[str, Path, List[str], List[Path]],
        *,
        per_unit: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        no_cache_dense_ybus: bool = False,
        lazy_row_groups: bool = False,
        row_group_cache_size: int = 2,
    ):
        self._per_unit = bool(per_unit)
        self._device = torch.device(device) if device is not None else None
        self._no_cache_dense_ybus = bool(no_cache_dense_ybus)
        self._lazy_row_groups = bool(lazy_row_groups)
        self._row_group_cache_size = max(1, int(row_group_cache_size))
        self._parquet_files = []
        self._row_group_metas = []
        self._row_group_ends = []
        self._row_group_cache = OrderedDict()
        self._num_rows = 0

        if isinstance(paths, (str, Path)):
            paths = [paths]

        self._binary_cols_master = [
            "bus_typ",
            "vn_kv",
            "Y_shunt_bus",
            "Branch_f_bus",
            "Branch_t_bus",
            "Branch_status",
            "Branch_tau",
            "Branch_shift_deg",
            "Branch_y_series_from",
            "Branch_y_series_to",
            "Branch_y_series_ft",
            "Branch_y_shunt_from",
            "Branch_y_shunt_to",
            "Is_trafo",
            "Branch_hv_is_f",
            "Branch_n",
            "Y_Lines",
            "Y_C_Lines",
            "u_start",
            "u_newton",
            "S_start",
            "S_newton",
        ]
        if not self._no_cache_dense_ybus:
            self._binary_cols_master.insert(self._binary_cols_master.index("u_start"), "Y_matrix")

        self._read_columns = [
            "bus_number",
            "branch_number",
            "S_base",
            "U_base",
            *self._binary_cols_master,
        ]

        if self._lazy_row_groups:
            self.rows = None
            self._build_lazy_row_group_index(paths)
        else:
            self.rows = []
            self._load_eager_rows(paths)

    def _present_columns(self, schema_names) -> List[str]:
        schema_names = set(schema_names)
        return [col for col in self._read_columns if col in schema_names]

    def _build_row_dict(self, r: Union[pd.Series, Dict[str, Any]], available_columns) -> Dict[str, Any]:
        to_t = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype, device=self._device)
        available_columns = set(available_columns)

        N = int(r["bus_number"])
        nl = int(r["branch_number"])

        S_base = float(r["S_base"])
        U_base_scalar = float(r["U_base"])

        bus_typ = _as_np(r["bus_typ"], np.int64).reshape(N)
        vn_kv = _as_np(r["vn_kv"], FLOAT_DTYPE).reshape(N)
        Vbase_bus = (vn_kv.astype(np.float64) * 1e3).astype(np.float64)

        Y_shunt_bus = _as_np(r["Y_shunt_bus"], COMPLEX_DTYPE).reshape(N)

        Branch_f_bus = _as_np(r["Branch_f_bus"], np.int64).reshape(nl)
        Branch_t_bus = _as_np(r["Branch_t_bus"], np.int64).reshape(nl)
        Branch_status = _as_np(r["Branch_status"], np.int8).reshape(nl)
        Branch_tau = _as_np(r["Branch_tau"], FLOAT_DTYPE).reshape(nl)
        Branch_shift_deg = _as_np(r["Branch_shift_deg"], FLOAT_DTYPE).reshape(nl)

        Branch_y_series_from = _as_np(r["Branch_y_series_from"], COMPLEX_DTYPE).reshape(nl)
        Branch_y_series_to = _as_np(r["Branch_y_series_to"], COMPLEX_DTYPE).reshape(nl)
        Branch_y_series_ft = _as_np(r["Branch_y_series_ft"], COMPLEX_DTYPE).reshape(nl)

        Branch_y_shunt_from = _as_np(r["Branch_y_shunt_from"], COMPLEX_DTYPE).reshape(nl)
        Branch_y_shunt_to = _as_np(r["Branch_y_shunt_to"], COMPLEX_DTYPE).reshape(nl)

        Is_trafo = _as_np(r["Is_trafo"], np.int8).reshape(nl)
        Branch_hv_is_f = _as_np(r["Branch_hv_is_f"], np.int8).reshape(nl)
        Branch_n = _as_np(r["Branch_n"], FLOAT_DTYPE).reshape(nl)

        Y_Lines = _as_np(r["Y_Lines"], COMPLEX_DTYPE).reshape(nl)
        Y_C_Lines = _as_np(r["Y_C_Lines"], FLOAT_DTYPE).reshape(nl)

        u_start = _as_np(r["u_start"], COMPLEX_DTYPE).reshape(N)
        u_newton = _as_np(r["u_newton"], COMPLEX_DTYPE).reshape(N)
        S_start = _as_np(r["S_start"], COMPLEX_DTYPE).reshape(N)
        S_newton = _as_np(r["S_newton"], COMPLEX_DTYPE).reshape(N)

        Ybus = None
        if not self._no_cache_dense_ybus:
            if "Y_matrix" in available_columns and r.get("Y_matrix", None) is not None:
                Ybus_si = _as_np(r["Y_matrix"], COMPLEX_DTYPE).reshape(N, N)
            else:
                Ybus_si = reconstruct_Y_pandapower_branchrows_direct_SI_np(
                    N,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Y_shunt_bus,
                    Vbase_bus=Vbase_bus,
                )

        if self._per_unit:
            if not self._no_cache_dense_ybus:
                Ybus = ybus_si_to_pu(Ybus_si, Vbase_bus, S_base)

            Y_shunt_bus = ysh_bus_si_to_pu(Y_shunt_bus, Vbase_bus, S_base)

            Vf = Vbase_bus[Branch_f_bus]
            Vt = Vbase_bus[Branch_t_bus]

            Branch_y_series_from = (
                Branch_y_series_from * (Vf ** 2 / S_base)
            ).astype(COMPLEX_DTYPE, copy=False)

            Branch_y_series_to = (
                Branch_y_series_to * (Vt ** 2 / S_base)
            ).astype(COMPLEX_DTYPE, copy=False)

            Branch_y_series_ft = (
                Branch_y_series_ft * (Vf * Vt / S_base)
            ).astype(COMPLEX_DTYPE, copy=False)

            Branch_y_shunt_from = (
                Branch_y_shunt_from * (Vf ** 2 / S_base)
            ).astype(COMPLEX_DTYPE, copy=False)

            Branch_y_shunt_to = (
                Branch_y_shunt_to * (Vt ** 2 / S_base)
            ).astype(COMPLEX_DTYPE, copy=False)

            Y_Lines = (
                Y_Lines * (Vf ** 2 / S_base)
            ).astype(COMPLEX_DTYPE, copy=False)

            Y_C_Lines = (
                Y_C_Lines * (Vf ** 2 / S_base)
            ).astype(FLOAT_DTYPE, copy=False)

            u_start = u_si_to_pu_per_bus(u_start, Vbase_bus)
            u_newton = u_si_to_pu_per_bus(u_newton, Vbase_bus)

            S_start = s_si_to_pu(S_start, S_base)
            S_newton = s_si_to_pu(S_newton, S_base)

        elif not self._no_cache_dense_ybus:
            Ybus = Ybus_si.astype(COMPLEX_DTYPE, copy=False)

        V_start = np.stack(
            [np.abs(u_start).astype(FLOAT_DTYPE), np.angle(u_start).astype(FLOAT_DTYPE)],
            axis=1
        )
        V_newton = np.stack(
            [np.abs(u_newton).astype(FLOAT_DTYPE), np.angle(u_newton).astype(FLOAT_DTYPE)],
            axis=1
        )

        row: Dict[str, Any] = {
            "N": N,
            "nl": nl,
            "S_base": to_t(np.array(S_base, dtype=FLOAT_DTYPE), dtype=torch.float32),
            "U_base": to_t(np.array(U_base_scalar, dtype=FLOAT_DTYPE), dtype=torch.float32),

            "bus_type": to_t(bus_typ, dtype=torch.int64),
            "vn_kv": to_t(vn_kv, dtype=torch.float32),
            "V_base_bus": to_t(Vbase_bus.astype(FLOAT_DTYPE), dtype=torch.float32),
            "Y_shunt_bus": to_t(Y_shunt_bus, dtype=torch.complex64),

            "Branch_f_bus": to_t(Branch_f_bus, dtype=torch.int64),
            "Branch_t_bus": to_t(Branch_t_bus, dtype=torch.int64),
            "Branch_status": to_t(Branch_status, dtype=torch.int8),
            "Branch_tau": to_t(Branch_tau, dtype=torch.float32),
            "Branch_shift_deg": to_t(Branch_shift_deg, dtype=torch.float32),

            "Branch_y_series_from": to_t(Branch_y_series_from, dtype=torch.complex64),
            "Branch_y_series_to": to_t(Branch_y_series_to, dtype=torch.complex64),
            "Branch_y_series_ft": to_t(Branch_y_series_ft, dtype=torch.complex64),
            "Branch_y_shunt_from": to_t(Branch_y_shunt_from, dtype=torch.complex64),
            "Branch_y_shunt_to": to_t(Branch_y_shunt_to, dtype=torch.complex64),

            "Is_trafo": to_t(Is_trafo, dtype=torch.int8),
            "Branch_hv_is_f": to_t(Branch_hv_is_f, dtype=torch.int8),
            "Branch_n": to_t(Branch_n, dtype=torch.float32),

            "Y_Lines": to_t(Y_Lines, dtype=torch.complex64),
            "Y_C_Lines": to_t(Y_C_Lines, dtype=torch.float32),

            "U_start": to_t(u_start, dtype=torch.complex64),
            "U_newton": to_t(u_newton, dtype=torch.complex64),
            "V_start": to_t(V_start, dtype=torch.float32),
            "V_newton": to_t(V_newton, dtype=torch.float32),
            "S_start": to_t(S_start, dtype=torch.complex64),
            "S_newton": to_t(S_newton, dtype=torch.complex64),
        }
        if not self._no_cache_dense_ybus:
            row["Ybus"] = to_t(Ybus, dtype=torch.complex64)

        return row

    def _load_eager_rows(self, paths: List[Union[str, Path]]) -> None:
        all_dfs = []

        for path in paths:
            print(f"Processing file {path} ...")
            present_columns = self._present_columns(pq.ParquetFile(str(path)).schema.names)
            df = pd.read_parquet(path, columns=present_columns, engine="pyarrow")

            binary_cols = [c for c in self._binary_cols_master if c in df.columns]
            df = decode_columns_mp_columnwise(df, binary_cols)
            print(f"Decoded columns from {path}: {binary_cols}")
            print(f"Raw shape: {df.shape}")

            if "u_newton" in df.columns:
                keep_mask = ~df["u_newton"].map(_all_zero_complex_fast).to_numpy()
                if not keep_mask.all():
                    removed = int((~keep_mask).sum())
                    df = df.loc[keep_mask].reset_index(drop=True)
                    print(f"Removed {removed} diverged rows -> {df.shape}")

            all_dfs.append(df)

        if len(all_dfs) == 0:
            raise ValueError("No parquet data loaded.")

        if len(all_dfs) == 1:
            df = all_dfs[0]
        else:
            df = pd.concat(all_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Final combined dataset shape: {df.shape}")

        available_columns = set(df.columns)
        for _, r in df.iterrows():
            self.rows.append(self._build_row_dict(r, available_columns))

    def _build_lazy_row_group_index(self, paths: List[Union[str, Path]]) -> None:
        total_valid = 0
        total_removed = 0

        for file_idx, path in enumerate(paths):
            print(f"Indexing parquet lazily from {path} ...")
            parquet_file = pq.ParquetFile(str(path))
            self._parquet_files.append(parquet_file)

            file_valid = 0
            file_removed = 0

            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(
                    row_group_idx,
                    columns=["bus_number", "branch_number", "u_newton"],
                )
                data = table.to_pydict()

                keep_positions = []
                signatures = []
                for row_pos, encoded_u_newton in enumerate(data["u_newton"]):
                    if _all_zero_complex_fast(npy_bytes_to_ndarray(encoded_u_newton)):
                        file_removed += 1
                        total_removed += 1
                        continue

                    keep_positions.append(row_pos)
                    signatures.append(
                        (
                            int(data["bus_number"][row_pos]),
                            int(data["branch_number"][row_pos]),
                        )
                    )

                if not keep_positions:
                    continue

                keep_positions_arr = np.asarray(keep_positions, dtype=np.int16)
                signatures_arr = np.asarray(signatures, dtype=np.int32)
                total_valid += len(keep_positions_arr)
                file_valid += len(keep_positions_arr)

                self._row_group_metas.append(
                    {
                        "file_idx": file_idx,
                        "row_group_idx": row_group_idx,
                        "keep_positions": keep_positions_arr,
                        "signatures": signatures_arr,
                    }
                )
                self._row_group_ends.append(total_valid)

            print(
                f"Indexed {file_valid} valid rows from {path}"
                f" across {parquet_file.num_row_groups} row groups"
                f" (removed {file_removed} diverged rows)"
            )

        if total_valid == 0:
            raise ValueError("No valid parquet rows remained after filtering diverged rows.")

        self._num_rows = total_valid
        print(
            f"Final combined dataset shape: ({self._num_rows}, streamed rows)"
            f" | removed {total_removed} diverged rows total"
        )

    def _decode_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self._binary_cols_master:
            if col in df.columns:
                df[col] = [npy_bytes_to_ndarray(b) for b in df[col].values]
        return df

    def _load_row_group_rows(self, meta_idx: int) -> List[Dict[str, Any]]:
        meta = self._row_group_metas[meta_idx]
        parquet_file = self._parquet_files[meta["file_idx"]]
        present_columns = self._present_columns(parquet_file.schema.names)

        table = parquet_file.read_row_group(
            meta["row_group_idx"],
            columns=present_columns,
        )
        df = table.to_pandas()
        df = self._decode_binary_columns(df)

        keep_positions = meta["keep_positions"]
        if len(keep_positions) != len(df):
            df = df.iloc[keep_positions].reset_index(drop=True)

        available_columns = set(df.columns)
        return [self._build_row_dict(r, available_columns) for _, r in df.iterrows()]

    def _get_lazy_meta_index(self, idx: int) -> int:
        return bisect_right(self._row_group_ends, idx)

    def _get_cached_row_group(self, meta_idx: int) -> List[Dict[str, Any]]:
        cached = self._row_group_cache.get(meta_idx)
        if cached is not None:
            self._row_group_cache.move_to_end(meta_idx)
            return cached

        rows = self._load_row_group_rows(meta_idx)
        self._row_group_cache[meta_idx] = rows
        self._row_group_cache.move_to_end(meta_idx)

        while len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)

        return rows

    def get_signature(self, idx: int):
        if self.rows is not None:
            row = self.rows[idx]
            return int(row["N"]), int(row["nl"])

        meta_idx = self._get_lazy_meta_index(idx)
        prev_end = 0 if meta_idx == 0 else self._row_group_ends[meta_idx - 1]
        within = idx - prev_end
        signature = self._row_group_metas[meta_idx]["signatures"][within]
        return int(signature[0]), int(signature[1])

    def __len__(self) -> int:
        if self.rows is not None:
            return len(self.rows)
        return self._num_rows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.rows is not None:
            return self.rows[idx]

        meta_idx = self._get_lazy_meta_index(idx)
        prev_end = 0 if meta_idx == 0 else self._row_group_ends[meta_idx - 1]
        within = idx - prev_end
        rows = self._get_cached_row_group(meta_idx)
        return rows[within]
