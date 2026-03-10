import io
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from read_npy_columns_optimized import decode_columns_mp_columnwise

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
        ysh_f  = complex(Branch_y_shunt_from[k])
        ysh_t  = complex(Branch_y_shunt_to[k])

        Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
        Ytt = (y_to   + ysh_t / 2.0)
        Yft = -y_ft / np.conj(a)
        Ytf = -y_ft / a

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
    __slots__ = ("rows", "_per_unit", "_device")

    def __init__(
        self,
        paths: Union[str, Path, List[str], List[Path]],
        *,
        per_unit: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self._per_unit = bool(per_unit)
        self._device = torch.device(device) if device is not None else None

        if isinstance(paths, (str, Path)):
            paths = [paths]

        all_dfs = []

        binary_cols_master = [
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
            "Y_matrix",
            "u_start",
            "u_newton",
            "S_start",
            "S_newton",
        ]

        for path in paths:
            print(f"Processing file {path} ...")
            df = pd.read_parquet(path, engine="pyarrow")

            binary_cols = [c for c in binary_cols_master if c in df.columns]
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

        to_t = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype, device=self._device)
        self.rows: List[Dict[str, Any]] = []

        for _, r in df.iterrows():
            N = int(r["bus_number"])
            nl = int(r["branch_number"])

            S_base = float(r["S_base"])
            U_base_scalar = float(r["U_base"])  # keep for metadata compatibility

            # ---- decode arrays ----
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
            Branch_y_series_to   = _as_np(r["Branch_y_series_to"], COMPLEX_DTYPE).reshape(nl)
            Branch_y_series_ft   = _as_np(r["Branch_y_series_ft"], COMPLEX_DTYPE).reshape(nl)

            Branch_y_shunt_from = _as_np(r["Branch_y_shunt_from"], COMPLEX_DTYPE).reshape(nl)
            Branch_y_shunt_to   = _as_np(r["Branch_y_shunt_to"], COMPLEX_DTYPE).reshape(nl)

            Is_trafo = _as_np(r["Is_trafo"], np.int8).reshape(nl)
            Branch_hv_is_f = _as_np(r["Branch_hv_is_f"], np.int8).reshape(nl)
            Branch_n = _as_np(r["Branch_n"], FLOAT_DTYPE).reshape(nl)

            Y_Lines = _as_np(r["Y_Lines"], COMPLEX_DTYPE).reshape(nl)
            Y_C_Lines = _as_np(r["Y_C_Lines"], FLOAT_DTYPE).reshape(nl)

            u_start = _as_np(r["u_start"], COMPLEX_DTYPE).reshape(N)
            u_newton = _as_np(r["u_newton"], COMPLEX_DTYPE).reshape(N)
            S_start = _as_np(r["S_start"], COMPLEX_DTYPE).reshape(N)
            S_newton = _as_np(r["S_newton"], COMPLEX_DTYPE).reshape(N)

            # ---- Ybus in SI ----
            if "Y_matrix" in df.columns and r.get("Y_matrix", None) is not None:
                Ybus_si = _as_np(r["Y_matrix"], COMPLEX_DTYPE).reshape(N, N)
            else:
                Ybus_si = reconstruct_Y_pandapower_branchrows_direct_SI_np(
                    N,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Y_shunt_bus,
                )

            # ---- per-unit conversion using CORRECT local bases ----
            if self._per_unit:
                # dense Ybus
                Ybus = ybus_si_to_pu(Ybus_si, Vbase_bus, S_base)

                # bus shunts
                Y_shunt_bus = ysh_bus_si_to_pu(Y_shunt_bus, Vbase_bus, S_base)

                # per-branch local bases
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

                # For line-only legacy features, use local from-bus base.
                # For non-lines these are usually zero anyway.
                Y_Lines = (
                    Y_Lines * (Vf ** 2 / S_base)
                ).astype(COMPLEX_DTYPE, copy=False)

                Y_C_Lines = (
                    Y_C_Lines * (Vf ** 2 / S_base)
                ).astype(FLOAT_DTYPE, copy=False)

                # voltages per bus
                u_start = u_si_to_pu_per_bus(u_start, Vbase_bus)
                u_newton = u_si_to_pu_per_bus(u_newton, Vbase_bus)

                # powers by system S_base
                S_start = s_si_to_pu(S_start, S_base)
                S_newton = s_si_to_pu(S_newton, S_base)

            else:
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
                "S_base": np.array(S_base, dtype=FLOAT_DTYPE),
                "U_base": np.array(U_base_scalar, dtype=FLOAT_DTYPE),

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

                "Ybus": to_t(Ybus, dtype=torch.complex64),

                "U_start": to_t(u_start, dtype=torch.complex64),
                "U_newton": to_t(u_newton, dtype=torch.complex64),
                "V_start": to_t(V_start, dtype=torch.float32),
                "V_newton": to_t(V_newton, dtype=torch.float32),
                "S_start": to_t(S_start, dtype=torch.complex64),
                "S_newton": to_t(S_newton, dtype=torch.complex64),
            }

            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]