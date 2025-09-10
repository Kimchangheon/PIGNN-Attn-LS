import ast, functools, json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from read_npy_columns_optimized import decode_columns_mp_columnwise
import time

FLOAT_DTYPE = np.float32
COMPLEX_DTYPE = np.complex64

# --------------------------------------------------------------------------- #
#                               helpers                                        #
# --------------------------------------------------------------------------- #
def _safe_list(val) -> list:
    """Always return Python list w/o eval’ing Python code."""
    if isinstance(val, str):
        # Fast path if JSON-like; fallback to literal_eval for Python-likes
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return ast.literal_eval(val)
    # Already list/array-like
    return list(val)

def _to_float_array(val, dtype=FLOAT_DTYPE) -> np.ndarray:
    arr = _safe_list(val)
    a = np.array(arr, dtype=object)
    return np.asarray(a.reshape(-1), dtype=dtype)

def _to_complex_array(val, dtype=COMPLEX_DTYPE) -> np.ndarray:
    """Convert many possible encodings into a flat complex numpy array.

    Supports:
      - native complex numbers
      - strings like '1+2j' or '1+2i' or '(1, 2)'
      - [real, imag] pairs
      - {'real': r, 'imag': i} or {'re': r, 'im': i}
      - plain reals (imag=0)
    """
    lst = _safe_list(val)
    obj = np.array(lst, dtype=object).reshape(-1)

    def to_c(e):
        # native complex or numpy complex
        if isinstance(e, complex) or np.issubdtype(type(e), np.complexfloating):
            return e
        # [real, imag] or (real, imag)
        if isinstance(e, (list, tuple)) and len(e) == 2:
            return complex(e[0], e[1])
        # dict with real/imag
        if isinstance(e, dict):
            if "real" in e and "imag" in e:
                return complex(e["real"], e["imag"])
            if "re" in e and "im" in e:
                return complex(e["re"], e["im"])
        # string cases
        if isinstance(e, str):
            s = e.strip().replace("i", "j")
            # try complex('a+bj')
            try:
                return complex(s)
            except Exception:
                # try '(a,b)' or 'a,b'
                s2 = s.strip("()[]")
                parts = [p.strip() for p in s2.split(",")]
                if len(parts) == 2:
                    return complex(float(parts[0]), float(parts[1]))
                raise
        # plain real number
        if isinstance(e, (int, float, np.integer, np.floating)):
            return complex(e, 0.0)
        # last resort
        return complex(float(e), 0.0)

    flat = np.array([to_c(e) for e in obj], dtype=dtype)
    return flat

def _is_zero_complex_list(val) -> bool:
    arr = _to_complex_array(val)
    return arr.size > 0 and np.all(arr == 0)

# --------------------------------------------------------------------------- #
#                              main Dataset                                    #
# --------------------------------------------------------------------------- #
class ChanghunDataset(Dataset):
    """
    Updated for new schema with combined complex columns.

    Every numeric/list column is eagerly turned into torch tensors *once* in
    __init__, so __getitem__ is just indexing.
    """
    __slots__ = ("rows", "_per_unit", "_device")

    # Columns that are list-like and need decoding
    _LIST_COLS = (
        "bus_typ Y_Lines Y_C_Lines Lines_connected Y_matrix u_start u_newton "
        "S_start S_newton I_newton"
    ).split()

    def __init__(self, paths: Union[str, Path, List[str], List[Path]], *, per_unit: bool = False, device: str = None):
        self._per_unit = per_unit
        self._device   = torch.device(device) if device else None

        # ---------- load & sanitise ------------------------------------------------
        # Create an empty list to store all dataframes
        all_dfs = []

        # Process each path
        for path in paths:
            print(f"Processing file {path}...")
            df = pd.read_parquet(path, engine="pyarrow")
            binary_cols = [
                'bus_typ', 'Y_Lines', 'Y_C_Lines', 'Lines_connected',
                'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton', 'I_newton'
            ]
            # Filter for columns that actually exist in the dataframe to avoid errors
            binary_cols_exist = [col for col in binary_cols if col in df.columns]
            print(f"File {path} - Columns to be decoded: {binary_cols_exist}")

            start_time = time.time()
            df = decode_columns_mp_columnwise(df, binary_cols_exist)
            colwise_time = time.time() - start_time
            print(f"File {path} - Method 3: New MP (column-wise) time: {colwise_time:.2f} seconds")

            print(f"File {path} - Parquet read →", df.shape)

            # vectorised decode list-columns
            for col in self._LIST_COLS:
                if col in df.columns:
                    df[col] = df[col].map(_safe_list)

            # remove diverged rows where u_newton is all zeros
            if "u_newton" in df.columns:
                keep = ~df["u_newton"].map(_is_zero_complex_list)
                if not keep.all():
                    df = df[keep].reset_index(drop=True)
                    print(f"File {path} - Removed {(~keep).sum()} diverged rows → {df.shape}")

            # ------------------------------------------------------------------
            # remove per-unit outliers in u_newton (IQR, aggregated across rows)
            # ------------------------------------------------------------------
            OUTLIER_K = 1.5  # IQR multiplier (same default as stats_pu.py)

            def _tolist(cell) -> list:
                if cell is None:
                    return []
                if isinstance(cell, (list, tuple, np.ndarray)):
                    return list(cell)
                return _safe_list(cell)

            def _iqr_bounds(data: Sequence[float], k: float) -> Optional[Tuple[float, float]]:
                arr = np.asarray(data, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return None
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                return float(q1 - k * iqr), float(q3 + k * iqr)

            def _report_outliers(name: str, data: list[float], k: float = 1.5):
                arr = np.asarray(data, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    print(f"No per-unit {name} data to analyze.")
                    return
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                lower, upper = q1 - k * iqr, q3 + k * iqr
                mask_out = (arr < lower) | (arr > upper)
                n_out = int(mask_out.sum())
                frac_out = n_out / arr.size * 100 if arr.size else 0.0
                print(f"\n>>> Per-Unit {name} Outlier Analysis <<<")
                print(f"  Samples: {arr.size},  Q1={q1:.3e}, Q3={q3:.3e}, IQR={iqr:.3e}")
                print(f"  Bounds = [{lower:.3e}, {upper:.3e}]")
                print(f"  Outliers = {n_out} ({frac_out:.2f}%)")
                print(f"  Span = [{arr.min():.3e}, {arr.max():.3e}]")

            if "u_newton" in df.columns:
                if "U_base" not in df.columns:
                    print("⚠️  Skipping per-unit outlier removal: 'U_base' column not found.")
                else:
                    # Gather per-unit samples across the dataset
                    u_bases = df["U_base"].astype(float).to_numpy()
                    all_real_pu: list[float] = []
                    all_imag_pu: list[float] = []

                    for idx, (cell, u_base) in enumerate(zip(df["u_newton"], u_bases)):
                        if not np.isfinite(u_base) or u_base == 0:
                            continue
                        try:
                            u = _to_complex_array(cell)  # robust converter you already have
                        except Exception:
                            continue
                        # keep finite only
                        finite_mask = np.isfinite(u.real) & np.isfinite(u.imag)
                        u = u[finite_mask]
                        if u.size == 0:
                            continue
                        all_real_pu.extend((u.real / u_base).tolist())
                        all_imag_pu.extend((u.imag / u_base).tolist())

                    # Print analysis and compute bounds
                    _report_outliers("u_newton Real", all_real_pu, k=OUTLIER_K)
                    _report_outliers("u_newton Imag", all_imag_pu, k=OUTLIER_K)

                    b_real = _iqr_bounds(all_real_pu, OUTLIER_K)
                    b_imag = _iqr_bounds(all_imag_pu, OUTLIER_K)

                    if b_real is None and b_imag is None:
                        print("No per-unit data available to define outlier bounds; skipping row removal.")
                    else:
                        lr, ur = b_real if b_real is not None else (-np.inf, np.inf)
                        li, ui = b_imag if b_imag is not None else (-np.inf, np.inf)

                        mask_row_outlier = np.zeros(len(df), dtype=bool)

                        for idx, (cell, u_base) in enumerate(zip(df["u_newton"], u_bases)):
                            if not np.isfinite(u_base) or u_base == 0:
                                continue
                            try:
                                u = _to_complex_array(cell)
                            except Exception:
                                continue
                            if u.size == 0:
                                continue
                            r_pu = np.asarray(u.real / u_base, dtype=float)
                            i_pu = np.asarray(u.imag / u_base, dtype=float)
                            r_pu = r_pu[np.isfinite(r_pu)]
                            i_pu = i_pu[np.isfinite(i_pu)]
                            bad_r = r_pu.size > 0 and (np.any(r_pu < lr) or np.any(r_pu > ur))
                            bad_i = i_pu.size > 0 and (np.any(i_pu < li) or np.any(i_pu > ui))
                            if bad_r or bad_i:
                                mask_row_outlier[idx] = True

                        n_bad = int(mask_row_outlier.sum())
                        if n_bad > 0:
                            df = df[~mask_row_outlier].reset_index(drop=True)
                            print(f"Removed {n_bad} per-unit outlier rows → {df.shape}")
                        else:
                            print("No rows contained per-unit u_newton outliers; nothing removed.")

            if "S_newton" in df.columns:
                if "S_base" not in df.columns:
                    print("⚠️  Skipping per-unit S_newton analysis: 'S_base' column not found.")
                else:
                    # Gather per-unit samples across the dataset
                    s_bases = df["S_base"].astype(float).to_numpy()
                    all_real_pu: list[float] = []
                    all_imag_pu: list[float] = []

                    for idx, (cell, s_base) in enumerate(zip(df["S_newton"], s_bases)):
                        if not np.isfinite(s_base) or s_base == 0:
                            continue
                        try:
                            s = _to_complex_array(cell)  # robust converter you already have
                        except Exception:
                            continue
                        # keep finite only
                        finite_mask = np.isfinite(s.real) & np.isfinite(s.imag)
                        s = s[finite_mask]
                        if s.size == 0:
                            continue
                        all_real_pu.extend((s.real / s_base).tolist())
                        all_imag_pu.extend((s.imag / s_base).tolist())

                    # Print analysis only
                    _report_outliers("S_newton Real", all_real_pu, k=OUTLIER_K)
                    _report_outliers("S_newton Imag", all_imag_pu, k=OUTLIER_K)

                    #removal part start
                    # b_real = _iqr_bounds(all_real_pu, OUTLIER_K)
                    # b_imag = _iqr_bounds(all_imag_pu, OUTLIER_K)
                    #
                    # if b_real is None and b_imag is None:
                    #     print("No per-unit S_newton data available to define outlier bounds; skipping row removal.")
                    # else:
                    #     lr, ur = b_real if b_real is not None else (-np.inf, np.inf)
                    #     li, ui = b_imag if b_imag is not None else (-np.inf, np.inf)
                    #
                    #     mask_row_outlier = np.zeros(len(df), dtype=bool)
                    #
                    #     for idx, (cell, s_base) in enumerate(zip(df["S_newton"], s_bases)):
                    #         if not np.isfinite(s_base) or s_base == 0:
                    #             continue
                    #         try:
                    #             s = _to_complex_array(cell)
                    #         except Exception:
                    #             continue
                    #         if s.size == 0:
                    #             continue
                    #         r_pu = np.asarray(s.real / s_base, dtype=float)
                    #         i_pu = np.asarray(s.imag / s_base, dtype=float)
                    #         r_pu = r_pu[np.isfinite(r_pu)]
                    #         i_pu = i_pu[np.isfinite(i_pu)]
                    #         bad_r = r_pu.size > 0 and (np.any(r_pu < lr) or np.any(r_pu > ur))
                    #         bad_i = i_pu.size > 0 and (np.any(i_pu < li) or np.any(i_pu > ui))
                    #         if bad_r or bad_i:
                    #             mask_row_outlier[idx] = True
                    #
                    #     n_bad = int(mask_row_outlier.sum())
                    #     if n_bad > 0:
                    #         df = df[~mask_row_outlier].reset_index(drop=True)
                    #         print(f"Removed {n_bad} per-unit S_newton outlier rows → {df.shape}")
                    #     else:
                    #         print("No rows contained per-unit S_newton outliers; nothing removed.")

            # Add processed dataframe to list
            all_dfs.append(df)

        if len(all_dfs) > 2:
            # Combine all dataframes and shuffle if there are more than 2
            df = pd.concat(all_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            # Just use the first dataframe if 2 or fewer
            df = all_dfs[0]

        print(f"Final combined dataset shape after merging and shuffling: {df.shape}")

        # # ---------- load & sanitise ------------------------------------------------
        # df = pd.read_parquet(path, engine="pyarrow")
        # binary_cols = [
        #     'bus_typ', 'Y_Lines', 'Y_C_Lines', 'Lines_connected',
        #     'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton', 'I_newton'
        # ]
        # # Filter for columns that actually exist in the dataframe to avoid errors
        # binary_cols_exist = [col for col in binary_cols if col in df.columns]
        # print(f"Columns to be decoded: {binary_cols_exist}")
        #
        # start_time = time.time()
        # df = decode_columns_mp_columnwise(df, binary_cols_exist)
        # colwise_time = time.time() - start_time
        # print(f"Method 3: New MP (column-wise) time: {colwise_time:.2f} seconds")
        #
        # print("Parquet read →", df.shape)
        #
        # # vectorised decode list-columns
        # for col in self._LIST_COLS:
        #     if col in df.columns:
        #         df[col] = df[col].map(_safe_list)
        #
        # # remove diverged rows where u_newton is all zeros
        # if "u_newton" in df.columns:
        #     keep = ~df["u_newton"].map(_is_zero_complex_list)
        #     if not keep.all():
        #         df = df[keep].reset_index(drop=True)
        #         print(f"Removed {(~keep).sum()} diverged rows → {df.shape}")
        #
        # # ------------------------------------------------------------------
        # # remove per-unit outliers in u_newton (IQR, aggregated across rows)
        # # ------------------------------------------------------------------
        # OUTLIER_K = 1.5  # IQR multiplier (same default as stats_pu.py)
        #
        # def _tolist(cell) -> list:
        #     if cell is None:
        #         return []
        #     if isinstance(cell, (list, tuple, np.ndarray)):
        #         return list(cell)
        #     return _safe_list(cell)
        #
        # def _iqr_bounds(data: Sequence[float], k: float) -> Optional[Tuple[float, float]]:
        #     arr = np.asarray(data, dtype=float)
        #     arr = arr[np.isfinite(arr)]
        #     if arr.size == 0:
        #         return None
        #     q1, q3 = np.percentile(arr, [25, 75])
        #     iqr = q3 - q1
        #     return float(q1 - k * iqr), float(q3 + k * iqr)
        #
        # def _report_outliers(name: str, data: list[float], k: float = 1.5):
        #     arr = np.asarray(data, dtype=float)
        #     arr = arr[np.isfinite(arr)]
        #     if arr.size == 0:
        #         print(f"No per-unit {name} data to analyze.")
        #         return
        #     q1, q3 = np.percentile(arr, [25, 75])
        #     iqr = q3 - q1
        #     lower, upper = q1 - k * iqr, q3 + k * iqr
        #     mask_out = (arr < lower) | (arr > upper)
        #     n_out = int(mask_out.sum())
        #     frac_out = n_out / arr.size * 100 if arr.size else 0.0
        #     print(f"\n>>> Per-Unit {name} Outlier Analysis <<<")
        #     print(f"  Samples: {arr.size},  Q1={q1:.3e}, Q3={q3:.3e}, IQR={iqr:.3e}")
        #     print(f"  Bounds = [{lower:.3e}, {upper:.3e}]")
        #     print(f"  Outliers = {n_out} ({frac_out:.2f}%)")
        #     print(f"  Span = [{arr.min():.3e}, {arr.max():.3e}]")
        #
        # if "u_newton" in df.columns:
        #     if "U_base" not in df.columns:
        #         print("⚠️  Skipping per-unit outlier removal: 'U_base' column not found.")
        #     else:
        #         # Gather per-unit samples across the dataset
        #         u_bases = df["U_base"].astype(float).to_numpy()
        #         all_real_pu: list[float] = []
        #         all_imag_pu: list[float] = []
        #
        #         for idx, (cell, u_base) in enumerate(zip(df["u_newton"], u_bases)):
        #             if not np.isfinite(u_base) or u_base == 0:
        #                 continue
        #             try:
        #                 u = _to_complex_array(cell)  # robust converter you already have
        #             except Exception:
        #                 continue
        #             # keep finite only
        #             finite_mask = np.isfinite(u.real) & np.isfinite(u.imag)
        #             u = u[finite_mask]
        #             if u.size == 0:
        #                 continue
        #             all_real_pu.extend((u.real / u_base).tolist())
        #             all_imag_pu.extend((u.imag / u_base).tolist())
        #
        #         # Print analysis and compute bounds
        #         _report_outliers("u_newton Real", all_real_pu, k=OUTLIER_K)
        #         _report_outliers("u_newton Imag", all_imag_pu, k=OUTLIER_K)
        #
        #         b_real = _iqr_bounds(all_real_pu, OUTLIER_K)
        #         b_imag = _iqr_bounds(all_imag_pu, OUTLIER_K)
        #
        #         if b_real is None and b_imag is None:
        #             print("No per-unit data available to define outlier bounds; skipping row removal.")
        #         else:
        #             lr, ur = b_real if b_real is not None else (-np.inf, np.inf)
        #             li, ui = b_imag if b_imag is not None else (-np.inf, np.inf)
        #
        #             mask_row_outlier = np.zeros(len(df), dtype=bool)
        #
        #             for idx, (cell, u_base) in enumerate(zip(df["u_newton"], u_bases)):
        #                 if not np.isfinite(u_base) or u_base == 0:
        #                     continue
        #                 try:
        #                     u = _to_complex_array(cell)
        #                 except Exception:
        #                     continue
        #                 if u.size == 0:
        #                     continue
        #                 r_pu = np.asarray(u.real / u_base, dtype=float)
        #                 i_pu = np.asarray(u.imag / u_base, dtype=float)
        #                 r_pu = r_pu[np.isfinite(r_pu)]
        #                 i_pu = i_pu[np.isfinite(i_pu)]
        #                 bad_r = r_pu.size > 0 and (np.any(r_pu < lr) or np.any(r_pu > ur))
        #                 bad_i = i_pu.size > 0 and (np.any(i_pu < li) or np.any(i_pu > ui))
        #                 if bad_r or bad_i:
        #                     mask_row_outlier[idx] = True
        #
        #             n_bad = int(mask_row_outlier.sum())
        #             if n_bad > 0:
        #                 df = df[~mask_row_outlier].reset_index(drop=True)
        #                 print(f"Removed {n_bad} per-unit outlier rows → {df.shape}")
        #             else:
        #                 print("No rows contained per-unit u_newton outliers; nothing removed.")
        #
        # if "S_newton" in df.columns:
        #     if "S_base" not in df.columns:
        #         print("⚠️  Skipping per-unit S_newton analysis: 'S_base' column not found.")
        #     else:
        #         # Gather per-unit samples across the dataset
        #         s_bases = df["S_base"].astype(float).to_numpy()
        #         all_real_pu: list[float] = []
        #         all_imag_pu: list[float] = []
        #
        #         for idx, (cell, s_base) in enumerate(zip(df["S_newton"], s_bases)):
        #             if not np.isfinite(s_base) or s_base == 0:
        #                 continue
        #             try:
        #                 s = _to_complex_array(cell)  # robust converter you already have
        #             except Exception:
        #                 continue
        #             # keep finite only
        #             finite_mask = np.isfinite(s.real) & np.isfinite(s.imag)
        #             s = s[finite_mask]
        #             if s.size == 0:
        #                 continue
        #             all_real_pu.extend((s.real / s_base).tolist())
        #             all_imag_pu.extend((s.imag / s_base).tolist())
        #
        #         # Print analysis only
        #         _report_outliers("S_newton Real", all_real_pu, k=OUTLIER_K)
        #         _report_outliers("S_newton Imag", all_imag_pu, k=OUTLIER_K)
        #
        #         #removal part start
        #         # b_real = _iqr_bounds(all_real_pu, OUTLIER_K)
        #         # b_imag = _iqr_bounds(all_imag_pu, OUTLIER_K)
        #         #
        #         # if b_real is None and b_imag is None:
        #         #     print("No per-unit S_newton data available to define outlier bounds; skipping row removal.")
        #         # else:
        #         #     lr, ur = b_real if b_real is not None else (-np.inf, np.inf)
        #         #     li, ui = b_imag if b_imag is not None else (-np.inf, np.inf)
        #         #
        #         #     mask_row_outlier = np.zeros(len(df), dtype=bool)
        #         #
        #         #     for idx, (cell, s_base) in enumerate(zip(df["S_newton"], s_bases)):
        #         #         if not np.isfinite(s_base) or s_base == 0:
        #         #             continue
        #         #         try:
        #         #             s = _to_complex_array(cell)
        #         #         except Exception:
        #         #             continue
        #         #         if s.size == 0:
        #         #             continue
        #         #         r_pu = np.asarray(s.real / s_base, dtype=float)
        #         #         i_pu = np.asarray(s.imag / s_base, dtype=float)
        #         #         r_pu = r_pu[np.isfinite(r_pu)]
        #         #         i_pu = i_pu[np.isfinite(i_pu)]
        #         #         bad_r = r_pu.size > 0 and (np.any(r_pu < lr) or np.any(r_pu > ur))
        #         #         bad_i = i_pu.size > 0 and (np.any(i_pu < li) or np.any(i_pu > ui))
        #         #         if bad_r or bad_i:
        #         #             mask_row_outlier[idx] = True
        #         #
        #         #     n_bad = int(mask_row_outlier.sum())
        #         #     if n_bad > 0:
        #         #         df = df[~mask_row_outlier].reset_index(drop=True)
        #         #         print(f"Removed {n_bad} per-unit S_newton outlier rows → {df.shape}")
        #         #     else:
        #         #         print("No rows contained per-unit S_newton outliers; nothing removed.")

        # ---------- convert to tensors --------------------------------------------
        self.rows: List[Dict[str, Any]] = []
        to_t  = functools.partial(torch.as_tensor, device=self._device)

        for _, r in df.iterrows():
            # infer N from bus_typ length if present; else from Y_matrix size
            if "bus_typ" in r and isinstance(r.bus_typ, list):
                N = int(len(r.bus_typ))
            else:
                ym_obj = np.array(_safe_list(r.Y_matrix), dtype=object)
                total = int(ym_obj.size)
                N = int(np.sqrt(total))

            # scalar bases (cast to desired float type)
            S_base = np.array(r.S_base if per_unit else 1.0, dtype=FLOAT_DTYPE)
            U_base = np.array(r.U_base if per_unit else 1.0, dtype=FLOAT_DTYPE)
            Y_base = np.array((S_base / (U_base ** 2)) if per_unit else 1.0, dtype=FLOAT_DTYPE)
            I_base = np.array((S_base / U_base) if per_unit else 1.0, dtype=FLOAT_DTYPE)  # adjust if you need √3

            # pre-normalised tensors / values
            row: Dict[str, Any] = {
                "S_base": S_base,
                "U_base": U_base,
                "N": N,
                "bus_type": to_t(r.bus_typ, dtype=torch.int64),
                "Lines_connected": to_t(r.Lines_connected, dtype=torch.bool),
            }

            # Admittance matrix (complex)
            Ybus = _to_complex_array(r.Y_matrix).reshape(N, N) / Y_base
            row["Ybus"] = to_t(Ybus, dtype=torch.complex64)

            # Line admittances (series & shunt)
            if "Y_Lines" in df.columns:
                Y_lines = _to_complex_array(r.Y_Lines) / Y_base
                row["Y_Lines"] = to_t(Y_lines, dtype=torch.complex64)

            if "Y_C_Lines" in df.columns:
                # keep as real (historically shunt caps were real-valued here)
                YC = _to_float_array(r.Y_C_Lines) / Y_base
                row["Y_C_Lines"] = to_t(YC)

            # Voltages
            U_start  = _to_complex_array(r.u_start)  / U_base
            U_newton = _to_complex_array(r.u_newton) / U_base
            row["U_start"]  = to_t(U_start,  dtype=torch.complex64)
            row["U_newton"] = to_t(U_newton, dtype=torch.complex64)

            # also provide magnitude/angle as float32 (N,2)
            for name, U in [("start", U_start), ("newton", U_newton)]:
                mag = np.abs(U).astype(FLOAT_DTYPE)
                ang = np.angle(U).astype(FLOAT_DTYPE)
                row[f"V_{name}"] = to_t(np.stack([mag, ang], axis=1))

            # Powers (complex)
            if "S_start" in df.columns:
                S_start = _to_complex_array(r.S_start) / S_base
                row["S_start"] = to_t(S_start, dtype=torch.complex64)

            if "S_newton" in df.columns:
                S_newton = _to_complex_array(r.S_newton) / S_base
                row["S_newton"] = to_t(S_newton, dtype=torch.complex64)

            # Currents (complex)
            if "I_newton" in df.columns:
                I_newton = _to_complex_array(r.I_newton) / I_base
                row["I_newton"] = to_t(I_newton, dtype=torch.complex64)

            self.rows.append(row)

    # ----------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]
