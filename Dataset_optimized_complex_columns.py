import ast, functools, json, math
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
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return ast.literal_eval(val)
    return list(val)

def _to_float_array(val, dtype=FLOAT_DTYPE) -> np.ndarray:
    arr = _safe_list(val)
    a = np.array(arr, dtype=object)
    return np.asarray(a.reshape(-1), dtype=dtype)

def _to_complex_array(val, dtype=COMPLEX_DTYPE) -> np.ndarray:
    """Convert many possible encodings into a flat complex numpy array."""
    lst = _safe_list(val)
    obj = np.array(lst, dtype=object).reshape(-1)

    def to_c(e):
        if isinstance(e, complex) or np.issubdtype(type(e), np.complexfloating):
            return e
        if isinstance(e, (list, tuple)) and len(e) == 2:
            return complex(e[0], e[1])
        if isinstance(e, dict):
            if "real" in e and "imag" in e:
                return complex(e["real"], e["imag"])
            if "re" in e and "im" in e:
                return complex(e["re"], e["im"])
        if isinstance(e, str):
            s = e.strip().replace("i", "j")
            try:
                return complex(s)
            except Exception:
                s2 = s.strip("()[]")
                parts = [p.strip() for p in s2.split(",")]
                if len(parts) == 2:
                    return complex(float(parts[0]), float(parts[1]))
                raise
        if isinstance(e, (int, float, np.integer, np.floating)):
            return complex(e, 0.0)
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
    If Y_matrix is absent, Ybus is set to None and N is inferred from other columns.
    """
    __slots__ = ("rows", "_per_unit", "_device")

    _LIST_COLS = (
        "bus_typ Y_Lines Y_C_Lines Lines_connected Y_matrix u_start u_newton "
        "S_start S_newton I_newton"
    ).split()

    def __init__(self, paths: Union[str, Path, List[str], List[Path]], *, per_unit: bool = False, device: str = None):
        self._per_unit = per_unit
        self._device   = torch.device(device) if device else None

        if isinstance(paths, (str, Path)):
            paths = [paths]

        all_dfs = []
        for path in paths:
            print(f"Processing file {path}...")
            df = pd.read_parquet(path, engine="pyarrow")

            binary_cols = [
                'bus_typ', 'Y_Lines', 'Y_C_Lines', 'Lines_connected',
                'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton', 'I_newton'
            ]
            binary_cols_exist = [col for col in binary_cols if col in df.columns]
            print(f"File {path} - Columns to be decoded: {binary_cols_exist}")

            start_time = time.time()
            df = decode_columns_mp_columnwise(df, binary_cols_exist)
            print(f"File {path} - Method 3: New MP (column-wise) time: {time.time() - start_time:.2f} seconds")
            print(f"File {path} - Parquet read →", df.shape)

            # ─────────────────────────────
            # 1) Remove diverged rows fast
            # ─────────────────────────────
            if "u_newton" in df.columns:
                def _all_zero_fast(cell) -> bool:
                    # Fast path if we already have a numeric ndarray from decode_columns_mp_columnwise
                    try:
                        a = np.asarray(cell)
                        if a.size == 0:
                            return True
                        if np.iscomplexobj(a):
                            return bool(np.all((a.real == 0) & (a.imag == 0)))
                        # real array or list of reals
                        return bool(np.all(a == 0))
                    except Exception:
                        # Fallback to robust (slower) path for irregular encodings
                        return _is_zero_complex_list(cell)

                keep_mask = ~df["u_newton"].map(_all_zero_fast).to_numpy()
                if not keep_mask.all():
                    df = df.loc[keep_mask].reset_index(drop=True)
                    print(f"File {path} - Removed {(~keep_mask).sum()} diverged rows → {df.shape}")

            # ──────────────────────────────────────────────
            # 2) Per-unit outlier analysis/removal (u_newton)
            #    (bounded memory via reservoir sampling)
            # ──────────────────────────────────────────────
            OUTLIER_K = 1.5
            SAMPLE_CAP = 2_000_000  # max elements kept per channel (real/imag)

            def _iqr_bounds(arr: np.ndarray, k: float) -> Optional[tuple[float, float]]:
                if arr.size == 0:
                    return None
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                return float(q1 - k * iqr), float(q3 + k * iqr)

            # exact reservoir sampling (Algorithm R), vectorized per batch
            def _reservoir_add(sample_r, sample_i, r, i, rng, total_seen):
                """
                Maintain fixed-size reservoirs sample_r/sample_i (size ≤ SAMPLE_CAP) with new arrays r/i.
                Uses Vitter's Algorithm R exactly:
                  - fill to capacity with a random subset of new elements
                  - for each subsequent element at position t (1-based stream index), draw j~Unif{1..t};
                    if j <= SAMPLE_CAP, replace reservoir[j-1] with this element.
                Args:
                  sample_r, sample_i: 1D float32 reservoirs (same length)
                  r, i:                1D float32 arrays of new real/imag values (same length)
                  rng:                 np.random.Generator
                  total_seen:          total number of elements processed so far (int)
                Returns:
                  (sample_r, sample_i, total_seen)
                """
                cap = SAMPLE_CAP
                n_new = r.size
                if n_new == 0:
                    return sample_r, sample_i, total_seen

                size = sample_r.size

                # ---- Fill phase: bring reservoir up to capacity (if not full) ----
                if size < cap:
                    need = cap - size
                    if n_new <= need:
                        # append all new data
                        sample_r = np.concatenate((sample_r, r))
                        sample_i = np.concatenate((sample_i, i))
                        total_seen += n_new
                        return sample_r, sample_i, total_seen
                    else:
                        # append a random subset of the new data to fill the reservoir
                        idx_fill = rng.choice(n_new, size=need, replace=False)
                        sample_r = np.concatenate((sample_r, r[idx_fill]))
                        sample_i = np.concatenate((sample_i, i[idx_fill]))
                        # keep the remaining elements for replacement phase
                        mask_fill = np.ones(n_new, dtype=bool)
                        mask_fill[idx_fill] = False
                        r = r[mask_fill]
                        i = i[mask_fill]
                        n_new = r.size
                        total_seen += need
                        size = cap  # now full

                # ---- Replacement phase (exact Algorithm R) ----
                # For the remaining n_new elements, their stream positions are:
                # t = total_seen + 1, total_seen + 2, ..., total_seen + n_new
                # Draw j_l ~ Unif{1..t_l}; replace if j_l <= cap
                t_vals = total_seen + np.arange(1, n_new + 1, dtype=np.int64)
                # rng.integers(low, high) draws in [low, high); need [1..t] inclusive → high=t+1
                j = rng.integers(1, t_vals + 1)
                sel = (j <= cap)
                if sel.any():
                    repl_idx = (j[sel] - 1).astype(np.int64)  # indices in [0..cap-1]
                    src_idx = np.nonzero(sel)[0].astype(np.int64)  # which new elements to use
                    sample_r[repl_idx] = r[src_idx]
                    sample_i[repl_idx] = i[src_idx]

                total_seen += n_new
                return sample_r, sample_i, total_seen
            if "u_newton" in df.columns:
                if "U_base" not in df.columns:
                    print("⚠️  Skipping per-unit outlier removal: 'U_base' column not found.")
                else:
                    u_bases = df["U_base"].astype("float64").to_numpy()

                    rng = np.random.default_rng(42)
                    sample_r = np.empty(0, dtype=np.float32)
                    sample_i = np.empty(0, dtype=np.float32)
                    total_seen = 0

                    # ---- Pass 1: build bounded-memory reservoir sample ----
                    for cell, u_base in zip(df["u_newton"], u_bases):
                        if not np.isfinite(u_base) or u_base == 0:
                            continue
                        # try to avoid heavy conversions
                        a = np.asarray(cell)
                        if a.size == 0:
                            continue
                        if not np.iscomplexobj(a):
                            # irregular encoding → use robust converter only when needed
                            a = _to_complex_array(cell)
                        r = np.asarray(a.real / u_base, dtype=np.float32)
                        i = np.asarray(a.imag / u_base, dtype=np.float32)
                        mask = np.isfinite(r) & np.isfinite(i)
                        if not mask.any():
                            continue
                        r = r[mask];
                        i = i[mask]

                        sample_r, sample_i, total_seen = _reservoir_add(sample_r, sample_i, r, i, rng, total_seen)

                    # Compute IQR bounds on the sample only
                    b_real = _iqr_bounds(sample_r, OUTLIER_K)
                    b_imag = _iqr_bounds(sample_i, OUTLIER_K)

                    # Optional: quick report (cheap; sample-based)
                    if sample_r.size or sample_i.size:
                        def _report(name, arr):
                            if arr.size == 0:
                                print(f"No per-unit {name} data to analyze.")
                                return
                            q1, q3 = np.percentile(arr, [25, 75])
                            iqr = q3 - q1
                            lower, upper = q1 - OUTLIER_K * iqr, q3 + OUTLIER_K * iqr
                            print(f"\n>>> Per-Unit {name} Outlier Analysis (sample) <<<")
                            print(f"  Samples: {arr.size},  Q1={q1:.3e}, Q3={q3:.3e}, IQR={iqr:.3e}")
                            print(f"  Bounds = [{lower:.3e}, {upper:.3e}]  (cap={SAMPLE_CAP})")

                        _report("u_newton Real", sample_r)
                        _report("u_newton Imag", sample_i)

                    if b_real is None and b_imag is None:
                        print("No per-unit data available to define outlier bounds; skipping row removal.")
                    else:
                        lr, ur = b_real if b_real is not None else (-np.inf, np.inf)
                        li, ui = b_imag if b_imag is not None else (-np.inf, np.inf)

                        # ---- Pass 2: flag rows using min/max early-exit (no large temporaries) ----
                        mask_row_outlier = np.zeros(len(df), dtype=bool)
                        for idx, (cell, u_base) in enumerate(zip(df["u_newton"], u_bases)):
                            if not np.isfinite(u_base) or u_base == 0:
                                continue
                            a = np.asarray(cell)
                            if a.size == 0:
                                continue
                            if not np.iscomplexobj(a):
                                a = _to_complex_array(cell)

                            r = np.asarray(a.real / u_base, dtype=np.float32)
                            i = np.asarray(a.imag / u_base, dtype=np.float32)
                            mask = np.isfinite(r) & np.isfinite(i)
                            if not mask.any():
                                continue
                            r = r[mask];
                            i = i[mask]

                            # quick min/max test before any per-element comparisons
                            if (r.size and (r.min() < lr or r.max() > ur)) or (
                                    i.size and (i.min() < li or i.max() > ui)):
                                mask_row_outlier[idx] = True

                        n_bad = int(mask_row_outlier.sum())
                        if n_bad > 0:
                            df = df.loc[~mask_row_outlier].reset_index(drop=True)
                            print(f"Removed {n_bad} per-unit outlier rows → {df.shape}")
                        else:
                            print("No rows contained per-unit u_newton outliers; nothing removed.")

            all_dfs.append(df)

        if len(all_dfs) > 2:
            df = pd.concat(all_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df = all_dfs[0]

        print(f"Final combined dataset shape after merging and shuffling: {df.shape}")

        self.rows: List[Dict[str, Any]] = []
        to_t  = functools.partial(torch.as_tensor, device=self._device)

        for _, r in df.iterrows():
            # ---- Infer N robustly ----
            if ("bus_typ" in df.columns) and isinstance(r.bus_typ, list) and len(r.bus_typ) > 0:
                N = int(len(r.bus_typ))
            elif ("Y_matrix" in df.columns) and r.Y_matrix is not None:
                ym_list = _safe_list(r.Y_matrix)
                total = int(len(ym_list))
                N = int(round(math.sqrt(total)))
            elif "Lines_connected" in df.columns:
                # Solve n(n-1)/2 = E
                e_len = int(len(_safe_list(r.Lines_connected)))
                N = int((1 + math.isqrt(1 + 8 * e_len)) // 2)
            else:
                raise ValueError("Cannot infer bus count N (need bus_typ, Y_matrix, or Lines_connected).")

            # scalar bases
            S_base = np.array(r.S_base if per_unit else 1.0, dtype=FLOAT_DTYPE)
            U_base = np.array(r.U_base if per_unit else 1.0, dtype=FLOAT_DTYPE)
            Y_base = np.array((S_base / (U_base ** 2)) if per_unit else 1.0, dtype=FLOAT_DTYPE)
            I_base = np.array((S_base / U_base) if per_unit else 1.0, dtype=FLOAT_DTYPE)

            row: Dict[str, Any] = {
                "S_base": S_base,
                "U_base": U_base,
                "N": N,
            }

            if "bus_typ" in df.columns:
                row["bus_type"] = to_t(r.bus_typ, dtype=torch.int64)
            if "Lines_connected" in df.columns:
                row["Lines_connected"] = to_t(r.Lines_connected, dtype=torch.bool)

            # ---- Ybus (may be None if Y_matrix missing) ----
            if "Y_matrix" in df.columns and r.Y_matrix is not None:
                Ybus = _to_complex_array(r.Y_matrix).reshape(N, N) / Y_base
                row["Ybus"] = to_t(Ybus, dtype=torch.complex64)
            else:
                row["Ybus"] = None  # <- requested behavior

            # Line admittances (series & shunt)
            if "Y_Lines" in df.columns:
                Y_lines = _to_complex_array(r.Y_Lines) / Y_base
                row["Y_Lines"] = to_t(Y_lines, dtype=torch.complex64)

            if "Y_C_Lines" in df.columns:
                YC = _to_float_array(r.Y_C_Lines) / Y_base
                row["Y_C_Lines"] = to_t(YC)

            # Voltages
            if "u_start" in df.columns:
                U_start  = _to_complex_array(r.u_start)  / U_base
                row["U_start"]  = to_t(U_start,  dtype=torch.complex64)
                # magnitude/angle
                mag = np.abs(U_start).astype(FLOAT_DTYPE)
                ang = np.angle(U_start).astype(FLOAT_DTYPE)
                row["V_start"] = to_t(np.stack([mag, ang], axis=1))

            if "u_newton" in df.columns:
                U_newton = _to_complex_array(r.u_newton) / U_base
                row["U_newton"] = to_t(U_newton, dtype=torch.complex64)
                mag = np.abs(U_newton).astype(FLOAT_DTYPE)
                ang = np.angle(U_newton).astype(FLOAT_DTYPE)
                row["V_newton"] = to_t(np.stack([mag, ang], axis=1))

            # Powers
            if "S_start" in df.columns:
                S_start = _to_complex_array(r.S_start) / S_base
                row["S_start"] = to_t(S_start, dtype=torch.complex64)
            if "S_newton" in df.columns:
                S_newton = _to_complex_array(r.S_newton) / S_base
                row["S_newton"] = to_t(S_newton, dtype=torch.complex64)

            # Currents (optional legacy)
            if "I_newton" in df.columns:
                I_newton = _to_complex_array(r.I_newton) / I_base
                row["I_newton"] = to_t(I_newton, dtype=torch.complex64)

            self.rows.append(row)

    # ----------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]