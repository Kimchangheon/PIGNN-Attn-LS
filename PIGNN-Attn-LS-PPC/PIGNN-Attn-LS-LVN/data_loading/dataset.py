from __future__ import annotations
import ast, functools, json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .npy_decode import decode_columns_mp_columnwise
import time

log = logging.getLogger("simplegnn")

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
    # Fast path: already a numpy array of complex or numeric dtype. Avoids
    # 260k-element Python iteration per call (the eager loop was the
    # dominant cost in lazy-mode __getitem__, ~123 ms/row -> ~5 ms/row).
    if isinstance(val, np.ndarray):
        if val.dtype.kind == 'c':
            return val.reshape(-1).astype(dtype, copy=False)
        if val.dtype.kind in ('f', 'i', 'u'):
            return val.reshape(-1).astype(dtype)

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

    def __init__(
        self,
        path: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        per_unit: bool = False,
        device: "Optional[Union[str, torch.device]]" = None,
        use_cache: bool = True,
        cache_dir: str | Path = "./.cache/datasets",
        rebuild_cache: bool = False,
    ):
        self._per_unit = per_unit
        self._device = torch.device(device) if device else None

        # Lazy load mode: store the parquet df with bytes-encoded columns
        # and build tensors per __getitem__ instead of pre-computing all
        # rows in __init__. Mandatory for large-N datasets where each row's
        # dense Y arrays are multi-MB (e.g. LVN 722-bus snapshots: ~4.5 MB
        # per row uncompressed -> 36k rows = 160 GB RAM and OOM-kill).
        # Auto-detected by path pattern; eager path (default) unchanged.
        _path_list = path if isinstance(path, (list, tuple)) else [path]
        self._lazy = any("LVN_converted" in str(p) for p in _path_list)
        if self._lazy:
            log.info("ChanghunDataset: lazy-load mode enabled (auto-detected LVN_converted)")
            dfs = []
            for p in _path_list:
                dfs.append(pd.read_parquet(p, engine="pyarrow"))
            self._df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            self._df_columns = set(self._df.columns)
            log.info("ChanghunDataset[lazy] %d rows, columns: %s",
                     len(self._df), sorted(self._df_columns))

            # Pre-decode all bytes columns to numpy arrays so __getitem__
            # only does sparse->dense reconstruction + tensor conversion.
            # Without this, single-threaded data loading at 138 ms per
            # sample dominates training time (33k samples * 138 ms = 75
            # min per pass, blocking even on the first eval). With
            # pre-decode, __getitem__ drops to ~20-30 ms.
            # Memory: sparse rows are ~50 KB each (active edges only), so
            # 36k LVN rows -> ~1.8 GB -- safe.
            cols_to_decode = [
                c for c in [
                    "bus_typ", "Y_Lines", "Y_C_Lines", "Lines_connected",
                    "Y_matrix", "u_start", "u_newton", "S_start", "S_newton",
                    "I_newton", "vn_log",
                    "active_pair_idx", "active_Y_series", "active_Y_shunt",
                ] if c in self._df_columns
            ]
            log.info("ChanghunDataset[lazy] pre-decoding %d binary columns...",
                     len(cols_to_decode))
            import time as _time
            _t0 = _time.time()
            self._df = decode_columns_mp_columnwise(self._df, cols_to_decode)
            log.info("ChanghunDataset[lazy] pre-decode done in %.1fs", _time.time() - _t0)
            return

        cache_path: Path | None = None
        if use_cache:
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / f"changhun_rows_{self._cache_key(path, per_unit=per_unit)}.pt"
            if cache_path.exists() and not rebuild_cache:
                # weights_only=False: torch 2.6+ flips the default to True, which
                # rejects pickled numpy reconstructors. The cache file is written
                # by this same project (trusted source), so opt out.
                payload = torch.load(
                    cache_path, map_location=(self._device or "cpu"), weights_only=False
                )
                rows = payload.get("rows") if isinstance(payload, dict) else None
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    self.rows = rows
                    return

        # ---------- load & sanitise ------------------------------------------------
        df = pd.read_parquet(path, engine="pyarrow")
        binary_cols = [
            'bus_typ', 'Y_Lines', 'Y_C_Lines', 'Lines_connected',
            'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton', 'I_newton',
            # Optional LVN-only: per-bus voltage class log10(vn_kv).
            # Decoder filters out columns that don't exist in the dataframe.
            'vn_log',
        ]
        # Filter for columns that actually exist in the dataframe to avoid errors
        binary_cols_exist = [col for col in binary_cols if col in df.columns]
        log.info("Columns to be decoded: %s", binary_cols_exist)

        start_time = time.time()
        df = decode_columns_mp_columnwise(df, binary_cols_exist)
        colwise_time = time.time() - start_time
        log.info("Method 3: New MP (column-wise) time: %.2f seconds", colwise_time)

        log.info("Parquet read -> %s", df.shape)

        # vectorised decode list-columns
        for col in self._LIST_COLS:
            if col in df.columns:
                df[col] = df[col].map(_safe_list)

        # remove diverged rows where u_newton is all zeros
        if "u_newton" in df.columns:
            keep = ~df["u_newton"].map(_is_zero_complex_list)
            if not keep.all():
                df = df[keep].reset_index(drop=True)
                log.info("Removed %s diverged rows -> %s", int((~keep).sum()), df.shape)

        # ------------------------------------------------------------------
        # remove per-unit outliers in u_newton (IQR, aggregated across rows)
        # ------------------------------------------------------------------
        # The row-level filter assumes a row-IQR designed for HVN's 4-32 bus
        # grids. For datasets with hundreds of buses per row (e.g. LVN's
        # fixed 722-bus snapshots), natural per-bus voltage variation flags
        # every row as containing >=1 outlier and all rows get dropped.
        #
        # Bypass triggers:
        #   GNN_SKIP_OUTLIER=1                : explicit env-var override
        #   any path contains "LVN_converted" : auto-detect for converted LVN
        # Both reach the same effect (OUTLIER_K=inf disables removal).
        import os as _os
        _path_list = path if isinstance(path, (list, tuple)) else [path]
        _is_lvn_converted = any("LVN_converted" in str(p) for p in _path_list)
        _is_ieee = any("IEEE_" in str(p) for p in _path_list)
        _env_skip = _os.environ.get("GNN_SKIP_OUTLIER", "0") == "1"
        if _env_skip or _is_lvn_converted or _is_ieee:
            reason = ("GNN_SKIP_OUTLIER=1" if _env_skip
                      else "auto-detected LVN_converted" if _is_lvn_converted
                      else "auto-detected IEEE benchmark")
            log.info("%s -> bypassing per-unit u_newton outlier filter", reason)
            OUTLIER_K = float("inf")
        else:
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
                log.info("No per-unit %s data to analyze.", name)
                return
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            mask_out = (arr < lower) | (arr > upper)
            n_out = int(mask_out.sum())
            frac_out = n_out / arr.size * 100 if arr.size else 0.0
            log.info(">>> Per-Unit %s Outlier Analysis <<< Samples=%s Q1=%.3e Q3=%.3e IQR=%.3e Bounds=[%.3e, %.3e] Outliers=%s (%.2f%%) Span=[%.3e, %.3e]",
                     name,
                     int(arr.size),
                     float(q1),
                     float(q3),
                     float(iqr),
                     float(lower),
                     float(upper),
                     int(n_out),
                     float(frac_out),
                     float(arr.min()),
                     float(arr.max()),
            )

        if "u_newton" in df.columns:
            if "U_base" not in df.columns:
                log.warning("Skipping per-unit outlier removal: 'U_base' column not found.")
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
                    log.info("No per-unit data available to define outlier bounds; skipping row removal.")
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
                        log.info("Removed %s per-unit outlier rows -> %s", int(n_bad), df.shape)
                    else:
                        log.info("No rows contained per-unit u_newton outliers; nothing removed.")


        # ---------- convert to tensors --------------------------------------------
        
        def _to_t_impl(x, device, dtype=torch.float32):
            if isinstance(x, type) and x in (np.float32, np.float64, np.int32, np.int64, np.bool_):
                raise TypeError(f"to_t received a numpy scalar TYPE instead of a value: {x}")
            if isinstance(x, np.generic):
                x = x.item()
            if isinstance(x, np.ndarray) and x.dtype == object:
                if getattr(dtype, "is_complex", False):
                    x = x.astype(np.complex64)
                elif dtype in (torch.int32, torch.int64):
                    x = x.astype(np.int64)
                elif dtype is torch.bool:
                    x = x.astype(np.bool_)
                else:
                    x = x.astype(np.float32)
            if isinstance(x, (float, int, bool, complex)):
                return torch.tensor(x, device=device, dtype=dtype)
            return torch.as_tensor(x, device=device, dtype=dtype)

        self.rows: List[Dict[str, Any]] = []
        to_t  = functools.partial(_to_t_impl, device=self._device)

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

            # Optional per-bus voltage class feature (LVN). HVN data has no
            # vn_log column -- we emit zeros so the model's input dim stays
            # consistent across datasets when bus_feat_extra is enabled.
            if "vn_log" in df.columns:
                row["vn_log"] = to_t(np.asarray(r.vn_log, dtype=FLOAT_DTYPE))
            else:
                row["vn_log"] = to_t(np.zeros(int(N), dtype=FLOAT_DTYPE))

            # Admittance matrix (complex). If the dataset ships a precomputed
            # Y_matrix, use it; otherwise build Ybus from Y_Lines +
            # Lines_connected here so the collate function always has it
            # (collate.torch.block_diag requires Ybus in every sample).
            #
            # Convention matches models.edge_selfattn.admittance.build_dense_Y:
            #   off-diag: Y[j,i] = Y[i,j] = -y_series
            #   diag:    Y[k,k] += y_series + y_shunt at each incident edge
            if "Y_matrix" in df.columns:
                Ybus = _to_complex_array(r.Y_matrix).reshape(N, N) / Y_base
                row["Ybus"] = to_t(Ybus, dtype=torch.complex64)
            else:
                _pairs = np.triu_indices(N, k=1)  # (j_arr, i_arr) with j<i
                _mask = np.asarray(r.Lines_connected, dtype=bool)
                _y_s = (_to_complex_array(r.Y_Lines)[_mask] / float(Y_base)).astype(np.complex64)
                _raw_yc = r.Y_C_Lines
                if isinstance(_raw_yc, np.ndarray) and np.iscomplexobj(_raw_yc):
                    _y_c = (_raw_yc[_mask] / float(Y_base)).astype(np.complex64)
                else:
                    _y_c_re = _to_float_array(_raw_yc)[_mask] / float(Y_base)
                    _y_c = (1j * _y_c_re).astype(np.complex64)
                _j = _pairs[0][_mask].astype(np.int64)
                _i = _pairs[1][_mask].astype(np.int64)
                Ybus = np.zeros((N, N), dtype=np.complex64)
                np.add.at(Ybus, (_j, _i), -_y_s)
                np.add.at(Ybus, (_i, _j), -_y_s)
                _diag = np.zeros(N, dtype=np.complex64)
                np.add.at(_diag, _j, _y_s)
                np.add.at(_diag, _i, _y_s)
                np.add.at(_diag, _j, _y_c)
                np.add.at(_diag, _i, _y_c)
                Ybus[np.arange(N), np.arange(N)] += _diag
                row["Ybus"] = to_t(Ybus, dtype=torch.complex64)

            # Line admittances (series & shunt)
            if "Y_Lines" in df.columns:
                Y_lines = _to_complex_array(r.Y_Lines) / Y_base
                row["Y_Lines"] = to_t(Y_lines, dtype=torch.complex64)

            if "Y_C_Lines" in df.columns:
                # historically Y_C_Lines was real-valued (just the susceptance B).
                # LVN converter stores it as imaginary-only complex; flatten back
                # to float by taking the imag part if the column is complex.
                raw_yc = r.Y_C_Lines
                if isinstance(raw_yc, np.ndarray) and np.iscomplexobj(raw_yc):
                    YC = raw_yc.imag.astype(FLOAT_DTYPE) / float(Y_base)
                else:
                    YC = _to_float_array(raw_yc) / Y_base
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

        # Cache a CPU copy so it is portable across machines.
        if cache_path is not None:
            rows_cpu: list[dict[str, Any]] = []
            for row in self.rows:
                row2: dict[str, Any] = {}
                for k, v in row.items():
                    if isinstance(v, torch.Tensor):
                        row2[k] = v.detach().cpu()
                    else:
                        row2[k] = v
                rows_cpu.append(row2)
            torch.save(
                {
                    "rows": rows_cpu,
                    "meta": {
                        "per_unit": bool(per_unit),
                        "n_rows": int(len(rows_cpu)),
                    },
                },
                cache_path,
            )

    @staticmethod
    def _cache_key(path: Union[str, Path, Sequence[Union[str, Path]]], *, per_unit: bool) -> str:
        """Stable cache key based on inputs and file fingerprints."""

        def as_paths(p) -> list[Path]:
            if isinstance(p, (list, tuple)):
                return [Path(x) for x in p]
            return [Path(p)]

        parts: list[str] = ["changhun_v1", f"per_unit={int(per_unit)}"]
        for p in sorted(as_paths(path), key=lambda x: str(x)):
            try:
                st = p.stat()
                parts.append(str(p.resolve()))
                parts.append(str(st.st_size))
                parts.append(str(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))
            except FileNotFoundError:
                # Still include the path so cache invalidates if it later appears.
                parts.append(str(p))

        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return digest[:24]

    # ----------------------------------------------------------------------- #
    def __len__(self) -> int:
        if self._lazy:
            return len(self._df)
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._lazy:
            return self._build_row_lazy(idx)
        return self.rows[idx]

    # ----------------------------------------------------------------------- #
    # Lazy row builder: decode bytes for one row and assemble the tensor
    # dict on demand. Used by __getitem__ when self._lazy is True. Mirrors
    # the eager loop body in __init__; kept separate to avoid risky refactor
    # of the eager path. Tensors are built on CPU; the train loop moves them
    # to the GPU when iterating batches.
    # ----------------------------------------------------------------------- #
    def _build_row_lazy(self, idx: int) -> Dict[str, torch.Tensor]:
        from .npy_decode import npy_bytes_to_ndarray

        r_raw = self._df.iloc[idx]
        cols = self._df_columns

        # Decode the binary columns for THIS row only.
        # Includes both the legacy dense format (Y_Lines / Y_C_Lines /
        # Lines_connected) and the new sparse format (active_pair_idx /
        # active_Y_series / active_Y_shunt). The sparse format avoids the
        # 160 GB pd.read_parquet OOM on 36k LVN rows -- dense arrays are
        # 99.7% zero and were materializing as Python bytes objects.
        binary_cols = [
            "bus_typ", "Y_Lines", "Y_C_Lines", "Lines_connected",
            "Y_matrix", "u_start", "u_newton", "S_start", "S_newton",
            "I_newton", "vn_log",
            "active_pair_idx", "active_Y_series", "active_Y_shunt",
        ]
        decoded: Dict[str, Any] = {}
        for c in binary_cols:
            if c not in cols:
                continue
            v = r_raw[c]
            if isinstance(v, (bytes, bytearray)):
                decoded[c] = npy_bytes_to_ndarray(v)
            else:
                decoded[c] = v

        # Reconstruct dense Y_Lines / Y_C_Lines / Lines_connected from sparse
        # triplets when the parquet ships sparse columns. Per-row scratch
        # arrays of length E = N*(N-1)/2 ~ 2.1 MB each -- created and freed
        # per __getitem__, so memory is bounded by batch_size, not row count.
        if "active_pair_idx" in decoded:
            # Infer N from bus_typ to know E
            _N = int(len(decoded["bus_typ"])) if "bus_typ" in decoded else int(r_raw["bus_number"])
            _E = _N * (_N - 1) // 2
            _idx = np.asarray(decoded["active_pair_idx"], dtype=np.int64)
            _Ys = np.asarray(decoded["active_Y_series"], dtype=np.complex64)
            _Yc = np.asarray(decoded["active_Y_shunt"], dtype=np.complex64)
            _lc = np.zeros(_E, dtype=bool)
            _yl = np.zeros(_E, dtype=np.complex64)
            _yc = np.zeros(_E, dtype=np.complex64)
            _lc[_idx] = True
            _yl[_idx] = _Ys
            _yc[_idx] = _Yc
            decoded["Lines_connected"] = _lc
            decoded["Y_Lines"] = _yl
            decoded["Y_C_Lines"] = _yc

        per_unit = self._per_unit
        device = self._device

        # Tensor builder (CPU; train loop handles device transfer).
        def to_t(x, dtype=torch.float32):
            if isinstance(x, np.ndarray) and x.dtype == object:
                if getattr(dtype, "is_complex", False):
                    x = x.astype(np.complex64)
                elif dtype in (torch.int32, torch.int64):
                    x = x.astype(np.int64)
                elif dtype is torch.bool:
                    x = x.astype(np.bool_)
                else:
                    x = x.astype(np.float32)
            if isinstance(x, (float, int, bool, complex)):
                return torch.tensor(x, dtype=dtype)
            return torch.as_tensor(x, dtype=dtype)

        # Infer N
        if "bus_typ" in decoded:
            N = int(len(decoded["bus_typ"]))
        else:
            N = int(r_raw["bus_number"])

        S_base = np.array(r_raw["S_base"] if per_unit else 1.0, dtype=FLOAT_DTYPE)
        U_base = np.array(r_raw["U_base"] if per_unit else 1.0, dtype=FLOAT_DTYPE)
        Y_base = np.array((S_base / (U_base ** 2)) if per_unit else 1.0, dtype=FLOAT_DTYPE)
        I_base = np.array((S_base / U_base) if per_unit else 1.0, dtype=FLOAT_DTYPE)

        row: Dict[str, Any] = {
            "S_base": S_base,
            "U_base": U_base,
            "N": N,
            "bus_type": to_t(decoded["bus_typ"], dtype=torch.int64),
            "Lines_connected": to_t(decoded["Lines_connected"], dtype=torch.bool),
        }

        if "vn_log" in decoded:
            row["vn_log"] = to_t(np.asarray(decoded["vn_log"], dtype=FLOAT_DTYPE))
        else:
            row["vn_log"] = to_t(np.zeros(N, dtype=FLOAT_DTYPE))

        # Ybus: build from Y_Lines + Lines_connected (LVN_converted never
        # carries the dense Y_matrix -- too big).
        if "Y_matrix" in decoded:
            Ybus = _to_complex_array(decoded["Y_matrix"]).reshape(N, N) / Y_base
            row["Ybus"] = to_t(Ybus, dtype=torch.complex64)
        else:
            _pairs = np.triu_indices(N, k=1)
            _mask = np.asarray(decoded["Lines_connected"], dtype=bool)
            _y_s = (_to_complex_array(decoded["Y_Lines"])[_mask] / float(Y_base)).astype(np.complex64)
            _raw_yc = decoded["Y_C_Lines"]
            if isinstance(_raw_yc, np.ndarray) and np.iscomplexobj(_raw_yc):
                _y_c = (_raw_yc[_mask] / float(Y_base)).astype(np.complex64)
            else:
                _y_c_re = _to_float_array(_raw_yc)[_mask] / float(Y_base)
                _y_c = (1j * _y_c_re).astype(np.complex64)
            _j = _pairs[0][_mask].astype(np.int64)
            _i = _pairs[1][_mask].astype(np.int64)
            Ybus = np.zeros((N, N), dtype=np.complex64)
            np.add.at(Ybus, (_j, _i), -_y_s)
            np.add.at(Ybus, (_i, _j), -_y_s)
            _diag = np.zeros(N, dtype=np.complex64)
            np.add.at(_diag, _j, _y_s)
            np.add.at(_diag, _i, _y_s)
            np.add.at(_diag, _j, _y_c)
            np.add.at(_diag, _i, _y_c)
            Ybus[np.arange(N), np.arange(N)] += _diag
            row["Ybus"] = to_t(Ybus, dtype=torch.complex64)

        if "Y_Lines" in decoded:
            Y_lines = _to_complex_array(decoded["Y_Lines"]) / Y_base
            row["Y_Lines"] = to_t(Y_lines, dtype=torch.complex64)
        if "Y_C_Lines" in decoded:
            raw_yc = decoded["Y_C_Lines"]
            if isinstance(raw_yc, np.ndarray) and np.iscomplexobj(raw_yc):
                YC = raw_yc.imag.astype(FLOAT_DTYPE) / float(Y_base)
            else:
                YC = _to_float_array(raw_yc) / Y_base
            row["Y_C_Lines"] = to_t(YC)

        U_start = _to_complex_array(decoded["u_start"]) / U_base
        U_newton = _to_complex_array(decoded["u_newton"]) / U_base
        row["U_start"] = to_t(U_start, dtype=torch.complex64)
        row["U_newton"] = to_t(U_newton, dtype=torch.complex64)
        for name, U in [("start", U_start), ("newton", U_newton)]:
            mag = np.abs(U).astype(FLOAT_DTYPE)
            ang = np.angle(U).astype(FLOAT_DTYPE)
            row[f"V_{name}"] = to_t(np.stack([mag, ang], axis=1))

        if "S_start" in decoded:
            S_start = _to_complex_array(decoded["S_start"]) / S_base
            row["S_start"] = to_t(S_start, dtype=torch.complex64)
        if "S_newton" in decoded:
            S_newton = _to_complex_array(decoded["S_newton"]) / S_base
            row["S_newton"] = to_t(S_newton, dtype=torch.complex64)
        if "I_newton" in decoded:
            I_newton = _to_complex_array(decoded["I_newton"]) / I_base
            row["I_newton"] = to_t(I_newton, dtype=torch.complex64)

        return row
