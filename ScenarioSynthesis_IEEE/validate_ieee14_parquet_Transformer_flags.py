#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io
import math
import numpy as np
import pyarrow.parquet as pq

# Import your generator to get a clean reference
from case_generator_improved import case_generation_ieee14_pandapower


def load_npy(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def print_vec_mismatch(name, a, b, atol=1e-8, rtol=0.0, max_show=10):
    """
    Helper: print indices of mismatches between two vectors.
    """
    if a.dtype.kind in "fc":
        mask = ~np.isclose(a, b, atol=atol, rtol=rtol)
    else:
        mask = (a != b)
    idx = np.where(mask)[0]
    n_bad = idx.size
    print(f"[FAIL] {name}: {n_bad} mismatches.")
    if n_bad:
        print("       First indices:", idx[:max_show])


def compare_vec(name, a, b, atol=1e-8, rtol=0.0):
    """
    Returns True if equal (or nearly equal), False otherwise, with printing.
    """
    if a.dtype.kind in "fc":
        ok = np.allclose(a, b, atol=atol, rtol=rtol)
    else:
        ok = np.array_equal(a, b)
    if ok:
        print(f"[OK]   {name}")
    else:
        print_vec_mismatch(name, a, b, atol=atol, rtol=rtol)
    return ok


def stamp_Y_from_vectors(
    N: int,
    Y_lines_SI: np.ndarray,
    Yc_per_end_SI: np.ndarray,
    is_trafo: np.ndarray,
    tau: np.ndarray,
    shift_deg: np.ndarray,
    y_tr_SI: np.ndarray,
    btr_total_SI: np.ndarray,
) -> np.ndarray:
    """
    Rebuild Y-bus (SI) from upper-tri vectors with the same stamping convention as the generator:

      * Lines:
          - Y_ii += y + j*b
          - Y_jj += y + j*b
          - Y_ij += -y
          - Y_ji += -y

      * Transformers:
          Off-nominal tap on the 'from' (lower-index) side:
          t = tau * exp(j*shift)
          y = y_tr_SI
          bT = total magnetizing susceptance (split bT/2 to each side)

          - Y_ii += (y + j*(bT/2)) / |t|^2
          - Y_jj +=  (y + j*(bT/2))
          - Y_ij += -(y / conj(t))
          - Y_ji += -(y / t)
    """
    Y = np.zeros((N, N), dtype=complex)
    iu, ju = np.triu_indices(N, 1)

    # Lines
    mask_line = (Y_lines_SI != 0)
    if np.any(mask_line):
        ii = iu[mask_line]
        jj = ju[mask_line]
        y = Y_lines_SI[mask_line]
        b = Yc_per_end_SI[mask_line]

        Y[ii, ii] += y + 1j * b
        Y[jj, jj] += y + 1j * b
        Y[ii, jj] += -y
        Y[jj, ii] += -y

    # Transformers
    mask_xf = (is_trafo == 1)
    if np.any(mask_xf):
        ii = iu[mask_xf]
        jj = ju[mask_xf]
        y = y_tr_SI[mask_xf]
        bT = btr_total_SI[mask_xf]
        t = tau[mask_xf] * np.exp(1j * np.deg2rad(shift_deg[mask_xf]))

        # Diagonals
        Y[ii, ii] += (y + 1j * (bT / 2.0)) / (np.abs(t) ** 2)
        Y[jj, jj] += (y + 1j * (bT / 2.0))

        # Off-diagonals
        Y[ii, jj] += -(y / np.conj(t))
        Y[jj, ii] += -(y / t)

    return Y

import numpy as np

def stamp_Y_from_vectors(
    N: int,
    Lines_connected: np.ndarray,
    Y_Lines: np.ndarray,
    Y_C_Lines: np.ndarray,
    Is_trafo: np.ndarray,
    Trafo_tau: np.ndarray,
    Trafo_shift_deg: np.ndarray,
    Trafo_y_series: np.ndarray,
    Trafo_b_total: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct nodal Y_matrix (engineering units, Siemens) from edge-level vectors.

    All admittances are in SI:
      - Y_Lines[k]       : series admittance (S) of line branch k
      - Y_C_Lines[k]     : per-end shunt susceptance (S) of line branch k
      - Lines_connected  : 1 if upper-tri pair is a physical line

      - Is_trafo[k]      : 1 if upper-tri pair is a transformer branch
      - Trafo_tau[k]     : off-nominal tap magnitude τ on the 'from'/i side
      - Trafo_shift_deg  : tap phase shift in degrees (electrical)
      - Trafo_y_series   : transformer series admittance (S)
      - Trafo_b_total    : total magnetizing susceptance (S) for that branch;
                           stamping uses b_total/2 on each side.

    Conventions:
      - Upper-tri indexing via: iu, ju = np.triu_indices(N, 1)
      - Transformers use the tap on the 'i' side (where i < j). This must match
        the way you computed Trafo_tau / Trafo_shift_deg / Trafo_y_series.
    """

    Y = np.zeros((N, N), dtype=np.complex128)
    iu, ju = np.triu_indices(N, 1)

    # --------- Lines (non-transformer branches) ----------
    # We treat as π-model:
    #   self:  Y_ii += y + j b_end
    #          Y_jj += y + j b_end
    #   mutual: Y_ij += -y, Y_ji += -y
    line_mask = (Lines_connected.astype(bool) & (Is_trafo == 0))
    if np.any(line_mask):
        ii = iu[line_mask]
        jj = ju[line_mask]

        y = Y_Lines[line_mask]      # series admittance (S)
        b = Y_C_Lines[line_mask]    # per-end shunt (S)

        Y[ii, ii] += y + 1j * b
        Y[jj, jj] += y + 1j * b
        Y[ii, jj] += -y
        Y[jj, ii] += -y

    # --------- Transformers ----------
    # Standard off-nominal tap model, tap on 'i' side:
    #   t = τ * e^{jφ}, φ in radians
    #
    #   Y_ii += (y_tr + j b_total/2) / |t|^2
    #   Y_jj +=  (y_tr + j b_total/2)
    #   Y_ij += -y_tr / conj(t)
    #   Y_ji += -y_tr / t
    xf_mask = (Is_trafo == 1)
    if np.any(xf_mask):
        ii = iu[xf_mask]
        jj = ju[xf_mask]

        y_tr = Trafo_y_series[xf_mask]         # series admittance (S)
        b_tot = Trafo_b_total[xf_mask]         # total magnetizing susceptance (S)
        tau = Trafo_tau[xf_mask]
        shift_rad = np.deg2rad(Trafo_shift_deg[xf_mask])

        t = tau * np.exp(1j * shift_rad)
        t_abs2 = np.abs(t) ** 2

        Y[ii, ii] += (y_tr + 1j * (b_tot / 2.0)) / t_abs2
        Y[jj, jj] += (y_tr + 1j * (b_tot / 2.0))

        Y[ii, jj] += -(y_tr / np.conj(t))
        Y[jj, ii] += -(y_tr / t)

    return Y


def main(parquet_path: str):
    print(f"[INFO] Validating Parquet file: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.num_row_groups
    print(f"[INFO] Row groups: {n_row_groups}")

    # ---- Load first row as representative sample ----
    tb0 = pf.read_row_group(0)
    row0 = {tb0.schema.names[i]: tb0.column(i)[0].as_py()
            for i in range(len(tb0.schema.names))}

    # Decode scalar/meta
    N0 = int(row0["bus_number"])
    gridtype0 = str(row0["gridtype"])
    U_base0 = float(row0["U_base"])
    S_base0 = float(row0["S_base"])

    print(f"[INFO] First row: N={N0}, gridtype={gridtype0}, U_base={U_base0}, S_base={S_base0}")

    # Decode arrays from first row
    bus_typ0 = load_npy(row0["bus_typ"])
    Y_Lines0 = load_npy(row0["Y_Lines"]).astype(complex)
    Y_C_Lines0 = load_npy(row0["Y_C_Lines"]).astype(float)
    Lines_connected0 = load_npy(row0["Lines_connected"]).astype(np.int8)

    vn_kv0 = load_npy(row0["vn_kv"]).astype(float)
    Is_trafo0 = load_npy(row0["Is_trafo"]).astype(np.int8)
    Trafo_tau0 = load_npy(row0["Trafo_tau"]).astype(float)
    Trafo_shift_deg0 = load_npy(row0["Trafo_shift_deg"]).astype(float)
    Trafo_y_series0 = load_npy(row0["Trafo_y_series"]).astype(complex)
    Trafo_b_total0 = load_npy(row0["Trafo_b_total"]).astype(float)

    u_start0 = load_npy(row0["u_start"]).astype(complex)
    S_start0 = load_npy(row0["S_start"]).astype(complex)
    u_newton0 = load_npy(row0["u_newton"]).astype(complex)
    S_newton0 = load_npy(row0["S_newton"]).astype(complex)

    if "Y_matrix" in row0:
        Y_matrix0 = load_npy(row0["Y_matrix"]).reshape(N0, N0).astype(complex)
    else:
        Y_matrix0 = None

    # Basic dimension checks
    ok = True
    C0 = N0 * (N0 - 1) // 2
    print(f"[INFO] upper-tri length C={C0}")

    if N0 != 14:
        print("[FAIL] bus_number != 14")
        ok = False
    else:
        print("[OK]   bus_number == 14")

    if gridtype0.lower() != "ieee14":
        print(f"[FAIL] gridtype is {gridtype0}, expected 'IEEE14'")
        ok = False
    else:
        print("[OK]   gridtype == 'IEEE14'")

    for name, arr in [
        ("bus_typ", bus_typ0),
        ("Y_Lines", Y_Lines0),
        ("Y_C_Lines", Y_C_Lines0),
        ("Lines_connected", Lines_connected0),
        ("vn_kv", vn_kv0),
        ("Is_trafo", Is_trafo0),
        ("Trafo_tau", Trafo_tau0),
        ("Trafo_shift_deg", Trafo_shift_deg0),
        ("Trafo_y_series", Trafo_y_series0),
        ("Trafo_b_total", Trafo_b_total0),
    ]:
        if name in ("bus_typ", "vn_kv"):
            exp_len = N0
        else:
            exp_len = C0
        if len(arr) != exp_len:
            print(f"[FAIL] len({name}) = {len(arr)} != {exp_len}")
            ok = False
        else:
            print(f"[OK]   len({name}) = {exp_len}")

    # ----- Build a clean reference via case_generation_ieee14_pandapower() -----
    print("\n[INFO] Building reference via case_generation_ieee14_pandapower() ...")
    (gridtype_ref,
     bus_typ_ref,
     S_ref,
     u_start_ref,
     Y_ref,
     is_connected_ref,
     Y_Lines_ref,
     Y_C_Lines_ref,
     Lines_connected_ref,
     U_base_ref,
     S_base_ref,
     vn_kv_ref,
     Is_trafo_ref,
     Trafo_tau_ref,
     Trafo_shift_deg_ref,
     Trafo_y_series_ref,
     Trafo_b_total_ref) = case_generation_ieee14_pandapower()

    # --- Compare scalar/meta fields against reference (where meaningful) ---
    if gridtype_ref.lower() != "ieee14":
        print(f"[WARN] Reference gridtype={gridtype_ref}, expected 'IEEE14' (from generator).")
    if not math.isnan(U_base0):
        print(f"[WARN] U_base in parquet is {U_base0}, expected NaN for multi-voltage preset.")

    if not np.isclose(S_base0, S_base_ref, rtol=0, atol=1e-6):
        print(f"[FAIL] S_base mismatch: parquet={S_base0}, ref={S_base_ref}")
        ok = False
    else:
        print("[OK]   S_base matches reference")

    # --- Compare structural / topology arrays to reference (these should be identical for all IEEE14_pp cases) ---
    print("\n[INFO] Comparing structural arrays of first row to reference ...")
    ok &= compare_vec("bus_typ", bus_typ0, bus_typ_ref)
    ok &= compare_vec("vn_kv", vn_kv0, vn_kv_ref)
    ok &= compare_vec("Lines_connected", Lines_connected0, Lines_connected_ref)
    ok &= compare_vec("Is_trafo", Is_trafo0, Is_trafo_ref)
    ok &= compare_vec("Y_Lines (series admittance, SI)", Y_Lines0, Y_Lines_ref)
    ok &= compare_vec("Y_C_Lines (per-end shunt, SI)", Y_C_Lines0, Y_C_Lines_ref)
    ok &= compare_vec("Trafo_tau", Trafo_tau0, Trafo_tau_ref)
    ok &= compare_vec("Trafo_shift_deg", Trafo_shift_deg0, Trafo_shift_deg_ref)
    ok &= compare_vec("Trafo_y_series (SI)", Trafo_y_series0, Trafo_y_series_ref)
    ok &= compare_vec("Trafo_b_total (SI)", Trafo_b_total0, Trafo_b_total_ref)

    # --- Rebuild Y from *parquet vectors* and compare with saved Y_matrix ---
    if Y_matrix0 is not None:
        print("\n[INFO] Reconstructing Y from vectors in parquet and comparing to saved Y_matrix ...")
        Y_recon = stamp_Y_from_vectors(
            N0,
            Lines_connected0,
            Y_Lines0,
            Y_C_Lines0,
            Is_trafo0,
            Trafo_tau0,
            Trafo_shift_deg0,
            Trafo_y_series0,
            Trafo_b_total0,
        )
        if np.allclose(Y_matrix0, Y_recon, atol=1e-8, rtol=0):
            print("[OK]   Y_matrix (saved) matches Y reconstructed from line/trafo vectors.")
        else:
            ok = False
            diff = Y_matrix0 - Y_recon
            print("[FAIL] Y_matrix vs reconstructed Y differ. Max |ΔY| =", np.max(np.abs(diff)))
            # show largest few differences
            flat_idx = np.argsort(np.abs(diff).ravel())[::-1][:10]
            for fi in flat_idx:
                i, j = divmod(fi, N0)
                print(f"       ΔY[{i},{j}] = {diff[i,j]}  (Y_saved={Y_matrix0[i,j]}, Y_rec={Y_recon[i,j]})")
    else:
        print("[WARN] Y_matrix not saved in parquet; skipping saved-vs-reconstructed check.")

    # --- Optional: compare Y_matrix to reference Y as well (global canonical check) ---
    if Y_matrix0 is not None:
        print("\n[INFO] Comparing saved Y_matrix to reference Y from generator ...")
        if Y_ref.shape != Y_matrix0.shape:
            print(f"[FAIL] Y_matrix shapes differ: parquet {Y_matrix0.shape}, ref {Y_ref.shape}")
            ok = False
        else:
            if np.allclose(Y_matrix0, Y_ref, atol=1e-8, rtol=0):
                print("[OK]   Y_matrix matches reference Y (within 1e-8)")
            else:
                ok = False
                diff = Y_matrix0 - Y_ref
                print("[FAIL] Y_matrix differs from reference Y, max |ΔY| =", np.max(np.abs(diff)))
    else:
        print("[WARN] Y_matrix not saved in parquet; skipping Y vs reference check.")

    # --- Soft checks on S_start and u_start (only exact for jitter_load=0, jitter_gen=0, rand_u_start=False) ---
    print("\n[INFO] Soft checks on first-row injections and initial voltages ...")
    slack_mask = (bus_typ0 == 1)
    pv_mask = (bus_typ0 == 2)
    pq_mask = (bus_typ0 == 3)

    # Slack: we *expect* S_start[slack] ≈ 0 + j0 in your NR formulation
    if S_start0[slack_mask].size:
        slack_S = S_start0[slack_mask]
        if np.allclose(slack_S, 0.0 + 0.0j, atol=1e-3):
            print("[OK]   Slack bus S_start ≈ 0 (as NR formulation expects)")
        else:
            print("[WARN] Slack bus S_start not zero. Values:", slack_S)

    # PV: Q part should be ~0 in S_spec (only P given)
    if S_start0[pv_mask].size:
        pv_Q = S_start0[pv_mask].imag
        if np.allclose(pv_Q, 0.0, atol=1e-3):
            print("[OK]   PV buses have Q_spec ≈ 0 in S_start")
        else:
            print("[WARN] PV buses have non-zero Q_spec in S_start. Q:", pv_Q)

    # u_start magnitudes vs Vbase (should be equal if rand_u_start=False)
    if np.allclose(np.abs(u_start0), vn_kv0 * 1e3, atol=1e-3):
        print("[OK]   |u_start| matches vn_kv*1e3 (flat start, 1.0 pu)")
    else:
        print("[WARN] |u_start| differs from vn_kv*1e3 – maybe rand_u_start=True or jitter in setpoints?")

    # --- Check subsequent rows for consistent structure (topology & parameters) ---
    print("\n[INFO] Checking all rows for consistent topology & parameters ...")
    total_rows = 0
    for rg_idx in range(n_row_groups):
        tb = pf.read_row_group(rg_idx)
        n_rows_rg = tb.num_rows
        total_rows += n_rows_rg

        for r in range(n_rows_rg):
            row = {tb.schema.names[i]: tb.column(i)[r].as_py()
                   for i in range(len(tb.schema.names))}
            N = int(row["bus_number"])
            if N != N0:
                print(f"[FAIL] Row-group {rg_idx}, row {r}: bus_number={N} != {N0}")
                ok = False
                continue

            bt = load_npy(row["bus_typ"])
            ln = load_npy(row["Lines_connected"]).astype(np.int8)
            it = load_npy(row["Is_trafo"]).astype(np.int8)
            yl = load_npy(row["Y_Lines"]).astype(complex)
            ycl = load_npy(row["Y_C_Lines"]).astype(float)
            vk = load_npy(row["vn_kv"]).astype(float)
            tt = load_npy(row["Trafo_tau"]).astype(float)
            ts = load_npy(row["Trafo_shift_deg"]).astype(float)
            ytr = load_npy(row["Trafo_y_series"]).astype(complex)
            btr = load_npy(row["Trafo_b_total"]).astype(float)

            if not np.array_equal(bt, bus_typ0):
                print(f"[FAIL] bus_typ differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.array_equal(ln, Lines_connected0):
                print(f"[FAIL] Lines_connected differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.array_equal(it, Is_trafo0):
                print(f"[FAIL] Is_trafo differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(yl, Y_Lines0, atol=1e-8, rtol=0):
                print(f"[FAIL] Y_Lines differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(ycl, Y_C_Lines0, atol=1e-8, rtol=0):
                print(f"[FAIL] Y_C_Lines differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(vk, vn_kv0, atol=1e-8, rtol=0):
                print(f"[FAIL] vn_kv differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(tt, Trafo_tau0, atol=1e-8, rtol=0):
                print(f"[FAIL] Trafo_tau differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(ts, Trafo_shift_deg0, atol=1e-8, rtol=0):
                print(f"[FAIL] Trafo_shift_deg differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(ytr, Trafo_y_series0, atol=1e-8, rtol=0):
                print(f"[FAIL] Trafo_y_series differs in row-group {rg_idx}, row {r}")
                ok = False
                break
            if not np.allclose(btr, Trafo_b_total0, atol=1e-8, rtol=0):
                print(f"[FAIL] Trafo_b_total differs in row-group {rg_idx}, row {r}")
                ok = False
                break

    print(f"[INFO] Total rows checked: {total_rows}")

    print("\n== SUMMARY ==")
    if ok:
        print("PASS: All IEEE14_pandapower cases have consistent topology/parameters,")
        print("      Y_matrix matches line/trafo vectors, and agree with the reference preset.")
        sys.exit(0)
    else:
        print("FAIL: See messages above for inconsistencies.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: validate_ieee14_pandapower_parquet.py <parquet_file>")
        sys.exit(2)
    main(sys.argv[1])