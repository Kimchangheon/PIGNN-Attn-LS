# This script is a “parquet doctor” for your IEEE14 datasets. You point it at a parquet file, it loads the first row, and then:
# 	•	In canonical mode it checks: “Does this row match my own canonical IEEE14 (single-base) definition?”
# 	•	In pandapower mode it checks: “Does this row match what pandapower would produce for case14 with multi-voltage & per-bus scaling?”
#
# If everything matches, it prints PASS. Otherwise, it shows exactly what’s inconsistent.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io, argparse
import numpy as np
import pyarrow.parquet as pq

# ---- Helpers ----
def load_npy(b):
    return np.load(io.BytesIO(b), allow_pickle=False)

def upper_index(N, i, j):
    if i == j: raise ValueError("i and j must differ")
    if i > j:  i, j = j, i
    return ((2*N - i - 1) * i) // 2 + (j - i - 1)

def build_ieee14_reference(S_BASE=100e6, U_BASE=100e3, N=14):
    """
    Canonical IEEE-14 reference (MATPOWER/PSAT branch table).
    Returns upper-tri vectors already converted to SI using single U_BASE.
    """
    SCALE = S_BASE / (U_BASE**2)
    branches = [
        (1,  2, 0.01938, 0.05917, 0.02640, 0.0,   0.0),
        (1,  5, 0.05403, 0.22304, 0.02460, 0.0,   0.0),
        (2,  3, 0.04699, 0.19797, 0.02190, 0.0,   0.0),
        (2,  4, 0.05811, 0.17632, 0.01870, 0.0,   0.0),
        (2,  5, 0.05695, 0.17388, 0.01700, 0.0,   0.0),
        (3,  4, 0.06701, 0.17103, 0.01730, 0.0,   0.0),
        (4,  5, 0.01335, 0.04211, 0.00640, 0.0,   0.0),
        (4,  7, 0.00000, 0.20912, 0.00000, 0.978, 0.0),  # tap
        (4,  9, 0.00000, 0.55618, 0.00000, 0.969, 0.0),  # tap
        (5,  6, 0.00000, 0.25202, 0.00000, 0.932, 0.0),  # tap
        (6, 11, 0.09498, 0.19890, 0.00000, 0.0,   0.0),
        (6, 12, 0.12291, 0.25581, 0.00000, 0.0,   0.0),
        (6, 13, 0.06615, 0.13027, 0.00000, 0.0,   0.0),
        (7,  8, 0.00000, 0.17615, 0.00000, 0.0,   0.0),
        (7,  9, 0.00000, 0.11001, 0.00000, 0.0,   0.0),
        (9, 10, 0.03181, 0.08450, 0.00000, 0.0,   0.0),
        (9, 14, 0.12711, 0.27038, 0.00000, 0.0,   0.0),
        (10,11, 0.08205, 0.19207, 0.00000, 0.0,   0.0),
        (12,13, 0.22092, 0.19988, 0.00000, 0.0,   0.0),
        (13,14, 0.17093, 0.34802, 0.00000, 0.0,   0.0),
    ]
    C = N*(N-1)//2
    Lines_connected = np.zeros(C, dtype=int)
    Is_trafo        = np.zeros(C, dtype=int)
    Y_Lines_SI      = np.zeros(C, dtype=complex)
    Y_C_Lines_SI    = np.zeros(C, dtype=float)
    Tau             = np.ones (C, dtype=float)
    Shift_deg       = np.zeros(C, dtype=float)
    Trafo_y_SI      = np.zeros(C, dtype=complex)
    Trafo_b_SI      = np.zeros(C, dtype=float)

    for (i1,j1,r,x,b_tot,tap,shift) in branches:
        i0, j0 = i1-1, j1-1
        k = upper_index(N, i0, j0)
        Lines_connected[k] = 1
        z = complex(r,x)
        y_pu = 0j if abs(z) < 1e-15 else 1.0/z
        if tap and abs(tap) > 1e-12:  # transformer
            Is_trafo[k]  = 1
            Tau[k]       = float(tap)
            Shift_deg[k] = float(shift)
            Trafo_y_SI[k]= y_pu * SCALE
            Trafo_b_SI[k]= float(b_tot) * SCALE   # 0 in this case
        else:  # line
            Y_Lines_SI[k]   = y_pu * SCALE
            Y_C_Lines_SI[k] = 0.5 * float(b_tot) * SCALE  # per-end
    return (Lines_connected, Is_trafo, Y_Lines_SI, Y_C_Lines_SI,
            Tau, Shift_deg, Trafo_y_SI, Trafo_b_SI)

def stamp_Y_from_vectors(N, Y_lines_SI, Yc_per_end_SI, is_trafo, tau, shift_deg, y_tr_SI, btr_total_SI):
    Y = np.zeros((N,N), dtype=complex)
    iu, ju = np.triu_indices(N, 1)

    # Lines
    mask_line = (Y_lines_SI != 0)
    if np.any(mask_line):
        ii = iu[mask_line]; jj = ju[mask_line]
        y  = Y_lines_SI[mask_line]
        b  = Yc_per_end_SI[mask_line]
        Y[ii,ii] += y + 1j*b
        Y[jj,jj] += y + 1j*b
        Y[ii,jj] += -y
        Y[jj,ii] += -y

    # Transformers (tap on 'ii' side)
    mask_xf = (is_trafo == 1)
    if np.any(mask_xf):
        ii = iu[mask_xf]; jj = ju[mask_xf]
        y  = y_tr_SI[mask_xf]
        bT = btr_total_SI[mask_xf]
        t  = tau[mask_xf] * np.exp(1j*np.deg2rad(shift_deg[mask_xf]))
        Y[ii,ii] += (y + 1j*(bT/2.0)) / (np.abs(t)**2)
        Y[jj,jj] +=  (y + 1j*(bT/2.0))
        Y[ii,jj] += -(y / np.conj(t))
        Y[jj,ii] += -(y / t)
    return Y

# ---- Validation runners ----
def validate_canonical(row):
    """Validate against canonical branch/tap table (single-base SI)."""
    ok = True
    def req(name, cond):
        nonlocal ok
        if not cond:
            ok = False; print(f"[FAIL] {name}")
        return cond
    def cmp_vec(name, a, b, atol=1e-8):
        nonlocal ok
        if a.dtype.kind in "fc":
            same = np.allclose(a, b, atol=atol, rtol=0)
        else:
            same = np.array_equal(a, b)
        if same: print(f"[OK]   {name}")
        else:
            ok = False
            idx = np.where(~np.isclose(a, b, atol=atol, rtol=0) if a.dtype.kind in "fc" else (a!=b))[0]
            print(f"[FAIL] {name}: {idx.size} mismatches. First 10 idx:", idx[:10])

    N = int(row['bus_number']); C = N*(N-1)//2
    bus_typ    = load_npy(row['bus_typ'])
    Y_lines    = load_npy(row['Y_Lines']).astype(complex)
    Yc_lines   = load_npy(row['Y_C_Lines']).astype(float)
    conn       = load_npy(row['Lines_connected']).astype(int)
    is_trafo   = load_npy(row['Is_trafo']).astype(int)
    tau        = load_npy(row['Trafo_tau']).astype(float)
    shift_deg  = load_npy(row['Trafo_shift_deg']).astype(float)
    y_tr       = load_npy(row['Trafo_y_series']).astype(complex)
    b_tr_tot   = load_npy(row['Trafo_b_total']).astype(float)
    Y_bytes    = row.get('Y_matrix', None)
    Y_parquet  = load_npy(Y_bytes).reshape(N,N) if Y_bytes is not None else None

    print(f"[INFO] canonical mode, N={N}, C={C}")
    for name,arr in [
        ("Y_Lines", Y_lines), ("Y_C_Lines",Yc_lines), ("Lines_connected",conn),
        ("Is_trafo", is_trafo), ("Trafo_tau",tau), ("Trafo_shift_deg",shift_deg),
        ("Trafo_y_series", y_tr), ("Trafo_b_total", b_tr_tot)
    ]: req(f"len({name})==C", len(arr)==C)

    (conn_ref, xf_ref, yline_ref, yc_ref, tau_ref, sh_ref, ytr_ref, btr_ref) = build_ieee14_reference()
    cmp_vec("Lines_connected", conn, conn_ref, atol=0)
    cmp_vec("Is_trafo", is_trafo, xf_ref, atol=0)
    cmp_vec("Trafo_tau", tau, tau_ref)
    cmp_vec("Trafo_shift_deg", shift_deg, sh_ref)
    cmp_vec("Y_Lines (SI)", Y_lines, yline_ref)
    cmp_vec("Y_C_Lines per-end (SI)", Yc_lines, yc_ref)
    cmp_vec("Trafo_y_series (SI)", y_tr, ytr_ref)
    cmp_vec("Trafo_b_total (SI)", b_tr_tot, btr_ref)

    Y_expected = stamp_Y_from_vectors(N, Y_lines, Yc_lines, is_trafo, tau, shift_deg, y_tr, b_tr_tot)
    if Y_parquet is not None:
        if np.allclose(Y_parquet, Y_expected, atol=1e-8, rtol=0):
            print("[OK]   Y_matrix matches expected stamping.")
        else:
            ok = False
            d = Y_parquet - Y_expected
            print("[FAIL] Y_matrix differs from expected. Max abs diff:", np.max(np.abs(d)))

    # IEEE-14 bus types (slack=1, PV={2,3,6,8})
    bus_typ_ref = np.array([1,2,2,3,3,2,3,2,3,3,3,3,3,3], dtype=int)
    if np.array_equal(bus_typ, bus_typ_ref): print("[OK]   bus_typ matches IEEE-14 types.")
    else: ok=False; print("[FAIL] bus_typ differs. Got:", bus_typ)

    print("\n== SUMMARY ==")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

def validate_pandapower(row):
    """
    Validate using pandapower as source of truth:
    - Read vn_kv from file, rebuild p.u. Ybus from pandapower.
    - Convert to SI with per-bus scaling and compare to saved Y_matrix.
    - Check line/trafo connectivity vectors against net.line / net.trafo.
    """
    import pandapower as pp
    import pandapower.networks as pn

    ok = True
    def req(name, cond):
        nonlocal ok
        if not cond:
            ok = False; print(f"[FAIL] {name}")
        return cond
    def cmp_vec(name, a, b, atol=0):
        nonlocal ok
        same = np.array_equal(a, b) if atol==0 else np.allclose(a, b, atol=atol, rtol=0)
        if same: print(f"[OK]   {name}")
        else:
            ok = False
            idx = np.where(~np.isclose(a, b, atol=atol, rtol=0) if atol else (a!=b))[0]
            print(f"[FAIL] {name}: {idx.size} mismatches. First 10 idx:", idx[:10])

    N = int(row['bus_number']); C = N*(N-1)//2
    vn_kv   = load_npy(row['vn_kv']).astype(float)
    Y_bytes = row.get('Y_matrix', None)
    Y_SI    = load_npy(Y_bytes).reshape(N,N) if Y_bytes is not None else None
    conn    = load_npy(row['Lines_connected']).astype(int)
    is_xf   = load_npy(row['Is_trafo']).astype(int)

    print(f"[INFO] pandapower mode, N={N}, C={C}")
    req("have Y_matrix", Y_SI is not None)

    # Build pandapower reference
    net = pn.case14()
    pp.runpp(net, calculate_voltage_angles=True)
    Ypu = net._ppc["internal"]["Ybus"].toarray().astype(np.complex128)
    S_base = float(net.sn_mva) * 1e6
    Vb = vn_kv * 1e3
    denom = np.outer(Vb, Vb)
    Y_SI_ref = Ypu * (S_base / denom)

    # Compare Y-matrix (SI)
    if np.allclose(Y_SI, Y_SI_ref, atol=1e-8, rtol=0):
        print("[OK]   Y_matrix matches pandapower (SI with per-bus scaling).")
    else:
        ok = False
        d = Y_SI - Y_SI_ref
        print("[FAIL] Y_matrix vs pandapower. Max abs diff:", np.max(np.abs(d)))

    # Upper-tri indexing map
    iu, ju = np.triu_indices(N, 1)
    pair_to_idx = {(int(i), int(j)): k for k,(i,j) in enumerate(zip(iu, ju))}

    # Build reference connectivity from net.line / net.trafo
    conn_ref = np.zeros(C, dtype=int)
    for _, r in net.line.iterrows():
        a,b = int(r.from_bus), int(r.to_bus)
        i,j = (a,b) if a<b else (b,a)
        if (i,j) in pair_to_idx:
            conn_ref[pair_to_idx[(i,j)]] = 1
    xf_ref = np.zeros(C, dtype=int)
    for _, r in net.trafo.iterrows():
        a,b = int(r.hv_bus), int(r.lv_bus)
        i,j = (a,b) if a<b else (b,a)
        if (i,j) in pair_to_idx:
            xf_ref[pair_to_idx[(i,j)]] = 1

    cmp_vec("Lines_connected (topology)", conn, conn_ref, atol=0)
    cmp_vec("Is_trafo (topology)", is_xf, xf_ref, atol=0)

    print("\n== SUMMARY ==")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Validate IEEE-14 parquet (canonical or pandapower).")
    ap.add_argument("parquet_file", help="Path to parquet file")
    ap.add_argument("--mode", choices=["auto", "canonical", "pandapower"], default="auto",
                    help="Validation mode (default: auto-detect via vn_kv uniqueness)")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.parquet_file)
    tb = pf.read_row_group(0)
    row = {tb.schema.names[i]: tb.column(i)[0].as_py() for i in range(len(tb.schema.names))}

    # quick sanity
    N = int(row['bus_number'])
    if N != 14:
        print(f"[FAIL] bus_number={N} (expected 14)")
        sys.exit(1)

    vn_kv = load_npy(row['vn_kv']).astype(float)
    unique_kv = np.unique(np.round(vn_kv, 6))

    mode = args.mode
    if mode == "auto":
        mode = "pandapower" if unique_kv.size > 1 else "canonical"
        print(f"[INFO] auto-selected mode: {mode}")

    if mode == "canonical":
        rc = validate_canonical(row)
    else:
        try:
            rc = validate_pandapower(row)
        except ImportError:
            print("[FAIL] pandapower is required for --mode pandapower (pip install pandapower).")
            rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    main()

# Canonical preset (single base):
# python validate_ieee14_parquet.py --mode canonical IEEE14_1_NR_plain.parquet

#Pandapower preset (multi-voltage, per-bus scaling):
# python validate_ieee14_parquet.py --mode pandapower ieee14_pandapower_1_NR_plain.parquet

# Auto detect
# python validate_ieee14_parquet.py IEEE14_1_NR_plain.parquet