#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from case_generator_improved import (
    case_generation_ieee14_pandapower,
    case_generation_ieee14_pandapower_stamped,
)

def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: {a.shape} vs {b.shape}")
        return np.inf
    if a.size == 0:
        return 0.0
    return np.max(np.abs(a - b))

def compare_arrays(name, a, b, atol=1e-12, rtol=1e-9):
    a = np.asarray(a)
    b = np.asarray(b)
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    mad = max_abs_diff(a, b)
    print(f"[{name}] allclose={ok}, max|Δ|={mad:.3e}, shape={a.shape}")
    if not ok:
        # show a few largest differences
        diff = np.abs(a - b)
        flat_idx = np.argsort(diff.ravel())[::-1]
        print(f"  Top 5 entries by |Δ{ name }|:")
        for k in flat_idx[:5]:
            idx = np.unravel_index(k, diff.shape)
            print(f"    {name}{idx} : a={a[idx]}, b={b[idx]}, |Δ|={diff[idx]:.3e}")
    return ok

def main():
    # --- Common parameters (must be identical for both calls) ---
    kwargs = dict(
        jitter_load=0.10,
        jitter_gen=0.10,
        pv_vset_range=(0.98, 1.04),
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        seed=12345,   # fixed seed for reproducibility
    )

    print("Running case_generation_ieee14_pandapower ...")
    res1 = case_generation_ieee14_pandapower(**kwargs)

    print("Running case_generation_ieee14_pandapower_stamped ...")
    res2 = case_generation_ieee14_pandapower_stamped(**kwargs)

    print("\nLength of returns:", len(res1), len(res2))
    if len(res1) != len(res2):
        print("ERROR: return lengths differ!")
        return

    # Unpack assuming 18-tuple including Y_shunt_bus
    (gridtype1, bus_typ1, s_multi1, u_start1, Y_matrix1, is_connected1,
     Y_Lines1, Y_C_Lines1, Lines_connected1, U_base1, S_base1,
     vn_kv1, Is_trafo1, Trafo_tau1, Trafo_shift_deg1,
     Trafo_y_series1, Trafo_b_total1, Y_shunt_bus1, Trafo_tap_on_i1) = res1

    (gridtype2, bus_typ2, s_multi2, u_start2, Y_matrix2, is_connected2,
     Y_Lines2, Y_C_Lines2, Lines_connected2, U_base2, S_base2,
     vn_kv2, Is_trafo2, Trafo_tau2, Trafo_shift_deg2,
     Trafo_y_series2, Trafo_b_total2, Y_shunt_bus2, Trafo_tap_on_i2) = res2

    print("\n=== Scalar / simple fields ===")
    print(f"gridtype: {gridtype1!r} vs {gridtype2!r} -> {'OK' if gridtype1 == gridtype2 else 'DIFF'}")
    print(f"is_connected: {is_connected1} vs {is_connected2} -> {'OK' if is_connected1 == is_connected2 else 'DIFF'}")
    print(f"U_base: {U_base1} vs {U_base2}")
    print(f"S_base: {S_base1} vs {S_base2}")

    print("\n=== Array comparisons (allclose) ===")
    all_ok = True
    all_ok &= compare_arrays("bus_typ",        bus_typ1,        bus_typ2)
    all_ok &= compare_arrays("s_multi",        s_multi1,        s_multi2)
    all_ok &= compare_arrays("u_start",        u_start1,        u_start2)
    all_ok &= compare_arrays("Y_matrix",       Y_matrix1,       Y_matrix2)
    all_ok &= compare_arrays("Y_Lines",        Y_Lines1,        Y_Lines2)
    all_ok &= compare_arrays("Y_C_Lines",      Y_C_Lines1,      Y_C_Lines2)
    all_ok &= compare_arrays("Lines_connected",Lines_connected1,Lines_connected2)
    all_ok &= compare_arrays("vn_kv",          vn_kv1,          vn_kv2)
    all_ok &= compare_arrays("Is_trafo",       Is_trafo1,       Is_trafo2)
    all_ok &= compare_arrays("Trafo_tau",      Trafo_tau1,      Trafo_tau2)
    all_ok &= compare_arrays("Trafo_shift_deg",Trafo_shift_deg1,Trafo_shift_deg2)
    all_ok &= compare_arrays("Trafo_y_series", Trafo_y_series1, Trafo_y_series2)
    all_ok &= compare_arrays("Trafo_b_total",  Trafo_b_total1,  Trafo_b_total2)
    all_ok &= compare_arrays("Y_shunt_bus",    Y_shunt_bus1,    Y_shunt_bus2)
    all_ok &= compare_arrays("Trafo_tap_on_i",    Trafo_tap_on_i1,    Trafo_tap_on_i2)

    print("\n=== Summary ===")
    if is_connected1 != is_connected2:
        all_ok = False

    if abs(U_base1 - U_base2) > 1e-9 or abs(S_base1 - S_base2) > 1e-3:
        print("U_base or S_base differ beyond tiny tolerance.")
        all_ok = False

    if all_ok:
        print("✅ All fields match within tolerance. The two generators are effectively identical.")
    else:
        print("❌ Differences detected. Check the detailed output above.")

if __name__ == "__main__":
    main()