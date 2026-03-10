if __name__ == "__main__":
    import numpy as np

    # import from your module if this "main" lives in a separate file:
    from case_generator_improved import case_generation_ieee14_pandapower

    def upper_index(N, i, j):
        if i == j:
            raise ValueError("i and j must differ")
        if i > j:
            i, j = j, i
        return ((2 * N - i - 1) * i) // 2 + (j - i - 1)

    def summarize(tag, out_tuple):
        (gridtype, bus_typ, s_multi, u_start, Y, is_connected,
         Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
         vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total) = out_tuple

        N = len(bus_typ)
        C = N * (N - 1) // 2
        print(f"\n=== {tag} ===")
        print(f"gridtype={gridtype}, N={N}, C={C}, is_connected={is_connected}")
        print(f"S_base={S_base:.3e} VA, U_base={U_base}")  # U_base is NaN in this preset (multi-voltage)
        print(f"Shapes: Y={Y.shape}, s_multi={s_multi.shape}, u_start={u_start.shape}")
        print(f"Upper arrays lens: Y_Lines={len(Y_Lines)}, Y_C_Lines={len(Y_C_Lines)}, Lines_connected={len(Lines_connected)}")

        # Bus-type counts
        n_slack = int((bus_typ == 1).sum())
        n_pv    = int((bus_typ == 2).sum())
        n_pq    = int((bus_typ == 3).sum())
        print(f"Bus types: slack={n_slack}, PV={n_pv}, PQ={n_pq}")

        # Voltages (start) in pu relative to per-bus base
        Vbase = vn_kv * 1e3
        vmag_pu = np.abs(u_start) / Vbase
        print(f"|U_start| pu: min={vmag_pu.min():.4f}, max={vmag_pu.max():.4f}, mean={vmag_pu.mean():.4f}")

        # P/Q stats (SI)
        P = np.real(s_multi); Q = np.imag(s_multi)
        print(f"P (W): sum={P.sum():.3e}, min={P.min():.3e}, max={P.max():.3e}")
        print(f"Q (var): sum={Q.sum():.3e}, min={Q.min():.3e}, max={Q.max():.3e}")

        # Symmetry residual of Y (IEEE-14 has no phase-shifting taps, so Y should be complex-symmetric)
        sym_res = np.max(np.abs(Y - Y.T))
        print(f"Y complex-symmetry residual (max |Y - Y^T|): {sym_res:.3e}")

        # Line-only stamping sanity: for each line edge (non-transformer), check Y[i,j] ≈ -Y_Lines[idx]
        iu, ju = np.triu_indices(N, 1)
        ok_pairs = []
        err_pairs = []
        for k, (i, j) in enumerate(zip(iu, ju)):
            if Lines_connected[k] == 1 and Is_trafo[k] == 0 and Y_Lines[k] != 0:
                err = complex(Y[i, j]) + complex(Y_Lines[k])  # should be ~0
                ok_pairs.append(abs(err))
            # (we skip transformer branches here, since their tap is already embedded in Y)
        if ok_pairs:
            print(f"Line off-diagonals check over {len(ok_pairs)} edges: "
                  f"max|Y[i,j]+Y_Lines|={np.max(ok_pairs):.3e}, mean={np.mean(ok_pairs):.3e}")
        else:
            print("No plain line edges to check (unexpected for IEEE-14).")

    # --------- Run baseline (no jitter) ----------
    out0 = case_generation_ieee14_pandapower(
        jitter_load=0.0,
        jitter_gen=0.0,
        pv_vset_range=None,
        rand_u_start=False,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        seed=1234
    )
    summarize("BASELINE (no jitter)", out0)

    # --------- Run with jitter ----------
    out1 = case_generation_ieee14_pandapower(
        jitter_load=0.10,              # ±10% loads
        jitter_gen=0.10,               # ±10% gen P
        pv_vset_range=(0.98, 1.04),    # PV |V| in [0.98, 1.04] pu
        rand_u_start=True,             # randomize start mag/angles (PV magnitudes honored)
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
        seed=5678
    )
    summarize("JITTERED (loads/gen/pv_vset/u_start)", out1)

    # Optional: show that S or u_start changed meaningfully
    _, _, s0, u0, Y0, *_ = out0
    _, _, s1, u1, Y1, *_ = out1
    print("\nΔ comparisons (baseline vs jittered):")
    print("max |ΔS| =", np.max(np.abs(s1 - s0)))
    print("max |ΔU_start| =", np.max(np.abs(u1 - u0)))
    print("max |ΔY| (should be ~0 because topology/params fixed) =", np.max(np.abs(Y1 - Y0)))
