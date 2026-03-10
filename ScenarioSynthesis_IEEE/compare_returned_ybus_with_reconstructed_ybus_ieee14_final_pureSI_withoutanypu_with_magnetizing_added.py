from case_generator_improved_no_pu_with_magnetizing_added import (
    case_generation_ieee14_pandapower,
    case_generation_ieee14_pandapower_stamped,
)
import numpy as np

def reconstruct_Y_ieee14_pandapower_branchwise(
    N: int,
    vn_kv: np.ndarray,
    S_base: float,
    Lines_connected: np.ndarray,
    Y_Lines: np.ndarray,
    Y_C_Lines: np.ndarray,
    Is_trafo: np.ndarray,
    Trafo_y_series: np.ndarray,
    Trafo_tau: np.ndarray,
    Trafo_shift_deg: np.ndarray,
    Trafo_y_shunt_from: np.ndarray,   # NEW: complex SI end shunt (from)
    Trafo_y_shunt_to: np.ndarray,     # NEW: complex SI end shunt (to)
    Y_shunt_bus: np.ndarray,
    Trafo_tap_on_i: np.ndarray,
) -> np.ndarray:
    Vbase = vn_kv.astype(float) * 1e3
    Y_SI = np.zeros((N, N), dtype=np.complex128)
    iu, ju = np.triu_indices(N, 1)

    # bus shunts
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    for k, (i, j) in enumerate(zip(iu, ju)):
        if Lines_connected[k] == 0:
            continue

        Vi, Vj = Vbase[i], Vbase[j]

        if Is_trafo[k] == 0:
            # line (unchanged, symmetric susceptance only)
            V_line = Vi
            Ys_SI = Y_Lines[k]
            B_end_SI = Y_C_Lines[k]
            Bc_SI = 2.0 * B_end_SI

            Ys_pu = Ys_SI * (V_line ** 2) / S_base
            Bc_pu = Bc_SI * (V_line ** 2) / S_base

            Ytt_pu = Ys_pu + 1j * Bc_pu / 2.0
            Yff_pu = Ytt_pu
            Yft_pu = -Ys_pu
            Ytf_pu = -Ys_pu

            Y_SI[i, i] += Yff_pu * S_base / (Vi * Vi)
            Y_SI[j, j] += Ytt_pu * S_base / (Vj * Vj)
            Y_SI[i, j] += Yft_pu * S_base / (Vi * Vj)
            Y_SI[j, i] += Ytf_pu * S_base / (Vj * Vi)

        else:
            # transformer: replicate pandapower makeYbus end-shunt logic (Bcf/Bct)
            # Determine from/to buses from tap orientation
            if int(Trafo_tap_on_i[k]) == 1:
                f, tbus = i, j
            else:
                f, tbus = j, i

            Vf, Vt = Vbase[f], Vbase[tbus]

            # series (assumed symmetric)
            V_h = max(Vi, Vj)
            Ys_pu = Trafo_y_series[k] * (V_h ** 2) / S_base

            # end shunts: convert SI back to pu on each end's base
            Bcf_pu = Trafo_y_shunt_from[k] * (Vf ** 2) / S_base
            Bct_pu = Trafo_y_shunt_to[k]   * (Vt ** 2) / S_base

            tau = float(Trafo_tau[k])
            theta = np.deg2rad(float(Trafo_shift_deg[k]))
            tapc = tau * np.exp(1j * theta)

            # pandapower makeYbus:
            Ytt_pu = Ys_pu + Bct_pu / 2.0
            Yff_pu = (Ys_pu + Bcf_pu / 2.0) / (tapc * np.conj(tapc))
            Yft_pu = -Ys_pu / np.conj(tapc)
            Ytf_pu = -Ys_pu / tapc

            Y_SI[f, f]       += Yff_pu * S_base / (Vf * Vf)
            Y_SI[tbus, tbus] += Ytt_pu * S_base / (Vt * Vt)
            Y_SI[f, tbus]    += Yft_pu * S_base / (Vf * Vt)
            Y_SI[tbus, f]    += Ytf_pu * S_base / (Vt * Vf)

    return Y_SI


def test_ieee14_pandapower_with_magnetizing():
    pfe_kw = 50.0
    i0_pct = 2.0

    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series, Trafo_y_shunt_from, Trafo_y_shunt_to,
     Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee14_pandapower(
        seed=0, trafo_pfe_kw=pfe_kw, trafo_i0_percent=i0_pct
    )

    N = len(bus_typ)
    Y_rec = reconstruct_Y_ieee14_pandapower_branchwise(
        N, vn_kv, S_base,
        Lines_connected, Y_Lines, Y_C_Lines,
        Is_trafo, Trafo_y_series, Trafo_tau, Trafo_shift_deg,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus,
        Trafo_tap_on_i,
    )

    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee14_pandapower_with_magnetizing ===")
    print("gridtype:", gridtype)
    print(f"forced trafo pfe_kw={pfe_kw}, i0_percent={i0_pct}")
    print("max |Y_rec - Y_matrix|:", diff)
    assert np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9)


def test_ieee14_pandapower_stamped_with_magnetizing():
    # same now (stamped wrapper uses pandapower Ybus ground truth)
    pfe_kw = 50.0
    i0_pct = 2.0

    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series, Trafo_y_shunt_from, Trafo_y_shunt_to,
     Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee14_pandapower_stamped(
        seed=0, trafo_pfe_kw=pfe_kw, trafo_i0_percent=i0_pct
    )

    N = len(bus_typ)
    Y_rec = reconstruct_Y_ieee14_pandapower_branchwise(
        N, vn_kv, S_base,
        Lines_connected, Y_Lines, Y_C_Lines,
        Is_trafo, Trafo_y_series, Trafo_tau, Trafo_shift_deg,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus,
        Trafo_tap_on_i,
    )

    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee14_pandapower_stamped_with_magnetizing ===")
    print("gridtype:", gridtype)
    print(f"forced trafo pfe_kw={pfe_kw}, i0_percent={i0_pct}")
    print("max |Y_rec - Y_matrix|:", diff)
    assert np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9)
