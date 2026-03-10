from case_generator_improved_no_pu import (
    case_generation_ieee14_pandapower,
    case_generation_ieee14_pandapower_stamped,
)
import numpy as np

def reconstruct_Y_ieee14_pandapower_branchwise(
    N: int,
    vn_kv: np.ndarray,
    S_base: float,
    Lines_connected: np.ndarray,
    Y_Lines: np.ndarray,        # SI series admittance for lines (S)
    Y_C_Lines: np.ndarray,      # SI per-end line charging (S)
    Is_trafo: np.ndarray,
    Trafo_y_series: np.ndarray, # SI series admittance HV-referred (S)
    Trafo_tau: np.ndarray,
    Trafo_shift_deg: np.ndarray,
    Trafo_b_total: np.ndarray,  # SI magnetizing (unused in Y_matrix for IEEE14)
    Y_shunt_bus: np.ndarray,    # SI bus shunt (S)
    Trafo_tap_on_i: np.ndarray, # int8 1 if tap on iu side else 0
    Trafo_hv_is_i: np.ndarray=None,  # optional
    Trafo_n: np.ndarray=None,        # optional
) -> np.ndarray:
    Vbase = vn_kv.astype(float) * 1e3
    Y_SI = np.zeros((N, N), dtype=np.complex128)

    iu, ju = np.triu_indices(N, 1)

    # bus shunts directly in SI
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    for k, (i, j) in enumerate(zip(iu, ju)):
        if Lines_connected[k] == 0:
            continue

        Vi, Vj = Vbase[i], Vbase[j]

        if Is_trafo[k] == 0:
            # ---- line: recover local pu, stamp, then scale entries into SI ----
            V_line = Vi  # IEEE14 lines have same base on both ends
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
            # ---- transformer: use HV-referred y_SI, recover local pu, stamp, scale ----
            V_h = max(Vi, Vj)
            y_hv_SI = Trafo_y_series[k]
            Ys_pu = y_hv_SI * (V_h ** 2) / S_base
            Bc_pu = 0.0  # IEEE14 transformer BR_B=0; keep extensible later

            tau = Trafo_tau[k]
            theta = np.deg2rad(Trafo_shift_deg[k])
            t = tau * np.exp(1j * theta)

            # from/to by tap orientation
            if int(Trafo_tap_on_i[k]) == 1:
                f, tbus = i, j
            else:
                f, tbus = j, i
            Vf, Vt = Vbase[f], Vbase[tbus]

            Ytt_pu = Ys_pu + 1j * Bc_pu / 2.0
            Yff_pu = Ytt_pu / (t * np.conj(t))
            Yft_pu = -Ys_pu / np.conj(t)
            Ytf_pu = -Ys_pu / t

            Y_SI[f, f]     += Yff_pu * S_base / (Vf * Vf)
            Y_SI[tbus,tbus]+= Ytt_pu * S_base / (Vt * Vt)
            Y_SI[f, tbus]  += Yft_pu * S_base / (Vf * Vt)
            Y_SI[tbus, f]  += Ytf_pu * S_base / (Vt * Vf)

            # if later you include trafo magnetizing in the actual Ybus:
            # you would add it here consistently (HV-referred, split half-half in pu then scale)

    return Y_SI


def test_ieee14_pandapower():
    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series, Trafo_b_total, Y_shunt_bus,
     Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee14_pandapower(seed=0)

    N = len(bus_typ)
    Y_rec = reconstruct_Y_ieee14_pandapower_branchwise(
        N, vn_kv, S_base,
        Lines_connected, Y_Lines, Y_C_Lines,
        Is_trafo, Trafo_y_series, Trafo_tau, Trafo_shift_deg,
        Trafo_b_total, Y_shunt_bus,
        Trafo_tap_on_i,
        Trafo_hv_is_i, Trafo_n,
    )
    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee14_pandapower ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    print("allclose:", np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9))


def test_ieee14_pandapower_stamped():
    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series, Trafo_b_total, Y_shunt_bus,
     Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee14_pandapower_stamped(seed=0)

    N = len(bus_typ)
    Y_rec = reconstruct_Y_ieee14_pandapower_branchwise(
        N, vn_kv, S_base,
        Lines_connected, Y_Lines, Y_C_Lines,
        Is_trafo, Trafo_y_series, Trafo_tau, Trafo_shift_deg,
        Trafo_b_total, Y_shunt_bus,
        Trafo_tap_on_i,
        Trafo_hv_is_i, Trafo_n,
    )
    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee14_pandapower_stamped ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    print("allclose:", np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9))