from case_generator_improved import (
    case_generation_ieee14_pandapower,
    case_generation_ieee14_pandapower_stamped,
)
import numpy as np


def reconstruct_Y_ieee14_pandapower(
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
    Trafo_b_total: np.ndarray,   # currently unused in stamping (Y_matrix doesn’t contain it)
    Y_shunt_bus: np.ndarray,
    Trafo_tap_on_i: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct Y_matrix in SI [S] from IEEE14 pandapower metadata.

    Strategy:
      - Convert SI branch/shunt data back to per-unit.
      - Stamp all branches in per-unit exactly like pypower.makeYbus.
      - Convert the final Y_pu to SI using:
            Y_SI[i,j] = Y_pu[i,j] * S_base / (Vbase_i * Vbase_j)

    Assumptions:
      - Y_Lines, Y_C_Lines are SI [S] for "pure" lines (no tap).
      - Trafo_y_series is SI [S], series admittance referred to HV side.
      - vn_kv is per-bus nominal voltage [kV].
      - Lines_connected / Is_trafo follow np.triu_indices upper-tri ordering.
      - Y_shunt_bus is the total bus shunt in SI [S].
      - Trafo_tap_on_i[k] == True  → tap is on iu[k] side;
        Trafo_tap_on_i[k] == False → tap is on ju[k] side.
    """
    # Per-bus base voltages
    Vbase = vn_kv.astype(float) * 1e3  # [V]
    Y_pu = np.zeros((N, N), dtype=np.complex128)

    # Upper-tri index mapping
    iu, ju = np.triu_indices(N, 1)

    # -------------
    # 1) Bus shunts: SI -> pu and stamp on diagonal in pu
    # -------------
    # Y_shunt_bus_SI[i] = (GS + jBS) * 1e6 / V_i^2
    # => Ysh_pu[i] = Y_shunt_bus_SI[i] * V_i^2 / S_base
    Ysh_pu = Y_shunt_bus * (Vbase * Vbase) / S_base
    Y_pu[np.diag_indices(N)] += Ysh_pu

    # -------------
    # 2) Line & transformer branches in pu
    # -------------
    for k in range(len(iu)):
        if Lines_connected[k] == 0:
            continue

        i = iu[k]
        j = ju[k]
        Vi = Vbase[i]
        Vj = Vbase[j]

        if Is_trafo[k] == 0:
            # --------------------
            # Pure line (π-model)
            # --------------------
            # For a line, we assumed in the generator:
            #   Y_Lines_SI = Ys_pu * S_base / V_line^2
            #   Y_C_Lines_SI (per-end) = (Bc_pu/2) * S_base / V_line^2
            # with V_line = Vbase[i] = Vbase[j].
            V_line = Vi  # they should be equal

            Ys_SI = Y_Lines[k]           # [S]
            B_end_SI = Y_C_Lines[k]      # [S] per end
            Bc_SI = 2.0 * B_end_SI       # total line charging [S]

            # Back to per-unit
            Ys_pu = Ys_SI * (V_line ** 2) / S_base
            Bc_pu = Bc_SI * (V_line ** 2) / S_base

            t_complex = 1.0 + 0j  # no tap, no phase shift

            Ytt_pu = Ys_pu + 1j * Bc_pu / 2.0
            Yff_pu = Ytt_pu  # since |t|=1
            Yft_pu = -Ys_pu
            Ytf_pu = -Ys_pu

            # Stamp into Y_pu
            Y_pu[i, i] += Yff_pu
            Y_pu[j, j] += Ytt_pu
            Y_pu[i, j] += Yft_pu
            Y_pu[j, i] += Ytf_pu

        else:
            # --------------------
            # Transformer branch
            # --------------------
            # Trafo_y_series[k] is SI [S], series admittance referred to HV side.
            # Let V_h be the HV nominal voltage; then:
            #   Trafo_y_series_SI = Ys_pu * S_base / V_h^2
            # => Ys_pu = Trafo_y_series_SI * V_h^2 / S_base
            if Vi >= Vj:
                V_h = Vi
            else:
                V_h = Vj

            Ys_SI_hv = Trafo_y_series[k]
            Ys_pu = Ys_SI_hv * (V_h ** 2) / S_base
            Bc_pu = 0.0  # we didn't encode BR_B for trafos in metadata

            # Tap & phase shift (attached to "from" side)
            tau = Trafo_tau[k]
            theta = np.deg2rad(Trafo_shift_deg[k])
            t_complex = tau * np.exp(1j * theta)

            # Determine which bus is "from" (tap side) and which is "to"
            # based on Trafo_tap_on_i.
            if Trafo_tap_on_i[k]:
                f = i
                t = j
            else:
                f = j
                t = i

            Vf = Vbase[f]
            Vt = Vbase[t]

            # Per-unit Y contributions (same as pypower.makeYbus):
            Ytt_pu = Ys_pu + 1j * Bc_pu / 2.0  # here Bc_pu=0 so Ytt_pu = Ys_pu
            Yff_pu = Ytt_pu / (t_complex * np.conj(t_complex))
            Yft_pu = -Ys_pu / np.conj(t_complex)
            Ytf_pu = -Ys_pu / t_complex

            # Stamp into Y_pu
            Y_pu[f, f] += Yff_pu
            Y_pu[t, t] += Ytt_pu
            Y_pu[f, t] += Yft_pu
            Y_pu[t, f] += Ytf_pu

            # NOTE on Trafo_b_total:
            #   Trafo_b_total is magnetizing susceptance in SI [S], HV side.
            #   The current Y_matrix (from ppc_int["Ybus"]) does NOT include this
            #   magnetizing branch, so we do NOT stamp it here. For IEEE14,
            #   Trafo_b_total == 0 anyway, so nothing is lost.

    # -------------
    # 3) Convert final Y_pu -> SI
    # -------------
    denom = np.outer(Vbase, Vbase)  # V_i * V_j
    Y_SI = Y_pu * (S_base / denom)

    return Y_SI


# =======================
# Tests
# =======================

def test_ieee14_pandapower():
    (gridtype,
     bus_typ,
     s_multi,
     u_start,
     Y_matrix,
     is_connected,
     Y_Lines,
     Y_C_Lines,
     Lines_connected,
     U_base,
     S_base,
     vn_kv,
     Is_trafo,
     Trafo_tau,
     Trafo_shift_deg,
     Trafo_y_series,
     Trafo_b_total,
     Y_shunt_bus,
     Trafo_tap_on_i) = case_generation_ieee14_pandapower(seed=0)

    N = len(bus_typ)
    Y_rec = reconstruct_Y_ieee14_pandapower(
        N=N,
        vn_kv=vn_kv,
        S_base=S_base,
        Lines_connected=Lines_connected,
        Y_Lines=Y_Lines,
        Y_C_Lines=Y_C_Lines,
        Is_trafo=Is_trafo,
        Trafo_y_series=Trafo_y_series,
        Trafo_tau=Trafo_tau,
        Trafo_shift_deg=Trafo_shift_deg,
        Trafo_b_total=Trafo_b_total,
        Y_shunt_bus=Y_shunt_bus,
        Trafo_tap_on_i=Trafo_tap_on_i,
    )

    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee14_pandapower ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    print("np.allclose:", np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9))


def test_ieee14_pandapower_stamped():
    (gridtype,
     bus_typ,
     s_multi,
     u_start,
     Y_matrix,
     is_connected,
     Y_Lines,
     Y_C_Lines,
     Lines_connected,
     U_base,
     S_base,
     vn_kv,
     Is_trafo,
     Trafo_tau,
     Trafo_shift_deg,
     Trafo_y_series,
     Trafo_b_total,
     Y_shunt_bus,
     Trafo_tap_on_i) = case_generation_ieee14_pandapower_stamped(seed=0)

    N = len(bus_typ)
    Y_rec = reconstruct_Y_ieee14_pandapower(
        N=N,
        vn_kv=vn_kv,
        S_base=S_base,
        Lines_connected=Lines_connected,
        Y_Lines=Y_Lines,
        Y_C_Lines=Y_C_Lines,
        Is_trafo=Is_trafo,
        Trafo_y_series=Trafo_y_series,
        Trafo_tau=Trafo_tau,
        Trafo_shift_deg=Trafo_shift_deg,
        Trafo_b_total=Trafo_b_total,
        Y_shunt_bus=Y_shunt_bus,
        Trafo_tap_on_i=Trafo_tap_on_i,
    )

    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee14_pandapower_stamped ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    print("np.allclose:", np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9))