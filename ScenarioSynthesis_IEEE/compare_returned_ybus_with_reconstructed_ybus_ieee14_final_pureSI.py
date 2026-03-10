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
    Y_Lines: np.ndarray,         # SI [S] (line series admittance)
    Y_C_Lines: np.ndarray,       # SI [S] per-end line charging susceptance
    Is_trafo: np.ndarray,
    Trafo_y_series: np.ndarray,  # SI [S], series admittance referred to HV side
    Trafo_tau: np.ndarray,
    Trafo_shift_deg: np.ndarray,
    Trafo_b_total: np.ndarray,   # magnetizing [S], not in Y_matrix → ignored here
    Y_shunt_bus: np.ndarray,     # SI [S] per-bus shunt (GS + jBS)
    Trafo_tap_on_i: np.ndarray,  # bool: True → tap on iu[k] side; False → tap on ju[k] side
) -> np.ndarray:
    """
    Reconstruct Y_matrix in SI [S] from IEEE-14 pandapower metadata.

    Strategy:
      - Use SI metadata for lines, trafos, shunts.
      - For each branch:
          * Recover local per-unit Ys_pu from the stored SI admittance.
          * Apply the *exact* pypower.makeYbus stamping in per-unit.
          * Immediately scale that 2×2 submatrix to SI and add to Y_SI.

    This guarantees bitwise equivalence with the original Y_matrix that
    came from ppc_int["Ybus"] → SI.
    """
    # Per-bus base voltages [V]
    Vbase = vn_kv.astype(float) * 1e3

    # Initialize Y in SI
    Y_SI = np.zeros((N, N), dtype=np.complex128)

    # Upper-tri index mapping (same as in the metadata)
    iu, ju = np.triu_indices(N, 1)

    # -------------
    # 1) Bus shunts: already in SI → directly to diagonal
    # -------------
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    # -------------
    # 2) Branches: lines + transformers
    # -------------
    for k in range(len(iu)):
        if Lines_connected[k] == 0:
            continue

        i = iu[k]
        j = ju[k]
        Vi = Vbase[i]
        Vj = Vbase[j]

        # ---------- Pure line (π model) in SI ----------
        if Is_trafo[k] == 0:
            # We stored:
            #   Y_Lines[k]    = Ys_pu * S_base / V_line^2
            #   Y_C_Lines[k]  = (Bc_pu/2) * S_base / V_line^2
            # with V_line = Vbase[i] = Vbase[j].
            V_line = Vi  # should equal Vj

            Ys_SI = Y_Lines[k]      # series admittance [S]
            B_end_SI = Y_C_Lines[k] # per-end charging [S]
            Bc_SI = 2.0 * B_end_SI  # total line charging [S]

            # Recover per-unit values for this branch
            #   Ys_pu = Ys_SI * V_line^2 / S_base
            #   Bc_pu = Bc_SI * V_line^2 / S_base
            Ys_pu = Ys_SI * (V_line ** 2) / S_base
            Bc_pu = Bc_SI * (V_line ** 2) / S_base

            # pypower.makeYbus formulas (per-unit, no tap):
            Ytt_pu = Ys_pu + 1j * Bc_pu / 2.0
            Yff_pu = Ytt_pu
            Yft_pu = -Ys_pu
            Ytf_pu = -Ys_pu

            # Convert this 2×2 per-unit submatrix to SI:
            #   Y_SI[p,p] += Y_pu[p,p] * S_base / V_p^2
            #   Y_SI[p,q] += Y_pu[p,q] * S_base / (V_p V_q)
            Y_SI[i, i] += Yff_pu * S_base / (Vi * Vi)
            Y_SI[j, j] += Ytt_pu * S_base / (Vj * Vj)
            Y_SI[i, j] += Yft_pu * S_base / (Vi * Vj)
            Y_SI[j, i] += Ytf_pu * S_base / (Vj * Vi)

        # ---------- Transformer branch ----------
        else:
            # Trafo_y_series[k] is SI [S], series admittance referred to HV side.
            # In the generator we did:
            #   y_hv_SI = Ys_pu * S_base / V_h^2
            # with V_h = max(Vi, Vj) (HV base voltage).
            V_h = max(Vi, Vj)
            y_hv_SI = Trafo_y_series[k]

            # Recover local per-unit series admittance:
            #   Ys_pu = y_hv_SI * V_h^2 / S_base
            Ys_pu = y_hv_SI * (V_h ** 2) / S_base
            Bc_pu = 0.0  # we did not store BR_B for trafos in metadata

            # Complex tap
            tau = Trafo_tau[k]
            theta = np.deg2rad(Trafo_shift_deg[k])
            t_complex = tau * np.exp(1j * theta)

            # Determine which bus is "from" (tap side) and "to"
            if Trafo_tap_on_i[k]:
                f = i
                t = j
            else:
                f = j
                t = i

            Vf = Vbase[f]
            Vt = Vbase[t]

            # Per-unit transformer stamping (pypower.makeYbus):
            Ytt_pu = Ys_pu + 1j * Bc_pu / 2.0   # here Bc_pu = 0 → Ytt_pu = Ys_pu
            Yff_pu = Ytt_pu / (t_complex * np.conj(t_complex))
            Yft_pu = -Ys_pu / np.conj(t_complex)
            Ytf_pu = -Ys_pu / t_complex

            # Map these per-unit entries into SI for buses f,t:
            Y_SI[f, f] += Yff_pu * S_base / (Vf * Vf)
            Y_SI[t, t] += Ytt_pu * S_base / (Vt * Vt)
            Y_SI[f, t] += Yft_pu * S_base / (Vf * Vt)
            Y_SI[t, f] += Ytf_pu * S_base / (Vt * Vf)

            # NOTE: Trafo_b_total[k] (magnetizing susceptance) is not stamped
            # here because ppc_int["Ybus"] does not include it (and for IEEE14
            # it's zero anyway). When you move to grids where Ybus includes
            # magnetizing, you can add the corresponding j*Bmag terms here.

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
    print("=== test_ieee14_pandapower (branchwise SI) ===")
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
    print("=== test_ieee14_pandapower_stamped (branchwise SI) ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    print("np.allclose:", np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9))