import numpy as np
import traceback

def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    """
    Computes mismatch, reduces by (slack / PV) rules, solves for update, reinserts PV-|U| zeros,
    and updates U in polar coordinates. Works with general complex (possibly non-symmetric) Y.
    """
    N, M = Y_admittance.shape

    I = Y_admittance @ U
    Ss = U * np.conj(I)

    slack = (bus_typ == 1)
    pv    = (bus_typ == 2)

    keep_upper = ~slack
    keep_lower = (~slack) & (~pv)
    keep_mask  = np.concatenate([keep_upper, keep_lower])

    PQs_all  = np.concatenate((np.real(Ss), np.imag(Ss)))
    PQs_keep = PQs_all[keep_mask]

    P_keep = P[keep_upper]
    Q_keep = Q[keep_lower]

    delta = np.concatenate((P_keep, Q_keep)) - PQs_keep
    J_red = JMatrix[np.ix_(keep_mask, keep_mask)]

    try:
        deltaU_red = np.linalg.solve(J_red, delta)
    except np.linalg.LinAlgError:
        print("Fehler (Jacobian singular)")
        raise  # let newtonrapson handle this as a failure
        deltaU_red = np.zeros_like(delta)

    deltaU = deltaU_red

    PV_indices = np.where(pv)[0]
    for position in PV_indices:
        deltaU = np.insert(deltaU, position + N - 2, 0.0)

    for k in range(N - 1):
        U[k + 1] = (np.abs(U[k + 1]) + deltaU[k + N - 1]) * np.exp(1j * (np.angle(U[k + 1]) + deltaU[k]))

    Us = U.copy()
    S  = Ss.copy()
    return Us, S


def JacobianMatrix3p(Y_admittance, U):
    """
    Vectorized Jacobian assembly (no Python loops).
    """
    N, M = Y_admittance.shape
    idx = np.arange(N)

    absU = np.abs(U)
    angU = np.angle(U)

    Ymag = np.abs(Y_admittance)
    Yang = np.angle(Y_admittance)

    theta   = angU[:, None] - angU[None, :] - Yang
    cos_th  = np.cos(theta)
    sin_th  = np.sin(theta)

    Ymag_d     = np.diag(Ymag)
    Yang_d     = np.diag(Yang)
    absU_col   = absU[:, None]
    absU_row   = absU[None, :]

    # dP/d|U|
    J2 = absU_col * Ymag * cos_th
    row_sum_cosu = (Ymag * absU_row * cos_th).sum(axis=1)
    diag_add     = absU * Ymag_d * np.cos(Yang_d)
    J2[idx, idx] = row_sum_cosu + diag_add

    # dP/dθ
    base_J1 = (absU_col * absU_row) * Ymag * sin_th
    J1 = base_J1.copy()
    row_sum = base_J1.sum(axis=1)
    diag_v  = np.diag(base_J1)
    J1[idx, idx] = -(row_sum - diag_v)

    # dQ/d|U|
    J4 = absU_col * Ymag * sin_th
    row_sum_sinUk = (Ymag * absU_row * sin_th).sum(axis=1)
    J4[idx, idx] = row_sum_sinUk - absU * Ymag_d * np.sin(Yang_d)

    # dQ/dθ
    base_J3 = (absU_col * absU_row) * Ymag * cos_th
    J3 = -base_J3
    row_sum_c = base_J3.sum(axis=1)
    diag_c    = np.diag(base_J3)
    J3[idx, idx] = row_sum_c - diag_c

    J = np.block([[J1, J2],
                  [J3, J4]])
    return J, J1, J2, J3, J4


def newtonrapson(bus_typ, Y_system, s_L, U, K=40):
    """
    Newton-Raphson with same I/O and prints as your version.
    """
    try:
        S = s_L
        P = np.real(S)
        Q = np.imag(S)

        N, M = Y_system.shape

        print(f'{N}- Bus', end='', flush=True)

        tol = 5e-4
        prev2 = None
        prev1 = None
        converged = False

        for _ in range(K):
            J, _, _, _, _ = JacobianMatrix3p(Y_system, U)
            print('.', end='', flush=True)

            try:
                Us, _ = VoltageCalculation3p(bus_typ, J, Y_system, U, P, Q)
            except np.linalg.LinAlgError:
                print("|Jacobian singular, abort NR|")
                converged = False
                break

            if prev2 is not None and prev1 is not None:
                if (np.max(np.abs(Us - prev1)) < tol) and (np.max(np.abs(prev1 - prev2)) < tol):
                    print("|converged successfully|")
                    U = Us
                    converged = True
                    break

            prev2 = prev1
            prev1 = Us
            U = Us
        else:
            print("|not-converged|")

        I = Y_system @ U
        S = U * np.conj(I)
        u_final = U if converged else (U * 0)

    except Exception as e:
        print(f"Fail in newtonrapson: {str(e)}")
        traceback.print_exc()
        u_final = []
        I = []
        S = []

    return u_final, I, S

# # relative tolerances
# def newtonrapson(bus_typ, Y_system, s_L, U, K=40):
#     try:
#         S = s_L
#         P = np.real(S)
#         Q = np.imag(S)
#
#         N, M = Y_system.shape
#
#         print(f'{N}- Bus', end='', flush=True)
#
#         # relative tolerances
#         tol_step = 1e-4   # relative voltage step
#         tol_mis  = 1e-3   # relative power mismatch
#
#         prev_U1 = None
#         converged = False
#
#         for it in range(K):
#             J, _, _, _, _ = JacobianMatrix3p(Y_system, U)
#             print('.', end='', flush=True)
#
#             Us, S_calc = VoltageCalculation3p(bus_typ, J, Y_system, U, P, Q)
#
#             # --- compute power mismatch norm ---
#             I = Y_system @ Us
#             S_inj = Us * np.conj(I)
#             P_calc = np.real(S_inj)
#             Q_calc = np.imag(S_inj)
#
#             slack = (bus_typ == 1)
#             pv    = (bus_typ == 2)
#             keep_upper = ~slack
#             keep_lower = (~slack) & (~pv)
#
#             mis_P = P[keep_upper] - P_calc[keep_upper]
#             mis_Q = Q[keep_lower] - Q_calc[keep_lower]
#
#             max_mis = np.max(np.abs(np.concatenate([mis_P, mis_Q])))
#             base_mis = max(1.0, np.max(np.abs(P)), np.max(np.abs(Q)))
#             rel_mis = max_mis / base_mis
#
#             # --- convergence tests ---
#             if prev_U1 is not None:
#                 step = np.max(np.abs(Us - prev_U1))
#                 base_U = max(1.0, np.max(np.abs(prev_U1)))
#                 rel_step = step / base_U
#             else:
#                 rel_step = np.inf
#
#             if (rel_mis < tol_mis) and (rel_step < tol_step):
#                 print("|converged successfully|")
#                 U = Us
#                 converged = True
#                 break
#
#             prev_U1 = Us
#             U = Us
#
#         if not converged:
#             print("|not-converged|")
#
#         I = Y_system @ U
#         S = U * np.conj(I)
#         u_final = U if converged else U  # <-- keep last iterate instead of zeroing
#
#     except Exception as e:
#         print(f"Fehler in newtonrapson: {str(e)}")
#         traceback.print_exc()
#         u_final = []
#         I = []
#         S = []
#
#     return u_final, I, S