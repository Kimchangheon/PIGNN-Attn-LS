import numpy as np
import traceback

import numpy as np

def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    """
    General NR voltage update that does NOT assume slack bus is index 0.
    Works with arbitrary PPC bus ordering.

    bus_typ convention:
      1 = slack
      2 = PV
      3 = PQ
    """
    N, _ = Y_admittance.shape

    I = Y_admittance @ U
    Ss = U * np.conj(I)

    slack = (bus_typ == 1)
    pv    = (bus_typ == 2)
    pq    = (bus_typ == 3)

    # State-variable bus sets
    ang_buses = np.where(~slack)[0]   # angle unknowns for all non-slack buses
    vm_buses  = np.where(pq)[0]       # magnitude unknowns for PQ buses only

    # Mismatch
    dP = P - Ss.real
    dQ = Q - Ss.imag

    mismatch = np.concatenate([dP[ang_buses], dQ[vm_buses]])

    # Reduced Jacobian
    keep = np.concatenate([ang_buses, N + vm_buses])
    J_red = JMatrix[np.ix_(keep, keep)]

    try:
        dx = np.linalg.solve(J_red, mismatch)
    except np.linalg.LinAlgError:
        print("Fehler (Jacobian singular)")
        raise

    n_ang = len(ang_buses)
    dth = dx[:n_ang]
    dvm = dx[n_ang:]

    U_new = U.copy()

    # update angles of all non-slack buses
    ang = np.angle(U_new)
    ang[ang_buses] += dth

    # update magnitudes of PQ buses only
    mag = np.abs(U_new)
    mag[vm_buses] += dvm

    U_new = mag * np.exp(1j * ang)

    I_new = Y_admittance @ U_new
    S_new = U_new * np.conj(I_new)

    return U_new, S_new

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


import numpy as np
import traceback

def newtonrapson(bus_typ, Y_system, s_L, U, K=40):
    """
    Newton-Raphson without assuming slack bus = index 0.
    """
    try:
        S_spec = np.asarray(s_L, dtype=np.complex128)
        U = np.asarray(U, dtype=np.complex128).copy()

        P = S_spec.real
        Q = S_spec.imag

        N, _ = Y_system.shape
        print(f"{N}- Bus", end="", flush=True)

        tol = 5e-4
        prev2 = None
        prev1 = None
        converged = False

        for _ in range(K):
            J, _, _, _, _ = JacobianMatrix3p(Y_system, U)
            print(".", end="", flush=True)

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

        # keep your original behavior if you want
        u_final = U if converged else (U * 0)

    except Exception as e:
        print(f"Fail in newtonrapson: {str(e)}")
        traceback.print_exc()
        u_final = []
        I = []
        S = []

    return u_final, I, S