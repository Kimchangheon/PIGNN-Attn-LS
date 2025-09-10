# Below is a drop‑in, faster version of your code.
# Same function names, inputs, and returned values.
# Same printed messages and update formula.
# Major speedups come from: vectorizing the full Jacobian (no Python triple loops), avoiding temporary diagonals/matrix deletes via boolean masks, using np.linalg.solve instead of explicit matrix inverse, and removing redundant operations.

# What changed (and why it’s faster)
# Vectorized Jacobian: Replaced the O(N³) nested loops with fully vectorized NumPy expressions using broadcasted angle differences. This is the main speedup.
# No explicit matrix inverse: Switched np.linalg.inv(J) @ delta to np.linalg.solve(J, delta), which is faster and numerically stabler. The try/except and printed message are preserved.
# Masking instead of repeated deletes: Constructed boolean masks once to slice both the mismatch vector and the Jacobian in one shot, mirroring the original delete pattern for slack and PV buses.
# Removed redundant temporaries: Used elementwise products (U * np.conj(I)) instead of np.diag(U) @ ... and avoided recomputing angles/magnitudes repeatedly.
# Same I/O and prints: Function signatures, return values, and printouts match your original logic, including the edge case where u_final becomes zeros if not converged.

import numpy as np
import traceback

def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    """
    Logic, inputs, and outputs identical to the original:
    - Computes Ss from currents, builds mismatch vector with slack/PV removal,
      solves for deltaU, reinserts PV-magnitude zeros, and updates U.
    """
    N, M = Y_admittance.shape

    # Currents and complex powers at buses
    I = Y_admittance @ U
    Ss = U * np.conj(I)  # faster than np.diag(U) @ np.conj(I)

    # Build masks (remove slack from P & Q; remove PV only from Q / |U| part)
    slack = (bus_typ == 1)
    pv    = (bus_typ == 2)

    # Upper half (P / angle) keeps all non-slack; lower half (Q / |U|) keeps non-slack, non-PV
    keep_upper = ~slack
    keep_lower = (~slack) & (~pv)
    keep_mask  = np.concatenate([keep_upper, keep_lower])

    # Computed P,Q (from current U)
    PQs_all  = np.concatenate((np.real(Ss), np.imag(Ss)))
    PQs_keep = PQs_all[keep_mask]

    # Specified P and Q after removing slack / PV exactly like original deletions
    P_keep = P[keep_upper]
    Q_keep = Q[keep_lower]

    # Mismatch
    delta = np.concatenate((P_keep, Q_keep)) - PQs_keep

    # Reduce J with the exact same pattern of row/col removals as the original deletes
    J_red = JMatrix[np.ix_(keep_mask, keep_mask)]

    # # Replace the np.linalg.solve block with the explicit inverse (like the original)
    # try:
    #     invJ = np.linalg.inv(J_red)
    #     deltaU_red = invJ @ delta
    # except np.linalg.LinAlgError:
    #     print("Fehler")
    #     deltaU_red = np.zeros_like(delta)

    # Solve (more stable & faster than explicit inverse)
    try:
        deltaU_red = np.linalg.solve(J_red, delta)
    except np.linalg.LinAlgError:
        print("Fehler")
        deltaU_red = np.zeros_like(delta)  # no‑op step if singular

    deltaU = deltaU_red

    # Reinsert zeros for removed PV-|U| unknowns (same indices as original loop)
    PV_indices = np.where(pv)[0]
    for position in PV_indices:
        deltaU = np.insert(deltaU, position + N - 2, 0.0)

    # # voltage limit applied
    # dtheta = deltaU[:N-1].copy()
    # dvm    = deltaU[N-1:].copy()
    #
    # # step limits
    # dtheta_max = 0.30                      # rad
    # dvm_frac   = 0.10                      # ≤10% of current |U|
    # vm_limits  = dvm_frac * np.abs(U[1:])  # skip slack
    #
    # dtheta = np.clip(dtheta, -dtheta_max, dtheta_max)
    # dvm    = np.clip(dvm,   -vm_limits,    vm_limits)
    #
    # def _apply_update(Uin: np.ndarray, alpha: float) -> np.ndarray:
    #     Uout = Uin.copy()
    #     # scaled step
    #     dth = alpha * dtheta
    #     dvm_s = alpha * dvm
    #     for k in range(N - 1):
    #         mag  = np.abs(Uout[k + 1])
    #         ang  = np.angle(Uout[k + 1])
    #         Uout[k + 1] = (mag + dvm_s[k]) * np.exp(1j * (ang + dth[k]))
    #     return Uout
    #
    # def _mismatch_inf_norm(Ucand: np.ndarray) -> float:
    #     I_cand  = Y_admittance @ Ucand
    #     S_cand  = Ucand * np.conj(I_cand)
    #     PQ_all  = np.concatenate((np.real(S_cand), np.imag(S_cand)))
    #     PQ_keep = PQ_all[keep_mask]
    #     mis     = np.concatenate((P_keep, Q_keep)) - PQ_keep
    #     return float(np.linalg.norm(mis, ord=np.inf))
    #
    # # Armijo backtracking on ∞-norm of mismatch
    # alpha  = 1.0 # – the step scaler. Start at 1; shrink if needed.
    # c1     = 1e-4 # – how much decrease you demand. Smaller = easier to satisfy.
    # shrink = 0.5 # – how aggressively you shrink; 0.5 = halve each time.
    # alpha_min = 1e-3 # – don’t bother with absurdly tiny steps below this.
    #
    # # Armijo backtracking on ∞-norm of mismatch
    # alpha = 1.0  # Keep at 1.0 as starting point
    # c1 = 1e-8  # Reduced from 1e-4 - much easier to satisfy
    # shrink = 1  # Increased from 0.5 - gentler step size reduction
    # alpha_min = 1e-4  # Reduced from 1e-3 - allow smaller steps
    #
    # F0 = _mismatch_inf_norm(U) # φ(U)
    # accepted = False
    # while alpha >= alpha_min:
    #     U_try = _apply_update(U, alpha)  # U + α * limited_step
    #     F1 = _mismatch_inf_norm(U_try)  # φ(U + α d)
    #     if F1 <= (1.0 - c1 * alpha) * F0:  # Armijo condition
    #         U = U_try
    #         accepted = True
    #         break
    #     alpha *= shrink # α ← 0.5 α
    #
    # if not accepted:
    #     # last-resort: take very small step only if it helps
    #     U_try = _apply_update(U, alpha_min)
    #     if _mismatch_inf_norm(U_try) < F0:
    #         U = U_try
    #
    # # U = _apply_update(U, 1.0)

    # Update voltages (same formula & indexing as original)
    for k in range(N - 1):
        U[k + 1] = (np.abs(U[k + 1]) + deltaU[k + N - 1]) * np.exp(1j * (np.angle(U[k + 1]) + deltaU[k]))

    Us = U.copy()
    S  = Ss.copy()
    return Us, S  # new voltages and nodal complex powers (S remains from pre-update, as in original)


def JacobianMatrix3p(Y_admittance, U):
    """
    Vectorized Jacobian assembly (no Python loops).
    Returns J, J1, J2, J3, J4 exactly like the original.
    """
    N, M = Y_admittance.shape
    idx = np.arange(N)

    # Magnitudes/angles
    absU = np.abs(U)
    angU = np.angle(U)

    Ymag = np.abs(Y_admittance)
    Yang = np.angle(Y_admittance)

    # Pairwise phase differences theta_nm = angle(U_n) - angle(U_m) - angle(Y_nm)
    theta   = angU[:, None] - angU[None, :] - Yang
    cos_th  = np.cos(theta)
    sin_th  = np.sin(theta)

    # Useful diagonals
    Ymag_d     = np.diag(Ymag)
    Yang_d     = np.diag(Yang)
    absU_col   = absU[:, None]
    absU_row   = absU[None, :]

    # ----- J2 = dP/d|U| -----
    # Off-diagonal base
    J2 = absU_col * Ymag * cos_th
    # Diagonal: 2|U_n||Y_nn|cos(∠Y_nn) + sum_{k≠n} |Y_nk||U_k|cos(theta_nk)
    row_sum_cosu = (Ymag * absU_row * cos_th).sum(axis=1)  # includes k=n term
    diag_add     = absU * Ymag_d * np.cos(Yang_d)
    J2[idx, idx] = row_sum_cosu + diag_add  # equals 2*A*cos + sum_{k≠n}(...)

    # ----- J1 = dP/dθ -----
    base_J1 = (absU_col * absU_row) * Ymag * sin_th
    J1 = base_J1.copy()
    row_sum = base_J1.sum(axis=1)
    diag_v  = np.diag(base_J1)
    J1[idx, idx] = -(row_sum - diag_v)

    # ----- J4 = dQ/d|U| -----
    J4 = absU_col * Ymag * sin_th  # off-diagonals as-is
    row_sum_sinUk = (Ymag * absU_row * sin_th).sum(axis=1)  # sum over k of |Y_nk||U_k|sin(theta_nk)
    J4[idx, idx] = row_sum_sinUk - absU * Ymag_d * np.sin(Yang_d)

    # ----- J3 = dQ/dθ -----
    base_J3 = (absU_col * absU_row) * Ymag * cos_th
    J3 = -base_J3
    row_sum_c = base_J3.sum(axis=1)
    diag_c    = np.diag(base_J3)
    J3[idx, idx] = row_sum_c - diag_c

    # Full Jacobian
    J = np.block([[J1, J2],
                  [J3, J4]])
    return J, J1, J2, J3, J4


# Parameter für die Lastflussberechnung
def newtonrapson(bus_typ, Y_system, s_L, U, K=40):
    """
    Logic preserved:
    - Prints identical progress markers.
    - Uses the same convergence check and returns the same tuple (u_final, I, S).
    - If not converged: prints the same message and returns u_final = zeros (like original),
      while I and S are computed from the last iterate U (same as original).
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

        for _ in range(K):  # Schrittverfahren
            J, _, _, _, _ = JacobianMatrix3p(Y_system, U)
            print('.', end='', flush=True)

            Us, _ = VoltageCalculation3p(bus_typ, J, Y_system, U, P, Q)

            # Convergence check (same logic as differences of last three U's)
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
            # not broken -> not converged within 40 iters
            print("|not-converged|")

        # Compute I, S from the final iterate U (same as original)
        I = Y_system @ U
        S = U * np.conj(I)

        # u_final matches original behavior:
        # - On convergence: last iterate
        # - On non-convergence: zeros (original overwrote last history row with zeros)
        u_final = U if converged else (U * 0)

    except Exception as e:
        print(f"Fehler in newtonrapson: {str(e)}")
        traceback.print_exc()
        u_final = []
        I = []
        S = []

    return u_final, I, S
