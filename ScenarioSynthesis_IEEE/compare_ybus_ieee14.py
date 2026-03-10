#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandapower as pp
import pandapower.networks as pn

# Use PYPOWER index constants instead of hard-coding column numbers
from pandapower.pypower.idx_brch import (
    F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
)
from pandapower.pypower.idx_bus import GS, BS, BASE_KV


def build_Y_stamped_from_ppc(ppc_int):
    """
    Rebuild Ybus (per-unit) from ppc_int['bus'] and ppc_int['branch']
    using the SAME formulas as pypower.makeYbus.

    This should match ppc_int['Ybus'] up to numerical noise.
    """
    baseMVA = float(ppc_int["baseMVA"])
    bus = np.asarray(ppc_int["bus"], dtype=float)
    branch = np.asarray(ppc_int["branch"], dtype=float)

    nb = bus.shape[0]   # number of buses
    nl = branch.shape[0]  # number of branches

    # Initialize Ybus in per-unit
    Ybus = np.zeros((nb, nb), dtype=np.complex128)

    # ----- Branch series admittance and line charging -----
    # Status (1 = in service, 0 = out of service)
    stat = branch[:, BR_STATUS]

    # Series admittance Ys = 1 / (R + jX)  (per-unit; 0 if out of service)
    R = branch[:, BR_R]
    X = branch[:, BR_X]
    Z = R + 1j * X
    Ys = np.zeros(nl, dtype=np.complex128)
    nonzero = (stat != 0) & (np.abs(Z) > 0)
    Ys[nonzero] = stat[nonzero] / Z[nonzero]

    # Total line charging susceptance (per-unit)
    Bc = stat * branch[:, BR_B]

    # ----- Tap ratios (including phase shift) -----
    tap = np.ones(nl, dtype=np.complex128)
    tap_raw = branch[:, TAP]
    nonunity_tap = (tap_raw != 0.0)
    tap[nonunity_tap] = tap_raw[nonunity_tap]

    shift_deg = branch[:, SHIFT]
    tap *= np.exp(1j * np.pi / 180.0 * shift_deg)  # include phase shift

    # ----- Diagonal & off-diagonal elements per branch -----
    Ytt = Ys + 1j * Bc / 2.0
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap

    # from- and to-bus indices (already internal indexing in ppc_int)
    f_bus = branch[:, F_BUS].astype(int)
    t_bus = branch[:, T_BUS].astype(int)

    # Stamp branches into Ybus
    for k in range(nl):
        f = f_bus[k]
        t = t_bus[k]

        # Skip branches with zero admittance (e.g., out of service)
        if Ys[k] == 0 and Bc[k] == 0:
            continue

        Ybus[f, f] += Yff[k]
        Ybus[t, t] += Ytt[k]
        Ybus[f, t] += Yft[k]
        Ybus[t, f] += Ytf[k]

    # ----- Bus shunts (GS + jBS) -----
    # Ysh = (GS + jBS) / baseMVA  (per-unit)
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    for i in range(nb):
        Ybus[i, i] += Ysh[i]

    return Ybus


def per_unit_to_SI(Y_pu, ppc_int):
    """
    Convert Ybus (per-unit) to SI [Siemens] using the same convention you used:

        Y_SI[i,j] = Y_pu[i,j] * (S_base / (Vbase_i * Vbase_j))

    where:
      - S_base = baseMVA * 1e6 [VA]
      - Vbase_i = BASE_KV_i * 1e3 [V] (per bus)
    """
    bus = np.asarray(ppc_int["bus"], dtype=float)
    baseMVA = float(ppc_int["baseMVA"])
    S_base = baseMVA * 1e6  # VA

    # per-bus nominal voltages
    vn_kv = bus[:, BASE_KV]
    Vbase = vn_kv * 1e3  # V

    denom = np.outer(Vbase, Vbase)  # V_i * V_j
    Y_SI = Y_pu * (S_base / denom)
    return Y_SI


def main():
    # --- Build IEEE-14 net and run a PF so that _ppc + Ybus are built
    net = pn.case14()
    pp.runpp(net, calculate_voltage_angles=True, init="flat")

    # pandapower stores PYPOWER case in net._ppc
    ppc = net._ppc

    # The internal system (reordered buses, etc.)
    if "internal" in ppc:
        ppc_int = ppc["internal"]
    else:
        ppc_int = ppc

    # ----- Ybus from PYPOWER (per-unit) -----
    Ypu_ppc_raw = ppc_int["Ybus"]
    # Handle both sparse and dense versions
    if hasattr(Ypu_ppc_raw, "toarray"):
        Ypu_ppc = Ypu_ppc_raw.toarray().astype(np.complex128)
    else:
        Ypu_ppc = np.asarray(Ypu_ppc_raw, dtype=np.complex128)

    # ----- Ybus from our own stamping (per-unit) -----
    Ypu_stamp = build_Y_stamped_from_ppc(ppc_int)

    # ----- Compare in per-unit -----
    dY_pu = Ypu_ppc - Ypu_stamp
    max_err_pu = np.max(np.abs(dY_pu))

    print("=== Per-unit comparison ===")
    print(f"max |Y_ppc_pu - Y_stamp_pu| = {max_err_pu:.6e}")

    # Show top 10 entries by absolute error
    flat_idx = np.argsort(-np.abs(dY_pu.flatten()))
    print("Top 10 entries by |ΔY_pu|:")
    n = Ypu_ppc.shape[0]
    for idx in flat_idx[:10]:
        i = idx // n
        j = idx % n
        print(
            f"  ΔY_pu[{i},{j}] = {dY_pu[i,j]: .6e} "
            f"(Y_ppc={Ypu_ppc[i,j]: .6e}, Y_stamp={Ypu_stamp[i,j]: .6e})"
        )

    # ----- Convert both to SI and compare -----
    Ysi_ppc = per_unit_to_SI(Ypu_ppc, ppc_int)
    Ysi_stamp = per_unit_to_SI(Ypu_stamp, ppc_int)

    dY_si = Ysi_ppc - Ysi_stamp
    max_err_si = np.max(np.abs(dY_si))

    print("\n=== SI comparison (Siemens) ===")
    print(f"max |Y_ppc_SI - Y_stamp_SI| = {max_err_si:.6e} S")

    flat_idx_si = np.argsort(-np.abs(dY_si.flatten()))
    print("Top 10 entries by |ΔY_SI|:")
    for idx in flat_idx_si[:10]:
        i = idx // n
        j = idx % n
        print(
            f"  ΔY_SI[{i},{j}] = {dY_si[i,j]: .6e} "
            f"(Y_ppc={Ysi_ppc[i,j]: .6e}, Y_stamp={Ysi_stamp[i,j]: .6e})"
        )


if __name__ == "__main__":
    main()