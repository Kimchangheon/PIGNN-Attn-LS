import numpy as np
from typing import Optional, Dict

# -----------------------------
# Helpers
# -----------------------------

def per_unit_to_SI(Y_pu: np.ndarray, ppc_int) -> np.ndarray:
    """Convert Ybus in pu to SI (Siemens) using per-bus Vbase."""
    from pandapower.pypower.idx_bus import BASE_KV
    bus = np.asarray(ppc_int["bus"], dtype=float)
    baseMVA = float(ppc_int["baseMVA"])
    vn_kv = bus[:, BASE_KV].astype(float)
    Vbase = vn_kv * 1e3
    S_base = baseMVA * 1e6
    return Y_pu.astype(np.complex128) * (S_base / np.outer(Vbase, Vbase))


def build_Y_stamped_from_ppc(ppc_int):
    """
    Rebuild Ybus (per-unit) from ppc_int['bus'] and ppc_int['branch'].

    Matches pandapower's extended PPC branch model when available:
      - BR_G (branch shunt conductance)
      - BR_B_ASYM / BR_G_ASYM (extra to-side shunt additions)
    """
    from pandapower.pypower.idx_bus import GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )
    try:
        from pandapower.pypower.idx_brch import BR_G
    except Exception:
        BR_G = None
    try:
        from pandapower.pypower.idx_brch import BR_B_ASYM
    except Exception:
        BR_B_ASYM = None
    try:
        from pandapower.pypower.idx_brch import BR_G_ASYM
    except Exception:
        BR_G_ASYM = None

    bus = np.asarray(ppc_int["bus"], dtype=float)
    branch = np.asarray(ppc_int["branch"], dtype=float)
    baseMVA = float(ppc_int["baseMVA"])
    nb = bus.shape[0]
    nl = branch.shape[0]
    Ybus = np.zeros((nb, nb), dtype=np.complex128)

    stat = branch[:, BR_STATUS]

    # series
    R = branch[:, BR_R]
    X = branch[:, BR_X]
    Z = R + 1j * X
    Ys = np.zeros(nl, dtype=np.complex128)
    nz = (stat != 0) & (np.abs(Z) > 1e-18)
    Ys[nz] = stat[nz] / Z[nz]

    # tap + shift
    tap = np.ones(nl, dtype=np.complex128)
    tap_raw = branch[:, TAP]
    nonunity = tap_raw != 0.0
    tap[nonunity] = tap_raw[nonunity]
    shift_deg = branch[:, SHIFT]
    tap *= np.exp(1j * np.pi / 180.0 * shift_deg)

    # shunts
    B = stat * branch[:, BR_B]
    if BR_G is not None and BR_G < branch.shape[1]:
        G = stat * branch[:, BR_G]
    else:
        G = np.zeros(nl, dtype=float)

    if BR_B_ASYM is not None and BR_B_ASYM < branch.shape[1]:
        B_asym = stat * branch[:, BR_B_ASYM]
    else:
        B_asym = np.zeros(nl, dtype=float)

    if BR_G_ASYM is not None and BR_G_ASYM < branch.shape[1]:
        G_asym = stat * branch[:, BR_G_ASYM]
    else:
        G_asym = np.zeros(nl, dtype=float)

    Bcf = (G + 1j * B)
    Bct = ((G + G_asym) + 1j * (B + B_asym))

    # stamps (pu)
    Yff = (Ys + Bcf / 2.0) / (tap * np.conj(tap))
    Ytt = Ys + Bct / 2.0
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap

    f_bus = branch[:, F_BUS].astype(int)
    t_bus = branch[:, T_BUS].astype(int)

    for k in range(nl):
        if stat[k] == 0:
            continue
        if (abs(Ys[k]) < 1e-18) and (abs(Bcf[k]) < 1e-18) and (abs(Bct[k]) < 1e-18):
            continue
        f = f_bus[k]
        t = t_bus[k]
        Ybus[f, f] += Yff[k]
        Ybus[t, t] += Ytt[k]
        Ybus[f, t] += Yft[k]
        Ybus[t, f] += Ytf[k]

    # bus shunts
    Ysh_pu = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    Ybus[np.arange(nb), np.arange(nb)] += Ysh_pu
    return Ybus


# -----------------------------
# IEEE30 Case generation (ppcY) with direct-SI metadata
# -----------------------------

def case_generation_ieee30_pandapower(
    jitter_load: float = 0.0,
    jitter_gen: float = 0.0,
    pv_vset_range=None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 5.0,
    mag_jitter_pq: float = 0.02,
    seed=None,
    trafo_pfe_kw: Optional[float] = None,
    trafo_i0_percent: Optional[float] = None,
    # NEW: optional PPC-level injection to force nonzero compiled shunt on transformer-like branches
    # Example: {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}
    force_branch_shunt_pu: Optional[Dict[str, float]] = None,
):
    import pandas as pd
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.powerflow import LoadflowNotConverged
    from pandapower.pypower.idx_bus import BASE_KV, GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )
    try:
        from pandapower.pypower.idx_brch import BR_G, BR_B_ASYM, BR_G_ASYM
    except Exception:
        BR_G = BR_B_ASYM = BR_G_ASYM = None

    def _col(row: np.ndarray, idx, default=0.0) -> float:
        if idx is None or idx >= row.shape[0]:
            return default
        return float(row[idx])

    rng = np.random.default_rng(seed)
    net = pn.case30()

    # Try forcing trafo magnetizing via net.trafo if it exists
    if len(getattr(net, "trafo", [])):
        if len(net.trafo):
            if trafo_pfe_kw is not None:
                net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
            if trafo_i0_percent is not None:
                net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

    # Optional jitter
    if jitter_load > 0 and len(net.load):
        s = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"] = net.load["p_mw"].to_numpy(float) * s
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s
    if jitter_gen > 0 and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s
    if pv_vset_range is not None and len(net.gen):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    try:
        pp.runpp(
            net, init="flat", calculate_voltage_angles=True,
            max_iteration=1, enforce_q_lims=False, tolerance_mva=1e9
        )
    except LoadflowNotConverged:
        pass

    ppc_int = net._ppc["internal"]
    baseMVA = float(ppc_int["baseMVA"])
    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    branch_ppc = np.asarray(ppc_int["branch"], dtype=float)

    N = bus_ppc.shape[0]
    vn_kv = bus_ppc[:, BASE_KV].astype(float)
    Vbase = vn_kv * 1e3
    S_base = baseMVA * 1e6

    # Optionally inject compiled shunt into transformer-like PPC branches (for robust tests)
    if force_branch_shunt_pu is not None:
        g_add = float(force_branch_shunt_pu.get("g", 0.0))
        b_add = float(force_branch_shunt_pu.get("b", 0.0))
        g_asym_add = float(force_branch_shunt_pu.get("g_asym", 0.0))
        b_asym_add = float(force_branch_shunt_pu.get("b_asym", 0.0))

        br2 = branch_ppc.copy()
        # transformer-like detection: tap/shift or different base KV
        tap_raw = br2[:, TAP]
        shift_raw = br2[:, SHIFT]
        fb = br2[:, F_BUS].astype(int)
        tb = br2[:, T_BUS].astype(int)
        Vi = Vbase[fb]
        Vj = Vbase[tb]
        is_tr = (np.abs(tap_raw) > 0) & (np.abs(tap_raw - 1.0) > 1e-12)
        is_tr |= (np.abs(shift_raw) > 1e-12)
        is_tr |= (np.abs(Vi - Vj) > 1e-6)

        br2[is_tr, BR_B] += b_add
        if BR_G is not None and BR_G < br2.shape[1]:
            br2[is_tr, BR_G] += g_add
        if BR_B_ASYM is not None and BR_B_ASYM < br2.shape[1]:
            br2[is_tr, BR_B_ASYM] += b_asym_add
        if BR_G_ASYM is not None and BR_G_ASYM < br2.shape[1]:
            br2[is_tr, BR_G_ASYM] += g_asym_add

        # recompute Ybus from the modified PPC (keep everything consistent)
        ppc_mod = dict(ppc_int)
        ppc_mod["branch"] = br2
        Ypu = build_Y_stamped_from_ppc(ppc_mod)
        Y_matrix = per_unit_to_SI(Ypu, ppc_mod)
        branch_used = br2
        bus_used = bus_ppc
        ppc_for_busshunt = ppc_mod
    else:
        # Ground-truth Ybus from pandapower (already computed)
        Ypu = ppc_int["Ybus"].toarray().astype(np.complex128)
        Y_matrix = Ypu * (S_base / np.outer(Vbase, Vbase))
        branch_used = branch_ppc
        bus_used = bus_ppc
        ppc_for_busshunt = ppc_int

    # Bus shunts in SI
    Ysh_pu = (bus_used[:, GS] + 1j * bus_used[:, BS]) / baseMVA
    Y_shunt_bus = Ysh_pu * (S_base / (Vbase * Vbase))

    # Bus types
    bus_typ = np.full(N, 3, dtype=int)
    if len(net.ext_grid):
        bus_typ[net.ext_grid["bus"].to_numpy(int)] = 1
    if len(net.gen):
        bus_typ[net.gen["bus"].to_numpy(int)] = 2

    # Injections
    S = np.zeros(N, dtype=np.complex128)
    if len(net.load):
        b = net.load.bus.to_numpy(int)
        P = net.load.p_mw.to_numpy(float) * 1e6
        Q = net.load.q_mvar.to_numpy(float) * 1e6
        S[b] -= (P + 1j * Q)
    if len(net.sgen):
        b = net.sgen.bus.to_numpy(int)
        P = net.sgen.p_mw.to_numpy(float) * 1e6
        Q = net.sgen.q_mvar.to_numpy(float) * 1e6
        S[b] += (P + 1j * Q)
    if len(net.gen):
        for _, r in net.gen.iterrows():
            S[int(r.bus)] += float(r.p_mw) * 1e6
    s_multi = S

    # u_start
    u_start = Vbase.astype(np.complex128)
    if rand_u_start:
        mag = Vbase.copy()
        if len(net.gen):
            for _, r in net.gen.iterrows():
                b = int(r.bus)
                vm = float(r.vm_pu) if pd.notna(r.vm_pu) else 1.0
                mag[b] = vm * Vbase[b]
        pq_mask = (bus_typ == 3)
        if pq_mask.any() and mag_jitter_pq > 0:
            mag[pq_mask] *= rng.uniform(
                1.0 - mag_jitter_pq, 1.0 + mag_jitter_pq, size=pq_mask.sum()
            )
        ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * np.pi / 180.0
        if len(net.ext_grid):
            ang[net.ext_grid.bus.to_numpy(int)] = 0.0
        u_start = mag * np.exp(1j * ang)

    # Upper-tri metadata
    iu, ju = np.triu_indices(N, 1)
    C = len(iu)
    pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}

    Lines_connected = np.zeros(C, dtype=np.int8)

    # lines
    Y_Lines = np.zeros(C, dtype=np.complex128)
    Y_C_Lines = np.zeros(C, dtype=np.float64)

    # trafos
    Is_trafo = np.zeros(C, dtype=np.int8)
    Trafo_tau = np.ones(C, dtype=np.float64)
    Trafo_shift_deg = np.zeros(C, dtype=np.float64)

    # direct-SI series "views"
    Trafo_y_series_from = np.zeros(C, dtype=np.complex128)  # Ys_pu*S_base/Vf^2
    Trafo_y_series_to   = np.zeros(C, dtype=np.complex128)  # Ys_pu*S_base/Vt^2
    Trafo_y_series_ft   = np.zeros(C, dtype=np.complex128)  # Ys_pu*S_base/(Vf*Vt)

    # end shunts in SI
    Trafo_y_shunt_from = np.zeros(C, dtype=np.complex128)
    Trafo_y_shunt_to   = np.zeros(C, dtype=np.complex128)

    Trafo_tap_on_i = np.zeros(C, dtype=np.int8)
    Trafo_hv_is_i = np.zeros(C, dtype=np.int8)
    Trafo_n = np.ones(C, dtype=np.float64)

    # Fill metadata from branch_used
    for row in branch_used:
        fb = int(row[F_BUS]); tb = int(row[T_BUS])
        r = float(row[BR_R]); x = float(row[BR_X])
        b = float(row[BR_B])
        g = _col(row, BR_G, 0.0)
        b_asym = _col(row, BR_B_ASYM, 0.0)
        g_asym = _col(row, BR_G_ASYM, 0.0)

        tap = float(row[TAP]); shift = float(row[SHIFT])
        status = int(row[BR_STATUS])
        if status == 0:
            continue

        if fb < tb:
            i, j = fb, tb
            tap_on_i = 1
        else:
            i, j = tb, fb
            tap_on_i = 0

        k = pair_to_idx.get((i, j), None)
        if k is None:
            continue
        Lines_connected[k] = 1

        if tap == 0.0:
            tap = 1.0

        Vi, Vj = Vbase[i], Vbase[j]
        Vh, Vl = max(Vi, Vj), min(Vi, Vj)
        Trafo_hv_is_i[k] = 1 if Vi >= Vj else 0
        Trafo_n[k] = (Vh / Vl) if Vl > 0 else 1.0

        # transformer-like detection: tap/shift or different base KV
        Vf = Vbase[fb]
        Vt = Vbase[tb]
        is_tr = (abs(tap - 1.0) > 1e-12) or (abs(shift) > 1e-12) or (abs(Vf - Vt) > 1e-6)

        z = complex(r, x)
        Ys_pu = 0j if abs(z) < 1e-12 else 1.0 / z

        if not is_tr:
            # line
            V_line = Vi
            scale = S_base / (V_line ** 2)
            Y_Lines[k] = Ys_pu * scale
            Y_C_Lines[k] = (0.5 * b) * scale
        else:
            # transformer-like
            Is_trafo[k] = 1
            Trafo_tau[k] = tap
            Trafo_shift_deg[k] = shift
            Trafo_tap_on_i[k] = tap_on_i

            Trafo_y_series_from[k] = Ys_pu * (S_base / (Vf ** 2))
            Trafo_y_series_to[k]   = Ys_pu * (S_base / (Vt ** 2))
            Trafo_y_series_ft[k]   = Ys_pu * (S_base / (Vf * Vt))

            Bcf_pu = (g + 1j * b)
            Bct_pu = ((g + g_asym) + 1j * (b + b_asym))
            Trafo_y_shunt_from[k] = Bcf_pu * (S_base / (Vf ** 2))
            Trafo_y_shunt_to[k]   = Bct_pu * (S_base / (Vt ** 2))

    is_connected = True
    U_base = float(Vbase[0])
    gridtype = "IEEE30_pandapower_ppcY"

    return (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Y_Lines, Y_C_Lines, Lines_connected.astype(np.int8), U_base, S_base,
        vn_kv.astype(np.float64), Is_trafo.astype(np.int8),
        Trafo_tau, Trafo_shift_deg,
        Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus.astype(np.complex128),
        Trafo_tap_on_i.astype(np.int8), Trafo_hv_is_i.astype(np.int8), Trafo_n.astype(np.float64),
    )


def case_generation_ieee30_pandapower_stamped(
    **kwargs
):
    """
    Same as ppcY version, but Y_matrix is built by stamping branch+bus (pu) then converting to SI.
    We reuse the ppcY generator but request the Y_matrix be computed from build_Y_stamped_from_ppc.
    Easiest: call ppcY generator with force_branch_shunt_pu=None and then overwrite Y_matrix
    from the same ppc model is more invasive; so we re-run here properly.
    """
    # For IEEE30, we can just call ppcY generator with force_branch_shunt_pu passthrough,
    # because when force_branch_shunt_pu is not None we already recompute Y_matrix via stamping.
    # But when it is None, ppcY uses ppc_int["Ybus"].
    # So to make "stamped" truly stamped always, we just set a tiny no-op dict and then undo it? Not ideal.
    # Instead: call ppcY generator with force_branch_shunt_pu={"g":0,"b":0,...} AND explicitly
    # recompute stamped Ybus even for the no-op case.
    #
    # To keep this concise, we implement a true stamped build by calling ppcY generator twice:
    # - first to get ppc_int equivalent and metadata is already built from branch_used
    # Here we do a cleaner approach: just force recompute using build_Y_stamped_from_ppc by passing
    # force_branch_shunt_pu={"g":0,"b":0,"g_asym":0,"b_asym":0} and set a flag that it is a no-op.
    #
    # This ensures Y_matrix is produced by stamping even when no injection is desired.

    noop = {"g": 0.0, "b": 0.0, "g_asym": 0.0, "b_asym": 0.0}
    if "force_branch_shunt_pu" in kwargs and kwargs["force_branch_shunt_pu"] is None:
        kwargs["force_branch_shunt_pu"] = noop
    elif "force_branch_shunt_pu" not in kwargs:
        kwargs["force_branch_shunt_pu"] = noop

    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
     Trafo_y_shunt_from, Trafo_y_shunt_to,
     Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee30_pandapower(**kwargs)

    gridtype = "IEEE30_pandapower_stamped"
    return (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
        vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
        Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n
    )


# -----------------------------
# Reconstruction: DIRECT SI stamping (no pu conversion)
# -----------------------------

def reconstruct_Y_pandapower_branchwise_direct_SI(
    N: int,
    vn_kv: np.ndarray,
    Lines_connected: np.ndarray,
    Y_Lines: np.ndarray,
    Y_C_Lines: np.ndarray,
    Is_trafo: np.ndarray,
    Trafo_tau: np.ndarray,
    Trafo_shift_deg: np.ndarray,
    Trafo_y_series_from: np.ndarray,
    Trafo_y_series_to: np.ndarray,
    Trafo_y_series_ft: np.ndarray,
    Trafo_y_shunt_from: np.ndarray,
    Trafo_y_shunt_to: np.ndarray,
    Y_shunt_bus: np.ndarray,
    Trafo_tap_on_i: np.ndarray,
) -> np.ndarray:
    N = int(N)
    Y_SI = np.zeros((N, N), dtype=np.complex128)
    iu, ju = np.triu_indices(N, 1)

    # bus shunts
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    for k, (i, j) in enumerate(zip(iu, ju)):
        if Lines_connected[k] == 0:
            continue

        if Is_trafo[k] == 0:
            # line
            y = Y_Lines[k]
            B_end = float(Y_C_Lines[k])  # per-end susceptance (SI)
            Yff = y + 1j * B_end
            Ytt = y + 1j * B_end
            Yft = -y
            Ytf = -y

            Y_SI[i, i] += Yff
            Y_SI[j, j] += Ytt
            Y_SI[i, j] += Yft
            Y_SI[j, i] += Ytf

        else:
            # transformer: determine ppc from/to side using tap orientation
            if int(Trafo_tap_on_i[k]) == 1:
                f, tbus = i, j
            else:
                f, tbus = j, i

            tau = float(Trafo_tau[k])
            theta = np.deg2rad(float(Trafo_shift_deg[k]))
            a = tau * np.exp(1j * theta)

            y_from = Trafo_y_series_from[k]
            y_to   = Trafo_y_series_to[k]
            y_ft   = Trafo_y_series_ft[k]
            ysh_f  = Trafo_y_shunt_from[k]
            ysh_t  = Trafo_y_shunt_to[k]

            Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
            Ytt = (y_to   + ysh_t / 2.0)
            Yft = -y_ft / np.conj(a)
            Ytf = -y_ft / a

            Y_SI[f, f]       += Yff
            Y_SI[tbus, tbus] += Ytt
            Y_SI[f, tbus]    += Yft
            Y_SI[tbus, f]    += Ytf

    return Y_SI


# -----------------------------
# Tests (plain + forced shunt injection)
# -----------------------------

def test_ieee30_pandapower():
    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
     Trafo_y_shunt_from, Trafo_y_shunt_to,
     Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee30_pandapower(seed=0)

    N = len(bus_typ)
    Y_rec = reconstruct_Y_pandapower_branchwise_direct_SI(
        N, vn_kv, Lines_connected,
        Y_Lines, Y_C_Lines,
        Is_trafo,
        Trafo_tau, Trafo_shift_deg,
        Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus, Trafo_tap_on_i
    )
    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee30_pandapower ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    assert np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9)


def test_ieee30_pandapower_stamped_forced_shunt():
    # Force a noticeable compiled shunt on transformer-like PPC branches (works even if net.trafo is empty)
    forced = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

    (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
     Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
     vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
     Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
     Trafo_y_shunt_from, Trafo_y_shunt_to,
     Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_ieee30_pandapower_stamped(
        seed=0, force_branch_shunt_pu=forced
    )

    N = len(bus_typ)
    Y_rec = reconstruct_Y_pandapower_branchwise_direct_SI(
        N, vn_kv, Lines_connected,
        Y_Lines, Y_C_Lines,
        Is_trafo,
        Trafo_tau, Trafo_shift_deg,
        Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus, Trafo_tap_on_i
    )
    diff = np.max(np.abs(Y_rec - Y_matrix))
    print("=== test_ieee30_pandapower_stamped_forced_shunt ===")
    print("gridtype:", gridtype)
    print("max |Y_rec - Y_matrix|:", diff)
    assert np.allclose(Y_rec, Y_matrix, rtol=1e-9, atol=1e-9)