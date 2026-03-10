import numpy as np
from typing import Optional, Dict, Callable, Any, Union

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
# Generic pandapower case generation (ppcY OR stamped)
# -----------------------------

CaseFn = Union[str, Callable[..., Any]]

def case_generation_pandapower(
    case_fn: CaseFn,
    case_kwargs: Optional[Dict[str, Any]] = None,
    *,
    # behavior
    ybus_mode: str = "ppcY",  # "ppcY" uses ppc_int["Ybus"] when possible; "stamped" always stamps from PPC
    seed=None,
    # jitters
    jitter_load: float = 0.0,
    jitter_gen: float = 0.0,
    pv_vset_range=None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 5.0,
    mag_jitter_pq: float = 0.02,
    # magnetizing via net.trafo (if exists)
    trafo_pfe_kw: Optional[float] = None,
    trafo_i0_percent: Optional[float] = None,
    # fallback: force PPC branch shunts on transformer-like branches
    # Example: {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}
    force_branch_shunt_pu: Optional[Dict[str, float]] = None,
):
    """
    Generic generator for pandapower test cases with DIRECT-SI metadata.

    Returns:
      (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
       Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
       vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
       Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
       Trafo_y_shunt_from, Trafo_y_shunt_to,
       Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n)

    Notes:
    - Uses upper-tri pairs (N*(N-1)/2) as your existing format does.
    - Transformer shunts use the pandapower-extended PPC columns when available.
    - "Direct SI" trafo series is stored as 3 SI views so reconstruction can stamp directly in SI.

    ybus_mode:
      - "ppcY": Y_matrix comes from ppc_int["Ybus"] (unless force_branch_shunt_pu is used)
      - "stamped": Y_matrix always computed by build_Y_stamped_from_ppc(ppc_used) -> per_unit_to_SI
    """
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

    # Resolve case_fn
    if isinstance(case_fn, str):
        fn = getattr(pn, case_fn, None)
        if fn is None:
            raise ValueError(f"Unknown pandapower case function name: {case_fn}")
        case_name = case_fn
    else:
        fn = case_fn
        case_name = getattr(case_fn, "__name__", "pandapower_case")

    case_kwargs = case_kwargs or {}

    rng = np.random.default_rng(seed)

    # ---- Build net ----
    net = fn(**case_kwargs)

    # ---- Apply trafo magnetizing if net.trafo exists and non-empty ----
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        if trafo_pfe_kw is not None:
            net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
        if trafo_i0_percent is not None:
            net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

    # ---- Optional jitter ----
    if jitter_load > 0 and hasattr(net, "load") and len(net.load):
        s = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"] = net.load["p_mw"].to_numpy(float) * s
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s

    if jitter_gen > 0 and hasattr(net, "gen") and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s

    if pv_vset_range is not None and hasattr(net, "gen") and len(net.gen):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    # ---- Build internal PPC ----
    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
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

    # ---- Optional PPC shunt forcing (for cases with no net.trafo) ----
    if force_branch_shunt_pu is not None:
        g_add = float(force_branch_shunt_pu.get("g", 0.0))
        b_add = float(force_branch_shunt_pu.get("b", 0.0))
        g_asym_add = float(force_branch_shunt_pu.get("g_asym", 0.0))
        b_asym_add = float(force_branch_shunt_pu.get("b_asym", 0.0))

        br2 = branch_ppc.copy()
        tap_raw = br2[:, TAP]
        shift_raw = br2[:, SHIFT]
        fb = br2[:, F_BUS].astype(int)
        tb = br2[:, T_BUS].astype(int)
        Vi = Vbase[fb]
        Vj = Vbase[tb]

        # transformer-like branches in PPC
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

        ppc_used = dict(ppc_int)
        ppc_used["branch"] = br2
        branch_used = br2
    else:
        ppc_used = ppc_int
        branch_used = branch_ppc

    # ---- Y_matrix in SI ----
    if (ybus_mode.lower() == "ppcy") and (force_branch_shunt_pu is None):
        Ypu = ppc_int["Ybus"].toarray().astype(np.complex128)
        Y_matrix = Ypu * (S_base / np.outer(Vbase, Vbase))
    else:
        # always recompute from PPC data (also required when we modify branch shunts)
        Ypu = build_Y_stamped_from_ppc(ppc_used)
        Y_matrix = per_unit_to_SI(Ypu, ppc_used)

    # ---- Bus shunts (SI) ----
    Ysh_pu = (bus_ppc[:, GS] + 1j * bus_ppc[:, BS]) / baseMVA
    Y_shunt_bus = Ysh_pu * (S_base / (Vbase * Vbase))

    # ---- Bus types ----
    bus_typ = np.full(N, 3, dtype=int)
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        bus_typ[net.ext_grid["bus"].to_numpy(int)] = 1
    if hasattr(net, "gen") and len(net.gen):
        bus_typ[net.gen["bus"].to_numpy(int)] = 2

    # ---- Injections (SI) ----
    S = np.zeros(N, dtype=np.complex128)
    if hasattr(net, "load") and len(net.load):
        b = net.load.bus.to_numpy(int)
        P = net.load.p_mw.to_numpy(float) * 1e6
        Q = net.load.q_mvar.to_numpy(float) * 1e6
        S[b] -= (P + 1j * Q)
    if hasattr(net, "sgen") and len(net.sgen):
        b = net.sgen.bus.to_numpy(int)
        P = net.sgen.p_mw.to_numpy(float) * 1e6
        Q = net.sgen.q_mvar.to_numpy(float) * 1e6
        S[b] += (P + 1j * Q)
    if hasattr(net, "gen") and len(net.gen):
        for _, r in net.gen.iterrows():
            S[int(r.bus)] += float(r.p_mw) * 1e6
    s_multi = S

    # ---- u_start ----
    u_start = Vbase.astype(np.complex128)
    if rand_u_start:
        mag = Vbase.copy()

        if hasattr(net, "gen") and len(net.gen):
            for _, r in net.gen.iterrows():
                b = int(r.bus)
                vm = float(r.vm_pu) if pd.notna(r.vm_pu) else 1.0
                mag[b] = vm * Vbase[b]

        pq_mask = (bus_typ == 3)
        if pq_mask.any() and mag_jitter_pq > 0:
            mag[pq_mask] *= rng.uniform(1.0 - mag_jitter_pq, 1.0 + mag_jitter_pq, size=pq_mask.sum())

        ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * np.pi / 180.0
        if hasattr(net, "ext_grid") and len(net.ext_grid):
            ang[net.ext_grid.bus.to_numpy(int)] = 0.0

        u_start = mag * np.exp(1j * ang)

    # ---- Upper-tri metadata arrays ----
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

    # ---- Fill metadata from branch_used ----
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

        # map to upper-tri and record whether PPC "from-bus" is the iu-side
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

        # transformer-like detection: tap/shift or different base kV
        Vf = Vbase[fb]
        Vt = Vbase[tb]
        is_tr = (abs(tap - 1.0) > 1e-12) or (abs(shift) > 1e-12) or (abs(Vf - Vt) > 1e-6)

        z = complex(r, x)
        Ys_pu = 0j if abs(z) < 1e-12 else 1.0 / z

        if not is_tr:
            # line: pure line charging stored per-end
            V_line = Vi
            scale = S_base / (V_line ** 2)
            Y_Lines[k] += Ys_pu * scale
            Y_C_Lines[k] += (0.5 * b) * scale
        else:
            Is_trafo[k] = 1
            Trafo_tau[k] = tap
            Trafo_shift_deg[k] = shift
            Trafo_tap_on_i[k] = tap_on_i

            # direct-SI series views (use PPC from/to bus bases)
            Trafo_y_series_from[k] += Ys_pu * (S_base / (Vf ** 2))
            Trafo_y_series_to[k]   += Ys_pu * (S_base / (Vt ** 2))
            Trafo_y_series_ft[k]   += Ys_pu * (S_base / (Vf * Vt))

            # end shunts (pandapower compiled) in SI on from/to bases
            Bcf_pu = (g + 1j * b)
            Bct_pu = ((g + g_asym) + 1j * (b + b_asym))
            Trafo_y_shunt_from[k] += Bcf_pu * (S_base / (Vf ** 2))
            Trafo_y_shunt_to[k]   += Bct_pu * (S_base / (Vt ** 2))

    is_connected = True
    U_base = float(Vbase[0])
    gridtype = f"{case_name}_pandapower_{'ppcY' if ybus_mode.lower()=='ppcy' else 'stamped'}"

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


def case_generation_pandapower_stamped(
    case_fn: CaseFn,
    case_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Convenience wrapper: always builds Y_matrix via stamping."""
    return case_generation_pandapower(case_fn, case_kwargs=case_kwargs, ybus_mode="stamped", **kwargs)


# -----------------------------
# Direct-SI reconstruction (unchanged; works for all cases)
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
    Y_SI = np.zeros((N, N), dtype=np.complex128)
    iu, ju = np.triu_indices(N, 1)

    # bus shunts
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    for k, (i, j) in enumerate(zip(iu, ju)):
        if Lines_connected[k] == 0:
            continue

        if Is_trafo[k] == 0:
            y = Y_Lines[k]
            B_end = float(Y_C_Lines[k])
            Yff = y + 1j * B_end
            Ytt = y + 1j * B_end
            Yft = -y
            Ytf = -y

            Y_SI[i, i] += Yff
            Y_SI[j, j] += Ytt
            Y_SI[i, j] += Yft
            Y_SI[j, i] += Ytf

        else:
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
# Example: run over your CASES list
# -----------------------------
if __name__ == "__main__":
    import pandapower.networks as pn

    CASES = [
        "case4gs",
        "case5",
        "case6ww",
        "case9",
        "case14",
        "case24_ieee_rts",
        "case30",
        "case_ieee30",
        "case33bw",
        "case39",
        "case57",
        "case89pegase",
        "case118",
        "case145",
        "case_illinois200",
        "case300",
        "case1354pegase",
        "case1888rte",
        "case2848rte",
        "case2869pegase",
        "case3120sp",
        "case6470rte",
        "case6495rte",
        "case6515rte",
        "case9241pegase",
        "GBnetwork",
        "GBreducednetwork",
        "iceland",
    ]

    # Cases with empty net.trafo in your run (use this list if you want a consistent "magnetizing-like" test):
    FORCE_SHUNT_CASES = {"case4gs", "case5", "case6ww", "case9", "case30", "case33bw"}

    for name in CASES:
        fn = getattr(pn, name)
        force = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0} if name in FORCE_SHUNT_CASES else None

        (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
         Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
         vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg,
         Trafo_y_series_from, Trafo_y_series_to, Trafo_y_series_ft,
         Trafo_y_shunt_from, Trafo_y_shunt_to,
         Y_shunt_bus, Trafo_tap_on_i, Trafo_hv_is_i, Trafo_n) = case_generation_pandapower(
            fn,
            case_kwargs={},
            ybus_mode="ppcY",
            seed=0,
            trafo_pfe_kw=50.0,
            trafo_i0_percent=2.0,
            force_branch_shunt_pu=force,
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
        print(name, gridtype, "max|diff|=", diff)