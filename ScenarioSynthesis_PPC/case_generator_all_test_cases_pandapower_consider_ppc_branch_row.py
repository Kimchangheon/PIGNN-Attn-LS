import numpy as np
from typing import Optional, Dict, Callable, Any, Union


# ============================================================
# Types
# ============================================================

CaseFn = Union[str, Callable[..., Any]]


# ============================================================
# Helpers
# ============================================================

def _optional_branch_indices():
    """Return optional pandapower extended branch column indices if available."""
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
    return BR_G, BR_B_ASYM, BR_G_ASYM


def _col(row: np.ndarray, idx, default=0.0) -> float:
    if idx is None or idx >= row.shape[0]:
        return default
    return float(row[idx])


def _resolve_case_function(case_fn: CaseFn):
    import pandapower.networks as pn

    if isinstance(case_fn, str):
        fn = getattr(pn, case_fn, None)
        if fn is None:
            raise ValueError(f"Unknown pandapower case function name: {case_fn}")
        case_name = case_fn
    else:
        fn = case_fn
        case_name = getattr(case_fn, "__name__", "pandapower_case")
    return fn, case_name


def _dense_ybus_from_ppc_internal(ppc_int) -> np.ndarray:
    """Safely extract dense Ybus from ppc internal dict."""
    Y = ppc_int["Ybus"]
    if hasattr(Y, "toarray"):
        return Y.toarray().astype(np.complex128)
    return np.asarray(Y, dtype=np.complex128)


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
      - BR_G
      - BR_B_ASYM
      - BR_G_ASYM
    """
    from pandapower.pypower.idx_bus import GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    BR_G, BR_B_ASYM, BR_G_ASYM = _optional_branch_indices()

    bus = np.asarray(ppc_int["bus"], dtype=float)
    branch = np.asarray(ppc_int["branch"], dtype=float)
    baseMVA = float(ppc_int["baseMVA"])

    nb = bus.shape[0]
    nl = branch.shape[0]
    Ybus = np.zeros((nb, nb), dtype=np.complex128)

    stat = branch[:, BR_STATUS]

    # series admittance
    R = branch[:, BR_R]
    X = branch[:, BR_X]
    Z = R + 1j * X
    Ys = np.zeros(nl, dtype=np.complex128)
    nz = (stat != 0) & (np.abs(Z) > 1e-18)
    Ys[nz] = stat[nz] / Z[nz]

    # tap + phase shift
    tap = np.ones(nl, dtype=np.complex128)
    tap_raw = branch[:, TAP]
    nonunity = tap_raw != 0.0
    tap[nonunity] = tap_raw[nonunity]
    shift_deg = branch[:, SHIFT]
    tap *= np.exp(1j * np.pi / 180.0 * shift_deg)

    # branch shunts
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

    # end shunts in pu before half-split
    Bcf = (G + 1j * B)
    Bct = ((G + G_asym) + 1j * (B + B_asym))

    # branch stamps in pu
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


# ============================================================
# Generic generator: one metadata row per PPC branch row
# ============================================================

def case_generation_pandapower(
    case_fn: CaseFn,
    case_kwargs: Optional[Dict[str, Any]] = None,
    *,
    ybus_mode: str = "ppcY",   # "ppcY" or "stamped"
    seed=None,
    jitter_load: float = 0.0,
    jitter_gen: float = 0.0,
    pv_vset_range=None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 5.0,
    mag_jitter_pq: float = 0.02,
    trafo_pfe_kw: Optional[float] = None,
    trafo_i0_percent: Optional[float] = None,
    # optional PPC-level forcing (best used when net.trafo is empty)
    force_branch_shunt_pu: Optional[Dict[str, float]] = None,
):
    """
    Generic pandapower case generator with ONE metadata row per PPC branch row.

    Important:
      - bus_typ is derived from compiled PPC bus types
      - s_multi is derived from compiled PPC bus/gen tables
      - u_start is built in PPC order using compiled PPC gen setpoints

    Returns:
      (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,

        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,

        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,

        Y_shunt_bus,

        Is_trafo, Branch_hv_is_f, Branch_n,

        Y_Lines, Y_C_Lines,
        U_base, S_base, vn_kv
      )

    Convention for bus_typ (your NR code):
      1 = slack
      2 = PV
      3 = PQ
    """
    import pandapower as pp
    from pandapower.powerflow import LoadflowNotConverged
    from pandapower.pypower.idx_bus import BASE_KV, GS, BS, BUS_TYPE, PD, QD
    from pandapower.pypower.idx_gen import GEN_BUS, PG, VG, GEN_STATUS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    BR_G, BR_B_ASYM, BR_G_ASYM = _optional_branch_indices()
    fn, case_name = _resolve_case_function(case_fn)

    case_kwargs = case_kwargs or {}
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1) Build network
    # ------------------------------------------------------------
    net = fn(**case_kwargs)

    # optional transformer magnetizing params
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        if trafo_pfe_kw is not None:
            net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
        if trafo_i0_percent is not None:
            net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

    # optional load jitter
    if jitter_load > 0 and hasattr(net, "load") and len(net.load):
        s = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"] = net.load["p_mw"].to_numpy(float) * s
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s

    # optional generator active-power jitter
    if jitter_gen > 0 and hasattr(net, "gen") and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s

    # optional PV voltage-setpoint jitter (on pandapower gens before compilation)
    if pv_vset_range is not None and hasattr(net, "gen") and len(net.gen):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    # ------------------------------------------------------------
    # 2) Compile PPC / Ybus
    # ------------------------------------------------------------
    runpp_exception = None
    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9,
        )
    except LoadflowNotConverged as e:
        runpp_exception = e
    except Exception as e:
        # keep the exception around, but still try to use compiled _ppc if available
        runpp_exception = e

    if not hasattr(net, "_ppc") or "internal" not in net._ppc:
        raise RuntimeError(
            f"Could not build internal PPC for case {case_name}. "
            f"Original runpp exception: {repr(runpp_exception)}"
        )

    ppc_int = net._ppc["internal"]
    baseMVA = float(ppc_int["baseMVA"])

    bus_ppc = np.asarray(ppc_int["bus"], dtype=float)
    branch_ppc = np.asarray(ppc_int["branch"], dtype=float)

    gen_ppc = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)
    if gen_ppc.ndim == 1 and gen_ppc.size == 0:
        gen_ppc = gen_ppc.reshape(0, 0)

    N = bus_ppc.shape[0]
    nl = branch_ppc.shape[0]

    vn_kv = bus_ppc[:, BASE_KV].astype(float)
    Vbase = vn_kv * 1e3
    S_base = baseMVA * 1e6

    # ------------------------------------------------------------
    # 3) Optional PPC branch-shunt forcing
    # ------------------------------------------------------------
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
        Vf0 = Vbase[fb]
        Vt0 = Vbase[tb]

        is_tr_like = (np.abs(tap_raw) > 0) & (np.abs(tap_raw - 1.0) > 1e-12)
        is_tr_like |= (np.abs(shift_raw) > 1e-12)
        is_tr_like |= (np.abs(Vf0 - Vt0) > 1e-6)

        br2[is_tr_like, BR_B] += b_add
        if BR_G is not None and BR_G < br2.shape[1]:
            br2[is_tr_like, BR_G] += g_add
        if BR_B_ASYM is not None and BR_B_ASYM < br2.shape[1]:
            br2[is_tr_like, BR_B_ASYM] += b_asym_add
        if BR_G_ASYM is not None and BR_G_ASYM < br2.shape[1]:
            br2[is_tr_like, BR_G_ASYM] += g_asym_add

        ppc_used = dict(ppc_int)
        ppc_used["branch"] = br2
        branch_used = br2
    else:
        ppc_used = ppc_int
        branch_used = branch_ppc

    # ------------------------------------------------------------
    # 4) Y_matrix in SI
    # ------------------------------------------------------------
    if (ybus_mode.lower() == "ppcy") and (force_branch_shunt_pu is None):
        Ypu = _dense_ybus_from_ppc_internal(ppc_int)
        Y_matrix = Ypu * (S_base / np.outer(Vbase, Vbase))
    else:
        Ypu = build_Y_stamped_from_ppc(ppc_used)
        Y_matrix = per_unit_to_SI(Ypu, ppc_used)

    # ------------------------------------------------------------
    # 5) Bus shunts in SI
    # ------------------------------------------------------------
    Ysh_pu = (bus_ppc[:, GS] + 1j * bus_ppc[:, BS]) / baseMVA
    Y_shunt_bus = Ysh_pu * (S_base / (Vbase * Vbase))

    # ------------------------------------------------------------
    # 6) bus_typ from compiled PPC bus table
    #    PYPOWER: 1=PQ, 2=PV, 3=REF
    #    Your NR : 1=slack, 2=PV, 3=PQ
    # ------------------------------------------------------------
    ppc_bus_type = bus_ppc[:, BUS_TYPE].astype(int)

    bus_typ = np.full(N, 3, dtype=np.int8)  # default PQ in your convention
    bus_typ[ppc_bus_type == 3] = 1          # REF -> slack
    bus_typ[ppc_bus_type == 2] = 2          # PV  -> PV
    bus_typ[ppc_bus_type == 1] = 3          # PQ  -> PQ

    # ------------------------------------------------------------
    # 7) s_multi from compiled PPC bus/gen tables, aligned to Ybus
    #
    #    injection convention:
    #      + generation
    #      - load
    # ------------------------------------------------------------
    s_multi = -(bus_ppc[:, PD] + 1j * bus_ppc[:, QD]) * 1e6

    if gen_ppc.size > 0 and gen_ppc.shape[1] > max(GEN_BUS, PG, GEN_STATUS):
        on = gen_ppc[:, GEN_STATUS] > 0
        gbus = gen_ppc[on, GEN_BUS].astype(int)
        pg = gen_ppc[on, PG].astype(float) * 1e6
        np.add.at(s_multi, gbus, pg)

    # ------------------------------------------------------------
    # 8) u_start in PPC order
    #
    #    Base start:
    #      - flat angles
    #      - magnitude = 1.0 pu * local Vbase
    #      - overwrite PV/slack magnitudes using compiled PPC VG
    # ------------------------------------------------------------
    mag = Vbase.copy()
    ang = np.zeros(N, dtype=np.float64)

    if gen_ppc.size > 0 and gen_ppc.shape[1] > max(GEN_BUS, VG, GEN_STATUS):
        on = gen_ppc[:, GEN_STATUS] > 0
        gbus = gen_ppc[on, GEN_BUS].astype(int)
        vg = gen_ppc[on, VG].astype(float)
        for b, vm in zip(gbus, vg):
            if 0 <= b < N and np.isfinite(vm) and vm > 0:
                mag[b] = vm * Vbase[b]

    if rand_u_start:
        # jitter PQ magnitudes only
        pq_mask = (bus_typ == 3)
        if pq_mask.any() and mag_jitter_pq > 0:
            mag[pq_mask] *= rng.uniform(
                1.0 - mag_jitter_pq,
                1.0 + mag_jitter_pq,
                size=int(pq_mask.sum())
            )

        # jitter non-slack angles only
        non_slack = (bus_typ != 1)
        if non_slack.any() and angle_jitter_deg > 0:
            ang[non_slack] = rng.uniform(
                -angle_jitter_deg,
                angle_jitter_deg,
                size=int(non_slack.sum())
            ) * np.pi / 180.0

    u_start = mag * np.exp(1j * ang)

    # ------------------------------------------------------------
    # 9) One metadata row per PPC branch row
    # ------------------------------------------------------------
    Branch_f_bus = branch_used[:, F_BUS].astype(np.int32)
    Branch_t_bus = branch_used[:, T_BUS].astype(np.int32)
    Branch_status = branch_used[:, BR_STATUS].astype(np.int8)

    Branch_tau = np.ones(nl, dtype=np.float64)
    Branch_shift_deg = np.zeros(nl, dtype=np.float64)

    Branch_y_series_from = np.zeros(nl, dtype=np.complex128)
    Branch_y_series_to = np.zeros(nl, dtype=np.complex128)
    Branch_y_series_ft = np.zeros(nl, dtype=np.complex128)

    Branch_y_shunt_from = np.zeros(nl, dtype=np.complex128)
    Branch_y_shunt_to = np.zeros(nl, dtype=np.complex128)

    Is_trafo = np.zeros(nl, dtype=np.int8)
    Branch_hv_is_f = np.zeros(nl, dtype=np.int8)
    Branch_n = np.ones(nl, dtype=np.float64)

    # informational line-only fields
    Y_Lines = np.zeros(nl, dtype=np.complex128)
    Y_C_Lines = np.zeros(nl, dtype=np.float64)

    for k, row in enumerate(branch_used):
        fb = int(row[F_BUS])
        tb = int(row[T_BUS])

        stat = float(row[BR_STATUS])
        r = float(row[BR_R])
        x = float(row[BR_X])

        b = stat * float(row[BR_B])
        g = stat * _col(row, BR_G, 0.0)
        b_asym = stat * _col(row, BR_B_ASYM, 0.0)
        g_asym = stat * _col(row, BR_G_ASYM, 0.0)

        tau = float(row[TAP])
        if tau == 0.0:
            tau = 1.0
        shift = float(row[SHIFT])

        Branch_tau[k] = tau
        Branch_shift_deg[k] = shift

        Vf = Vbase[fb]
        Vt = Vbase[tb]
        Vh = max(Vf, Vt)
        Vl = min(Vf, Vt)

        Branch_hv_is_f[k] = 1 if Vf >= Vt else 0
        Branch_n[k] = (Vh / Vl) if Vl > 0 else 1.0

        is_tr = (abs(tau - 1.0) > 1e-12) or (abs(shift) > 1e-12) or (abs(Vf - Vt) > 1e-6)
        Is_trafo[k] = 1 if is_tr else 0

        z = complex(r, x)
        Ys_pu = 0j if (stat == 0.0 or abs(z) < 1e-12) else (stat / z)

        # direct-SI series views
        Branch_y_series_from[k] = Ys_pu * (S_base / (Vf ** 2))
        Branch_y_series_to[k] = Ys_pu * (S_base / (Vt ** 2))
        Branch_y_series_ft[k] = Ys_pu * (S_base / (Vf * Vt))

        # direct-SI end shunts
        Bcf_pu = (g + 1j * b)
        Bct_pu = ((g + g_asym) + 1j * (b + b_asym))
        Branch_y_shunt_from[k] = Bcf_pu * (S_base / (Vf ** 2))
        Branch_y_shunt_to[k] = Bct_pu * (S_base / (Vt ** 2))

        # informational line-only fields
        if not is_tr:
            Y_Lines[k] = Branch_y_series_ft[k]
            Y_C_Lines[k] = 0.5 * b * (S_base / (Vf ** 2))

    is_connected = bool(N > 0)

    # kept only for backward compatibility with your old interface
    U_base = float(Vbase[0])

    gridtype = f"{case_name}_pandapower_{'ppcY' if ybus_mode.lower() == 'ppcy' else 'stamped'}"

    return (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,

        Branch_f_bus, Branch_t_bus, Branch_status,
        Branch_tau, Branch_shift_deg,

        Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
        Branch_y_shunt_from, Branch_y_shunt_to,

        Y_shunt_bus.astype(np.complex128),

        Is_trafo, Branch_hv_is_f, Branch_n,

        Y_Lines, Y_C_Lines,
        U_base, S_base, vn_kv.astype(np.float64),
    )


def case_generation_pandapower_stamped(
    case_fn: CaseFn,
    case_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Wrapper that always builds Y_matrix by stamping PPC branch+bus data."""
    return case_generation_pandapower(
        case_fn=case_fn,
        case_kwargs=case_kwargs,
        ybus_mode="stamped",
        **kwargs,
    )


# ============================================================
# Reconstruction: direct SI, one metadata row per PPC branch
# ============================================================

def reconstruct_Y_pandapower_branchrows_direct_SI(
    N: int,
    Branch_f_bus: np.ndarray,
    Branch_t_bus: np.ndarray,
    Branch_status: np.ndarray,
    Branch_tau: np.ndarray,
    Branch_shift_deg: np.ndarray,
    Branch_y_series_from: np.ndarray,
    Branch_y_series_to: np.ndarray,
    Branch_y_series_ft: np.ndarray,
    Branch_y_shunt_from: np.ndarray,
    Branch_y_shunt_to: np.ndarray,
    Y_shunt_bus: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct Ybus directly in SI, using ONE metadata row per PPC branch row.

    Direct SI stamps:
      Yff = (y_from + ysh_from/2) / |a|^2
      Ytt =  y_to   + ysh_to/2
      Yft = -y_ft / conj(a)
      Ytf = -y_ft / a

    where:
      a = tau * exp(j*theta)
    """
    N = int(N)
    nl = len(Branch_f_bus)

    Y_SI = np.zeros((N, N), dtype=np.complex128)
    Y_SI[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    for k in range(nl):
        if int(Branch_status[k]) == 0:
            continue

        f = int(Branch_f_bus[k])
        t = int(Branch_t_bus[k])

        tau = float(Branch_tau[k])
        theta = np.deg2rad(float(Branch_shift_deg[k]))
        a = tau * np.exp(1j * theta)

        y_from = complex(Branch_y_series_from[k])
        y_to = complex(Branch_y_series_to[k])
        y_ft = complex(Branch_y_series_ft[k])
        ysh_f = complex(Branch_y_shunt_from[k])
        ysh_t = complex(Branch_y_shunt_to[k])

        Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
        Ytt = (y_to + ysh_t / 2.0)
        Yft = -y_ft / np.conj(a)
        Ytf = -y_ft / a

        Y_SI[f, f] += Yff
        Y_SI[t, t] += Ytt
        Y_SI[f, t] += Yft
        Y_SI[t, f] += Ytf

    return Y_SI


# ============================================================
# Example usage over your CASES list
# ============================================================

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

    # cases where net.trafo is empty, so forced PPC branch shunt is useful
    FORCE_SHUNT_CASES = {
        "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
    }

    CASE_KWARGS: Dict[str, Dict[str, Any]] = {
        # "case1888rte": {"ref_bus_idx": 1246},
        # "case2848rte": {"ref_bus_idx": 271},
        # "case6470rte": {"ref_bus_idx": 5988},
        # "case6515rte": {"ref_bus_idx": 6171},
    }

    print("=== branch-row direct-SI reconstruction check ===")

    for name in CASES:
        fn = getattr(pn, name)
        case_kwargs = CASE_KWARGS.get(name, {})

        force = None
        if name in FORCE_SHUNT_CASES:
            force = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

        try:
            (
                gridtype,
                bus_typ,
                s_multi,
                u_start,
                Y_matrix,
                is_connected,

                Branch_f_bus,
                Branch_t_bus,
                Branch_status,
                Branch_tau,
                Branch_shift_deg,

                Branch_y_series_from,
                Branch_y_series_to,
                Branch_y_series_ft,
                Branch_y_shunt_from,
                Branch_y_shunt_to,

                Y_shunt_bus,

                Is_trafo,
                Branch_hv_is_f,
                Branch_n,

                Y_Lines,
                Y_C_Lines,
                U_base,
                S_base,
                vn_kv,
            ) = case_generation_pandapower(
                case_fn=fn,
                case_kwargs=case_kwargs,
                ybus_mode="ppcY",
                seed=0,
                trafo_pfe_kw=50.0,
                trafo_i0_percent=2.0,
                force_branch_shunt_pu=force,
            )

            N = len(bus_typ)
            nl = len(Branch_f_bus)

            Y_rec = reconstruct_Y_pandapower_branchrows_direct_SI(
                N,
                Branch_f_bus,
                Branch_t_bus,
                Branch_status,
                Branch_tau,
                Branch_shift_deg,
                Branch_y_series_from,
                Branch_y_series_to,
                Branch_y_series_ft,
                Branch_y_shunt_from,
                Branch_y_shunt_to,
                Y_shunt_bus,
            )

            diff = np.max(np.abs(Y_rec - Y_matrix))
            n_tr = int(np.sum(Is_trafo))
            n_on = int(np.sum(Branch_status))

            print(
                f"{name:16s} | "
                f"gridtype={gridtype:30s} | "
                f"N={N:5d} nl={nl:6d} on={n_on:6d} trafo_like={n_tr:6d} | "
                f"forced_shunt={'yes' if force is not None else 'no ':3s} | "
                f"max|diff|={diff:.12g}"
            )

        except Exception as e:
            print(f"{name:16s} | FAILED | {repr(e)}")

    print("=== done ===")