import math
import time
import numpy as np
from collections import deque
from matplotlib.collections import LineCollection

# -------------------------------
# Helpers: geometry / connectivity
# -------------------------------

def plot_random_connections(bus_number, lines_connected):
    try:
        import matplotlib
        import os
        # headless backend just in case someone runs on a node without display
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("plot_random_connections() requires matplotlib; set pic=False or install matplotlib.") from e

    """Draw buses on a circle and connect those pairs flagged by lines_connected (0/1)."""
    angles = np.linspace(0, 2 * np.pi, bus_number, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='blue')

    for i in range(bus_number):
        ax.text(x[i], y[i], f'Bus {i + 1}', ha='right', va='bottom')

    iu, ju = np.triu_indices(bus_number, 1)
    connected_mask = (lines_connected == 1)
    ii = iu[connected_mask]
    jj = ju[connected_mask]
    if ii.size:
        segments = np.stack([np.column_stack([x[ii], y[ii]]),
                             np.column_stack([x[jj], y[jj]])], axis=1)
        lc = LineCollection(segments, linewidths=1.0, colors='gray')
        ax.add_collection(lc)

        mid_x = (x[ii] + x[jj]) / 2.0
        mid_y = (y[ii] + y[jj]) / 2.0
        for mx, my, a, b in zip(mid_x, mid_y, ii, jj):
            ax.text(mx, my, f'L {a + 1}-{b + 1}', color='red')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Zufällige Verbindungen zwischen {bus_number} Bussen')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def is_bus_one_connected_to_all_others(bus_number, lines_connected):
    """BFS reachability from Bus 1."""
    iu, ju = np.triu_indices(bus_number, 1)
    sel = (lines_connected == 1)
    adj = [[] for _ in range(bus_number)]
    for a, b in zip(iu[sel], ju[sel]):
        adj[a].append(b); adj[b].append(a)

    visited = [False] * bus_number
    q = deque([0])
    while q:
        node = q.popleft()
        if visited[node]:
            continue
        visited[node] = True
        for nb in adj[node]:
            if not visited[nb]:
                q.append(nb)
    return all(visited)


def create_adjacency_matrix(bus_number, lines_connected):
    """Symmetric 0/1 adjacency from upper-triangle encoding."""
    adj = np.zeros((bus_number, bus_number), dtype=int)
    iu, ju = np.triu_indices(bus_number, 1)
    adj[iu, ju] = lines_connected
    adj[ju, iu] = lines_connected
    return adj


# -------------------------------
# Multi-voltage + transformer logic
# -------------------------------

def _choose_voltage_levels(gridtype: str, Bus_number: int, fixed: bool):
    """
    Return per-bus nominal voltages vn_kv (in kV) and a set of allowed levels,
    loosely inspired by IEEE cases.
    """
    if fixed:
        np.random.seed(Bus_number * 17)

    # You can tune these to match IEEE14/IEEE30 more closely
    if gridtype == "LVN":
        allowed = np.array([0.4], dtype=float)  # single-level LV
    elif gridtype == "MVN":
        allowed = np.array([10.0, 20.0, 0.4], dtype=float)
    else:  # HVN and hybrids: include a few levels (IEEE14-ish)
        allowed = np.array([135.0, 14.0, 0.208], dtype=float)

    # Choose 1..3 distinct levels to appear in this sample, then assign per bus
    n_levels = min(len(allowed), max(1, np.random.randint(1, 4)))
    lvls = np.random.choice(allowed, size=n_levels, replace=False)
    vn_kv = np.random.choice(lvls, size=Bus_number, replace=True)
    return vn_kv  # shape (N,), kV per bus


def _sample_line_impedances(num_connections, min_real, max_real, min_imag, max_imag,
                            min_length, max_length, fixed, debugging):
    """Return Z_Lines (ohm) and per-connection line length l_r (km)."""
    if debugging:
        Z_Lines = np.array(
            [0.02 + 0.06j, 0.08 + 0.24j, 0, 0, 0.06 + 0.18j,
             0.06 + 0.18j, 0.04 + 0.12j, 0.01 + 0.03j, 0,
             0.08 + 0.24j], dtype=complex
        )
        l_r = np.ones_like(Z_Lines.real)  # unused in debugging shunt path
    else:
        if fixed:
            np.random.seed(num_connections)
        r = np.random.uniform(min_real, max_real, num_connections)     # ohm/km
        x = np.random.uniform(min_imag, max_imag, num_connections)     # ohm/km
        l_r = np.random.uniform(min_length, max_length, num_connections)  # km
        Z_Lines = r * l_r + 1j * (x * l_r)  # ohm
    return Z_Lines.astype(complex), l_r


def _sample_line_shunt(num_connections, gridtype, l_r, fixed, debugging):
    """Return per-end shunt susceptance Y_C_Lines (S) for π-model lines."""
    if debugging:
        # Debug path used your older Z_Base. Keep SI: convert approx to S by picking omega*C.
        # We'll just give reasonable small susceptances in S:
        Y_C_Lines = np.array([0.06, 0.05, 0, 0, 0.04, 0.04, 0.03, 0.02, 0, 0.05], dtype=float) * 1e-3
        return Y_C_Lines

    if fixed:
        np.random.seed(num_connections)

    if gridtype == "LVN":
        Cpkm_min, Cpkm_max = 20e-9, 60e-9  # F/km
    elif gridtype == "MVN":
        Cpkm_min, Cpkm_max = 8e-9, 14e-9   # F/km
    else:
        Cpkm_min, Cpkm_max = 8e-9, 10e-9   # F/km

    Cpkm = np.random.uniform(Cpkm_min, Cpkm_max, size=num_connections)  # F/km
    f = 50.0
    omega = 2 * np.pi * f
    B_total_S = omega * Cpkm * l_r     # S  (total per line)
    Y_C_Lines = 0.5 * B_total_S        # S per-end
    return Y_C_Lines


def _sample_transformers(num_connections, connected, is_trafo, vn_kv, fixed):
    """
    For connections flagged as transformers, sample:
      - series leakage reactance via vk_percent & S_n (gives y_series in S)
      - total magnetizing b_total (small)
      - off-nominal tap tau ~ U[0.93, 1.07]
      - phase shift deg ~ 0 (can widen later)
    We place the tap on the HV side by construction.
    Returns arrays aligned with upper-triangle order; zeros elsewhere.
    """
    if fixed:
        np.random.seed(num_connections * 37)

    iu, ju = np.triu_indices(len(vn_kv), 1)
    hv_side = vn_kv[iu] >= vn_kv[ju]  # True if i is HV, else j is HV

    # Defaults (zeros for non-trafo edges)
    y_series = np.zeros(num_connections, dtype=complex)  # S
    b_total  = np.zeros(num_connections, dtype=float)    # S (total magnetizing)
    tau      = np.ones(num_connections, dtype=float)     # off-nominal magnitude
    shift_deg= np.zeros(num_connections, dtype=float)    # angle in degrees
    tap_on_i = np.zeros(num_connections, dtype=bool)     # True if tap on i side (HV side)

    # Sample only where is_trafo == 1
    mask = (connected == 1) & (is_trafo == 1)
    if not np.any(mask):
        return y_series, b_total, tau, shift_deg, tap_on_i

    # Short-circuit voltage percent vk% ~ U[6, 12]; nameplate power S_n pick near network scale
    vk_percent = np.random.uniform(6.0, 12.0, size=num_connections)  # [%]
    S_n = np.random.uniform(30e6, 120e6, size=num_connections)       # 30..120 MVA, in VA

    # Base impedance seen from HV side: Zb = V_hv^2 / S_n (V in Volts)
    # Choose the HV side voltage in Volts:
    V_hv_V = (np.maximum(vn_kv[iu], vn_kv[ju]) * 1e3)  # kV -> V
    Zb = (V_hv_V ** 2) / S_n                           # ohm
    X_eq = (vk_percent / 100.0) * Zb                   # ohm
    y_series_all = 1.0 / (1j * X_eq)                   # S (purely reactive for simplicity)

    # Magnetizing susceptance: tiny fraction of series, total per branch
    b_total_all = 0.02 * np.abs(y_series_all)          # 2% of |y| as a crude total magnetizing

    # Off-nominal tap & phase-shift
    tau_all   = np.random.uniform(0.93, 1.07, size=num_connections)   # IEEE-like small deviation
    shift_all = np.zeros(num_connections, dtype=float)                # keep 0 deg unless you want PS

    # Fill where trafo:
    y_series[mask] = y_series_all[mask]
    b_total[mask]  = b_total_all[mask]
    tau[mask]      = tau_all[mask]
    shift_deg[mask]= shift_all[mask]
    tap_on_i[mask] = hv_side[mask]  # put the tap on whichever end is HV
    return y_series, b_total, tau, shift_deg, tap_on_i


def _build_Y_with_lines_and_transformers(N, connected, is_trafo, Z_Lines, Y_C_Lines,
                                         vn_kv, y_tr, b_tr_total, tau, shift_deg, tap_on_i):
    """
    Build the global Y (S) combining:
      * Lines (π model, per-end shunt) for same-voltage pairs
      * Transformers with series y_tr, magnetizing b_tr_total, and complex tap
        t = tau * exp(j theta) * (V_hv / V_lv). Tap located on HV side.
    The 'connected' and 'is_trafo' are 0/1 vectors in upper-triangle order (len = N*(N-1)/2).
    """
    Y = np.zeros((N, N), dtype=complex)
    iu, ju = np.triu_indices(N, 1)
    mask = (connected == 1)
    if not np.any(mask):
        return Y  # no branches

    # For vectorized application
    pairs_i = iu[mask]; pairs_j = ju[mask]

    # Lines (same level): stamp standard π-model
    line_mask = mask & (is_trafo == 0)
    if np.any(line_mask):
        i = iu[line_mask]; j = ju[line_mask]
        y_series = np.zeros_like(Z_Lines, dtype=complex)
        nz = (Z_Lines != 0)
        y_series[nz] = 1.0 / Z_Lines[nz]
        y = y_series[line_mask]                  # S
        b_end = Y_C_Lines[line_mask]             # S, per end
        # Diagonals
        Y[i, i] += y + 1j * b_end
        Y[j, j] += y + 1j * b_end
        # Mutual
        Y[i, j] += -y
        Y[j, i] += -y

    # Transformers (different level): off-nominal tap only (nominal ratio already in y_tr)
    trafo_mask = mask & (is_trafo == 1)
    if np.any(trafo_mask):
        i = iu[trafo_mask];
        j = ju[trafo_mask]

        Vi = vn_kv[i] * 1e3  # V
        Vj = vn_kv[j] * 1e3  # V
        hv_is_i = tap_on_i[trafo_mask]

        # We still need to know which node is HV vs LV,
        # but we DO NOT include Vhv/Vlv in t any more.
        ihv = np.where(hv_is_i, i, j)
        ilv = np.where(hv_is_i, j, i)

        y = y_tr[trafo_mask]  # series admittance in S (referred to HV side)
        btot = b_tr_total[trafo_mask]  # total magnetizing susceptance in S

        theta = np.deg2rad(shift_deg[trafo_mask])
        t = tau[trafo_mask] * np.exp(1j * theta)  # off-nominal tap only

        # Diagonals
        Y[ihv, ihv] += (y + 1j * (btot / 2.0)) / (np.abs(t) ** 2)
        Y[ilv, ilv] += (y + 1j * (btot / 2.0))

        # Mutuals
        Y[ihv, ilv] += -(y / np.conj(t))
        Y[ilv, ihv] += -(y / t)
    return Y


# -------------------------------
# Your original utility functions
# -------------------------------

def insert_values_in_matrix(matrix, connections, values):
    """Return symmetric matrix where entries with adjacency==1 take 'values' (upper-tri order)."""
    n = matrix.shape[0]
    out = np.zeros((n, n), dtype=float)
    iu, ju = np.triu_indices(n, 1)
    mask = (matrix[iu, ju] == 1)
    if np.any(mask):
        sel_vals = values[mask]
        out[iu[mask], ju[mask]] = sel_vals
        out[ju[mask], iu[mask]] = sel_vals
    return out


def insert_values_in_matrix_komplex(matrix, connections, values):
    """Complex-valued variant."""
    n = matrix.shape[0]
    out = np.zeros((n, n), dtype=complex)
    iu, ju = np.triu_indices(n, 1)
    mask = (matrix[iu, ju] == 1)
    if np.any(mask):
        sel_vals = values[mask]
        out[iu[mask], ju[mask]] = sel_vals
        out[ju[mask], iu[mask]] = sel_vals
    return out


def create_bus_typ(busnummer, fixed):
    """Slack=1, others randomly PV(2)/PQ(3)."""
    bus_typ = np.zeros(busnummer, dtype=int)
    bus_typ[0] = 1
    if busnummer > 1:
        for i in range(1, busnummer):
            if fixed:
                np.random.seed(busnummer * i)
            bus_typ[i] = np.random.choice([2, 3])
    return bus_typ


def generate_PQ(bus_typ, gridtype):
    """Generate P,Q per bus according to types and gridtype (in MW/Mvar, will convert to W/var)."""
    num_buses = len(bus_typ)
    P = np.zeros(num_buses)
    Q = np.zeros(num_buses)
    np.random.seed(time.time_ns() % (2 ** 32))

    if gridtype == "LVN":
        P_min, P_max = -50 * 0.0005, 50 * 0.0005
        Q_min, Q_max = -25 * 0.0005, 25 * 0.0005
    elif gridtype == "MVN":
        P_min, P_max = -5, 5
        Q_min, Q_max = -2, 2
    elif gridtype == "MVN20kv":
        P_min, P_max = -25, 25
        Q_min, Q_max = -12.5, 12.5
    else:  # HVN
        P_min, P_max = -300, 300
        Q_min, Q_max = -150, 150

    for i, t in enumerate(bus_typ):
        if t == 1:
            P[i] = 0; Q[i] = 0
        elif t == 2:
            P[i] = np.round(np.random.uniform(P_min, P_max, 1)).astype(int)
            Q[i] = 0
        elif t == 3:
            P[i] = np.round(np.random.uniform(P_min, P_max, 1)).astype(int)
            Q[i] = np.round(np.random.uniform(Q_min, Q_max, 1)).astype(int)
    return P, Q


def generate_u_start_for_buses(bustype, vn_kv):
    """Initial voltage guess per bus (complex), near each bus's nominal vn_kv."""
    u_start = np.zeros_like(bustype, dtype=complex)
    np.random.seed(time.time_ns() % (2 ** 32))
    for i, bus_typ in enumerate(bustype):
        Vnom = vn_kv[i] * 1e3  # V
        if bus_typ in (1, 2):
            u_start[i] = np.round(np.random.uniform(0.95, 1.05), 3) * Vnom + 0j
        elif bus_typ == 3:
            u_start[i] = 1.00 * Vnom + 0j
        else:
            raise ValueError(f"Ungültiger Bus-Typ für Bus {i + 1}. (1,2,3)")
    return u_start


# -------------------------------
# Main case generator (updated)
# -------------------------------

def case_generation(gridtype, Bus_number, fixed, debugging, pic):
    """
    Returns:
      gridtype, bus_typ, s_multi (W+jVar), u_start (V), Y_matrix (S), is_connected,
      Y_Lines (S) [upper-tri order], Y_C_Lines (S, per-end) [upper-tri], Lines_connected (0/1),
      U_base (now per-bus nominal, for compatibility we return mean), int(S_base),
      vn_kv (kV per bus), Is_trafo (0/1 upper-tri), Trafo_tau, Trafo_shift_deg, Trafo_y_series (S), Trafo_b_total (S)
    """
    if debugging:
        Bus_number = 5

    num_connections = math.comb(Bus_number, 2)

    # Multi-voltage levels per bus
    vn_kv = _choose_voltage_levels(gridtype, Bus_number, fixed)  # kV per bus

    # Choose global apparent power scale (just a number for your metadata)
    if gridtype == "LVN":
        S_base = 1e6
    elif gridtype == "MVN":
        S_base = 10e6
    else:
        S_base = 100e6

    # Impedance sampling ranges by gridtype (ohm/km)
    if gridtype == "LVN":
        RX = np.random.uniform(2, 10)
        min_real, max_real = 0.05, 0.5
        min_imag, max_imag = min_real / RX, max_real / RX
        min_length, max_length = 1, 5
    elif gridtype == "MVN":
        min_real, max_real = 0.5, 0.6
        min_imag, max_imag = 0.3, 0.35
        min_length, max_length = 1, 20
    else:  # HVN
        min_real, max_real = 0.15, 0.2
        min_imag, max_imag = 0.35, 0.45
        min_length, max_length = 1, 50

    # Per-connection Z and shunts for lines (upper-tri order)
    Z_Lines, l_r = _sample_line_impedances(num_connections, min_real, max_real,
                                           min_imag, max_imag, min_length, max_length,
                                           fixed, debugging)
    Y_C_Lines = _sample_line_shunt(num_connections, gridtype, l_r, fixed, debugging)

    # Connectivity pattern
    np.random.seed(time.time_ns() % (2 ** 32))
    if debugging:
        Lines_connected = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 1], dtype=int)
    else:
        Lines_connected = np.random.randint(2, size=num_connections)

    is_connected = is_bus_one_connected_to_all_others(Bus_number, Lines_connected)
    Conection_matrix = create_adjacency_matrix(Bus_number, Lines_connected)

    # Mark which connected pairs are transformers (different voltage levels)
    iu, ju = np.triu_indices(Bus_number, 1)
    Is_trafo = np.zeros(num_connections, dtype=int)
    diff_level = (np.abs(vn_kv[iu] - vn_kv[ju]) > 1e-6)
    # Only connections that exist AND cross levels become trafos:
    Is_trafo[(Lines_connected == 1) & diff_level] = 1

    # Ensure at least one trafo exists if we have multiple levels
    if np.any(diff_level) and not np.any(Is_trafo == 1):
        # force one random cross-level pair to be connected & trafo
        cand = np.where(diff_level)[0]
        idx = int(np.random.choice(cand))
        Lines_connected[idx] = 1
        Is_trafo[idx] = 1
        Conection_matrix = create_adjacency_matrix(Bus_number, Lines_connected)
        is_connected = is_bus_one_connected_to_all_others(Bus_number, Lines_connected)

    # Sample transformer params for trafo edges
    (Trafo_y_series, Trafo_b_total, Trafo_tau,
     Trafo_shift_deg, Trafo_tap_on_i) = _sample_transformers(
        num_connections, Lines_connected, Is_trafo, vn_kv, fixed
    )

    # Build Y (S) from both lines and trafos
    Y_matrix = _build_Y_with_lines_and_transformers(
        Bus_number, Lines_connected, Is_trafo,
        Z_Lines, Y_C_Lines,
        vn_kv, Trafo_y_series, Trafo_b_total, Trafo_tau, Trafo_shift_deg, Trafo_tap_on_i
    )

    # Bus types & injections
    if debugging:
        bus_typ = np.array([1, 2, 3, 3, 3])
        P = np.array([0, -50, -65, -35, -45])
        Q = np.array([0, 0, -10, -10, -15])
    else:
        bus_typ = create_bus_typ(Bus_number, fixed)
        P, Q = generate_PQ(bus_typ, gridtype)

    # Convert MW/Mvar to W/var
    P, Q = P * 1e6, Q * 1e6
    s_multi = P + 1j * Q

    # Initial voltages near each bus nominal
    u_start = generate_u_start_for_buses(bus_typ, vn_kv)

    if pic:
        plot_random_connections(Bus_number, Lines_connected)

    # For backward compatibility, legacy U_base = mean of vn_kv*kV->V
    U_base_legacy = float(np.mean(vn_kv) * 1e3)

    return (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
            # legacy arrays (upper-tri)
            (1.0 / np.where(Z_Lines != 0, Z_Lines, np.inf)).astype(complex),  # Y_Lines (S)
            Y_C_Lines, Lines_connected,
            U_base_legacy, int(S_base),
            # new metadata for multi-level + trafos
            vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total)

# ============================
# IEEE-14 PRESET (thin wrapper)
# ============================

def case_generation_ieee14(fixed=True, pic=False):
    """
    Canonical IEEE-14 topology + branch data + taps.
    - Topology and taps are fixed to the MATPOWER IEEE-14 case.
    - Phase shift = 0 for all tapped branches.
    - P/Q are synthesized with your existing generator (HVN ranges) so you can
      create many scenarios on the same canonical network. (Ask if you want the
      canonical P/Q table instead.)
    - Units:
        We stamp Y in per-unit first (classic), then convert to SI using one
        global base (S_base = 100 MVA, U_base = 100 kV). That keeps everything
        consistent for your Newton solver (V in Volts, S in Watts, Y in Siemens).
    Returns:
      (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
       Y_Lines, Y_C_Lines, Lines_connected, U_base, int(S_base),
       vn_kv, Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total)
    """
    import numpy as np

    gridtype = "IEEE14"
    N = 14

    # -------- Bus types (IEEE-14): 1=slack, 2=PV, 3=PQ (0-based index)
    # Slack at 1; PV at 2,3,6,8; the rest PQ
    bus_typ = np.array([1, 2, 2, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3], dtype=int)

    # -------- Branch list (MATPOWER case14 style, 1-based buses)
    # (i, j, r, x, b_total, tap, shift_deg)
    branches = [
        (1,  2, 0.01938, 0.05917, 0.02640, 0.0,   0.0),
        (1,  5, 0.05403, 0.22304, 0.02460, 0.0,   0.0),
        (2,  3, 0.04699, 0.19797, 0.02190, 0.0,   0.0),
        (2,  4, 0.05811, 0.17632, 0.01870, 0.0,   0.0),
        (2,  5, 0.05695, 0.17388, 0.01700, 0.0,   0.0),
        (3,  4, 0.06701, 0.17103, 0.01730, 0.0,   0.0),
        (4,  5, 0.01335, 0.04211, 0.00640, 0.0,   0.0),
        (4,  7, 0.00000, 0.20912, 0.00000, 0.978, 0.0),  # tap
        (4,  9, 0.00000, 0.55618, 0.00000, 0.969, 0.0),  # tap
        (5,  6, 0.00000, 0.25202, 0.00000, 0.932, 0.0),  # tap
        (6, 11, 0.09498, 0.19890, 0.00000, 0.0,   0.0),
        (6, 12, 0.12291, 0.25581, 0.00000, 0.0,   0.0),
        (6, 13, 0.06615, 0.13027, 0.00000, 0.0,   0.0),
        (7,  8, 0.00000, 0.17615, 0.00000, 0.0,   0.0),
        (7,  9, 0.00000, 0.11001, 0.00000, 0.0,   0.0),
        (9, 10, 0.03181, 0.08450, 0.00000, 0.0,   0.0),
        (9, 14, 0.12711, 0.27038, 0.00000, 0.0,   0.0),
        (10,11, 0.08205, 0.19207, 0.00000, 0.0,   0.0),
        (12,13, 0.22092, 0.19988, 0.00000, 0.0,   0.0),
        (13,14, 0.17093, 0.34802, 0.00000, 0.0,   0.0),
    ]

    # Upper-tri index helpers
    iu, ju = np.triu_indices(N, 1) # all upper-triangle pairs (i, j) with i < j for an N×N matrix.
    num_connections = len(iu) # number of potential branches (all pairs).

    # -------- Build Lines_connected (upper-tri 0/1), plus arrays for series/shunt/taps
    Lines_connected = np.zeros(num_connections, dtype=int)
    Is_trafo        = np.zeros(num_connections, dtype=int)
    Y_series_pu     = np.zeros(num_connections, dtype=complex)  # for line edges
    Yc_per_end_pu   = np.zeros(num_connections, dtype=float)    # per-end shunt (pu)
    Trafo_tau       = np.ones (num_connections, dtype=float)
    Trafo_shift_deg = np.zeros(num_connections, dtype=float)
    Trafo_y_series_pu = np.zeros(num_connections, dtype=complex)
    Trafo_b_total_pu  = np.zeros(num_connections, dtype=float)

    # Helper: map (i,j) -> index in upper-tri vector
    def _idx(i0, j0):  # 0-based buses
        if i0 > j0:
            i0, j0 = j0, i0
        # position of pair (i0,j0) in the flattened upper-tri
        return int((2*N - i0 - 1) * i0 // 2 + (j0 - i0 - 1))

    # Fill arrays from branch table (per-unit)
    for (i1, j1, r, x, b_tot, tap, shift) in branches:
        i0, j0 = i1 - 1, j1 - 1
        k = _idx(i0, j0)
        Lines_connected[k] = 1

        # Series admittance (pu)
        z = complex(r, x)
        y = 0j if abs(z) < 1e-15 else 1.0 / z

        if tap and abs(tap) > 1e-12:
            # Transformer branch (off-nominal magnitude tap, zero phase-shift)
            Is_trafo[k]        = 1
            Trafo_tau[k]       = float(tap)
            Trafo_shift_deg[k] = float(shift)  # 0 here
            Trafo_y_series_pu[k] = y
            Trafo_b_total_pu[k]  = float(b_tot)  # total magnetizing (IEEE-14 taps have 0)
        else:
            # Regular line (π model): store line series & per-end shunt
            Y_series_pu[k]   = y
            Yc_per_end_pu[k] = 0.5 * float(b_tot)  # half to each end

    # -------- Stamp Y in per-unit first (MATPOWER-style)
    Y_pu = np.zeros((N, N), dtype=complex)

    # Lines (no tap)
    mask_line = (Lines_connected == 1) & (Is_trafo == 0)
    if np.any(mask_line):
        ii = iu[mask_line]; jj = ju[mask_line]
        y  = Y_series_pu[mask_line]
        b  = Yc_per_end_pu[mask_line]
        # diagonals
        Y_pu[ii, ii] += y + 1j * b
        Y_pu[jj, jj] += y + 1j * b
        # off-diagonals
        Y_pu[ii, jj] += -y
        Y_pu[jj, ii] += -y

    # Transformers (off-nominal magnitude, zero phase-shift)
    mask_xf = (Lines_connected == 1) & (Is_trafo == 1)
    if np.any(mask_xf):
        ii = iu[mask_xf]; jj = ju[mask_xf]
        y  = Trafo_y_series_pu[mask_xf]
        bT = Trafo_b_total_pu[mask_xf]
        t  = Trafo_tau[mask_xf] * np.exp(1j * np.deg2rad(Trafo_shift_deg[mask_xf]))  # θ=0 here
        # Diagonals
        Y_pu[ii, ii] += (y + 1j * (bT / 2.0)) / (np.abs(t) ** 2)
        Y_pu[jj, jj] +=  (y + 1j * (bT / 2.0))
        # Mutuals
        Y_pu[ii, jj] += -(y / np.conj(t))
        Y_pu[jj, ii] += -(y / t)

    # -------- Convert per-unit to SI with one global base
    S_base = 100e6     # 100 MVA
    U_base = 100e3     # 100 kV  (single global base -> consistent scaling)
    Y_SI = Y_pu * (S_base / (U_base ** 2))  # S
    Y_matrix = Y_SI

    # -------- Initial voltages and injections (synthetic, using your HVN ranges)
    from typing import Tuple
    def _generate_PQ(bus_typ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # reuse your HVN ranges
        P_min, P_max = -300, 300   # MW
        Q_min, Q_max = -150, 150   # Mvar
        P = np.zeros(N)
        Q = np.zeros(N)
        rng = np.random.default_rng(int(time.time()) & 0xffffffff)
        for i, t in enumerate(bus_typ):
            if t == 1:   # slack
                P[i] = 0; Q[i] = 0
            elif t == 2: # PV: specify P, Q=0
                P[i] = int(rng.uniform(P_min, P_max))
                Q[i] = 0
            else:        # PQ
                P[i] = int(rng.uniform(P_min, P_max))
                Q[i] = int(rng.uniform(Q_min, Q_max))
        # MW/Mvar -> W/var
        return P * 1e6, Q * 1e6

    P_W, Q_var = _generate_PQ(bus_typ)
    s_multi = P_W + 1j * Q_var

    # Initial voltages near 1.0 pu -> convert to SI with U_base
    u_start = np.ones(N, dtype=complex) * U_base
    # jitter PV/PQ a bit (leave slack as 1.0∠0)
    for k in range(1, N):
        mag = float(np.clip(1.0 + np.random.uniform(-0.05, 0.05), 0.9, 1.1))
        ang = float(np.random.uniform(-3.0, 3.0)) * np.pi/180.0
        u_start[k] = (mag * U_base) * np.exp(1j * ang)

    # Sum of per-end shunts at buses, in SI (for legacy column Y_C_Bus you used earlier)
    # We follow your Parquet columns: Y_Lines (series) / Y_C_Lines (per-end) in SI, upper-tri order.
    Y_Lines_SI   = Y_series_pu * (S_base / (U_base ** 2))
    Y_C_Lines_SI = Yc_per_end_pu * (S_base / (U_base ** 2))
    Trafo_y_series_SI = Trafo_y_series_pu * (S_base / (U_base ** 2))
    Trafo_b_total_SI  = Trafo_b_total_pu  * (S_base / (U_base ** 2))

    # vn_kv metadata: we keep a single level (100 kV) so nominal ratio is NOT double-counted
    vn_kv = np.full(N, U_base / 1e3, dtype=float)

    is_connected = True  # IEEE-14 is connected

    # Legacy U_base/S_base in returns: keep them for compatibility
    U_base_legacy = float(U_base)

    return (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
            Y_Lines_SI, Y_C_Lines_SI, Lines_connected.astype(int),
            U_base_legacy, int(S_base),
            # new metadata (multi-level/trafo)
            vn_kv, Is_trafo.astype(int), Trafo_tau, Trafo_shift_deg,
            Trafo_y_series_SI, Trafo_b_total_SI)



def stamp_Y_from_vectors_with_shunt(
    N: int,
    Lines_connected: np.ndarray,
    Y_Lines: np.ndarray,
    Is_trafo: np.ndarray,
    Trafo_y_series: np.ndarray,
    Y_shunt_bus: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct nodal Y_matrix (Siemens) from edge-level series admittances and
    per-bus shunt admittances.

    All admittances are in SI:
      - Y_Lines[k]        : effective series admittance (S) for line branches
      - Trafo_y_series[k] : effective series admittance (S) for trafo branches
      - Lines_connected[k]: 1 if there is a physical line on that upper-tri pair
      - Is_trafo[k]       : 1 if that upper-tri pair is a transformer branch
      - Y_shunt_bus[i]    : total shunt admittance to ground at bus i (S)

    Assumes upper-tri index ordering consistent with np.triu_indices(N, 1).
    """

    Y = np.zeros((N, N), dtype=np.complex128)
    iu, ju = np.triu_indices(N, 1)

    for k, (i, j) in enumerate(zip(iu, ju)):
        if not (Lines_connected[k] or Is_trafo[k]):
            continue

        # Choose series admittance from the proper vector
        if Is_trafo[k]:
            y = Trafo_y_series[k]
        else:
            y = Y_Lines[k]

        if y == 0:
            continue

        # Symmetric mutual coupling
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    # Add bus shunts on the diagonal
    Y[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    return Y


def case_generation_ieee14_pandapower(
    jitter_load: float = 0.0,          # e.g. 0.10 => multiply each (p_mw, q_mvar) by N(1, 0.10)
    jitter_gen: float = 0.0,           # e.g. 0.10 => multiply each gen p_mw by N(1, 0.10)
    pv_vset_range = None,              # e.g. (0.98, 1.04) => uniform PV |V| setpoints in pu
    rand_u_start: bool = False,        # randomize initial |U|/angle (PV |U| honored)
    angle_jitter_deg: float = 5.0,     # +/- deg jitter for non-slack start angles
    mag_jitter_pq: float = 0.02,       # +/- fraction for PQ start |U| (e.g. 0.02 => ±2%)
    seed = None                        # RNG seed for reproducibility
):
    """
    IEEE-14 preset with per-bus vn_kv from pandapower, plus optional jitter.

    IMPORTANT (Goal B):
      - Y_matrix is built EXCLUSIVELY by _build_Y_with_lines_and_transformers
        using edge vectors in engineering units (ohm / S).
      - We do NOT use pandapower's internal Ybus for Y_matrix.
      - pandapower is only a convenient data source (topology, vn_kv, trafo nameplate, etc).

    Returns:
      gridtype (str), bus_typ (N,), s_multi (N,), u_start (N,), Y_matrix (N,N),
      is_connected (bool), Y_Lines (C,), Y_C_Lines (C,), Lines_connected (C,),
      U_base (float), S_base (float),
      vn_kv (N,), Is_trafo (C,), Trafo_tau (C,), Trafo_shift_deg (C,),
      Trafo_y_series (C,), Trafo_b_total (C,)
    """
    import numpy as np
    import pandas as pd
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.powerflow import LoadflowNotConverged

    rng = np.random.default_rng(seed)

    # --- Build the base IEEE-14 net
    net = pn.case14()

    # ----- JITTER: loads (p_mw, q_mvar) -----
    if jitter_load > 0 and len(net.load):
        s = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"]   = net.load["p_mw"].to_numpy(float)   * s
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s

    # ----- JITTER: generator active powers (p_mw)
    if jitter_gen > 0 and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s

    # ----- JITTER: PV voltage magnitude setpoints (vm_pu)
    if pv_vset_range is not None and len(net.gen):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    # --- We may still call runpp if you want internal converged voltages later,
    #     but for the Y-matrix we no longer use net._ppc["internal"]["Ybus"].
    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9
        )
    except LoadflowNotConverged:
        print("exception LoadflowNotConverged (ignored for Y building)")
        pass

    N = len(net.bus)

    # per-bus nominal voltages (kV -> V)
    vn_kv = net.bus["vn_kv"].to_numpy(dtype=float)
    Vbase = vn_kv * 1e3  # V

    # System base power in VA (just metadata)
    S_base = float(net.sn_mva) * 1e6

    # --- Bus types: 1=slack, 2=PV, 3=PQ
    bus_typ = np.full(N, 3, dtype=int)
    if len(net.ext_grid):
        bus_typ[net.ext_grid["bus"].to_numpy()] = 1
    if len(net.gen):
        bus_typ[net.gen["bus"].to_numpy()] = 2

    # --- Specified complex injections S (W + j var), SI
    S = np.zeros(N, dtype=np.complex128)

    # Loads: negative injections
    if len(net.load):
        b = net.load.bus.to_numpy(int)
        P = net.load.p_mw.to_numpy(float) * 1e6
        Q = net.load.q_mvar.to_numpy(float) * 1e6
        S[b] -= (P + 1j * Q)

    # Static gens (positive)
    if len(net.sgen):
        b = net.sgen.bus.to_numpy(int)
        P = net.sgen.p_mw.to_numpy(float) * 1e6
        Q = net.sgen.q_mvar.to_numpy(float) * 1e6
        S[b] += (P + 1j * Q)

    # Synchronous gens at PV: add active power; Q solved by NR
    if len(net.gen):
        for _, r in net.gen.iterrows():
            S[int(r.bus)] += float(r.p_mw) * 1e6

    s_multi = S

    # --- Start voltages (engineering units, V)
    u_start = Vbase.astype(np.complex128)
    if rand_u_start:
        mag = Vbase.copy()

        if len(net.gen):
            vm_by_bus = {}
            for _, r in net.gen.iterrows():
                vm_val = float(r.vm_pu) if pd.notna(r.vm_pu) else 1.0
                vm_by_bus[int(r.bus)] = vm_val
            for b, vm in vm_by_bus.items():
                mag[b] = vm * Vbase[b]

        pq_mask = (bus_typ == 3)
        if pq_mask.any() and mag_jitter_pq > 0:
            mag[pq_mask] *= rng.uniform(1.0 - mag_jitter_pq,
                                        1.0 + mag_jitter_pq,
                                        size=pq_mask.sum())

        ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * (np.pi / 180.0)
        if len(net.ext_grid):
            ang[net.ext_grid.bus.to_numpy(int)] = 0.0

        u_start = mag * np.exp(1j * ang)

    # --- Upper-tri indexing helpers
    iu, ju = np.triu_indices(N, 1)
    C = len(iu)

    # --- LINE edge features (store Z_Lines in ohm, shunt in S)
    Lines_connected = np.zeros(C, dtype=np.int8)
    Z_Lines = np.zeros(C, dtype=np.complex128)   # total series impedance (ohm)
    Y_Lines = np.zeros(C, dtype=np.complex128)   # series admittance (S) - for output only
    Y_C_Lines = np.zeros(C, dtype=np.float64)    # per-end shunt susceptance (S)

    pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}

    f = float(getattr(net, "f_hz", 50.0))
    two_pi_f = 2.0 * np.pi * f

    if len(net.line):
        for _, r in net.line.iterrows():
            a, b = int(r.from_bus), int(r.to_bus)
            i, j = (a, b) if a < b else (b, a)
            idx = pair_to_idx.get((i, j), None)
            if idx is None:
                continue
            Lines_connected[idx] = 1

            # Series Z_total = (r + jx) * length_km  -> [ohm]
            try:
                rpkm = float(r.r_ohm_per_km)
                xpkm = float(r.x_ohm_per_km)
                Lkm  = float(r.length_km)
                Z = complex(rpkm, xpkm) * Lkm
                Z_Lines[idx] = Z
                if Z != 0:
                    Y_Lines[idx] = 1.0 / Z  # S, just for output
            except Exception:
                pass

            # Shunt total B_total = ω * C_total ; per-end = 0.5 * B_total
            try:
                c_nf_per_km = float(r.c_nf_per_km)
                Lkm = float(r.length_km)
                C_total = c_nf_per_km * 1e-9 * Lkm  # Farads
                B_total = two_pi_f * C_total        # Siemens
                Y_C_Lines[idx] = 0.5 * B_total
            except Exception:
                pass

    # ---------- Transformer vectors (engineering units, NaN-safe) ----------
    def _upper_index_map(n_buses: int):
        iu_, ju_ = np.triu_indices(n_buses, 1)
        C_ = len(iu_)
        p2i = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu_, ju_))}
        return iu_, ju_, C_, p2i

    def compute_trafo_vectors_from_net(net_):
        """
        Build transformer vectors aligned to the upper-tri (iu,ju):
          - Is_trafo:       0/1
          - Trafo_tau:      off-nominal magnitude (tap)
          - Trafo_shift_deg: phase shift in degrees
          - Trafo_y_series:  series admittance (Siemens), HV side base
          - Trafo_b_total:   total magnetizing susceptance (Siemens)
          - Trafo_tap_on_i:  bool, True if tap/HV side is the 'i' node of (i,j)
        """
        N_ = len(net_.bus)
        iu_, ju_, C_, p2i_ = _upper_index_map(N_)

        Is_trafo_        = np.zeros(C_, dtype=np.int8)
        Trafo_tau_       = np.ones(C_, dtype=np.float64)
        Trafo_shift_deg_ = np.zeros(C_, dtype=np.float64)
        Trafo_y_series_  = np.zeros(C_, dtype=np.complex128)
        Trafo_b_total_   = np.zeros(C_, dtype=np.float64)
        Trafo_tap_on_i_  = np.zeros(C_, dtype=bool)

        if len(net_.trafo) == 0:
            return Is_trafo_, Trafo_tau_, Trafo_shift_deg_, Trafo_y_series_, Trafo_b_total_, Trafo_tap_on_i_

        for _, r in net_.trafo.iterrows():
            hv = int(r.hv_bus)
            lv = int(r.lv_bus)
            # map to upper-tri index (i_ < j_)
            if hv < lv:
                i_, j_ = hv, lv
                hv_is_i = True
            else:
                i_, j_ = lv, hv
                hv_is_i = False
            k_ = p2i_.get((i_, j_))
            if k_ is None:
                continue

            Is_trafo_[k_] = 1
            Trafo_tap_on_i_[k_] = hv_is_i  # True if 'i_' is HV/tap side

            # ---- tap magnitude & phase (NaN-safe) ----
            tap_pos_raw   = r.tap_pos          if "tap_pos"          in r.index else np.nan
            tap_neut_raw  = r.tap_neutral      if "tap_neutral"      in r.index else 0.0
            step_pct_raw  = r.tap_step_percent if "tap_step_percent" in r.index else 0.0
            step_deg_raw  = r.tap_step_degree  if "tap_step_degree"  in r.index else 0.0
            shift_raw     = r.shift_degree     if "shift_degree"     in r.index else 0.0

            tap_pos  = float(tap_pos_raw)  if pd.notna(tap_pos_raw)  else np.nan
            tap_neut = float(tap_neut_raw) if pd.notna(tap_neut_raw) else 0.0
            step_pct = float(step_pct_raw) if pd.notna(step_pct_raw) else 0.0
            step_deg = float(step_deg_raw) if pd.notna(step_deg_raw) else 0.0
            fixed_deg = float(shift_raw)   if pd.notna(shift_raw)    else 0.0

            if np.isfinite(tap_pos) and step_pct != 0.0:
                n = tap_pos - tap_neut
                Trafo_tau_[k_] = 1.0 + (n * step_pct / 100.0)
                Trafo_shift_deg_[k_] = fixed_deg + n * step_deg
            else:
                Trafo_tau_[k_] = 1.0
                Trafo_shift_deg_[k_] = fixed_deg

            # ---- series admittance on HV side ----
            try:
                sn_mva_raw   = r.sn_mva    if "sn_mva"    in r.index else net_.sn_mva
                vn_hv_kv_raw = r.vn_hv_kv  if "vn_hv_kv"  in r.index else net_.bus.vn_kv.loc[hv]
                vk_pct_raw   = r.vk_percent if "vk_percent" in r.index else np.nan
                vkr_pct_raw  = r.vkr_percent if "vkr_percent" in r.index else np.nan

                sn_mva   = float(sn_mva_raw)
                vn_hv_kv = float(vn_hv_kv_raw)
                vk_pct   = float(vk_pct_raw)  if pd.notna(vk_pct_raw)  else np.nan
                vkr_pct  = float(vkr_pct_raw) if pd.notna(vkr_pct_raw) else np.nan

                if np.isfinite(vk_pct) and np.isfinite(vkr_pct) and vk_pct > 0.0:
                    r_pu = vkr_pct / 100.0
                    x_pu_sq = (vk_pct / 100.0) ** 2 - r_pu ** 2
                    x_pu = np.sqrt(x_pu_sq) if x_pu_sq > 0 else 0.0

                    V_ll = vn_hv_kv * 1e3  # V
                    S_va = sn_mva * 1e6    # VA
                    Z_base = (V_ll ** 2) / S_va
                    Z_series = (r_pu + 1j * x_pu) * Z_base
                    if Z_series != 0:
                        Trafo_y_series_[k_] = 1.0 / Z_series
            except Exception:
                pass

            # ---- magnetizing susceptance (total) from pfe_kw & i0_percent ----
            try:
                pfe_kw_raw = r.pfe_kw     if "pfe_kw"     in r.index else 0.0
                i0_pct_raw = r.i0_percent if "i0_percent" in r.index else 0.0
                sn_mva_raw   = r.sn_mva    if "sn_mva"    in r.index else net_.sn_mva
                vn_hv_kv_raw = r.vn_hv_kv  if "vn_hv_kv"  in r.index else net_.bus.vn_kv.loc[hv]

                pfe_kw  = float(pfe_kw_raw) if pd.notna(pfe_kw_raw) else 0.0
                i0_pct  = float(i0_pct_raw) if pd.notna(i0_pct_raw) else 0.0
                sn_mva  = float(sn_mva_raw)
                vn_hv_kv = float(vn_hv_kv_raw)

                V_ll = vn_hv_kv * 1e3
                S_va = sn_mva * 1e6
                I_base = S_va / (np.sqrt(3.0) * V_ll)
                I0 = (i0_pct / 100.0) * I_base
                I_c = (pfe_kw * 1e3) / (np.sqrt(3.0) * V_ll)
                Im_sq = I0 ** 2 - I_c ** 2
                I_m = np.sqrt(Im_sq) if Im_sq > 0 else 0.0

                B_phase = (np.sqrt(3.0) * I_m) / V_ll  # S
                Trafo_b_total_[k_] = B_phase
            except Exception:
                pass

        return Is_trafo_, Trafo_tau_, Trafo_shift_deg_, Trafo_y_series_, Trafo_b_total_, Trafo_tap_on_i_


    # Compute transformer vectors (SI)
    (Is_trafo,
     Trafo_tau,
     Trafo_shift_deg,
     Trafo_y_series,
     Trafo_b_total,
     Trafo_tap_on_i) = compute_trafo_vectors_from_net(net)

    # there is some branch (line or transformer) here
    Lines_connected = np.where(Is_trafo == 1, 1, Lines_connected).astype(np.int8)

    # --- Build Y_matrix using the SAME stamping model as your generic generator
    Y_matrix = _build_Y_with_lines_and_transformers(
        N,
        Lines_connected,
        Is_trafo,
        Z_Lines,
        Y_C_Lines,
        vn_kv,
        Trafo_y_series,
        Trafo_b_total,
        Trafo_tau,
        Trafo_shift_deg,
        Trafo_tap_on_i
    )

    gridtype = "IEEE14"
    U_base = float("nan")   # not meaningful with multi-voltage scaling
    is_connected = True

    return (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
            Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
            vn_kv.astype(np.float64),
            Is_trafo.astype(np.int8), Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total)

# case generator with getting S with U*Y
def case_generation_ieee14_pandapower(
    jitter_load: float = 0.0,          # e.g. 0.10 => multiply each (p_mw, q_mvar) by N(1, 0.10)
    jitter_gen: float = 0.0,           # e.g. 0.10 => multiply each gen p_mw by N(1, 0.10)
    pv_vset_range = None,              # e.g. (0.98, 1.04) => uniform PV |V| setpoints in pu
    rand_u_start: bool = True,        # randomize initial |U|/angle (PV |U| honored)
    angle_jitter_deg: float = 5.0,     # +/- deg jitter for non-slack start angles
    mag_jitter_pq: float = 0.02,       # +/- fraction for PQ start |U| (e.g. 0.02 => ±2%)
    seed = None,                       # RNG seed for reproducibility
    self_consistent_S: bool = True    # <<< NEW: Experiment A flag
):
    """
    IEEE-14 preset with per-bus vn_kv from pandapower, plus optional jitter.

    IMPORTANT (Goal B):
      - Y_matrix is built EXCLUSIVELY by _build_Y_with_lines_and_transformers
        using edge vectors in engineering units (ohm / S).
      - We do NOT use pandapower's internal Ybus for Y_matrix.
      - pandapower is only a convenient data source (topology, vn_kv, trafo nameplate, etc).

    If self_consistent_S=True (Experiment A):
      - s_multi is overridden and built from Y_matrix and u_start:
          I0 = Y_matrix @ u_start
          S0 = u_start * conj(I0)
        so that (Y_matrix, s_multi, u_start) is a mathematically exact solution
        for your Newton-Raphson formulation.
    """
    import numpy as np
    import pandas as pd
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.powerflow import LoadflowNotConverged

    rng = np.random.default_rng(seed)

    # --- Build the base IEEE-14 net
    net = pn.case14()

    # ----- JITTER: loads (p_mw, q_mvar) -----
    if jitter_load > 0 and len(net.load):
        s = rng.normal(1.0, jitter_load, size=len(net.load))
        net.load["p_mw"]   = net.load["p_mw"].to_numpy(float)   * s
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s

    # ----- JITTER: generator active powers (p_mw)
    if jitter_gen > 0 and len(net.gen):
        s = rng.normal(1.0, jitter_gen, size=len(net.gen))
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s

    # ----- JITTER: PV voltage magnitude setpoints (vm_pu)
    if pv_vset_range is not None and len(net.gen):
        lo, hi = pv_vset_range
        net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))

    # --- Optional runpp (only for net internals, NOT for Y_matrix)
    try:
        pp.runpp(
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9
        )
    except LoadflowNotConverged:
        print("exception LoadflowNotConverged (ignored for Y building)")
        pass

    N = len(net.bus)

    # per-bus nominal voltages (kV -> V)
    vn_kv = net.bus["vn_kv"].to_numpy(dtype=float)
    Vbase = vn_kv * 1e3  # V

    # System base power in VA (just metadata)
    S_base = float(net.sn_mva) * 1e6

    # --- Bus types: 1=slack, 2=PV, 3=PQ
    bus_typ = np.full(N, 3, dtype=int)
    if len(net.ext_grid):
        bus_typ[net.ext_grid["bus"].to_numpy()] = 1
    if len(net.gen):
        bus_typ[net.gen["bus"].to_numpy()] = 2

    # --- Specified complex injections S (W + j var), SI
    S = np.zeros(N, dtype=np.complex128)

    # Loads: negative injections
    if len(net.load):
        b = net.load.bus.to_numpy(int)
        P = net.load.p_mw.to_numpy(float) * 1e6
        Q = net.load.q_mvar.to_numpy(float) * 1e6
        S[b] -= (P + 1j * Q)

    # Static gens (positive)
    if len(net.sgen):
        b = net.sgen.bus.to_numpy(int)
        P = net.sgen.p_mw.to_numpy(float) * 1e6
        Q = net.sgen.q_mvar.to_numpy(float) * 1e6
        S[b] += (P + 1j * Q)

    # Synchronous gens at PV: add active power; Q solved by NR
    if len(net.gen):
        for _, r in net.gen.iterrows():
            S[int(r.bus)] += float(r.p_mw) * 1e6

    # This is the "pandapower-defined" S; we may override it later.
    s_multi = S

    # --- Start voltages (engineering units, V)
    u_start = Vbase.astype(np.complex128)
    if rand_u_start:
        mag = Vbase.copy()

        if len(net.gen):
            vm_by_bus = {}
            for _, r in net.gen.iterrows():
                vm_val = float(r.vm_pu) if pd.notna(r.vm_pu) else 1.0
                vm_by_bus[int(r.bus)] = vm_val
            for b, vm in vm_by_bus.items():
                mag[b] = vm * Vbase[b]

        pq_mask = (bus_typ == 3)
        if pq_mask.any() and mag_jitter_pq > 0:
            mag[pq_mask] *= rng.uniform(1.0 - mag_jitter_pq,
                                        1.0 + mag_jitter_pq,
                                        size=pq_mask.sum())

        ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * (np.pi / 180.0)
        if len(net.ext_grid):
            ang[net.ext_grid.bus.to_numpy(int)] = 0.0

        u_start = mag * np.exp(1j * ang)

    # --- Upper-tri indexing helpers
    iu, ju = np.triu_indices(N, 1)
    C = len(iu)

    # --- LINE edge features (store Z_Lines in ohm, shunt in S)
    Lines_connected = np.zeros(C, dtype=np.int8)
    Z_Lines = np.zeros(C, dtype=np.complex128)   # total series impedance (ohm)
    Y_Lines = np.zeros(C, dtype=np.complex128)   # series admittance (S) - for output only
    Y_C_Lines = np.zeros(C, dtype=np.float64)    # per-end shunt susceptance (S)

    pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}

    f = float(getattr(net, "f_hz", 50.0))
    two_pi_f = 2.0 * np.pi * f

    if len(net.line):
        for _, r in net.line.iterrows():
            a, b = int(r.from_bus), int(r.to_bus)
            i, j = (a, b) if a < b else (b, a)
            idx = pair_to_idx.get((i, j), None)
            if idx is None:
                continue
            Lines_connected[idx] = 1

            # Series Z_total = (r + jx) * length_km  -> [ohm]
            try:
                rpkm = float(r.r_ohm_per_km)
                xpkm = float(r.x_ohm_per_km)
                Lkm  = float(r.length_km)
                Z = complex(rpkm, xpkm) * Lkm
                Z_Lines[idx] = Z
                if Z != 0:
                    Y_Lines[idx] = 1.0 / Z  # S, just for output
            except Exception:
                pass

            # Shunt total B_total = ω * C_total ; per-end = 0.5 * B_total
            try:
                c_nf_per_km = float(r.c_nf_per_km)
                Lkm = float(r.length_km)
                C_total = c_nf_per_km * 1e-9 * Lkm  # Farads
                B_total = two_pi_f * C_total        # Siemens
                Y_C_Lines[idx] = 0.5 * B_total
            except Exception:
                pass

    # ---------- Transformer vectors (engineering units, NaN-safe) ----------
    def _upper_index_map(n_buses: int):
        iu_, ju_ = np.triu_indices(n_buses, 1)
        C_ = len(iu_)
        p2i = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu_, ju_))}
        return iu_, ju_, C_, p2i

    def compute_trafo_vectors_from_net(net_):
        """
        Build transformer vectors aligned to the upper-tri (iu,ju):
          - Is_trafo:       0/1
          - Trafo_tau:      off-nominal magnitude (tap)
          - Trafo_shift_deg: phase shift in degrees
          - Trafo_y_series:  series admittance (Siemens), HV side base
          - Trafo_b_total:   total magnetizing susceptance (Siemens)
          - Trafo_tap_on_i:  bool, True if tap/HV side is the 'i' node of (i,j)
        """
        N_ = len(net_.bus)
        iu_, ju_, C_, p2i_ = _upper_index_map(N_)

        Is_trafo_        = np.zeros(C_, dtype=np.int8)
        Trafo_tau_       = np.ones(C_, dtype=np.float64)
        Trafo_shift_deg_ = np.zeros(C_, dtype=np.float64)
        Trafo_y_series_  = np.zeros(C_, dtype=np.complex128)
        Trafo_b_total_   = np.zeros(C_, dtype=np.float64)
        Trafo_tap_on_i_  = np.zeros(C_, dtype=bool)

        if len(net_.trafo) == 0:
            return Is_trafo_, Trafo_tau_, Trafo_shift_deg_, Trafo_y_series_, Trafo_b_total_, Trafo_tap_on_i_

        for _, r in net_.trafo.iterrows():
            hv = int(r.hv_bus)
            lv = int(r.lv_bus)
            # map to upper-tri index (i_ < j_)
            if hv < lv:
                i_, j_ = hv, lv
                hv_is_i = True
            else:
                i_, j_ = lv, hv
                hv_is_i = False
            k_ = p2i_.get((i_, j_))
            if k_ is None:
                continue

            Is_trafo_[k_] = 1
            Trafo_tap_on_i_[k_] = hv_is_i  # True if 'i_' is HV/tap side

            # ---- tap magnitude & phase (NaN-safe) ----
            tap_pos_raw   = r.tap_pos          if "tap_pos"          in r.index else np.nan
            tap_neut_raw  = r.tap_neutral      if "tap_neutral"      in r.index else 0.0
            step_pct_raw  = r.tap_step_percent if "tap_step_percent" in r.index else 0.0
            step_deg_raw  = r.tap_step_degree  if "tap_step_degree"  in r.index else 0.0
            shift_raw     = r.shift_degree     if "shift_degree"     in r.index else 0.0

            tap_pos  = float(tap_pos_raw)  if pd.notna(tap_pos_raw)  else np.nan
            tap_neut = float(tap_neut_raw) if pd.notna(tap_neut_raw) else 0.0
            step_pct = float(step_pct_raw) if pd.notna(step_pct_raw) else 0.0
            step_deg = float(step_deg_raw) if pd.notna(step_deg_raw) else 0.0
            fixed_deg = float(shift_raw)   if pd.notna(shift_raw)    else 0.0

            if np.isfinite(tap_pos) and step_pct != 0.0:
                n = tap_pos - tap_neut
                Trafo_tau_[k_] = 1.0 + (n * step_pct / 100.0)
                Trafo_shift_deg_[k_] = fixed_deg + n * step_deg
            else:
                Trafo_tau_[k_] = 1.0
                Trafo_shift_deg_[k_] = fixed_deg

            # ---- series admittance on HV side ----
            try:
                sn_mva_raw   = r.sn_mva    if "sn_mva"    in r.index else net_.sn_mva
                vn_hv_kv_raw = r.vn_hv_kv  if "vn_hv_kv"  in r.index else net_.bus.vn_kv.loc[hv]
                vk_pct_raw   = r.vk_percent if "vk_percent" in r.index else np.nan
                vkr_pct_raw  = r.vkr_percent if "vkr_percent" in r.index else np.nan

                sn_mva   = float(sn_mva_raw)
                vn_hv_kv = float(vn_hv_kv_raw)
                vk_pct   = float(vk_pct_raw)  if pd.notna(vk_pct_raw)  else np.nan
                vkr_pct  = float(vkr_pct_raw) if pd.notna(vkr_pct_raw) else np.nan

                if np.isfinite(vk_pct) and np.isfinite(vkr_pct) and vk_pct > 0.0:
                    r_pu = vkr_pct / 100.0
                    x_pu_sq = (vk_pct / 100.0) ** 2 - r_pu ** 2
                    x_pu = np.sqrt(x_pu_sq) if x_pu_sq > 0 else 0.0

                    V_ll = vn_hv_kv * 1e3  # V
                    S_va = sn_mva * 1e6    # VA
                    Z_base = (V_ll ** 2) / S_va
                    Z_series = (r_pu + 1j * x_pu) * Z_base
                    if Z_series != 0:
                        Trafo_y_series_[k_] = 1.0 / Z_series
            except Exception:
                pass

            # ---- magnetizing susceptance (total) from pfe_kw & i0_percent ----
            try:
                pfe_kw_raw = r.pfe_kw     if "pfe_kw"     in r.index else 0.0
                i0_pct_raw = r.i0_percent if "i0_percent" in r.index else 0.0
                sn_mva_raw   = r.sn_mva    if "sn_mva"    in r.index else net_.sn_mva
                vn_hv_kv_raw = r.vn_hv_kv  if "vn_hv_kv"  in r.index else net_.bus.vn_kv.loc[hv]

                pfe_kw  = float(pfe_kw_raw) if pd.notna(pfe_kw_raw) else 0.0
                i0_pct  = float(i0_pct_raw) if pd.notna(i0_pct_raw) else 0.0
                sn_mva  = float(sn_mva_raw)
                vn_hv_kv = float(vn_hv_kv_raw)

                V_ll = vn_hv_kv * 1e3
                S_va = sn_mva * 1e6
                I_base = S_va / (np.sqrt(3.0) * V_ll)
                I0 = (i0_pct / 100.0) * I_base
                I_c = (pfe_kw * 1e3) / (np.sqrt(3.0) * V_ll)
                Im_sq = I0 ** 2 - I_c ** 2
                I_m = np.sqrt(Im_sq) if Im_sq > 0 else 0.0

                B_phase = (np.sqrt(3.0) * I_m) / V_ll  # S
                Trafo_b_total_[k_] = B_phase
            except Exception:
                pass

        return Is_trafo_, Trafo_tau_, Trafo_shift_deg_, Trafo_y_series_, Trafo_b_total_, Trafo_tap_on_i_

    # Compute transformer vectors (SI)
    (Is_trafo,
     Trafo_tau,
     Trafo_shift_deg,
     Trafo_y_series,
     Trafo_b_total,
     Trafo_tap_on_i) = compute_trafo_vectors_from_net(net)

    # ensure branch is marked connected if it's a trafo
    Lines_connected = np.where(Is_trafo == 1, 1, Lines_connected).astype(np.int8)

    # --- Build Y_matrix using the SAME stamping model as your generic generator
    Y_matrix = _build_Y_with_lines_and_transformers(
        N,
        Lines_connected,
        Is_trafo,
        Z_Lines,
        Y_C_Lines,
        vn_kv,
        Trafo_y_series,
        Trafo_b_total,
        Trafo_tau,
        Trafo_shift_deg,
        Trafo_tap_on_i
    )

    gridtype = "IEEE14"
    U_base = float("nan")   # not meaningful with multi-voltage scaling
    is_connected = True

    # ==========================================================
    # === EXPERIMENT A: override s_multi to be self-consistent ===
    # ==========================================================
    if self_consistent_S:
        # Use your own stamped Y and current u_start to define S that is
        # *exactly* compatible with (Y_matrix, u_start).
        I0 = Y_matrix @ u_start
        S0 = u_start * np.conj(I0)    # complex injections at each bus (W + j var)

        # Build P, Q arrays as your NR code expects
        P_spec = np.real(S0).copy()
        Q_spec = np.imag(S0).copy()

        # Note: in your NR, Q at PV buses is *ignored* (their mismatch is not used),
        # so we can safely leave Q_spec[pv] = imag(S0[pv]) or set them to 0.
        # To keep pure mathematical consistency, we leave them as imag(S0).
        s_multi = P_spec + 1j * Q_spec

    return (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
            Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
            vn_kv.astype(np.float64),
            Is_trafo.astype(np.int8), Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total)

# Buidling Y_matrix from net._ppc
# def case_generation_ieee14_pandapower(
#     jitter_load: float = 0.0,          # e.g. 0.10 => multiply each (p_mw, q_mvar) by N(1, 0.10)
#     jitter_gen: float = 0.0,           # e.g. 0.10 => multiply each gen p_mw by N(1, 0.10)
#     pv_vset_range = None,             # e.g. (0.98, 1.04) => uniform PV |V| setpoints in pu
#     rand_u_start: bool = False,       # randomize initial |U|/angle (PV |U| honored)
#     angle_jitter_deg: float = 5.0,    # +/- deg jitter for non-slack start angles
#     mag_jitter_pq: float = 0.02,      # +/- fraction for PQ start |U| (e.g. 0.02 => ±2%)
#     seed = None                       # RNG seed for reproducibility
# ):
#     """
#     IEEE-14 preset in SI units with per-bus vn_kv from pandapower, plus optional jitter.
#
#     Returns:
#       gridtype (str), bus_typ (N,), s_multi (N,), u_start (N,), Y_matrix (N,N),
#       is_connected (bool), Y_Lines (C,), Y_C_Lines (C,), Lines_connected (C,),
#       U_base (float), S_base (float),
#       vn_kv (N,), Is_trafo (C,), Trafo_tau (C,), Trafo_shift_deg (C,),
#       Trafo_y_series (C,), Trafo_b_total (C,)
#     """
#     import numpy as np
#     import pandas as pd
#     import pandapower as pp
#     import pandapower.networks as pn
#     from pandapower.powerflow import LoadflowNotConverged
#
#     rng = np.random.default_rng(seed)
#
#     # --- Build the base IEEE-14 net
#     net = pn.case14()
#
#     # ----- JITTER: loads (p_mw, q_mvar) -----
#     if jitter_load > 0 and len(net.load):
#         s = rng.normal(1.0, jitter_load, size=len(net.load))
#         # keep signs and units consistent (pandapower stores positive consumption)
#         net.load["p_mw"]   = net.load["p_mw"].to_numpy(float)   * s
#         net.load["q_mvar"] = net.load["q_mvar"].to_numpy(float) * s
#
#     # ----- JITTER: generator active powers (p_mw)
#     if jitter_gen > 0 and len(net.gen):
#         s = rng.normal(1.0, jitter_gen, size=len(net.gen))
#         net.gen["p_mw"] = net.gen["p_mw"].to_numpy(float) * s
#
#     # ----- JITTER: PV voltage magnitude setpoints (vm_pu)
#     if pv_vset_range is not None and len(net.gen):
#         lo, hi = pv_vset_range
#         net.gen["vm_pu"] = rng.uniform(lo, hi, size=len(net.gen))
#
#     # --- Build PPC & Ybus once (no need to converge fully; 1 iter builds matrices)
#     try:
#         pp.runpp(
#             net,
#             init="flat",
#             calculate_voltage_angles=True,
#             max_iteration=1,      # just build model/Ybus
#             enforce_q_lims=False, # don't let Q-lims stop model build
#             tolerance_mva=1e9
#         )
#     except LoadflowNotConverged:
#         # ok: we only need net._ppc["internal"]["Ybus"]
#         print("exception LoadflowNotConverged ")
#         pass
#
#     N = len(net.bus)
#
#     # per-bus nominal voltages (kV -> V)
#     vn_kv = net.bus["vn_kv"].to_numpy(dtype=float)
#     Vbase = vn_kv * 1e3  # V
#
#     # System base power in VA
#     S_base = float(net.sn_mva) * 1e6
#
#     # --- Per-unit Ybus -> SI Ybus using per-bus scaling
#     Ypu = net._ppc["internal"]["Ybus"].toarray().astype(np.complex128)
#     denom = np.outer(Vbase, Vbase)            # V_i * V_j
#     Y_matrix = Ypu * (S_base / denom)         # Siemens
#
#     # --- Bus types: 1=slack, 2=PV, 3=PQ
#     # Sets types from tables: ext_grid → slack, gen buses → PV, rest → PQ.
#     # (In IEEE-14 this yields: 1 slack; 2,3,6,8 as PV; others PQ.)
#     bus_typ = np.full(N, 3, dtype=int)
#     if len(net.ext_grid):
#         bus_typ[net.ext_grid["bus"].to_numpy()] = 1
#     if len(net.gen):
#         bus_typ[net.gen["bus"].to_numpy()] = 2
#
#     # --- Specified complex injections S (W + j var), SI
#     S = np.zeros(N, dtype=np.complex128)  # Start with zeros for all buses.
#
#     # Loads: negative injections
#     if len(net.load):
#         b = net.load.bus.to_numpy(int)
#         P = net.load.p_mw.to_numpy(float) * 1e6
#         Q = net.load.q_mvar.to_numpy(float) * 1e6
#         S[b] -= (P + 1j * Q)
#
#     # Static gens (positive)
#     if len(net.sgen):
#         b = net.sgen.bus.to_numpy(int)
#         P = net.sgen.p_mw.to_numpy(float) * 1e6
#         Q = net.sgen.q_mvar.to_numpy(float) * 1e6
#         S[b] += (P + 1j * Q)
#
#     # Synchronous gens at PV: add active power; Q is to be solved
#     if len(net.gen):
#         for _, r in net.gen.iterrows():
#             S[int(r.bus)] += float(r.p_mw) * 1e6
#
#     # (Slack and PV Q=0 conventions are enforced in NR, not here.)
#     s_multi = S
#
#     # --- Start voltages
#     # default: flat start at 1.0 pu on each bus => |U| = Vbase, angle = 0
#     u_start = Vbase.astype(np.complex128)
#
#     if rand_u_start:
#         mag = Vbase.copy()
#
#         # Honor PV bus |V| targets from net.gen.vm_pu
#         if len(net.gen):
#             vm_by_bus = {}
#             for _, r in net.gen.iterrows():
#                 vm_val = float(r.vm_pu) if pd.notna(r.vm_pu) else 1.0
#                 vm_by_bus[int(r.bus)] = vm_val
#             for b, vm in vm_by_bus.items():
#                 mag[b] = vm * Vbase[b]
#
#         # PQ buses: jitter magnitude by ±mag_jitter_pq
#         pq_mask = (bus_typ == 3)
#         if pq_mask.any() and mag_jitter_pq > 0:
#             mag[pq_mask] *= rng.uniform(1.0 - mag_jitter_pq,
#                                         1.0 + mag_jitter_pq,
#                                         size=pq_mask.sum())
#
#         # Angles: slack stays 0; others jitter ±angle_jitter_deg
#         ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * (np.pi / 180.0)
#         if len(net.ext_grid):
#             ang[net.ext_grid.bus.to_numpy(int)] = 0.0
#
#         u_start = mag * np.exp(1j * ang)
#
#     # --- Upper-tri indexing helpers
#     iu, ju = np.triu_indices(N, 1)
#     C = len(iu)
#
#     # --- LINE edge features in SI (optional; Y_matrix is authoritative)
#     Lines_connected = np.zeros(C, dtype=np.int8)
#     Y_Lines = np.zeros(C, dtype=np.complex128)   # series admittance (S)
#     Y_C_Lines = np.zeros(C, dtype=np.float64)    # per-end shunt susceptance (S)
#
#     # pair -> index map
#     pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}
#
#     # frequency (for shunt B from capacitance)
#     f = float(getattr(net, "f_hz", 50.0))
#     two_pi_f = 2.0 * np.pi * f
#
#     if len(net.line):
#         for _, r in net.line.iterrows():
#             a, b = int(r.from_bus), int(r.to_bus)
#             i, j = (a, b) if a < b else (b, a)
#             idx = pair_to_idx.get((i, j), None)
#             if idx is None:
#                 continue
#             Lines_connected[idx] = 1
#
#             # Series Z_total = (r + jx) * length_km  ->  Y = 1/Z
#             try:
#                 rpkm = float(r.r_ohm_per_km)
#                 xpkm = float(r.x_ohm_per_km)
#                 Lkm  = float(r.length_km)
#                 Z = complex(rpkm, xpkm) * Lkm
#                 if Z != 0:
#                     Y_Lines[idx] = 1.0 / Z
#             except Exception:
#                 pass
#
#             # Shunt: total B_total = ω * C_total ; per-end = 0.5 * B_total
#             try:
#                 c_nf_per_km = float(r.c_nf_per_km)
#                 Lkm = float(r.length_km)
#                 C_total = c_nf_per_km * 1e-9 * Lkm  # Farads
#                 B_total = two_pi_f * C_total        # Siemens
#                 Y_C_Lines[idx] = 0.5 * B_total
#             except Exception:
#                 pass
#
#     # ---------- Transformer vectors (NaN-safe) ----------
#     def _upper_index_map(n_buses: int):
#         iu_, ju_ = np.triu_indices(n_buses, 1)
#         C_ = len(iu_)
#         p2i = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu_, ju_))}
#         return iu_, ju_, C_, p2i
#
#     def compute_trafo_vectors_from_net(net_):
#         """
#         Build transformer vectors aligned to the upper-tri (iu,ju):
#           - Is_trafo:   0/1
#           - Trafo_tau:  off-nominal magnitude (from tap settings if available, else 1.0)
#           - Trafo_shift_deg: phase shift in degrees (tap-based or fixed shift), finite (no NaN)
#           - Trafo_y_series:  series admittance (Siemens) computed on HV side
#           - Trafo_b_total:   total magnetizing susceptance (Siemens)
#         """
#         N_ = len(net_.bus)
#         iu_, ju_, C_, p2i_ = _upper_index_map(N_)
#
#         Is_trafo_ = np.zeros(C_, dtype=np.int8)
#         Trafo_tau_ = np.ones(C_, dtype=np.float64)
#         Trafo_shift_deg_ = np.zeros(C_, dtype=np.float64)
#         Trafo_y_series_ = np.zeros(C_, dtype=np.complex128)
#         Trafo_b_total_ = np.zeros(C_, dtype=np.float64)
#
#         if len(net_.trafo) == 0:
#             return Is_trafo_, Trafo_tau_, Trafo_shift_deg_, Trafo_y_series_, Trafo_b_total_
#
#         for _, r in net_.trafo.iterrows():
#             hv = int(r.hv_bus)
#             lv = int(r.lv_bus)
#             i_, j_ = (hv, lv) if hv < lv else (lv, hv)
#             k_ = p2i_.get((i_, j_))
#             if k_ is None:
#                 continue
#
#             Is_trafo_[k_] = 1
#
#             # ---- tap magnitude & phase (NaN-safe) ----
#             tap_pos_raw   = r.tap_pos       if "tap_pos"       in r.index else np.nan
#             tap_neut_raw  = r.tap_neutral   if "tap_neutral"   in r.index else 0.0
#             step_pct_raw  = r.tap_step_percent if "tap_step_percent" in r.index else 0.0
#             step_deg_raw  = r.tap_step_degree  if "tap_step_degree"  in r.index else 0.0
#             shift_raw     = r.shift_degree  if "shift_degree"  in r.index else 0.0
#
#             tap_pos  = float(tap_pos_raw)  if (pd.notna(tap_pos_raw))  else np.nan
#             tap_neut = float(tap_neut_raw) if (pd.notna(tap_neut_raw)) else 0.0
#             step_pct = float(step_pct_raw) if (pd.notna(step_pct_raw)) else 0.0
#             step_deg = float(step_deg_raw) if (pd.notna(step_deg_raw)) else 0.0
#             fixed_deg = float(shift_raw)   if (pd.notna(shift_raw))    else 0.0
#
#             if np.isfinite(tap_pos) and step_pct != 0.0:
#                 n = tap_pos - tap_neut
#                 Trafo_tau_[k_] = 1.0 + (n * step_pct / 100.0)
#                 Trafo_shift_deg_[k_] = fixed_deg + n * step_deg
#             else:
#                 Trafo_tau_[k_] = 1.0
#                 Trafo_shift_deg_[k_] = fixed_deg
#
#             # ---- series admittance on HV side ----
#             try:
#                 sn_mva_raw   = r.sn_mva    if "sn_mva"    in r.index else net_.sn_mva
#                 vn_hv_kv_raw = r.vn_hv_kv  if "vn_hv_kv"  in r.index else net_.bus.vn_kv.loc[hv]
#                 vk_pct_raw   = r.vk_percent if "vk_percent" in r.index else np.nan
#                 vkr_pct_raw  = r.vkr_percent if "vkr_percent" in r.index else np.nan
#
#                 sn_mva   = float(sn_mva_raw)
#                 vn_hv_kv = float(vn_hv_kv_raw)
#                 vk_pct   = float(vk_pct_raw)  if pd.notna(vk_pct_raw)  else np.nan
#                 vkr_pct  = float(vkr_pct_raw) if pd.notna(vkr_pct_raw) else np.nan
#
#                 if np.isfinite(vk_pct) and np.isfinite(vkr_pct) and vk_pct > 0.0:
#                     r_pu = vkr_pct / 100.0
#                     x_pu_sq = (vk_pct / 100.0) ** 2 - r_pu ** 2
#                     x_pu = np.sqrt(x_pu_sq) if x_pu_sq > 0 else 0.0
#
#                     V_ll = vn_hv_kv * 1e3  # line-to-line [V]
#                     S_va = sn_mva * 1e6    # [VA]
#                     Z_base = (V_ll ** 2) / S_va
#                     Z_series = (r_pu + 1j * x_pu) * Z_base
#                     if Z_series != 0:
#                         Trafo_y_series_[k_] = 1.0 / Z_series
#             except Exception:
#                 pass
#
#             # ---- magnetizing susceptance (total) from pfe_kw & i0_percent ----
#             try:
#                 pfe_kw_raw = r.pfe_kw     if "pfe_kw"     in r.index else 0.0
#                 i0_pct_raw = r.i0_percent if "i0_percent" in r.index else 0.0
#                 sn_mva_raw   = r.sn_mva    if "sn_mva"    in r.index else net_.sn_mva
#                 vn_hv_kv_raw = r.vn_hv_kv  if "vn_hv_kv"  in r.index else net_.bus.vn_kv.loc[hv]
#
#                 pfe_kw  = float(pfe_kw_raw) if pd.notna(pfe_kw_raw) else 0.0
#                 i0_pct  = float(i0_pct_raw) if pd.notna(i0_pct_raw) else 0.0
#                 sn_mva  = float(sn_mva_raw)
#                 vn_hv_kv = float(vn_hv_kv_raw)
#
#                 V_ll = vn_hv_kv * 1e3
#                 S_va = sn_mva * 1e6
#                 I_base = S_va / (np.sqrt(3.0) * V_ll)
#                 I0 = (i0_pct / 100.0) * I_base
#                 I_c = (pfe_kw * 1e3) / (np.sqrt(3.0) * V_ll)
#                 Im_sq = I0 ** 2 - I_c ** 2
#                 I_m = np.sqrt(Im_sq) if Im_sq > 0 else 0.0
#
#                 # per-phase susceptance (Siemens); stamping will split as b/2 each side
#                 B_phase = (np.sqrt(3.0) * I_m) / V_ll
#                 Trafo_b_total_[k_] = B_phase
#             except Exception:
#                 pass
#
#         return Is_trafo_, Trafo_tau_, Trafo_shift_deg_, Trafo_y_series_, Trafo_b_total_
#
#     # Compute transformer vectors once (NaN-safe)
#     Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total = compute_trafo_vectors_from_net(net)
#
#     gridtype = "IEEE14"
#     U_base = float("nan")   # not meaningful with multi-voltage scaling
#     is_connected = True
#
#     return (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
#             Y_Lines, Y_C_Lines, Lines_connected, U_base, S_base,
#             vn_kv.astype(np.float64),
#             Is_trafo, Trafo_tau, Trafo_shift_deg, Trafo_y_series, Trafo_b_total)

    # # --- Per-unit Ybus -> SI Ybus using per-bus scaling
    # Ypu = net._ppc["internal"]["Ybus"].toarray().astype(np.complex128)
    # denom = np.outer(Vbase, Vbase)            # V_i * V_j
    # Y_matrix = Ypu * (S_base / denom)         # Siemens