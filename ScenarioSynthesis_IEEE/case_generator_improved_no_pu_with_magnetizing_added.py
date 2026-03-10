import math
import time
import numpy as np
from collections import deque
from matplotlib.collections import LineCollection
import pandas as pd

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


import numpy as np
import pandapower as pp
import pandapower.networks as pn

from pandapower.pypower.idx_bus import BASE_KV, GS, BS
from pandapower.pypower.idx_brch import (
    F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
)


# ---------------------------------------------------------------------
# Helper: rebuild Ybus from ppci (per-unit), like pypower.makeYbus
# ---------------------------------------------------------------------
import numpy as np

def build_Y_stamped_from_ppc(ppc_int):
    """
    Rebuild Ybus (per-unit) from ppc_int['bus'] and ppc_int['branch'].

    Matches pandapower's extended PPC branch model when available:
      - BR_G (shunt conductance)
      - BR_B_ASYM / BR_G_ASYM (to-side asym shunt additions)

    Falls back to classic PYPOWER behavior if these columns are not present.
    """
    from pandapower.pypower.idx_bus import GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    # Optional pandapower-extended columns (may not exist in older versions)
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

    # --- Series admittance Ys = 1/(R+jX) (per-unit), 0 if out of service ---
    R = branch[:, BR_R]
    X = branch[:, BR_X]
    Z = R + 1j * X
    Ys = np.zeros(nl, dtype=np.complex128)
    nonzero = (stat != 0) & (np.abs(Z) > 1e-18)
    Ys[nonzero] = stat[nonzero] / Z[nonzero]

    # --- Tap ratios + phase shift ---
    tap = np.ones(nl, dtype=np.complex128)
    tap_raw = branch[:, TAP]
    nonunity = tap_raw != 0.0
    tap[nonunity] = tap_raw[nonunity]
    shift_deg = branch[:, SHIFT]
    tap *= np.exp(1j * np.pi / 180.0 * shift_deg)

    # --- Branch shunts (pandapower-extended if available) ---
    B = stat * branch[:, BR_B]  # total susceptance base term

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

    # From-side and To-side shunt admittances (pu)
    Bcf = (G + 1j * B)
    Bct = ((G + G_asym) + 1j * (B + B_asym))

    # --- Stamp branch contributions (pandapower makeYbus-style) ---
    Yff = (Ys + Bcf / 2.0) / (tap * np.conj(tap))
    Ytt = Ys + Bct / 2.0
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap

    f_bus = branch[:, F_BUS].astype(int)
    t_bus = branch[:, T_BUS].astype(int)

    for k in range(nl):
        if stat[k] == 0:
            continue
        # Skip if everything is ~0
        if (abs(Ys[k]) < 1e-18) and (abs(Bcf[k]) < 1e-18) and (abs(Bct[k]) < 1e-18):
            continue

        f = f_bus[k]
        t = t_bus[k]
        Ybus[f, f] += Yff[k]
        Ybus[t, t] += Ytt[k]
        Ybus[f, t] += Yft[k]
        Ybus[t, f] += Ytf[k]

    # --- Bus shunts (GS + jBS)/baseMVA ---
    Ysh_pu = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    Ybus[np.arange(nb), np.arange(nb)] += Ysh_pu

    return Ybus


# ---------------------------------------------------------------------
# Helper: convert Ybus (pu) -> SI with per-bus Vbase
# ---------------------------------------------------------------------
def per_unit_to_SI(Y_pu, ppc_int):
    """
    Convert Ybus (per-unit) to SI [Siemens] using:

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


# ---------------------------------------------------------------------
# Helper: bus shunt vector in SI (per bus)
# ---------------------------------------------------------------------
def get_Y_shunt_bus_SI(ppc_int):
    """
    Extract per-bus shunt admittance Y_shunt_bus in SI [S] from ppci.

    From MATPOWER convention:
        Ysh_pu[i] = (GS_i + j BS_i) / baseMVA

    Then:
        Ysh_SI[i] = Ysh_pu[i] * (S_base / Vbase_i^2)
                  = (GS_i + j BS_i) * 1e6 / Vbase_i^2
    """
    bus = np.asarray(ppc_int["bus"], dtype=float)
    baseMVA = float(ppc_int["baseMVA"])

    # System base power
    S_base = baseMVA * 1e6  # VA

    # per-bus nominal voltage
    vn_kv = bus[:, BASE_KV]
    Vbase = vn_kv * 1e3  # V

    GS_MW = bus[:, GS]   # MW
    BS_MVAr = bus[:, BS] # MVAr

    # per-unit shunt admittance
    Ysh_pu = (GS_MW + 1j * BS_MVAr) / baseMVA

    # convert to SI
    Y_shunt_SI = Ysh_pu * (S_base / (Vbase * Vbase))
    return Y_shunt_SI.astype(np.complex128)



from pandapower.powerflow import LoadflowNotConverged

# add to your imports if not already
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.powerflow import LoadflowNotConverged
from pandapower.pypower.idx_bus import BASE_KV, GS, BS
from pandapower.pypower.idx_brch import (
    F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
)
from typing import Union


import numpy as np

def case_generation_ieee14_pandapower(
    jitter_load: float = 0.0,
    jitter_gen: float = 0.0,
    pv_vset_range=None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 5.0,
    mag_jitter_pq: float = 0.02,
    seed=None,
    trafo_pfe_kw: Union[float, None] = None,
    trafo_i0_percent: Union[float, None] = None,
):
    """
    Like before, but now exports *end-specific* transformer shunt admittances
    that pandapower uses in makeYbus:

      Bcf_pu = BR_G + j*BR_B
      Bct_pu = (BR_G+BR_G_ASYM) + j*(BR_B+BR_B_ASYM)

    We export these in SI (S) as Trafo_y_shunt_from/to so reconstruction can
    match ppc_int["Ybus"] exactly when magnetizing is nonzero.
    """
    import pandas as pd
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.powerflow import LoadflowNotConverged
    from pandapower.pypower.idx_bus import BASE_KV, GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    # optional columns exist in newer pandapower
    try:
        from pandapower.pypower.idx_brch import BR_G, BR_G_ASYM, BR_B_ASYM
    except Exception:
        BR_G = BR_G_ASYM = BR_B_ASYM = None

    def _col(row, idx, default=0.0):
        if idx is None:
            return default
        if idx >= row.shape[0]:
            return default
        return float(row[idx])

    rng = np.random.default_rng(seed)

    net = pn.case14()

    # force nonzero magnetizing
    if len(net.trafo):
        if trafo_pfe_kw is not None:
            net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
        if trafo_i0_percent is not None:
            net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

    # jitter (unchanged)
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
            net,
            init="flat",
            calculate_voltage_angles=True,
            max_iteration=1,
            enforce_q_lims=False,
            tolerance_mva=1e9
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

    # ground-truth Ybus (pu -> SI)
    Ypu = ppc_int["Ybus"].toarray().astype(np.complex128)
    Y_matrix = Ypu * (S_base / np.outer(Vbase, Vbase))

    # bus shunts (GS/BS) (pu -> SI)
    Ysh_pu = (bus_ppc[:, GS] + 1j * bus_ppc[:, BS]) / baseMVA
    Y_shunt_bus = Ysh_pu * (S_base / (Vbase * Vbase))

    # bus types
    bus_typ = np.full(N, 3, dtype=int)
    if len(net.ext_grid):
        bus_typ[net.ext_grid["bus"].to_numpy(int)] = 1
    if len(net.gen):
        bus_typ[net.gen["bus"].to_numpy(int)] = 2

    # injections
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
            mag[pq_mask] *= rng.uniform(1.0 - mag_jitter_pq, 1.0 + mag_jitter_pq, size=pq_mask.sum())
        ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * np.pi / 180.0
        if len(net.ext_grid):
            ang[net.ext_grid.bus.to_numpy(int)] = 0.0
        u_start = mag * np.exp(1j * ang)

    # upper-tri metadata
    iu, ju = np.triu_indices(N, 1)
    C = len(iu)
    pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}

    Lines_connected = np.zeros(C, dtype=np.int8)

    Y_Lines = np.zeros(C, dtype=np.complex128)
    Y_C_Lines = np.zeros(C, dtype=np.float64)

    Is_trafo = np.zeros(C, dtype=np.int8)
    Trafo_tau = np.ones(C, dtype=np.float64)
    Trafo_shift_deg = np.zeros(C, dtype=np.float64)
    Trafo_y_series = np.zeros(C, dtype=np.complex128)

    # NEW: end-specific transformer shunt admittances in SI
    Trafo_y_shunt_from = np.zeros(C, dtype=np.complex128)
    Trafo_y_shunt_to   = np.zeros(C, dtype=np.complex128)

    Trafo_tap_on_i = np.zeros(C, dtype=np.int8)
    Trafo_hv_is_i = np.zeros(C, dtype=np.int8)
    Trafo_n = np.ones(C, dtype=np.float64)

    for row in branch_ppc:
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

        # map to upper-tri, keep "from-bus side" as tap orientation
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

        Vi = Vbase[i]; Vj = Vbase[j]
        Vh = max(Vi, Vj); Vl = min(Vi, Vj)
        Trafo_hv_is_i[k] = 1 if Vi >= Vj else 0
        Trafo_n[k] = (Vh / Vl) if Vl > 0 else 1.0

        # robust trafo detection
        is_tr = (abs(tap - 1.0) > 1e-12) or (abs(shift) > 1e-12) or (abs(Vi - Vj) > 1e-6)

        z = complex(r, x)
        y_pu = 0j if abs(z) < 1e-12 else 1.0 / z

        if not is_tr:
            V_line = Vi
            scale = S_base / (V_line ** 2)
            Y_Lines[k] = y_pu * scale
            Y_C_Lines[k] = (0.5 * b) * scale
        else:
            Is_trafo[k] = 1
            Trafo_tau[k] = tap
            Trafo_shift_deg[k] = shift
            Trafo_tap_on_i[k] = tap_on_i

            # store series in SI (HV-referred like you did)
            scale_tr = S_base / (Vh ** 2)
            Trafo_y_series[k] = y_pu * scale_tr

            # NEW: store end-specific shunts in SI referred to the *actual* from/to bus bases
            Vf = Vbase[fb]
            Vt = Vbase[tb]

            Bcf_pu = (g + 1j * b)
            Bct_pu = ((g + g_asym) + 1j * (b + b_asym))

            Trafo_y_shunt_from[k] = Bcf_pu * (S_base / (Vf ** 2))
            Trafo_y_shunt_to[k]   = Bct_pu * (S_base / (Vt ** 2))

    is_connected = True
    U_base = float(Vbase[0])
    gridtype = "IEEE14_pandapower_ppcY"

    return (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Y_Lines, Y_C_Lines, Lines_connected.astype(np.int8), U_base, S_base,
        vn_kv.astype(np.float64), Is_trafo.astype(np.int8), Trafo_tau, Trafo_shift_deg,
        Trafo_y_series, Trafo_y_shunt_from, Trafo_y_shunt_to,  # NEW
        Y_shunt_bus.astype(np.complex128),
        Trafo_tap_on_i.astype(np.int8), Trafo_hv_is_i.astype(np.int8), Trafo_n.astype(np.float64),
    )


# -----------------------------
# Case generation (stamped) — REAL stamped Ybus, not wrapper
# -----------------------------

def case_generation_ieee14_pandapower_stamped(
    jitter_load: float = 0.0,
    jitter_gen: float = 0.0,
    pv_vset_range=None,
    rand_u_start: bool = False,
    angle_jitter_deg: float = 5.0,
    mag_jitter_pq: float = 0.02,
    seed=None,
    trafo_pfe_kw: Union[float, None] = None,
    trafo_i0_percent: Union[float, None] = None,
):
    """
    Same metadata as the ppcY function, but Y_matrix is built by:
      build_Y_stamped_from_ppc(ppc_int) -> per_unit_to_SI(...)
    """
    import pandas as pd
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.powerflow import LoadflowNotConverged
    from pandapower.pypower.idx_bus import BASE_KV, GS, BS
    from pandapower.pypower.idx_brch import (
        F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    )

    # Optional pandapower-extended branch columns
    try:
        from pandapower.pypower.idx_brch import BR_G, BR_B_ASYM, BR_G_ASYM
    except Exception:
        BR_G = BR_B_ASYM = BR_G_ASYM = None

    def _col(row: np.ndarray, idx, default=0.0) -> float:
        if idx is None:
            return default
        if idx >= row.shape[0]:
            return default
        return float(row[idx])

    rng = np.random.default_rng(seed)
    net = pn.case14()

    if len(net.trafo):
        if trafo_pfe_kw is not None:
            net.trafo.loc[:, "pfe_kw"] = float(trafo_pfe_kw)
        if trafo_i0_percent is not None:
            net.trafo.loc[:, "i0_percent"] = float(trafo_i0_percent)

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

    # Stamped Ybus (pu -> SI)
    Y_pu_stamp = build_Y_stamped_from_ppc(ppc_int)
    Y_matrix = per_unit_to_SI(Y_pu_stamp, ppc_int)

    # Bus shunts in SI
    Ysh_pu = (bus_ppc[:, GS] + 1j * bus_ppc[:, BS]) / baseMVA
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
            mag[pq_mask] *= rng.uniform(1.0 - mag_jitter_pq, 1.0 + mag_jitter_pq, size=pq_mask.sum())
        ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg, size=N) * np.pi / 180.0
        if len(net.ext_grid):
            ang[net.ext_grid.bus.to_numpy(int)] = 0.0
        u_start = mag * np.exp(1j * ang)

    # Upper-tri metadata (same as ppcY)
    iu, ju = np.triu_indices(N, 1)
    C = len(iu)
    pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}

    Lines_connected = np.zeros(C, dtype=np.int8)

    Y_Lines = np.zeros(C, dtype=np.complex128)
    Y_C_Lines = np.zeros(C, dtype=np.float64)

    Is_trafo = np.zeros(C, dtype=np.int8)
    Trafo_tau = np.ones(C, dtype=np.float64)
    Trafo_shift_deg = np.zeros(C, dtype=np.float64)
    Trafo_y_series = np.zeros(C, dtype=np.complex128)
    Trafo_y_shunt_from = np.zeros(C, dtype=np.complex128)
    Trafo_y_shunt_to   = np.zeros(C, dtype=np.complex128)

    Trafo_tap_on_i = np.zeros(C, dtype=np.int8)
    Trafo_hv_is_i = np.zeros(C, dtype=np.int8)
    Trafo_n = np.ones(C, dtype=np.float64)

    for row in branch_ppc:
        fb = int(row[F_BUS])
        tb = int(row[T_BUS])
        r = float(row[BR_R])
        x = float(row[BR_X])
        b = float(row[BR_B])
        g = _col(row, BR_G, 0.0)
        b_asym = _col(row, BR_B_ASYM, 0.0)
        g_asym = _col(row, BR_G_ASYM, 0.0)

        tap = float(row[TAP])
        shift = float(row[SHIFT])
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

        is_tr = (abs(tap - 1.0) > 1e-12) or (abs(shift) > 1e-12) or (abs(Vi - Vj) > 1e-6)

        z = complex(r, x)
        y_pu = 0j if abs(z) < 1e-12 else 1.0 / z

        if not is_tr:
            V_line = Vi
            scale = S_base / (V_line ** 2)
            Y_Lines[k] = y_pu * scale
            Y_C_Lines[k] = (0.5 * b) * scale
        else:
            Is_trafo[k] = 1
            Trafo_tau[k] = tap
            Trafo_shift_deg[k] = shift
            Trafo_tap_on_i[k] = tap_on_i

            scale_tr = S_base / (Vh ** 2)
            Trafo_y_series[k] = y_pu * scale_tr

            Vf = Vbase[fb]
            Vt = Vbase[tb]
            Bcf_pu = (g + 1j * b)
            Bct_pu = ((g + g_asym) + 1j * (b + b_asym))
            Trafo_y_shunt_from[k] = Bcf_pu * (S_base / (Vf ** 2))
            Trafo_y_shunt_to[k]   = Bct_pu * (S_base / (Vt ** 2))

    is_connected = True
    U_base = float(Vbase[0])
    gridtype = "IEEE14_pandapower_stamped"

    return (
        gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
        Y_Lines, Y_C_Lines, Lines_connected.astype(np.int8), U_base, S_base,
        vn_kv.astype(np.float64), Is_trafo.astype(np.int8),
        Trafo_tau, Trafo_shift_deg, Trafo_y_series,
        Trafo_y_shunt_from, Trafo_y_shunt_to,
        Y_shunt_bus.astype(np.complex128),
        Trafo_tap_on_i.astype(np.int8), Trafo_hv_is_i.astype(np.int8), Trafo_n.astype(np.float64),
    )

