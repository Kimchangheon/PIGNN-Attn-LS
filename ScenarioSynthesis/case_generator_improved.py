import math
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.collections import LineCollection


def plot_random_connections(bus_number, lines_connected):
    """Draw buses on a circle and connect those pairs flagged by lines_connected (0/1)."""
    # Coordinates on the unit circle
    angles = np.linspace(0, 2 * np.pi, bus_number, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='blue')

    # Labels for each bus
    for i in range(bus_number):
        ax.text(x[i], y[i], f'Bus {i + 1}', ha='right', va='bottom')

    # Build connected segments in one go (faster than many plt.plot calls)
    iu, ju = np.triu_indices(bus_number, 1)
    connected_mask = (lines_connected == 1)
    ii = iu[connected_mask]
    jj = ju[connected_mask]
    if ii.size:
        segments = np.stack([np.column_stack([x[ii], y[ii]]),
                             np.column_stack([x[jj], y[jj]])], axis=1)
        lc = LineCollection(segments, linewidths=1.0, colors='gray')
        ax.add_collection(lc)

        # Line labels at midpoints (kept as per original behavior)
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
    """
    Check if Bus 1 (index 0) can reach all others via connections.
    Faster BFS using adjacency lists + deque. Logic preserved.
    """
    iu, ju = np.triu_indices(bus_number, 1)
    sel = (lines_connected == 1)
    adj = [[] for _ in range(bus_number)]
    for a, b in zip(iu[sel], ju[sel]):
        adj[a].append(b)
        adj[b].append(a)

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
    """Vectorized construction of symmetric 0/1 adjacency matrix from upper-triangle encoding."""
    adj = np.zeros((bus_number, bus_number), dtype=int)
    iu, ju = np.triu_indices(bus_number, 1)
    adj[iu, ju] = lines_connected
    adj[ju, iu] = lines_connected
    return adj


def insert_values_in_matrix(matrix, connections, values):
    """
    Return a symmetric matrix where entries corresponding to adjacency==1
    are filled with the position-aligned 'values' (upper-triangle order),
    zeros elsewhere. (connections is unused logically in original; preserved in signature.)
    """
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
    """Complex-valued variant of insert_values_in_matrix (logic identical)."""
    n = matrix.shape[0]
    out = np.zeros((n, n), dtype=complex)
    iu, ju = np.triu_indices(n, 1)
    mask = (matrix[iu, ju] == 1)
    if np.any(mask):
        sel_vals = values[mask]
        out[iu[mask], ju[mask]] = sel_vals
        out[ju[mask], iu[mask]] = sel_vals
    return out


def build_Y_matrix(matrix, Y_C_Bus, Line_matrix):
    """
    Build nodal admittance matrix Y:
      - Off-diagonal: -Y_line for connected pairs, 0 otherwise
      - Diagonal: sum of incident line admittances + j * Y_C_Bus
    Vectorized and fully symmetric.
    """
    n = matrix.shape[0]
    Y = np.zeros((n, n), dtype=complex)

    # Off-diagonals
    iu, ju = np.triu_indices(n, 1)
    mask = (matrix[iu, ju] == 1)
    if np.any(mask):
        off = -Line_matrix[iu[mask], ju[mask]]
        Y[iu[mask], ju[mask]] = off
        Y[ju[mask], iu[mask]] = off

    # Diagonal
    diag = (Line_matrix * matrix).sum(axis=1) + 1j * Y_C_Bus
    np.fill_diagonal(Y, diag)
    return Y


def create_bus_typ(busnummer, fixed):
    """
    Slack = 1 at index 0; others randomly 2 (PV) or 3 (PQ).
    Keep per-index seeding behavior for reproducibility when 'fixed' is True.
    """
    bus_typ = np.zeros(busnummer, dtype=int)
    bus_typ[0] = 1  # Slack
    if busnummer > 1:
        for i in range(1, busnummer):
            if fixed:
                np.random.seed(busnummer * i)
            bus_typ[i] = np.random.choice([2, 3])
    return bus_typ


def generate_PQ(bus_typ, gridtype):
    """
    Generate P,Q per bus according to types and gridtype.
    Kept per-bus sampling loop to preserve randomness characteristics.
    """
    num_buses = len(bus_typ)
    P = np.zeros(num_buses)
    Q = np.zeros(num_buses)
    np.random.seed(time.time_ns() % (2 ** 32))

    if gridtype == "LVN":
        P_min, P_max = -50 * 0.0005, 50 * 0.0005
        Q_min, Q_max = -25 * 0.0005, 25 * 0.0005
    elif gridtype == "MVN":  # 10 kV
        P_min, P_max = -5, 5
        Q_min, Q_max = -2, 2
    elif gridtype == "MVN20kv":
        P_min, P_max = -25, 25
        Q_min, Q_max = -12.5, 12.5
    else:  # HVN (110 kV)
        P_min, P_max = -300, 300
        Q_min, Q_max = -150, 150

    for i, t in enumerate(bus_typ):
        if t == 1:  # Slack
            P[i] = 0
            Q[i] = 0
        elif t == 2:  # PV
            P[i] = np.round(np.random.uniform(P_min, P_max, 1)).astype(int)
            Q[i] = 0
        elif t == 3:  # PQ
            P[i] = np.round(np.random.uniform(P_min, P_max, 1)).astype(int)
            Q[i] = np.round(np.random.uniform(Q_min, Q_max, 1)).astype(int)
    return P, Q


def generate_u_start_for_buses(bustype):
    """
    Initial voltage guess per bus (complex). Keep per-bus sampling loop.
    """
    u_start = np.zeros_like(bustype, dtype=complex)
    np.random.seed(time.time_ns() % (2 ** 32))
    for i, bus_typ in enumerate(bustype):
        if bus_typ in (1, 2):
            u_start[i] = np.round(np.random.uniform(0.9, 1.1), 2) + 0j
        elif bus_typ == 3:
            u_start[i] = 1.0 + 0j
        else:
            raise ValueError(f"Ungültiger Bus-Typ für Bus {i + 1}. Die Bus-Typen sind 1, 2 oder 3.")
    return u_start


def case_generation(gridtype, Bus_number, fixed, debugging, pic):
    """
    Main case generator with identical behavior and outputs, but faster internals.
    Returns:
      gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
      Y_Lines, Y_C_Lines, Lines_connected, U_base, int(S_base)
    """
    if debugging:
        Bus_number = 5

    num_connections = math.comb(Bus_number, 2)

    # Base values and per-grid parameters
    if fixed:
        np.random.seed(num_connections)

    if gridtype =="HMVN" :
        gridtype = np.random.choice(["HVN", "MVN"], p=[3 / 11, 8 / 11])

    if gridtype == "LVN":
        RX = np.random.uniform(2, 10)
        min_real, max_real = 0.05, 0.5
        min_imag, max_imag = min_real / RX, max_real / RX
        min_length, max_length = 1, 5  # km
        U_base = 400
        S_base = 1 * 1e6
    elif gridtype == "MVN":
        # min_real, max_real = 0.50, 0.55
        # min_imag, max_imag = 0.34, 0.35
        min_real, max_real = 0.5, 0.6
        min_imag, max_imag = 0.3, 0.35
        min_length, max_length = 1, 20  # km
        U_base = 10e3
        S_base = 10 * 1e6
    else:  # HVN
        min_real, max_real = 0.15, 0.2
        min_imag, max_imag = 0.35, 0.45
        min_length, max_length = 1, 50  # km
        U_base = 110e3
        S_base = 100 * 1e6

    if debugging:
        Z_Lines = np.array(
            [0.02 + 0.06j, 0.08 + 0.24j, 0, 0, 0.06 + 0.18j,
             0.06 + 0.18j, 0.04 + 0.12j, 0.01 + 0.03j, 0,
             0.08 + 0.24j], dtype=complex
        )
        Z_Base = U_base ** 2 / S_base
        Z_Lines = Z_Lines * Z_Base
    else:
        if fixed:
            np.random.seed(num_connections)
        # Draw per-connection impedances (same distributional logic as original)
        r = np.random.uniform(min_real, max_real, num_connections)
        x = np.random.uniform(min_imag, max_imag, num_connections)
        l_r = np.random.uniform(min_length, max_length, num_connections)
        # l_x = np.random.uniform(min_length, max_length, num_connections)
        Z_Lines = r * l_r + 1j * (x * l_r)

    Z_Base = U_base ** 2 / S_base
    Y_Base = 1 / Z_Base  # (kept for parity; not returned)

    # Vectorized reciprocal with zero handling
    Y_Lines = np.zeros_like(Z_Lines, dtype=complex)
    nz = (Z_Lines != 0)
    Y_Lines[nz] = 1 / Z_Lines[nz]

    # Connectivity pattern
    np.random.seed(time.time_ns() % (2 ** 32))
    if debugging:
        Lines_connected = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 1], dtype=int)
        print("Lines_conected ", Lines_connected)
    else:
        Lines_connected = np.random.randint(2, size=num_connections)

    if debugging:
        plot_random_connections(Bus_number, Lines_connected)

    is_connected = is_bus_one_connected_to_all_others(Bus_number, Lines_connected)
    Conection_matrix = create_adjacency_matrix(Bus_number, Lines_connected)

    if debugging:
        Y_C_Lines = np.array([0.06, 0.05, 0, 0, 0.04, 0.04, 0.03, 0.02, 0, 0.05], dtype=float) / Z_Base
        print("Lines_conected ", Lines_connected)
    else:
        if fixed:
            np.random.seed(num_connections)

        # Per-km shunt capacitance ranges from Oswald (F/km)
        if gridtype == "LVN":
            Cpkm_min, Cpkm_max = 20e-9, 60e-9  # 20–60 nF/km (overhead-ish)
        elif gridtype == "MVN":
            Cpkm_min, Cpkm_max = 8e-9, 14e-9  # 8–14 nF/km (Oswald 1..30 kV)
            # Cpkm_min, Cpkm_max = 0, 0  # 8–14 nF/km (Oswald 1..30 kV)
        else:  # HVN (110 kV class)
            Cpkm_min, Cpkm_max = 8e-9, 10e-9  # ~9 nF/km

        Cpkm = np.random.uniform(Cpkm_min, Cpkm_max, size=num_connections)  # F/km
        f = 50.0
        omega = 2 * np.pi * f
        # l_r is your sampled line length (km). Use it to scale shunt: B_total = ω C' L
        B_total_S = omega * Cpkm * l_r  # siemens (total per line)
        Y_C_Lines = 0.5 * B_total_S  # per-end shunt (π-model), siemens

        # Y_C_Lines = np.random.uniform(0.01, 0.10, size=num_connections) / Z_Base

    # Sum of line charging per bus
    Y_C_Bus = insert_values_in_matrix(Conection_matrix, Lines_connected, Y_C_Lines).sum(axis=0)

    # Line admittance matrix and full Y
    Line_matrix = insert_values_in_matrix_komplex(Conection_matrix, Lines_connected, Y_Lines)
    Y_matrix = build_Y_matrix(Conection_matrix, Y_C_Bus, Line_matrix)

    if debugging:
        bus_typ = np.array([1, 2, 3, 3, 3])
    else:
        bus_typ = create_bus_typ(Bus_number, fixed)

    if debugging:
        P = np.array([0, -50, -65, -35, -45])
        Q = np.array([0, 0, -10, -10, -15])
    else:
        P, Q = generate_PQ(bus_typ, gridtype)

    P, Q = P * 1e6, Q * 1e6  # to Watts/Vars
    if debugging:
        u_start = np.array([1.1 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
    else:
        u_start = generate_u_start_for_buses(bus_typ)
    u_start = u_start * U_base

    s_multi = P + 1j * Q

    if pic:
        plot_random_connections(Bus_number, Lines_connected)

    return (gridtype, bus_typ, s_multi, u_start, Y_matrix, is_connected,
            Y_Lines, Y_C_Lines, Lines_connected, U_base, int(S_base))
