import warnings
warnings.filterwarnings('ignore')

import pandapower as pp
import pandapower.converter.cim.cim2pp.from_cim as cim2pp
from pandapower.topology import unsupplied_buses
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeYbus import makeYbus
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.io import savemat
import copy
import time

# ============================================================
# CONFIGURATION
# ============================================================
CGMES_FILE = r'/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/CGMES_to_PandaPower_clean/LVN_PowerFactory_fixed.zip'

# ============================================================
# LOAD NETWORK
# ============================================================
print("=" * 60)
print("LOADING NETWORK")
print("=" * 60)
net = cim2pp.from_cim(file_list=CGMES_FILE, cgmes_version='2.4.15', ignore_errors=True)
print(f"Network loaded!")
print(f"  Buses:      {len(net.bus)}")
print(f"  Lines:      {len(net.line)}")
print(f"  Trafos 2W:  {len(net.trafo)}")
print(f"  Trafos 3W:  {len(net.trafo3w)}")
print(f"  Switches:   {len(net.switch)}")

# ============================================================
# FIXES
# ============================================================
isolated = unsupplied_buses(net)
net.line.loc[
    (net.line['r_ohm_per_km'] == 0) & (net.line['x_ohm_per_km'] == 0),
    'in_service'
] = False
if len(net.gen) > 0:
    net.gen.loc[(net.gen['vm_pu'] < 0.8) | (net.gen['vm_pu'] > 1.2), 'vm_pu'] = 1.0
print(f"\nFixes applied:")
print(f"  Isolated buses:          {len(isolated)}")
print(f"  Zero-impedance lines:    {((net.line['r_ohm_per_km'] == 0) & (net.line['x_ohm_per_km'] == 0)).sum()}")


# ============================================================
# HELPER: set _options (required by _pd2ppc without runpp)
# ============================================================
def set_pp_options(net_obj):
    net_obj._options = {
        'calculate_voltage_angles':     True,
        'trafo_model':                  't',
        'check_connectivity':           False,
        'mode':                         'pf',
        'switch_rx_ratio':              2,
        'recycle':                      None,
        'delta':                        0,
        'voltage_depend_loads':         False,
        'trafo3w_losses':               'hv',
        'init_vm_pu':                   'flat',
        'init_va_degree':               'flat',
        'distributed_slack':            False,
        'enforce_p_lims':               False,
        'enforce_q_lims':               False,
        'p_lim_default':                1e9,
        'q_lim_default':                1e9,
        'neglect_open_switch_branches': False,
        'consider_line_temperature':    False,
        'tdpf':                         False,
        'tdpf_update_r_theta':          False,
        'tdpf_delay_s':                 None,
        'numba':                        False,
        'lightsim2grid':                False,
        'algorithm':                    'nr',
        'max_iteration':                10,
        'tolerance_mva':                1e-8,
        'v_debug':                      False,
        'sequence':                     None,
    }


# ============================================================
# MODEL A: BUS-BRANCH Y-BUS
# Standard pandapower: closed switches merge buses
# Result: ~722 x 722
# ============================================================
print("\n" + "=" * 60)
print("MODEL A: BUS-BRANCH Y-BUS")
print("=" * 60)

net_bb = copy.deepcopy(net)
net_bb.bus.loc[list(isolated), 'in_service'] = False

pp.runpp(net_bb, algorithm='nr', init='dc', max_iteration=100,
         tolerance_mva=1e-6, calculate_voltage_angles=True,
         check_connectivity=True, numba=False)
print("  Power flow converged!")

ppc_bb, ppci_bb = _pd2ppc(net_bb)
Ybus_bb, _, _   = makeYbus(ppci_bb['baseMVA'], ppci_bb['bus'], ppci_bb['branch'])

# Bus mapping: pandapower index → ppci index
bus_lookup_bb = net_bb._pd2ppc_lookups['bus']
active_pp_idx = net_bb.bus[net_bb.bus['in_service']].index
bm_bb = net_bb.bus.loc[active_pp_idx, ['name', 'vn_kv']].copy()
bm_bb['ppci_idx'] = bus_lookup_bb[active_pp_idx]
bm_bb = bm_bb[bm_bb['ppci_idx'] >= 0].sort_values('ppci_idx').reset_index()
bm_bb.rename(columns={'index': 'pp_idx'}, inplace=True)

print(f"  Dimension:   {Ybus_bb.shape[0]} x {Ybus_bb.shape[1]}")
print(f"  Non-zero:    {Ybus_bb.nnz}")
print(f"  Bus mapping: {len(bm_bb)} active buses")


# ============================================================
# MODEL B: NODE-BREAKER Y-BUS
# Every connectivity node = one matrix row/column
# Switches are explicit admittances (not merged)
# Result: ~3798 x 3798
# ============================================================
print("\n" + "=" * 60)
print("MODEL B: NODE-BREAKER Y-BUS")
print("=" * 60)

# --- Identify switch-only buses ---
# A switch-only bus has NO line/trafo/impedance connection.
# _pd2ppc removes these when all switches are open → need dummy anchors.
connected_via_branch = set()
for _, r in net.line[net.line['in_service']].iterrows():
    connected_via_branch.update([r['from_bus'], r['to_bus']])
for _, r in net.trafo[net.trafo['in_service']].iterrows():
    connected_via_branch.update([r['hv_bus'], r['lv_bus']])
for _, r in net.trafo3w[net.trafo3w['in_service']].iterrows():
    connected_via_branch.update([r['hv_bus'], r['mv_bus'], r['lv_bus']])
for _, r in net.impedance[net.impedance['in_service']].iterrows():
    connected_via_branch.update([r['from_bus'], r['to_bus']])
for _, r in net.ext_grid[net.ext_grid['in_service']].iterrows():
    connected_via_branch.add(r['bus'])

sw_bb       = net.switch[net.switch['et'] == 'b']
sw_buses    = set(sw_bb['bus'].tolist() + sw_bb['element'].tolist())
all_buses   = set(net.bus.index)
switch_only = (all_buses - connected_via_branch) & sw_buses
need_anchor = switch_only | (isolated - connected_via_branch)

print(f"  Switch-only buses needing dummy anchor: {len(need_anchor)}")

# --- Build static network (all switches open + dummy anchors) ---
net_nb = copy.deepcopy(net)

# Disable loads/gens at isolated buses (power balance not needed for Y-bus)
for table in ['load', 'gen', 'sgen', 'ext_grid']:
    if len(net_nb[table]) > 0:
        mask = net_nb[table]['bus'].isin(isolated)
        net_nb[table].loc[mask, 'in_service'] = False

# Dummy slack at every isolated bus (required by _pd2ppc)
for b in isolated:
    pp.create_ext_grid(net_nb, bus=b, vm_pu=1.0, va_degree=0.0,
                       in_service=True, name=f"DUMMY_SLACK_{b}")

# Dummy anchor: high-impedance branch keeps switch-only buses in _pd2ppc
sw_neighbor = {}
for _, sw in sw_bb.iterrows():
    b1, b2 = sw['bus'], sw['element']
    if b1 not in sw_neighbor: sw_neighbor[b1] = b2
    if b2 not in sw_neighbor: sw_neighbor[b2] = b1

for bus in need_anchor:
    neighbor = sw_neighbor.get(bus)
    if neighbor is not None:
        # Impedance 1e8 pu → Y = 1e-8 pu → physically negligible
        pp.create_impedance(net_nb, from_bus=bus, to_bus=neighbor,
                            rft_pu=0.0, xft_pu=1e8,
                            rtf_pu=0.0, xtf_pu=1e8,
                            sn_mva=net_nb.sn_mva, in_service=True,
                            name=f"DUMMY_IMP_{bus}")
    else:
        pp.create_shunt(net_nb, bus=bus, p_mw=0.0, q_mvar=-1e-8,
                        name=f"DUMMY_SHUNT_{bus}")

# Save original switch states and open all switches
# → prevents any bus merging inside _pd2ppc
original_closed = sw_bb['closed'].copy()
net_nb.switch.loc[sw_bb.index, 'closed'] = False

# Build static Y-bus (lines + trafos + impedances, NO switches)
set_pp_options(net_nb)
t0 = time.time()
ppc_nb, ppci_nb = _pd2ppc(net_nb)
Ybus_static, _, _ = makeYbus(ppci_nb['baseMVA'], ppci_nb['bus'], ppci_nb['branch'])
t_static = time.time() - t0

bus_lookup_nb = net_nb._pd2ppc_lookups['bus']
n_total       = Ybus_static.shape[0]
n_buses       = len(net.bus)
n_t3w         = len(net.trafo3w)

print(f"  Static Y-bus (pandapower, no switches): {Ybus_static.shape[0]} x {Ybus_static.shape[1]}")
print(f"  = {n_buses} buses + {n_t3w} trafo3w star nodes")
print(f"  Build time (once): {t_static:.2f}s")


# --- Switch admittance matrix ---
# open  → no entry (exactly 0, physically correct)
# closed → small impedance z = 1e-9 + j1e-6 (not exactly 0 → avoids singularity)
def build_switch_matrix(switch_states):
    """
    Builds sparse admittance matrix for switches only.
    Called every time topology changes – takes only milliseconds.
    switch_states: pandas Series {switch_idx: True/False}
    """
    rows, cols, vals = [], [], []
    for sw_idx, closed in switch_states.items():
        sw = net.switch.loc[sw_idx]
        if sw['et'] != 'b':
            continue
        i = bus_lookup_nb[sw['bus']]
        j = bus_lookup_nb[sw['element']]
        if i < 0 or j < 0 or i >= n_total or j >= n_total:
            continue
        if not closed:
            continue                          # open switch = 0, no entry needed
        y = 1.0 / complex(1e-9, 1e-6)        # closed = near short-circuit
        rows.extend([i, j, i, j])
        cols.extend([i, j, j, i])
        vals.extend([y, y, -y, -y])
    if not vals:
        return sparse.csr_matrix((n_total, n_total), dtype=complex)
    return sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_total, n_total)
    ).tocsr()


def get_Ybus_nb(switch_states):
    """
    Returns full node-breaker Y-bus for given switch states.
    Y_total = Y_static (lines/trafos) + Y_switches
    Only Y_switches is rebuilt on topology change.
    """
    return Ybus_static + build_switch_matrix(switch_states)


# Build nominal Y-bus (original switch states from CGMES)
t0 = time.time()
Ybus_nb    = get_Ybus_nb(original_closed)
t_nominal  = time.time() - t0

# Bus mapping for node-breaker
bm_nb = net_nb.bus[['name', 'vn_kv']].copy()
bm_nb['pp_idx']   = bm_nb.index
bm_nb['ppci_idx'] = bus_lookup_nb[bm_nb.index]
bm_nb['isolated'] = bm_nb.index.isin(isolated)
bm_nb['sw_only']  = bm_nb.index.isin(switch_only)
bm_nb = bm_nb[bm_nb['ppci_idx'] >= 0].sort_values('ppci_idx').reset_index(drop=True)

print(f"  Dimension (nominal):   {Ybus_nb.shape[0]} x {Ybus_nb.shape[1]}")
print(f"  Non-zero:              {Ybus_nb.nnz}")
print(f"  Switch update time:    {t_nominal*1000:.1f}ms")
print(f"  Bus mapping: {len(bm_nb)} buses  "
      f"(active: {(~bm_nb['isolated']).sum()}, isolated: {bm_nb['isolated'].sum()})")


# ============================================================
# CHECK: Original switch states through _pd2ppc → must equal bus-branch
# ============================================================
print("\n" + "=" * 60)
print("CHECK: ORIGINAL SWITCH STATES → _pd2ppc → should give 722×722")
print("=" * 60)

net_check = copy.deepcopy(net)
net_check.bus.loc[list(isolated), 'in_service'] = False
# No switch manipulation → keeps original states (2600 closed, 1030 open)

pp.runpp(net_check, algorithm='nr', init='dc', max_iteration=100,
         tolerance_mva=1e-6, calculate_voltage_angles=True,
         check_connectivity=True, numba=False)

ppc_check, ppci_check = _pd2ppc(net_check)
Ybus_check, _, _ = makeYbus(
    ppci_check['baseMVA'], ppci_check['bus'], ppci_check['branch']
)

print(f"  Bus-branch reference:                {Ybus_bb.shape[0]} x {Ybus_bb.shape[1]}")
print(f"  Check (original switches, _pd2ppc):  {Ybus_check.shape[0]} x {Ybus_check.shape[1]}")

diff_check = abs(Ybus_check.shape[0] - Ybus_bb.shape[0])
if diff_check == 0:
    print(f"  ✅ Perfect match!")
else:
    print(f"  ⚠️  Difference: {diff_check}")

print(f"\n  Switch states:")
print(f"    closed: {original_closed.sum()}")
print(f"    open:   {(~original_closed).sum()}")


# ============================================================
# CONSISTENCY CHECK: T.T @ Ybus_nb @ T ≈ Ybus_bb
# Aggregation matrix T maps nb nodes → bb nodes
# If consistent: reducing node-breaker model gives bus-branch model
# ============================================================
print("\n" + "=" * 60)
print("CONSISTENCY CHECK: T.T @ Ybus_nb @ T ≈ Ybus_bb")
print("=" * 60)

n_nb = Ybus_nb.shape[0]
n_bb = Ybus_bb.shape[0]

# Build nb→bb mapping
nb_to_bb = {}

# 1) Regular buses via pp_idx as common key
for _, row in bm_nb.iterrows():
    pp_idx  = row['pp_idx']
    ppci_nb = row['ppci_idx']
    if ppci_nb < 0 or ppci_nb >= n_nb:
        continue
    if pp_idx not in net_bb.bus.index:
        continue
    if not net_bb.bus.loc[pp_idx, 'in_service']:
        continue
    ppci_bb = bus_lookup_bb[pp_idx]
    if ppci_bb < 0 or ppci_bb >= n_bb:
        continue
    nb_to_bb[ppci_nb] = ppci_bb

# 2) trafo3w star nodes (last n_t3w entries in both matrices)
for k in range(n_t3w):
    ppci_nb_star = n_nb - n_t3w + k
    ppci_bb_star = n_bb - n_t3w + k
    nb_to_bb[ppci_nb_star] = ppci_bb_star

print(f"  Mapped nb→bb pairs:    {len(nb_to_bb)}")
print(f"  Unmapped (isolated):   {n_nb - len(nb_to_bb)}")

# Build aggregation matrix T (n_nb × n_bb)
T_rows = list(nb_to_bb.keys())
T_cols = list(nb_to_bb.values())
T_vals = [1.0] * len(T_rows)

T = sparse.coo_matrix(
    (T_vals, (T_rows, T_cols)),
    shape=(n_nb, n_bb)
).tocsr()

# Kron reduction
Ybus_nb_csr  = Ybus_nb.tocsr().astype(complex)
Ybus_reduced = (T.T @ Ybus_nb_csr @ T).toarray()
Ybus_bb_arr  = Ybus_bb.toarray()

# Compare
diff_abs = np.abs(Ybus_reduced - Ybus_bb_arr)

both_nonzero = (np.abs(Ybus_bb_arr)  > 1e-10) & \
               (np.abs(Ybus_reduced)  > 1e-10)
only_in_bb   = (np.abs(Ybus_bb_arr)  > 1e-10) & \
               (np.abs(Ybus_reduced) <= 1e-10)
only_in_red  = (np.abs(Ybus_bb_arr) <= 1e-10) & \
               (np.abs(Ybus_reduced)  > 1e-10)

diff_rel = np.zeros_like(diff_abs)
diff_rel[both_nonzero] = (diff_abs[both_nonzero] /
                          np.abs(Ybus_bb_arr[both_nonzero]))

diag_bb  = np.diag(Ybus_bb_arr)
diag_red = np.diag(Ybus_reduced)
diag_nz  = np.abs(diag_bb) > 1e-10
diag_rel = np.abs(diag_bb[diag_nz] - diag_red[diag_nz]) / np.abs(diag_bb[diag_nz])

print(f"\n  Entries in both (non-zero):       {both_nonzero.sum()}")
print(f"  Only in Ybus_bb (missing):        {only_in_bb.sum()}")
print(f"  Only in Ybus_red (extra):         {only_in_red.sum()}")
print(f"\n  Relative diff (both non-zero):")
print(f"    Max:    {diff_rel[both_nonzero].max():.4e}")
print(f"    Mean:   {diff_rel[both_nonzero].mean():.4e}")
print(f"    Median: {np.median(diff_rel[both_nonzero]):.4e}")
print(f"\n  Diagonal (self-admittances):")
print(f"    Max rel diff:  {diag_rel.max():.4e}")
print(f"    Mean rel diff: {diag_rel.mean():.4e}")

# Verdict
tol = 1e-2
max_rel = diff_rel[both_nonzero].max() if both_nonzero.any() else 0.0
print(f"\n  Verdict:")
if only_in_bb.sum() == 0 and max_rel < tol:
    print(f"  ✅ PASS: max rel diff = {max_rel:.2e} < {tol:.0e}")
    print(f"     Ybus_nb is physically consistent with Ybus_bb")
elif only_in_bb.sum() == 0:
    print(f"  ⚠️  Structural match, numerical diff = {max_rel:.2e}")
    print(f"     Expected: switch impedance approximation (1e-6 pu)")
else:
    print(f"  ❌ {only_in_bb.sum()} entries missing → mapping incomplete")

    # Show which bb nodes are not covered
    bb_covered  = set(nb_to_bb.values())
    miss_rows_i, miss_cols_i = np.where(only_in_bb)
    missing_bb  = set(miss_rows_i.tolist() + miss_cols_i.tolist()) - bb_covered
    print(f"     BB ppci nodes with no nb mapping: {sorted(missing_bb)[:10]}")


# ============================================================
# EXPORT
# ============================================================
print("\n" + "=" * 60)
print("EXPORT")
print("=" * 60)

# Model A: bus-branch
sparse.save_npz('Ybus_bus_branch.npz', Ybus_bb)
bm_bb[['pp_idx', 'ppci_idx', 'name', 'vn_kv']].to_csv(
    'bus_mapping_bus_branch.csv', index=False, encoding='utf-8-sig')

# Model B: node-breaker
sparse.save_npz('Ybus_node_breaker.npz', Ybus_nb)
sparse.save_npz('Ybus_static.npz',       Ybus_static)
bm_nb[['pp_idx', 'ppci_idx', 'name', 'vn_kv',
        'isolated', 'sw_only']].to_csv(
    'bus_mapping_node_breaker.csv', index=False, encoding='utf-8-sig')

# Switch list with matrix indices
sw_export = sw_bb[['bus', 'element', 'closed']].copy()
sw_export['from_ppci'] = bus_lookup_nb[sw_export['bus'].values]
sw_export['to_ppci']   = bus_lookup_nb[sw_export['element'].values]
sw_export.to_csv('switch_list.csv', encoding='utf-8-sig')

# MATLAB export
savemat('Ybus_both.mat', {
    'Ybus_bus_branch':   Ybus_bb,
    'Ybus_node_breaker': Ybus_nb,
    'Ybus_static':       Ybus_static,
})

print(f"✅ Ybus_bus_branch.npz        ({Ybus_bb.shape[0]}×{Ybus_bb.shape[1]})")
print(f"✅ Ybus_node_breaker.npz      ({Ybus_nb.shape[0]}×{Ybus_nb.shape[1]})")
print(f"✅ Ybus_static.npz            ({Ybus_static.shape[0]}×{Ybus_static.shape[1]})")
print(f"✅ bus_mapping_bus_branch.csv")
print(f"✅ bus_mapping_node_breaker.csv")
print(f"✅ switch_list.csv")
print(f"✅ Ybus_both.mat")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
  Model A – Bus-Branch:
    Dimension:       {Ybus_bb.shape[0]:>5} x {Ybus_bb.shape[1]}
    Non-zero:        {Ybus_bb.nnz:>5}
    Description:     Closed switches merge buses (standard pandapower)

  Model B – Node-Breaker:
    Dimension:       {Ybus_nb.shape[0]:>5} x {Ybus_nb.shape[1]}
    Non-zero:        {Ybus_nb.nnz:>5}
    Description:     Every connectivity node = one matrix row/column

    Static part:     {Ybus_static.shape[0]:>5} x {Ybus_static.shape[1]}  (built once: {t_static:.2f}s)
    Switch update:   {t_nominal*1000:>5.1f} ms per topology change

    Usage:
      Ybus = get_Ybus_nb(switch_states)
      where switch_states is a Series {{switch_idx: True/False}}

  Check:
    Original switches → _pd2ppc → {Ybus_check.shape[0]}×{Ybus_check.shape[1]} == {Ybus_bb.shape[0]}×{Ybus_bb.shape[1]}?  {'✅' if diff_check == 0 else '⚠️'}

  Consistency:
    T.T @ Ybus_nb @ T ≈ Ybus_bb?  max rel diff = {max_rel:.2e}  {'✅' if max_rel < tol else '⚠️'}
""")
