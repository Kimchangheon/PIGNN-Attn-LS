#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import pandapower as pp
import pandapower.converter.cim.cim2pp.from_cim as cim2pp
from pandapower.topology import unsupplied_buses
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeYbus import makeYbus


# ============================================================
# CONFIG
# ============================================================
CGMES_FILE = r"/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/CGMES_to_PandaPower_clean/LVN_PowerFactory_fixed.zip"
CGMES_VERSION = "2.4.15"
IGNORE_ERRORS = True

RUNPP_KWARGS = dict(
    algorithm="nr",
    init="dc",
    max_iteration=100,
    tolerance_mva=1e-6,
    calculate_voltage_angles=True,
    check_connectivity=True,
    numba=False,
)

ABS_TOL = 1e-10
REL_TOL = 1e-8
TOPK_DIFFS = 20


# ============================================================
# HELPERS
# ============================================================
def to_csr_complex(x):
    if sp.issparse(x):
        return x.tocsr().astype(np.complex128)
    return sp.csr_matrix(np.asarray(x, dtype=np.complex128))


def sparse_abs_max(mat):
    if mat.nnz == 0:
        return 0.0
    return float(np.max(np.abs(mat.data)))


def rounded_array(a, decimals=12):
    return np.round(np.asarray(a, dtype=float), decimals=decimals)


def table_equal(a, b, decimals=12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    return np.array_equal(np.round(a, decimals), np.round(b, decimals))


def get_dense_if_small(mat, max_dim=5000):
    if mat.shape[0] <= max_dim and mat.shape[1] <= max_dim:
        return mat.toarray()
    return None


def largest_diff_entries(diff_csr, topk=20):
    coo = diff_csr.tocoo()
    if coo.nnz == 0:
        return pd.DataFrame(columns=["row", "col", "abs_diff", "real_diff", "imag_diff"])

    abs_vals = np.abs(coo.data)
    order = np.argsort(-abs_vals)[:topk]

    rows = []
    for idx in order:
        rows.append({
            "row": int(coo.row[idx]),
            "col": int(coo.col[idx]),
            "abs_diff": float(abs_vals[idx]),
            "real_diff": float(coo.data[idx].real),
            "imag_diff": float(coo.data[idx].imag),
        })
    return pd.DataFrame(rows)


def compare_sparse(A, B, abs_tol=1e-10, rel_tol=1e-8):
    A = to_csr_complex(A)
    B = to_csr_complex(B)

    out = {
        "shape_A": A.shape,
        "shape_B": B.shape,
        "nnz_A": int(A.nnz),
        "nnz_B": int(B.nnz),
        "same_shape": A.shape == B.shape,
    }

    if A.shape != B.shape:
        out.update({
            "exact_equal": False,
            "allclose": False,
            "max_abs_diff": np.nan,
            "fro_norm_diff": np.nan,
            "nnz_diff_matrix": np.nan,
        })
        return out, None

    D = (A - B).tocsr()
    out["nnz_diff_matrix"] = int(D.nnz)
    out["max_abs_diff"] = sparse_abs_max(D)

    if D.nnz == 0:
        out["exact_equal"] = True
        out["allclose"] = True
        out["fro_norm_diff"] = 0.0
        return out, D

    out["exact_equal"] = False
    out["fro_norm_diff"] = float(np.linalg.norm(D.data))

    # allclose on sparse pattern union
    A_dense = get_dense_if_small(A)
    B_dense = get_dense_if_small(B)
    if A_dense is not None and B_dense is not None:
        out["allclose"] = bool(np.allclose(A_dense, B_dense, atol=abs_tol, rtol=rel_tol))
    else:
        # sparse fallback: check entrywise on union
        Acoo = A.tocoo()
        Bcoo = B.tocoo()

        amap = {(int(r), int(c)): v for r, c, v in zip(Acoo.row, Acoo.col, Acoo.data)}
        bmap = {(int(r), int(c)): v for r, c, v in zip(Bcoo.row, Bcoo.col, Bcoo.data)}
        keys = set(amap.keys()) | set(bmap.keys())

        ok = True
        for k in keys:
            av = amap.get(k, 0.0 + 0.0j)
            bv = bmap.get(k, 0.0 + 0.0j)
            if not np.allclose(av, bv, atol=abs_tol, rtol=rel_tol):
                ok = False
                break
        out["allclose"] = ok

    return out, D


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# OPTIONAL: options for _pd2ppc without direct runpp dependence
# ============================================================
def set_pp_options(net_obj):
    net_obj._options = {
        "calculate_voltage_angles": True,
        "trafo_model": "t",
        "check_connectivity": False,
        "mode": "pf",
        "switch_rx_ratio": 2,
        "recycle": None,
        "delta": 0,
        "voltage_depend_loads": False,
        "trafo3w_losses": "hv",
        "init_vm_pu": "flat",
        "init_va_degree": "flat",
        "distributed_slack": False,
        "enforce_p_lims": False,
        "enforce_q_lims": False,
        "p_lim_default": 1e9,
        "q_lim_default": 1e9,
        "neglect_open_switch_branches": False,
        "consider_line_temperature": False,
        "tdpf": False,
        "tdpf_update_r_theta": False,
        "tdpf_delay_s": None,
        "numba": False,
        "lightsim2grid": False,
        "algorithm": "nr",
        "max_iteration": 10,
        "tolerance_mva": 1e-8,
        "v_debug": False,
        "sequence": None,
    }


# ============================================================
# FIXES
# ============================================================
def apply_basic_fixes(net):
    isolated = unsupplied_buses(net)

    zero_line_mask = (
        (net.line["r_ohm_per_km"] == 0) &
        (net.line["x_ohm_per_km"] == 0)
    )
    n_zero_lines = int(zero_line_mask.sum())
    if n_zero_lines > 0:
        net.line.loc[zero_line_mask, "in_service"] = False

    n_gen_vm_fixed = 0
    if len(net.gen) > 0:
        bad_vm = (net.gen["vm_pu"] < 0.8) | (net.gen["vm_pu"] > 1.2)
        n_gen_vm_fixed = int(bad_vm.sum())
        if n_gen_vm_fixed > 0:
            net.gen.loc[bad_vm, "vm_pu"] = 1.0

    return {
        "isolated": isolated,
        "n_isolated": len(isolated),
        "n_zero_lines_disabled": n_zero_lines,
        "n_gen_vm_fixed": n_gen_vm_fixed,
    }


# ============================================================
# BUILD YOUR ppcY
# ============================================================
def build_your_ppcy(net):
    net1 = copy.deepcopy(net)

    pp.runpp(net1, **RUNPP_KWARGS)

    if "_ppc" not in net1 or "internal" not in net1._ppc:
        raise RuntimeError("net._ppc['internal'] not available after runpp")

    ppc_int = net1._ppc["internal"]

    Y_ppcy = to_csr_complex(ppc_int["Ybus"])
    bus = np.asarray(ppc_int["bus"], dtype=float)
    gen = np.asarray(ppc_int.get("gen", np.empty((0, 0))), dtype=float)
    branch = np.asarray(ppc_int["branch"], dtype=float)

    lookups = copy.deepcopy(net1.get("_pd2ppc_lookups", {}))

    return {
        "net": net1,
        "ppc_int": ppc_int,
        "Ybus": Y_ppcy,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "lookups": lookups,
    }


# ============================================================
# BUILD COLLEAGUE MODEL A
# ============================================================
def build_colleague_model_a(net):
    net2 = copy.deepcopy(net)

    # same preprocessing as in your colleague code
    isolated = unsupplied_buses(net2)
    net2.bus.loc[list(isolated), "in_service"] = False

    pp.runpp(net2, **RUNPP_KWARGS)

    ppc_bb, ppci_bb = _pd2ppc(net2)
    Y_bb, _, _ = makeYbus(ppci_bb["baseMVA"], ppci_bb["bus"], ppci_bb["branch"])

    lookups = copy.deepcopy(net2.get("_pd2ppc_lookups", {}))

    return {
        "net": net2,
        "ppc": ppc_bb,
        "ppci": ppci_bb,
        "Ybus": to_csr_complex(Y_bb),
        "bus": np.asarray(ppci_bb["bus"], dtype=float),
        "gen": np.asarray(ppci_bb.get("gen", np.empty((0, 0))), dtype=float),
        "branch": np.asarray(ppci_bb["branch"], dtype=float),
        "lookups": lookups,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print_section("LOAD CGMES")
    net = cim2pp.from_cim(
        file_list=CGMES_FILE,
        cgmes_version=CGMES_VERSION,
        ignore_errors=IGNORE_ERRORS
    )
    print(f"Loaded: {CGMES_FILE}")
    print(f"Buses:    {len(net.bus)}")
    print(f"Lines:    {len(net.line)}")
    print(f"Trafo2W:  {len(net.trafo)}")
    print(f"Trafo3W:  {len(net.trafo3w)}")
    print(f"Switches: {len(net.switch)}")

    print_section("APPLY BASIC FIXES ON BASE NET")
    fix_info = apply_basic_fixes(net)
    for k, v in fix_info.items():
        if k == "isolated":
            print(f"{k}: <set of size {len(v)}>")
        else:
            print(f"{k}: {v}")

    print_section("BUILD YOUR ppcY")
    yours = build_your_ppcy(net)
    print(f"Ybus shape: {yours['Ybus'].shape}, nnz={yours['Ybus'].nnz}")
    print(f"bus shape: {yours['bus'].shape}")
    print(f"gen shape: {yours['gen'].shape}")
    print(f"branch shape: {yours['branch'].shape}")

    print_section("BUILD COLLEAGUE MODEL A")
    colleague = build_colleague_model_a(net)
    print(f"Ybus shape: {colleague['Ybus'].shape}, nnz={colleague['Ybus'].nnz}")
    print(f"bus shape: {colleague['bus'].shape}")
    print(f"gen shape: {colleague['gen'].shape}")
    print(f"branch shape: {colleague['branch'].shape}")

    print_section("COMPARE Ybus")
    ycmp, ydiff = compare_sparse(
        yours["Ybus"],
        colleague["Ybus"],
        abs_tol=ABS_TOL,
        rel_tol=REL_TOL,
    )
    for k, v in ycmp.items():
        print(f"{k}: {v}")

    print_section("COMPARE INTERNAL TABLES")
    print(f"bus table equal (round12):    {table_equal(yours['bus'], colleague['bus'])}")
    print(f"gen table equal (round12):    {table_equal(yours['gen'], colleague['gen'])}")
    print(f"branch table equal (round12): {table_equal(yours['branch'], colleague['branch'])}")

    print_section("COMPARE LOOKUPS")
    your_bus_lookup = np.asarray(yours["lookups"].get("bus", []))
    col_bus_lookup = np.asarray(colleague["lookups"].get("bus", []))

    print(f"your bus lookup len:      {len(your_bus_lookup)}")
    print(f"colleague bus lookup len: {len(col_bus_lookup)}")

    if your_bus_lookup.shape == col_bus_lookup.shape:
        same_lookup = np.array_equal(your_bus_lookup, col_bus_lookup)
        print(f"bus lookup exactly equal: {same_lookup}")
        if not same_lookup:
            diff_idx = np.where(your_bus_lookup != col_bus_lookup)[0]
            print(f"number of differing lookup entries: {len(diff_idx)}")
            print(f"first 20 differing pp bus indices: {diff_idx[:20].tolist()}")
    else:
        print("bus lookup shapes differ")

    if ydiff is not None:
        print_section(f"TOP {TOPK_DIFFS} LARGEST Ybus DIFFERENCES")
        df_top = largest_diff_entries(ydiff, topk=TOPK_DIFFS)
        if len(df_top) == 0:
            print("No differences.")
        else:
            print(df_top.to_string(index=False))

    print_section("INTERPRETATION")
    if ycmp["same_shape"] and ycmp["allclose"]:
        print("Result: your ppcY and colleague Model A are numerically the same up to tolerance.")
    elif ycmp["same_shape"]:
        print("Result: same dimension, but numerically different.")
        print("Check whether one path deactivates buses / lines differently before _pd2ppc.")
        print("The biggest clues are:")
        print("- bus lookup differences")
        print("- bus/gen/branch table differences")
        print("- largest Ybus differing entries above")
    else:
        print("Result: dimensions differ.")
        print("That usually means the two paths produced different internal fused solver models.")

    print("\nDone.")


if __name__ == "__main__":
    main()