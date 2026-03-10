"""
check_ppc_magnetizing_propagation.py

the cases where net.trafo rows =0
case4gs
case5
case6ww
case9
case30 (PYPOWER-derived Washington 30 bus dynamic test case)
case33bw

For each pandapower test case listed, this script checks:

1) Does net.trafo exist? (attribute + number of rows)
2) If we force nonzero trafo magnetizing values (pfe_kw, i0_percent),
   do the PPC shunt columns change?

We report deltas for:
- bus shunts: GS, BS
- branch shunts: BR_G, BR_B, BR_G_ASYM, BR_B_ASYM (if present in your pandapower version)

Run:
  python check_ppc_magnetizing_propagation.py
"""

import numpy as np

import pandapower as pp
import pandapower.networks as pn
from pandapower.powerflow import LoadflowNotConverged

from pandapower.pypower.idx_bus import GS, BS
from pandapower.pypower.idx_brch import BR_B, BR_STATUS, TAP, SHIFT, F_BUS, T_BUS

# Optional pandapower-extended branch shunt columns (may not exist in older versions)
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


def _get_col(arr: np.ndarray, idx, default=0.0) -> np.ndarray:
    """Safely get a column from a 2D numpy array, or return a default vector."""
    if idx is None:
        return np.full(arr.shape[0], default, dtype=float)
    if idx >= arr.shape[1]:
        return np.full(arr.shape[0], default, dtype=float)
    return arr[:, idx].astype(float, copy=True)


def snapshot_ppc_shunts(net):
    """
    Run a very light powerflow (or attempt to) to build net._ppc["internal"],
    then return the shunt-related PPC columns we care about.
    """
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
        # PPC is usually still built; continue
        pass
    except Exception:
        # still try to read ppc if it exists
        pass

    if "_ppc" not in net or "internal" not in net._ppc:
        return None

    ppc_int = net._ppc["internal"]
    bus = np.asarray(ppc_int["bus"], dtype=float)
    br = np.asarray(ppc_int["branch"], dtype=float)

    out = {
        "nb": int(bus.shape[0]),
        "nl": int(br.shape[0]),
        "GS": _get_col(bus, GS, 0.0),  # MW in PPC convention
        "BS": _get_col(bus, BS, 0.0),  # MVAr in PPC convention
        "BR_B": _get_col(br, BR_B, 0.0),  # pu
        "BR_G": _get_col(br, BR_G, 0.0),  # pu (if present)
        "BR_B_ASYM": _get_col(br, BR_B_ASYM, 0.0),  # pu (if present)
        "BR_G_ASYM": _get_col(br, BR_G_ASYM, 0.0),  # pu (if present)
        "BR_STATUS": _get_col(br, BR_STATUS, 0.0),
        "TAP": _get_col(br, TAP, 0.0),
        "SHIFT": _get_col(br, SHIFT, 0.0),
        "F_BUS": _get_col(br, F_BUS, 0.0),
        "T_BUS": _get_col(br, T_BUS, 0.0),
    }
    return out


def make_case(name: str):
    """
    Construct a pandapower network for a given case name, using default args.
    Returns net or None if not available.
    """
    fn = getattr(pn, name, None)
    if fn is None:
        return None
    try:
        return fn()
    except TypeError:
        # Some cases accept kwargs; try default anyway
        return fn(**{})
    except Exception:
        return None


def apply_magnetizing(net, pfe_kw=50.0, i0_percent=2.0):
    """
    Force nonzero magnetizing parameters onto net.trafo if it exists and has rows.
    Creates columns if missing.
    """
    if not hasattr(net, "trafo"):
        return False
    if net.trafo is None or len(net.trafo) == 0:
        return False

    net.trafo.loc[:, "pfe_kw"] = float(pfe_kw)
    net.trafo.loc[:, "i0_percent"] = float(i0_percent)
    return True


def main():
    tol_bus = 1e-9
    tol_br = 1e-12

    print("=== PPC magnetizing propagation check ===")
    print(f"Optional branch columns present in this pandapower build:")
    print(f"  BR_G: {'yes' if BR_G is not None else 'no'}")
    print(f"  BR_B_ASYM: {'yes' if BR_B_ASYM is not None else 'no'}")
    print(f"  BR_G_ASYM: {'yes' if BR_G_ASYM is not None else 'no'}")
    print()

    for name in CASES:
        net0 = make_case(name)
        if net0 is None:
            print(f"[{name}]  SKIP (not available / failed to build)")
            continue

        has_trafo_attr = hasattr(net0, "trafo")
        n_trafo0 = int(len(net0.trafo)) if has_trafo_attr and net0.trafo is not None else 0

        snap0 = snapshot_ppc_shunts(net0)
        if snap0 is None:
            print(f"[{name}]  SKIP (ppc_int not available after runpp)")
            continue

        # Build modified net
        net1 = make_case(name)
        if net1 is None:
            print(f"[{name}]  SKIP (could not rebuild for modified test)")
            continue

        has_trafo_attr_1 = hasattr(net1, "trafo")
        n_trafo1 = int(len(net1.trafo)) if has_trafo_attr_1 and net1.trafo is not None else 0
        did_apply = apply_magnetizing(net1, pfe_kw=50.0, i0_percent=2.0)

        snap1 = snapshot_ppc_shunts(net1)
        if snap1 is None:
            print(f"[{name}]  SKIP (ppc_int missing for modified net)")
            continue

        # Compute deltas
        dGS = snap1["GS"] - snap0["GS"]
        dBS = snap1["BS"] - snap0["BS"]
        dBRB = snap1["BR_B"] - snap0["BR_B"]
        dBRG = snap1["BR_G"] - snap0["BR_G"]
        dBRB_A = snap1["BR_B_ASYM"] - snap0["BR_B_ASYM"]
        dBRG_A = snap1["BR_G_ASYM"] - snap0["BR_G_ASYM"]

        max_dGS = float(np.max(np.abs(dGS))) if dGS.size else 0.0
        max_dBS = float(np.max(np.abs(dBS))) if dBS.size else 0.0
        max_dBRB = float(np.max(np.abs(dBRB))) if dBRB.size else 0.0
        max_dBRG = float(np.max(np.abs(dBRG))) if dBRG.size else 0.0
        max_dBRB_A = float(np.max(np.abs(dBRB_A))) if dBRB_A.size else 0.0
        max_dBRG_A = float(np.max(np.abs(dBRG_A))) if dBRG_A.size else 0.0

        nz_gs = int(np.sum(np.abs(dGS) > tol_bus))
        nz_bs = int(np.sum(np.abs(dBS) > tol_bus))
        nz_brb = int(np.sum(np.abs(dBRB) > tol_br))
        nz_brg = int(np.sum(np.abs(dBRG) > tol_br))
        nz_brb_a = int(np.sum(np.abs(dBRB_A) > tol_br))
        nz_brg_a = int(np.sum(np.abs(dBRG_A) > tol_br))

        changed_bus = (max_dGS > tol_bus) or (max_dBS > tol_bus)
        changed_branch = (max_dBRB > tol_br) or (max_dBRG > tol_br) or (max_dBRB_A > tol_br) or (max_dBRG_A > tol_br)

        print(f"[{name}]")
        print(f"  buses={snap0['nb']}, branches={snap0['nl']}")
        print(f"  net.trafo exists={has_trafo_attr}, rows={n_trafo0}  | applied pfe/i0? {did_apply}")
        print(f"  Δ bus shunts:   max|ΔGS|={max_dGS:.6g} (nz={nz_gs}), max|ΔBS|={max_dBS:.6g} (nz={nz_bs})")
        print(f"  Δ branch shunts:max|ΔBR_B|={max_dBRB:.6g} (nz={nz_brb}), max|ΔBR_G|={max_dBRG:.6g} (nz={nz_brg})")
        print(f"                 max|ΔBR_B_ASYM|={max_dBRB_A:.6g} (nz={nz_brb_a}), max|ΔBR_G_ASYM|={max_dBRG_A:.6g} (nz={nz_brg_a})")
        print(f"  RESULT: bus_shunts_changed={changed_bus}, branch_shunts_changed={changed_branch}")
        print()

    print("=== done ===")


if __name__ == "__main__":
    main()