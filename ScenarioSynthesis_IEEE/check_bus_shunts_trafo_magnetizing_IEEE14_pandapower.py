import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.idx_bus import GS, BS

def main():
    # --- Build IEEE-14 network
    net = pn.case14()

    print("=== Pandapower-level shunts ===")
    if len(net.shunt):
        print(net.shunt)
    else:
        print("net.shunt is empty -> no explicit shunt elements.")

    # --- Run a (loose) power flow to populate net._ppc
    try:
        pp.runpp(net, init="flat", max_iteration=1, enforce_q_lims=False, tolerance_mva=1e9)
    except pp.powerflow.LoadflowNotConverged:
        # We don't care about convergence here, just want the ppc structure
        print("Loadflow did not converge (ignored).")

    ppc_int = net._ppc["internal"]
    bus_ppc = ppc_int["bus"]

    GS_ppc = bus_ppc[:, GS]
    BS_ppc = bus_ppc[:, BS]

    print("\n=== GS / BS from internal ppc (per bus) ===")
    for i, (gs, bs) in enumerate(zip(GS_ppc, BS_ppc)):
        print(f"Bus {i}: GS = {gs:.6g}, BS = {bs:.6g}")

    print("\nCheck if all GS, BS are ~0:")
    print("allclose(GS_ppc, 0) ->", np.allclose(GS_ppc, 0.0))
    print("allclose(BS_ppc, 0) ->", np.allclose(BS_ppc, 0.0))

    # --- Transformer magnetizing parameters at pandapower level
    print("\n=== Transformer magnetizing parameters (pandapower net.trafo) ===")
    if len(net.trafo):
        cols = ["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv",
                "vk_percent", "vkr_percent", "pfe_kw", "i0_percent"]
        print(net.trafo[cols])

        pfe = net.trafo["pfe_kw"].to_numpy(dtype=float)
        i0  = net.trafo["i0_percent"].to_numpy(dtype=float)

        # Replace NaN with 0 for the check (NaN means "not specified")
        pfe_check = np.nan_to_num(pfe)
        i0_check  = np.nan_to_num(i0)

        print("\nCheck if pfe_kw and i0_percent are effectively zero:")
        print("pfe_kw:", pfe)
        print("i0_percent:", i0)
        print("allclose(pfe_kw (NaN->0), 0) ->", np.allclose(pfe_check, 0.0))
        print("allclose(i0_percent (NaN->0), 0) ->", np.allclose(i0_check, 0.0))
    else:
        print("No transformers defined in net.trafo (which would be weird for case14).")

if __name__ == "__main__":
    main()