import os
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.converter.cim.cim2pp.from_cim as cim2pp
from pandapower.topology import unsupplied_buses


# ============================================================
# Pandas display settings (show full tables)
# ============================================================
pd.set_option("display.max_columns", None)      # show all columns
pd.set_option("display.width", 2000)            # large width to avoid wrapping
pd.set_option("display.max_colwidth", None)     # do not truncate long strings (important for RDFID)
pd.set_option("display.expand_frame_repr", False)  # prevent line breaks

# ============================================================
# 1. Load CGMES model
# ============================================================
cgmes_file = r"/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS/CGMES_to_PandaPower_clean/LVN_PowerFactory_fixed.zip"

net = cim2pp.from_cim(
    file_list=cgmes_file,
    cgmes_version="2.4.15",
    ignore_errors=True
)

print("Network successfully loaded!")

# ============================================================
# Helper functions
# ============================================================
def print_section(title: str):
    """Print a clearly separated section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def get_rdfid_column(df: pd.DataFrame):
    """
    Try to find the CGMES RDFID / mRID column.
    Depending on the converter version, the identifier may have different names.
    """
    candidates = ["origin_id", "cim_id", "rdf_id", "uuid"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_identifier_columns(df: pd.DataFrame):
    """
    Return a prioritized list of useful identifier columns for comparison.
    The RDFID-like column is preferred, followed by name-like columns.
    """
    rdf_col = get_rdfid_column(df)
    candidates = []

    if rdf_col is not None:
        candidates.append(rdf_col)

    for col in ["name", "origin_id", "cim_id", "rdf_id", "uuid", "id"]:
        if col in df.columns and col not in candidates:
            candidates.append(col)

    return candidates


def build_display_df(df: pd.DataFrame, base_cols: list, include_pp_index: bool = True) -> pd.DataFrame:
    """
    Build a display DataFrame that always includes:
    - pandapower index
    - RDFID (if available)
    - name (if available)
    plus selected base columns.

    This ensures consistent comparison with external tools like PowerFactory.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()

    selected_cols = []

    # Always include pandapower index
    if include_pp_index:
        out.insert(0, "pp_index", out.index)
        selected_cols.append("pp_index")

    # Add RDFID-like column (preferred)
    rdf_col = get_rdfid_column(out)
    if rdf_col is not None and rdf_col not in selected_cols:
        selected_cols.append(rdf_col)

    # Always include name if available (NOT only as fallback)
    if "name" in out.columns and "name" not in selected_cols:
        selected_cols.append("name")

    # Add other identifier-like columns (optional but useful)
    for col in ["cim_id", "rdf_id", "uuid", "id"]:
        if col in out.columns and col not in selected_cols:
            selected_cols.append(col)

    # Add requested base columns
    for col in base_cols:
        if col in out.columns and col not in selected_cols:
            selected_cols.append(col)

    return out[selected_cols]


def print_element_preview(df: pd.DataFrame, title: str, base_cols: list, n: int = 10):
    """
    Print the first n rows of a DataFrame with identifiers and selected columns.
    """
    print_subsection(title)

    if df is None or len(df) == 0:
        print("No entries available.")
        return

    display_df = build_display_df(df, base_cols=base_cols, include_pp_index=True)
    print(display_df.head(n))


def print_available_columns(df: pd.DataFrame, element_name: str):
    """
    Print all available columns for a given element table.
    This is useful when inspecting CGMES import mappings.
    """
    print_subsection(f"Available columns in net.{element_name}")
    if df is None:
        print("Element table is None.")
        return
    print(list(df.columns))


# ============================================================
# 2. Initial data preview for comparison
# ============================================================
print_section("INITIAL DATA PREVIEW FOR COMPARISON")

print_element_preview(
    net.bus,
    title="First 10 buses",
    base_cols=["vn_kv", "type", "zone", "in_service"]
)

print_element_preview(
    net.load,
    title="First 10 loads",
    base_cols=["bus", "p_mw", "q_mvar", "scaling", "in_service"]
)

print_element_preview(
    net.gen,
    title="First 10 generators",
    base_cols=["bus", "p_mw", "vm_pu", "slack", "in_service"]
)

print_element_preview(
    net.sgen,
    title="First 10 static generators",
    base_cols=["bus", "p_mw", "q_mvar", "scaling", "in_service"]
)

print_element_preview(
    net.line,
    title="First 10 lines",
    base_cols=["from_bus", "to_bus", "length_km", "r_ohm_per_km", "x_ohm_per_km", "max_i_ka", "in_service"]
)

print_element_preview(
    net.trafo,
    title="First 10 transformers",
    base_cols=["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "in_service"]
)

# Optional debug output to inspect imported CGMES mappings
print_available_columns(net.bus, "bus")
print_available_columns(net.load, "load")
print_available_columns(net.gen, "gen")
print_available_columns(net.sgen, "sgen")
print_available_columns(net.line, "line")
print_available_columns(net.trafo, "trafo")

# ============================================================
# 3. Diagnostics
# ============================================================
print_section("DIAGNOSTICS")

# 3a) Analyze switches
print_subsection("Switch overview")
print(f"Total number of switches: {len(net.switch)}")

if len(net.switch) > 0 and "closed" in net.switch.columns:
    print(f"Closed switches: {net.switch['closed'].sum()}")
    print(f"Open switches:   {(~net.switch['closed']).sum()}")

if len(net.switch) > 0 and "et" in net.switch.columns:
    print("\nSwitch element types:")
    print(net.switch["et"].value_counts())

# 3b) Find unsupplied buses
print_subsection("Unsupplied bus check")
try:
    isolated = unsupplied_buses(net)
    print(f"Number of unsupplied buses: {len(isolated)}")
    if len(isolated) > 0:
        print(f"First 10 unsupplied bus indices: {list(isolated)[:10]}")
except Exception as e:
    print(f"Topology check failed: {e}")
    isolated = set()

# 3c) Check out-of-service elements
print_subsection("Out-of-service elements")
print(f"Buses:        {(~net.bus['in_service']).sum() if 'in_service' in net.bus.columns else 'N/A'}")
print(f"Lines:        {(~net.line['in_service']).sum() if 'in_service' in net.line.columns else 'N/A'}")
print(f"Transformers: {(~net.trafo['in_service']).sum() if 'in_service' in net.trafo.columns else 'N/A'}")
print(f"Loads:        {(~net.load['in_service']).sum() if 'in_service' in net.load.columns else 'N/A'}")
print(f"Generators:   {(~net.gen['in_service']).sum() if 'in_service' in net.gen.columns else 'N/A'}")
print(f"Static gens:  {(~net.sgen['in_service']).sum() if 'in_service' in net.sgen.columns else 'N/A'}")

# 3d) Check voltage levels
print_subsection("Voltage levels")
if len(net.bus) > 0 and "vn_kv" in net.bus.columns:
    print(net.bus["vn_kv"].value_counts().sort_index())

# 3e) Check external grids / slack buses
print_subsection("External grid / slack buses")
if len(net.ext_grid) > 0:
    ext_grid_cols = [c for c in ["bus", "vm_pu", "va_degree", "in_service"] if c in net.ext_grid.columns]
    print(net.ext_grid[ext_grid_cols])

    if "bus" in net.ext_grid.columns and "vn_kv" in net.bus.columns:
        valid_buses = [b for b in net.ext_grid["bus"] if b in net.bus.index]
        print("\nAssociated nominal bus voltages:")
        print(net.bus.loc[valid_buses, "vn_kv"])
else:
    print("No external grid elements found.")

# 3f) Generator checks
print_subsection("Generator checks")
if len(net.gen) > 0:
    if "p_mw" in net.gen.columns:
        print(f"Total active power of generators: {net.gen['p_mw'].sum():.6f} MW")
    if "vm_pu" in net.gen.columns:
        invalid_vm = net.gen[(net.gen["vm_pu"] < 0.8) | (net.gen["vm_pu"] > 1.2)]
        print(f"Generators with suspicious vm_pu outside [0.8, 1.2]: {len(invalid_vm)}")
else:
    print("No generators found.")

# 3g) Static generator checks
print_subsection("Static generator checks")
if len(net.sgen) > 0:
    if "p_mw" in net.sgen.columns:
        print(f"Total active power of static generators: {net.sgen['p_mw'].sum():.6f} MW")
    if "q_mvar" in net.sgen.columns:
        print(f"Total reactive power of static generators: {net.sgen['q_mvar'].sum():.6f} MVAr")
else:
    print("No static generators found.")

# 3h) Load checks
print_subsection("Load checks")
if len(net.load) > 0:
    total_p = net.load["p_mw"].sum() if "p_mw" in net.load.columns else np.nan
    total_q = net.load["q_mvar"].sum() if "q_mvar" in net.load.columns else np.nan
    negative_loads = (net.load["p_mw"] < 0).sum() if "p_mw" in net.load.columns else "N/A"
    print(f"Total load: P = {total_p:.6f} MW, Q = {total_q:.6f} MVAr")
    print(f"Loads with negative active power: {negative_loads}")
else:
    print("No loads found.")

# ============================================================
# 4. Apply robustness fixes
# ============================================================
print_section("APPLYING FIXES")

# Fix 1: Set unsupplied buses out of service
try:
    isolated = unsupplied_buses(net)
    if len(isolated) > 0:
        print(f"Setting {len(isolated)} unsupplied buses to in_service = False")
        net.bus.loc[list(isolated), "in_service"] = False
    else:
        print("No unsupplied buses need to be disabled.")
except Exception as e:
    print(f"Unsupplied bus handling failed: {e}")

# Fix 2: Deactivate zero-impedance lines
if len(net.line) > 0 and all(c in net.line.columns for c in ["r_ohm_per_km", "x_ohm_per_km"]):
    zero_imp_lines = net.line[
        (net.line["r_ohm_per_km"] == 0) &
        (net.line["x_ohm_per_km"] == 0)
    ]
    if len(zero_imp_lines) > 0:
        print(f"Found {len(zero_imp_lines)} zero-impedance lines. Setting them out of service.")
        net.line.loc[zero_imp_lines.index, "in_service"] = False
    else:
        print("No zero-impedance lines found.")

# Fix 3: Correct suspicious generator voltage setpoints
if len(net.gen) > 0 and "vm_pu" in net.gen.columns:
    vm_invalid = net.gen[(net.gen["vm_pu"] < 0.8) | (net.gen["vm_pu"] > 1.2)]
    if len(vm_invalid) > 0:
        print(f"Found {len(vm_invalid)} generators with invalid vm_pu. Setting vm_pu = 1.0")
        net.gen.loc[vm_invalid.index, "vm_pu"] = 1.0
    else:
        print("No invalid generator voltage setpoints found.")

# ============================================================
# 5. Power flow attempts
# ============================================================
print_section("POWER FLOW")

powerflow_success = False
powerflow_mode = None

print("Attempt 1: Newton-Raphson with DC initialization")
try:
    pp.runpp(
        net,
        algorithm="nr",
        init="dc",
        max_iteration=100,
        tolerance_mva=1e-6,
        calculate_voltage_angles=True,
        check_connectivity=True,
        numba=False
    )
    powerflow_success = True
    powerflow_mode = "AC NR with DC init"
    print("Success: Power flow converged with DC initialization.")

except pp.powerflow.LoadflowNotConverged:
    print("Attempt 1 failed.")

    print("\nAttempt 2: Newton-Raphson with flat start")
    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="flat",
            max_iteration=100,
            tolerance_mva=1e-5,
            calculate_voltage_angles=True,
            check_connectivity=True,
            numba=False
        )
        powerflow_success = True
        powerflow_mode = "AC NR with flat start"
        print("Success: Power flow converged with flat start.")

    except pp.powerflow.LoadflowNotConverged:
        print("Attempt 2 failed.")

        print("\nAttempt 3: DC power flow")
        try:
            pp.rundcpp(net, check_connectivity=True)
            powerflow_success = True
            powerflow_mode = "DC power flow"
            print("Success: DC power flow completed.")
            print("Note: Voltage magnitudes and reactive power results are not available in DC mode.")
        except Exception as e:
            print(f"Attempt 3 failed: {e}")

# ============================================================
# 6. Results and comparison views
# ============================================================
print_section("RESULTS")

if powerflow_success and "res_bus" in net and len(net.res_bus) > 0:
    print(f"Power flow mode: {powerflow_mode}")

    # --------------------------------------------------------
    # Bus result comparison table
    # --------------------------------------------------------
    print_subsection("First 10 bus results for comparison")

    bus_compare = build_display_df(
        net.bus,
        base_cols=["vn_kv", "type", "zone", "in_service"],
        include_pp_index=True
    )

    if "vm_pu" in net.res_bus.columns:
        bus_compare["res_vm_pu"] = net.res_bus["vm_pu"]
    if "va_degree" in net.res_bus.columns:
        bus_compare["res_va_degree"] = net.res_bus["va_degree"]
    if "p_mw" in net.res_bus.columns:
        bus_compare["res_p_mw"] = net.res_bus["p_mw"]
    if "q_mvar" in net.res_bus.columns:
        bus_compare["res_q_mvar"] = net.res_bus["q_mvar"]

    print(bus_compare.head(10))

    # Voltage extremes
    if "vm_pu" in net.res_bus.columns:
        print_subsection("Bus voltage extremes")
        print(f"Minimum voltage: {net.res_bus['vm_pu'].min():.6f} pu at bus {net.res_bus['vm_pu'].idxmin()}")
        print(f"Maximum voltage: {net.res_bus['vm_pu'].max():.6f} pu at bus {net.res_bus['vm_pu'].idxmax()}")

        v_low = net.res_bus[net.res_bus["vm_pu"] < 0.95]
        v_high = net.res_bus[net.res_bus["vm_pu"] > 1.05]
        print(f"Undervoltage buses (< 0.95 pu): {len(v_low)}")
        print(f"Overvoltage buses  (> 1.05 pu): {len(v_high)}")

    # --------------------------------------------------------
    # Load result comparison table
    # --------------------------------------------------------
    if "res_load" in net and len(net.res_load) > 0:
        print_subsection("First 10 load results for comparison")

        load_compare = build_display_df(
            net.load,
            base_cols=["bus", "p_mw", "q_mvar", "scaling", "in_service"],
            include_pp_index=True
        )

        if "p_mw" in net.res_load.columns:
            load_compare["res_p_mw"] = net.res_load["p_mw"]
        if "q_mvar" in net.res_load.columns:
            load_compare["res_q_mvar"] = net.res_load["q_mvar"]

        print(load_compare.head(10))

    # --------------------------------------------------------
    # Generator result comparison table
    # --------------------------------------------------------
    if "res_gen" in net and len(net.res_gen) > 0:
        print_subsection("First 10 generator results for comparison")

        gen_compare = build_display_df(
            net.gen,
            base_cols=["bus", "p_mw", "vm_pu", "slack", "in_service"],
            include_pp_index=True
        )

        if "p_mw" in net.res_gen.columns:
            gen_compare["res_p_mw"] = net.res_gen["p_mw"]
        if "q_mvar" in net.res_gen.columns:
            gen_compare["res_q_mvar"] = net.res_gen["q_mvar"]
        if "vm_pu" in net.res_gen.columns:
            gen_compare["res_vm_pu"] = net.res_gen["vm_pu"]
        if "va_degree" in net.res_gen.columns:
            gen_compare["res_va_degree"] = net.res_gen["va_degree"]

        print(gen_compare.head(10))

    # --------------------------------------------------------
    # Static generator result comparison table
    # --------------------------------------------------------
    if "res_sgen" in net and len(net.res_sgen) > 0:
        print_subsection("First 10 static generator results for comparison")

        sgen_compare = build_display_df(
            net.sgen,
            base_cols=["bus", "p_mw", "q_mvar", "scaling", "in_service"],
            include_pp_index=True
        )

        if "p_mw" in net.res_sgen.columns:
            sgen_compare["res_p_mw"] = net.res_sgen["p_mw"]
        if "q_mvar" in net.res_sgen.columns:
            sgen_compare["res_q_mvar"] = net.res_sgen["q_mvar"]

        print(sgen_compare.head(10))

    # --------------------------------------------------------
    # Line result comparison table
    # --------------------------------------------------------
    if "res_line" in net and len(net.res_line) > 0:
        print_subsection("First 10 line results for comparison")

        line_compare = build_display_df(
            net.line,
            base_cols=["from_bus", "to_bus", "length_km", "r_ohm_per_km", "x_ohm_per_km", "max_i_ka", "in_service"],
            include_pp_index=True
        )

        for col in ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "i_from_ka", "i_to_ka", "loading_percent"]:
            if col in net.res_line.columns:
                line_compare[f"res_{col}"] = net.res_line[col]

        print(line_compare.head(10))

        if "loading_percent" in net.res_line.columns:
            max_loading_idx = net.res_line["loading_percent"].idxmax()
            max_loading = net.res_line["loading_percent"].max()
            overloaded = net.res_line[net.res_line["loading_percent"] > 100]
            print_subsection("Line loading summary")
            print(f"Maximum line loading: {max_loading:.6f}% at line {max_loading_idx}")
            print(f"Overloaded lines (> 100%): {len(overloaded)}")

    # --------------------------------------------------------
    # Transformer result comparison table
    # --------------------------------------------------------
    if "res_trafo" in net and len(net.res_trafo) > 0:
        print_subsection("First 10 transformer results for comparison")

        trafo_compare = build_display_df(
            net.trafo,
            base_cols=["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "in_service"],
            include_pp_index=True
        )

        for col in ["p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar", "i_hv_ka", "i_lv_ka", "loading_percent"]:
            if col in net.res_trafo.columns:
                trafo_compare[f"res_{col}"] = net.res_trafo[col]

        print(trafo_compare.head(10))

        if "loading_percent" in net.res_trafo.columns:
            print_subsection("Transformer loading summary")
            print(f"Maximum transformer loading: {net.res_trafo['loading_percent'].max():.6f}%")

else:
    print("No converged power flow results are available.")

# ============================================================
# 7. Optional export tables for external comparison
# ============================================================
print_section("OPTIONAL COMPARISON EXPORT TABLES")

try:
    bus_compare_export = build_display_df(
        net.bus,
        base_cols=["vn_kv", "type", "zone", "in_service"],
        include_pp_index=True
    )
    if "res_bus" in net and len(net.res_bus) > 0:
        for col in ["vm_pu", "va_degree", "p_mw", "q_mvar"]:
            if col in net.res_bus.columns:
                bus_compare_export[f"res_{col}"] = net.res_bus[col]

    load_compare_export = build_display_df(
        net.load,
        base_cols=["bus", "p_mw", "q_mvar", "scaling", "in_service"],
        include_pp_index=True
    )
    if "res_load" in net and len(net.res_load) > 0:
        for col in ["p_mw", "q_mvar"]:
            if col in net.res_load.columns:
                load_compare_export[f"res_{col}"] = net.res_load[col]

    gen_compare_export = build_display_df(
        net.gen,
        base_cols=["bus", "p_mw", "vm_pu", "slack", "in_service"],
        include_pp_index=True
    )
    if "res_gen" in net and len(net.res_gen) > 0:
        for col in ["p_mw", "q_mvar", "vm_pu", "va_degree"]:
            if col in net.res_gen.columns:
                gen_compare_export[f"res_{col}"] = net.res_gen[col]

    sgen_compare_export = build_display_df(
        net.sgen,
        base_cols=["bus", "p_mw", "q_mvar", "scaling", "in_service"],
        include_pp_index=True
    )
    if "res_sgen" in net and len(net.res_sgen) > 0:
        for col in ["p_mw", "q_mvar"]:
            if col in net.res_sgen.columns:
                sgen_compare_export[f"res_{col}"] = net.res_sgen[col]

    line_compare_export = build_display_df(
        net.line,
        base_cols=["from_bus", "to_bus", "length_km", "r_ohm_per_km", "x_ohm_per_km", "max_i_ka", "in_service"],
        include_pp_index=True
    )
    if "res_line" in net and len(net.res_line) > 0:
        for col in ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "i_from_ka", "i_to_ka", "loading_percent"]:
            if col in net.res_line.columns:
                line_compare_export[f"res_{col}"] = net.res_line[col]

    trafo_compare_export = build_display_df(
        net.trafo,
        base_cols=["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "in_service"],
        include_pp_index=True
    )
    if "res_trafo" in net and len(net.res_trafo) > 0:
        for col in ["p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar", "i_hv_ka", "i_lv_ka", "loading_percent"]:
            if col in net.res_trafo.columns:
                trafo_compare_export[f"res_{col}"] = net.res_trafo[col]

    print("Comparison tables prepared in memory:")
    print("  - bus_compare_export")
    print("  - load_compare_export")
    print("  - gen_compare_export")
    print("  - sgen_compare_export")
    print("  - line_compare_export")
    print("  - trafo_compare_export")

except Exception as e:
    print(f"Failed to prepare comparison tables: {e}")

# ============================================================
# 8. Save network
# ============================================================
print_section("EXPORT")

output_file = "network_converged.json"
pp.to_json(net, output_file)
print(f"Network saved to JSON: {output_file}")