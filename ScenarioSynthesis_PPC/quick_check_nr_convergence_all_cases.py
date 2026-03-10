#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import traceback
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandapower.networks as pn

from case_generator_all_test_cases_pandapower_consider_ppc_branch_row import case_generation_pandapower
from newton_raphson_improved import newtonrapson


CASES = [
    # "case4gs",
    # "case5",
    # "case6ww",
    # "case9",
    # "case14",
    # "case24_ieee_rts",
    # "case30",
    # "case_ieee30",
    # "case33bw",
    # "case39",
    # "case57",
    # "case89pegase",
    # "case118",
    # "case145",
    # "case_illinois200",
    # "case300",
    # "case1354pegase",
    "case1888rte",
    # "case2848rte",
    # "case2869pegase",
    # "case3120sp",
    "case6470rte",
    "case6495rte",
    "case6515rte",
    # "case9241pegase",
    # "GBnetwork",
    # "GBreducednetwork",
    # "iceland",
]

# Use only for cases without net.trafo, if you want a nonzero compiled branch shunt
FORCE_SHUNT_CASES = {
    "case4gs", "case5", "case6ww", "case9", "case30", "case33bw"
}

# Optional special kwargs for some huge RTE cases if needed
CASE_KWARGS = {
    # "case1888rte": {"ref_bus_idx": 1246},
    # "case2848rte": {"ref_bus_idx": 271},
    # "case6470rte": {"ref_bus_idx": 5988},
    # "case6495rte": {"ref_bus_idx": None},
    # "case6515rte": {"ref_bus_idx": 6171},
}

SCENARIO_PRESETS = {
    "easy": dict(  # Easy scenario with minimal variability
        jitter_load=0.02,
        jitter_gen=0.01,
        pv_vset_range=(0.995, 1.005),
        rand_u_start=False,
        angle_jitter_deg=0.5,
        mag_jitter_pq=0.002,
    ),
    "no_change": dict(  # Scenario with no changes at all
        jitter_load=0.0,
        jitter_gen=0.0,
        pv_vset_range=(1.0, 1.0),
        rand_u_start=False,
        angle_jitter_deg=0.0,
        mag_jitter_pq=0.0,
    ),

    "A": dict(
        jitter_load=0.05,
        jitter_gen=0.03,
        pv_vset_range=(0.99, 1.02),
        rand_u_start=True,
        angle_jitter_deg=3.0,
        mag_jitter_pq=0.01,
    ),
    "B": dict(
        jitter_load=0.10,
        jitter_gen=0.05,
        pv_vset_range=(0.98, 1.03),
        rand_u_start=True,
        angle_jitter_deg=5.0,
        mag_jitter_pq=0.02,
    ),
    "C": dict(
        jitter_load=0.15,
        jitter_gen=0.08,
        pv_vset_range=(0.97, 1.04),
        rand_u_start=True,
        angle_jitter_deg=7.0,
        mag_jitter_pq=0.03,
    ),
}


def is_converged_solution(u_newton) -> bool:
    a = np.asarray(u_newton)
    if a.size == 0:
        return False
    if np.allclose(a, 0.0):
        return False
    return True


def _worker_init():
    # Defensive: keep every worker single-threaded internally
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_one_case_one_seed(task):
    """
    task = (
        case_name,
        seed,
        scenario_cfg,
        K,
        use_force_shunt_when_no_trafo,
        trafo_pfe_kw,
        trafo_i0_percent,
    )
    """
    (
        case_name,
        seed,
        scenario_cfg,
        K,
        use_force_shunt_when_no_trafo,
        trafo_pfe_kw,
        trafo_i0_percent,
    ) = task

    try:
        fn = getattr(pn, case_name)
        case_kwargs = CASE_KWARGS.get(case_name, {}).copy()

        force = None
        if use_force_shunt_when_no_trafo and case_name in FORCE_SHUNT_CASES:
            force = {"g": 0.0, "b": 0.2, "g_asym": 0.0, "b_asym": 0.0}

        gen_kwargs = dict(
            case_fn=fn,
            case_kwargs=case_kwargs,
            ybus_mode="ppcY",
            seed=seed,
            trafo_pfe_kw=trafo_pfe_kw,
            trafo_i0_percent=trafo_i0_percent,
            force_branch_shunt_pu=force,
        )
        gen_kwargs.update(scenario_cfg)

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
        ) = case_generation_pandapower(**gen_kwargs)

        N = len(bus_typ)

        if not is_connected:
            return {
                "case": case_name,
                "seed": seed,
                "N": N,
                "gridtype": gridtype,
                "converged": False,
                "reason": "generator_returned_not_connected",
            }

        u_newton, _I_unused, S_newton = newtonrapson(
            np.asarray(bus_typ).copy(),
            np.asarray(Y_matrix).copy(),
            np.asarray(s_multi).copy(),
            np.asarray(u_start).copy(),
            K=K,
        )

        conv = is_converged_solution(u_newton)

        return {
            "case": case_name,
            "seed": seed,
            "N": N,
            "gridtype": gridtype,
            "converged": conv,
            "reason": "ok" if conv else "all_zero_solution_or_empty",
        }

    except Exception as e:
        return {
            "case": case_name,
            "seed": seed,
            "N": None,
            "gridtype": None,
            "converged": False,
            "reason": f"NR_exception: {repr(e)}",
        }


def build_tasks(
    scenarios_per_case=10,
    scenario_level="A",
    K=40,
    use_force_shunt_when_no_trafo=True,
    trafo_pfe_kw=50.0,
    trafo_i0_percent=2.0,
):
    if scenario_level not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario_level={scenario_level}. Use one of {list(SCENARIO_PRESETS.keys())}")

    scenario_cfg = SCENARIO_PRESETS[scenario_level]
    tasks = []

    for case_name in CASES:
        for s in range(scenarios_per_case):
            seed = 1000 + s
            tasks.append((
                case_name,
                seed,
                scenario_cfg,
                K,
                use_force_shunt_when_no_trafo,
                trafo_pfe_kw,
                trafo_i0_percent,
            ))
    return tasks


def summarize_results(results, scenarios_per_case):
    by_case = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)

    summary = []
    zero_conv_cases = []
    all_fail_details = {}

    for case_name in CASES:
        rows = by_case.get(case_name, [])
        rows = sorted(rows, key=lambda x: x["seed"])

        n_conv = sum(1 for r in rows if r["converged"])
        n_fail = len(rows) - n_conv
        N = None
        gridtype = None

        fail_reasons = []
        for r in rows:
            if N is None and r["N"] is not None:
                N = r["N"]
            if gridtype is None and r["gridtype"] is not None:
                gridtype = r["gridtype"]
            if not r["converged"]:
                fail_reasons.append(r["reason"])

        at_least_once = (n_conv > 0)

        summary.append({
            "case": case_name,
            "N": N,
            "gridtype": gridtype,
            "converged": n_conv,
            "failed": n_fail,
            "at_least_once": at_least_once,
        })

        if not at_least_once:
            zero_conv_cases.append(case_name)
            all_fail_details[case_name] = fail_reasons

    return summary, zero_conv_cases, all_fail_details


def quick_screen_all_cases_mp(
    scenarios_per_case=10,
    scenario_level="A",
    K=40,
    workers=0,
    chunksize=1,
    use_force_shunt_when_no_trafo=True,
    trafo_pfe_kw=None,
    trafo_i0_percent=None,
):
    tasks = build_tasks(
        scenarios_per_case=scenarios_per_case,
        scenario_level=scenario_level,
        K=K,
        use_force_shunt_when_no_trafo=use_force_shunt_when_no_trafo,
        trafo_pfe_kw=trafo_pfe_kw,
        trafo_i0_percent=trafo_i0_percent,
    )

    if workers <= 0:
        workers = os.cpu_count() or 1

    print("=" * 90)
    print("Quick NR convergence screening (multiprocessing)")
    print(f"tasks              = {len(tasks)}")
    print(f"cases              = {len(CASES)}")
    print(f"scenarios_per_case = {scenarios_per_case}")
    print(f"scenario_level     = {scenario_level}")
    print(f"scenario_cfg       = {SCENARIO_PRESETS[scenario_level]}")
    print(f"K                  = {K}")
    print(f"workers            = {workers}")
    print(f"chunksize          = {chunksize}")
    print("=" * 90)

    if os.name == "nt":
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")

    results = []
    with ctx.Pool(processes=workers, initializer=_worker_init, maxtasksperchild=200) as pool:
        for idx, res in enumerate(pool.imap_unordered(run_one_case_one_seed, tasks, chunksize=chunksize), 1):
            results.append(res)
            status = "converged" if res["converged"] else f"failed ({res['reason']})"
            print(f"[{idx:4d}/{len(tasks)}] {res['case']} seed={res['seed']} -> {status}")

    summary, zero_conv_cases, all_fail_details = summarize_results(results, scenarios_per_case)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    summary_sorted = sorted(summary, key=lambda x: (x["converged"], x["N"] if x["N"] is not None else 10**9))
    for row in summary_sorted:
        print(
            f"{row['case']:16s} | "
            f"N={str(row['N']):>5s} | "
            f"conv={row['converged']:2d}/{scenarios_per_case} | "
            f"at_least_once={row['at_least_once']}"
        )

    print("\nCases with ZERO convergence in the tested scenarios:")
    if zero_conv_cases:
        for c in zero_conv_cases:
            print(f"  - {c}")
    else:
        print("  None")

    print("\nFailure reason samples for zero-convergence cases:")
    if zero_conv_cases:
        for c in zero_conv_cases:
            reasons = all_fail_details[c]
            uniq = []
            for r in reasons:
                if r not in uniq:
                    uniq.append(r)
            print(f"\n[{c}]")
            for r in uniq[:5]:
                print(f"  {r}")
    else:
        print("  None")

    return results, summary, zero_conv_cases, all_fail_details


if __name__ == "__main__":
    quick_screen_all_cases_mp(
        scenarios_per_case=3,
        scenario_level="no_change",
        K=40,
        workers=0,   # 0 => all CPU cores
        chunksize=1,
        use_force_shunt_when_no_trafo=True,
        # trafo_pfe_kw=50.0,
        # trafo_i0_percent=2.0,
    )