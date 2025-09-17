#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired throughput benchmark: Newton–Raphson (CPU, multiprocessing) vs GNS (GPU).

Goals:
  - Use all CPU cores for NR (e.g., 16 workers).
  - Fully leverage RTX 3070 (8GB) for GNS with safe streaming micro-batches.
  - Same cases per N for both solvers.
  - Report throughput, wall time, and speedup across N = [startN, endN].

Usage example:
  python bench_pair_throughput.py \
      --PARQUET /proj/.../your.parquet \
      --device cuda \
      --startN 4 --endN 1024 \
      --cases_per_N 4096 \
      --batch_style blockdiag \
      --models GNSMsg_EdgeSelfAttn_armijo \
      --nr_workers 16 \
      --max_nodes_per_forward 17408 \
      --amp --adaptive_oom_backoff \
      --warmup 2 --repeat 5
"""

# -------- BLAS/OpenMP single-thread + abort diagnostics (must be BEFORE numpy/torch) --------
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
import faulthandler as _fa
try:
    _fa.enable(all_threads=True)
except Exception:
    pass
# --------------------------------------------------------------------------------------------

import argparse, os, sys, time, math, datetime, csv
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import torch
from torch import nn

torch.backends.cudnn.benchmark = True

# ---- Pin PyTorch CPU threads in parent (workers get their own pin via initializer below) ----
def _pin_torch_threads(n: int = 1):
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
# Avoid set_num_interop_threads on older builds that sometimes abort
_pin_torch_threads(1)
# --------------------------------------------------------------------------------------------

# ---------------- Models (unified classes with use_armijo) ----------------
from GNN.SimpleGNN.GNSMsg_armijo_5think import GNSMsg

_GNSMsg_EdgeSelfAttn = None
_try_modules = ["GNN.SimpleGNN.GNSMsg_SelfAttention_armijo_stable_5think"]
_last_err = None
for mod in _try_modules:
    try:
        _m = __import__(mod, fromlist=["GNSMsg_EdgeSelfAttn"])
        _GNSMsg_EdgeSelfAttn = getattr(_m, "GNSMsg_EdgeSelfAttn")
        break
    except Exception as e:
        _last_err = e
if _GNSMsg_EdgeSelfAttn is None:
    raise ImportError("Could not import GNSMsg_EdgeSelfAttn from: " + ", ".join(_try_modules) + f"\n{_last_err}")

# ---------------- Dataset & Collate ----------------
# If you use the _no_remove dataset, import that instead:
try:
    from GNN.SimpleGNN.Dataset_optimized_complex_columns_no_remove import ChanghunDataset
except Exception:
    from GNN.SimpleGNN.Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag

# ---------------- Newton–Raphson ----------------
try:
    from newton_solver import newtonrapson
except Exception as e:
    print("ERROR: import newtonrapson failed. Put your NR in newton_solver.py.\n", e)
    sys.exit(1)

# ---------------- Helpers ----------------
def pick_device(which: str) -> torch.device:
    return torch.device("cuda" if which == "auto" and torch.cuda.is_available() else which)

def to_complex_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.is_complex():
            return x.numpy()
        if x.ndim >= 1 and x.shape[-1] == 2:
            return (x[..., 0].numpy() + 1j * x[..., 1].numpy())
        return x.numpy()
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return x
        if x.ndim >= 1 and x.shape[-1] == 2:
            return (x[..., 0] + 1j * x[..., 1])
        return x
    raise TypeError("Unsupported type for to_complex_numpy")

def ensure_complex_from_keys(sample, key_complex, key_real=None, key_imag=None):
    if key_complex in sample: return sample[key_complex]
    if key_real and key_imag and key_real in sample and key_imag in sample:
        return sample[key_real] + 1j * sample[key_imag]
    raise KeyError(f"Need {key_complex} or ({key_real},{key_imag})")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def median_mean_timing(fn, *, device, warmup=2, repeat=5):
    times = []
    for _ in range(warmup):
        fn();  torch.cuda.synchronize() if device.type == "cuda" else None
    for _ in range(repeat):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0))
    arr = np.array(times, dtype=float)
    return float(np.median(arr)), float(np.mean(arr))

def get_one_index_per_N(dataset, N_values):
    picks = {}
    for idx in range(len(dataset)):
        N = int(dataset[idx]["N"])
        if N in N_values and N not in picks:
            picks[N] = idx
            if len(picks) == len(N_values): break
    return picks

def _tile_to_k(seq, K):
    if not seq: return []
    reps = (K + len(seq) - 1) // len(seq)
    return (seq * reps)[:K]

def get_indices_for_N(dataset, N, K, *, strategy="cycle"):
    idxs = [i for i in range(len(dataset)) if int(dataset[i]["N"]) == int(N)]
    if not idxs:
        return []
    if len(idxs) >= K:
        return idxs[:K]
    if strategy == "first":
        return idxs + [idxs[0]] * (K - len(idxs))
    elif strategy == "random":
        rng = np.random.default_rng(12345)
        extra = rng.choice(idxs, size=K - len(idxs), replace=True).tolist()
        return idxs + extra
    else:
        return _tile_to_k(idxs, K)

# ---------------- NR multiprocessing ----------------
def _init_worker_singlethread():
    # Pin threads inside each child process, too.
    _pin_torch_threads(1)

def _nr_worker(args):
    bus_typ, Y_np, S_np, U_np = args
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _u, _I, _S = newtonrapson(bus_typ, Y_np, S_np, U_np)
    return 0

def make_nr_batch_args(samples):
    args_list = []
    for s in samples:
        bus_typ = s["bus_type"].cpu().numpy().astype(np.int64).squeeze()
        Ybus = ensure_complex_from_keys(s, "Ybus", "Ybus_real", "Ybus_imag")
        if "U_start" in s: U0 = s["U_start"]
        elif "V_start" in s:
            V = s["V_start"]; U0 = V[..., 0] * torch.exp(1j * V[..., 1])
        else: raise KeyError("Need U_start or V_start")
        S = s["S_start"] if "S_start" in s else (s["P_start"] + 1j * s["Q_start"])
        args_list.append((bus_typ, to_complex_numpy(Ybus), to_complex_numpy(S), to_complex_numpy(U0)))
    return args_list

def time_nr_once(args_list, *, workers):
    workers = (os.cpu_count() or 1) if workers in (None, -1) else max(1, workers)
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=workers, initializer=_init_worker_singlethread) as pool:
        pool.map(_nr_worker, args_list, chunksize=max(1, len(args_list)//(workers*4) or 1))
    t1 = time.perf_counter()
    return t1 - t0  # seconds

# ---------------- GNS forward (streaming micro-batches) ----------------
def _streamed_gns_forward(model, samples, *, batch_style, device, micro_B,
                          use_amp=False, adaptive_backoff=True, verbose=False):
    """
    Process 'samples' in chunks of size micro_B, building each chunk on-the-fly.
    Never stores >1 chunk on GPU. If adaptive_backoff: halve chunk size on CUDA OOM.
    """
    B = len(samples)
    i = 0
    first_msg = True

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if (use_amp and device.type == "cuda") else torch.cuda.amp.autocast(enabled=False)
    )

    while i < B:
        b = min(micro_B, B - i)
        while True:
            try:
                with autocast_ctx:
                    if batch_style == "plain":
                        def stack_key(key): return torch.stack([s[key] for s in samples[i:i+b]], dim=0).to(device)
                        bt = stack_key("bus_type")
                        lc = stack_key("Lines_connected")
                        Y  = stack_key("Ybus")
                        Ys = stack_key("Y_Lines")
                        Yc = stack_key("Y_C_Lines")
                        Ss = stack_key("S_start")
                        Vs = stack_key("V_start")
                        with torch.no_grad():
                            out = model(bt, lc, Y, Ys, Yc, Ss, Vs, None)
                            if isinstance(out, tuple): _ = out[0]
                        del bt, lc, Y, Ys, Yc, Ss, Vs, out
                    else:
                        bch = collate_blockdiag(samples[i:i+b])
                        def g2(k):
                            v = bch[k]
                            return v.to(device) if isinstance(v, torch.Tensor) else v
                        with torch.no_grad():
                            out = model(g2("bus_type"), g2("Lines_connected"), g2("Ybus"),
                                        g2("Y_Lines"), g2("Y_C_Lines"), g2("S_start"),
                                        g2("V_start"), g2("sizes"))
                            if isinstance(out, tuple): _ = out[0]
                        del bch, out
                break  # success
            except RuntimeError as e:
                is_oom = ("out of memory" in str(e).lower()) or ("cudaErrorMemoryAllocation" in str(e))
                if not (adaptive_backoff and is_oom and b > 1):
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                b_next = max(1, b // 2)
                if verbose or first_msg:
                    print(f"    [adaptive] CUDA OOM at chunk size {b}; retrying with {b_next}")
                    first_msg = False
                b = b_next
                continue
        i += b

# ---------------- Batch size chooser ----------------
def _elem_size_bytes_from_sample(sample):
    if "Ybus" in sample and isinstance(sample["Ybus"], torch.Tensor):
        return int(sample["Ybus"].element_size())
    return 8  # fallback ≈ complex64

def choose_start_micro_B(N, B, max_nodes, batch_style, elem_size, device):
    """
    Initial micro_B from user cap + free GPU memory.
    Nodes cap: floor(max_nodes / N).
    Memory cap: keep M=(N*micro_B) or per-item tensors under a budget of ~60% free mem.
    """
    b_nodes = max(1, min(B, max_nodes // max(1, N)))

    if device.type != "cuda":
        return b_nodes

    try:
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        free_bytes = 0

    if free_bytes <= 0:
        return b_nodes

    safety = 0.60
    budget = int(free_bytes * safety)

    if batch_style == "blockdiag":
        # dominant memory ~ (N*micro_B)^2 * elem_size
        b_mem = int(max(1, (int((budget / max(1, elem_size)) ** 0.5) // max(1, N))))
    else:
        # rough: micro_B * N^2 * elem_size
        denom = elem_size * (N * N)
        b_mem = int(max(1, budget // max(1, denom)))

    return max(1, min(B, b_nodes, b_mem))

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Paired throughput benchmark: NR (CPU) vs GNS (GPU)")

    ap.add_argument("--models", nargs="+",
                    default=["GNSMsg_EdgeSelfAttn_armijo"],
                    choices=["GNSMsg", "GNSMsg_armijo", "GNSMsg_EdgeSelfAttn", "GNSMsg_EdgeSelfAttn_armijo"],
                    help="Which GNS models to benchmark")

    ap.add_argument("--PARQUET", nargs="+", required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--startN", type=int, default=4)
    ap.add_argument("--endN", type=int, default=1024)

    # Total cases per N (same for NR and GNS)
    ap.add_argument("--cases_per_N", type=int, default=4096,
                    help="How many cases to solve per N. Will replicate if dataset lacks enough distinct samples.")

    # Model HParams
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--d_hi", type=int, default=16)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--vlimit", dest="vlimit", action="store_true")
    ap.add_argument("--no-vlimit", dest="vlimit", action="store_false")
    ap.set_defaults(vlimit=True)

    # Attention HParams
    ap.add_argument("--edge_d_model", type=int, default=None)
    ap.add_argument("--edge_heads", type=int, default=4)
    ap.add_argument("--edge_layers", type=int, default=1)
    ap.add_argument("--edge_dropout", type=float, default=0.0)

    # Controls
    ap.add_argument("--batch_style", type=str, default="blockdiag", choices=["blockdiag", "plain"],
                    help="blockdiag uses collate_blockdiag (recommended)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Use fp16 autocast on CUDA for GNS")
    ap.add_argument("--step", type=int, default=1, help="Step size for N range")
    # Micro-batching
    ap.add_argument("--max_nodes_per_forward", type=int, default=17408,
                    help="Derive initial micro_B via floor(max_nodes/N); also bounded by free GPU memory.")
    ap.add_argument("--adaptive_oom_backoff", action="store_true",
                    help="Halve micro_B on CUDA OOM and retry within a call.")
    ap.add_argument("--verbose_adaptive", action="store_true")

    # NR multiprocessing
    ap.add_argument("--nr_workers", type=int, default=16)

    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    # Keep PyTorch CPU ops from oversubscribing cores while we use multiprocessing
    _pin_torch_threads(1)

    # Load dataset once (lazily returns tensors already on device if you coded it so)
    full_ds = ChanghunDataset(args.PARQUET, per_unit=False, device="cpu")  # keep raw on CPU; we move per-chunk
    print(f"Loaded dataset with {len(full_ds)} samples.")

    Ns = list(range(args.startN, args.endN + 1, args.step))
    idx_by_N = get_one_index_per_N(full_ds, set(Ns))
    missing = [N for N in Ns if N not in idx_by_N]
    if missing:
        print(f"WARNING: No sample for N={missing} (skipped).")

    def model_kwargs_common():
        return dict(d=args.d, d_hi=args.d_hi, K=args.K, pinn=False, gamma=args.gamma, v_limit=args.vlimit)

    def _mk_GNSMsg_kwargs(use_armijo: bool):
        kw = model_kwargs_common();  kw["use_armijo"] = use_armijo;  return kw

    def _mk_EdgeSelfAttn_kwargs(use_armijo: bool):
        kw = model_kwargs_common()
        kw.update(dict(use_armijo=use_armijo, d_model=args.edge_d_model, n_heads=args.edge_heads,
                       num_attn_layers=args.edge_layers, attn_dropout=args.edge_dropout))
        return kw

    model_specs = {
        "GNSMsg": ("GNSMsg", _mk_GNSMsg_kwargs(False)),
        "GNSMsg_armijo": ("GNSMsg", _mk_GNSMsg_kwargs(True)),
        "GNSMsg_EdgeSelfAttn": ("GNSMsg_EdgeSelfAttn", _mk_EdgeSelfAttn_kwargs(False)),
        "GNSMsg_EdgeSelfAttn_armijo": ("GNSMsg_EdgeSelfAttn", _mk_EdgeSelfAttn_kwargs(True)),
    }
    model_specs = {k: v for k, v in model_specs.items() if k in args.models}

    # Build models on device (CUDA for speed)
    models = {}
    for name, (clsname, kwargs) in model_specs.items():
        try:
            cls = _GNSMsg_EdgeSelfAttn if clsname == "GNSMsg_EdgeSelfAttn" else GNSMsg
            m = cls(**kwargs).to(device).eval()
            if args.compile:
                try:
                    m = torch.compile(m, mode="reduce-overhead", fullgraph=True)
                except Exception as e:
                    print(f"Warning: Could not compile model {name}: {e}")
            models[name] = (clsname, kwargs, m)
            print(f"[OK] {name:28s} params={count_parameters(m):,}")
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    if not models:
        print("No models to benchmark.");  sys.exit(1)

    rows = []
    out_dir = "./results/bench_pair"
    os.makedirs(out_dir, exist_ok=True)

    for N in Ns:
        if N not in idx_by_N:
            continue

        # Build the SAME cases for NR and GNS
        K_total = int(args.cases_per_N)
        idxs = get_indices_for_N(full_ds, N, K_total, strategy="cycle")
        if not idxs:
            print(f"N={N}: no cases available; skipping")
            continue
        samples = [full_ds[i] for i in idxs]  # KEEP on CPU; we move chunk-by-chunk

        # --- NR timing (CPU workers) ---
        nr_args_list = make_nr_batch_args(samples)  # pure numpy/complex
        nr_sec = time_nr_once(nr_args_list, workers=args.nr_workers)
        nr_thru = K_total / nr_sec if nr_sec > 0 else float("nan")

        for name, (clsname, kwargs, model) in models.items():
            # Choose safe starting micro_B for GNS
            elem_size = _elem_size_bytes_from_sample(samples[0])
            micro_B0 = choose_start_micro_B(
                N=N, B=K_total, max_nodes=args.max_nodes_per_forward,
                batch_style=args.batch_style, elem_size=elem_size, device=device
            )
            micro_B0 = max(1, micro_B0)
            if micro_B0 < K_total:
                print(f"N={N}: GNS streaming micro_B={micro_B0} (cap={args.max_nodes_per_forward}, elem={elem_size}B)")

            # Build a callable that runs ALL K_total samples once (for timing)
            def _gns_run_all_once():
                _streamed_gns_forward(model, samples,
                                      batch_style=args.batch_style,
                                      device=device,
                                      micro_B=micro_B0,
                                      use_amp=args.amp,
                                      adaptive_backoff=args.adaptive_oom_backoff,
                                      verbose=args.verbose_adaptive)

            # Time it (median/mean of a few repeats)
            med_sec, mean_sec = median_mean_timing(_gns_run_all_once, device=device,
                                                   warmup=args.warmup, repeat=args.repeat)
            gns_sec = med_sec
            gns_thru = K_total / gns_sec if gns_sec > 0 else float("nan")
            speedup = nr_sec / gns_sec if (gns_sec > 0 and nr_sec > 0) else float("nan")

            rows.append({
                "N": N,
                "cases": K_total,
                "nr_workers": args.nr_workers,
                "gns_model": name,
                "gns_batch_style": args.batch_style,
                "gns_amp": int(args.amp),
                "micro_B_start": micro_B0,
                "max_nodes_cap": args.max_nodes_per_forward,
                "nr_sec_total": round(nr_sec, 6),
                "gns_sec_total": round(gns_sec, 6),
                "nr_cases_per_sec": round(nr_thru, 3),
                "gns_cases_per_sec": round(gns_thru, 3),
                "speedup_nr_over_gns": round(speedup, 3),
            })

            print(
                f"N={N:4d} | cases={K_total:5d} | NR {nr_sec:7.3f}s "
                f"({nr_thru:7.1f} cases/s) | {name:28s} "
                f"{gns_sec:7.3f}s ({gns_thru:7.1f} cases/s) | speedup x{speedup:5.2f}"
            )

    # Save CSV
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"pair_throughput_{stamp}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "N","cases","nr_workers","gns_model","gns_batch_style","gns_amp",
            "micro_B_start","max_nodes_cap",
            "nr_sec_total","gns_sec_total","nr_cases_per_sec","gns_cases_per_sec",
            "speedup_nr_over_gns",
        ])
        writer.writeheader();  writer.writerows(rows)

    print(f"\nSaved results → {out_csv}")
    print("Done.")

if __name__ == "__main__":
    main()
