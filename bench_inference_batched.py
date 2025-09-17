#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batchable + multiprocessing benchmark for GNS variants vs Newton–Raphson.

Key flags:
  --batch_size INT
  --batch_mode {replicate,distinct}
  --batch_style {blockdiag,plain}   # blockdiag uses your collate_* (recommended)
  --nr_workers INT                  # -1 = all cores
  --gns_workers INT                 # CPU-only parallel GNS

Examples
  GPU, block-diag batching:
    python bench_inference.py --device auto --batch_size 64 --batch_style blockdiag

  CPU parallel NR + GNS:
    python bench_inference.py --device cpu --batch_size 32 --batch_style blockdiag \
        --nr_workers -1 --gns_workers 4
"""

import argparse, os, sys, time, math, io, datetime, csv
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import torch
from torch import nn

torch.backends.cudnn.benchmark = True

# ---------------- Models (unified classes with use_armijo) ----------------
from GNN.SimpleGNN.GNSMsg_armijo_5think import GNSMsg  # unified: has use_armijo

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

# ---------------- Dataset & Collate (block-diag) ----------------
from GNN.SimpleGNN.Dataset_optimized_complex_columns_no_remove import ChanghunDataset
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
            return x.numpy().astype(np.complex128)
        if x.ndim >= 1 and x.shape[-1] == 2:
            return (x[..., 0].numpy() + 1j * x[..., 1].numpy()).astype(np.complex128)
        return x.numpy().astype(np.complex128)
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return x.astype(np.complex128)
        if x.ndim >= 1 and x.shape[-1] == 2:
            return (x[..., 0] + 1j * x[..., 1]).astype(np.complex128)
        return x.astype(np.complex128)
    raise TypeError("Unsupported type for to_complex_numpy")

def ensure_complex_from_keys(sample, key_complex, key_real=None, key_imag=None):
    if key_complex in sample: return sample[key_complex]
    if key_real and key_imag and key_real in sample and key_imag in sample:
        return sample[key_real] + 1j * sample[key_imag]
    raise KeyError(f"Need {key_complex} or ({key_real},{key_imag})")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def time_callable_ms(fn, *, device, warmup=3, repeat=10):
    times = []
    for _ in range(warmup):
        fn();  torch.cuda.synchronize() if device.type == "cuda" else None
    for _ in range(repeat):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
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
    else:  # "cycle" (round-robin)
        return _tile_to_k(idxs, K)



# ---------------- Newton–Raphson multiprocessing ----------------
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

def time_nr_batch(args_list, *, workers, warmup=3, repeat=10):
    workers = (os.cpu_count() or 1) if workers in (None, -1) else max(1, workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for _ in range(warmup):
            pool.map(_nr_worker, args_list, chunksize=max(1, len(args_list)//(workers*4) or 1))
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            pool.map(_nr_worker, args_list, chunksize=max(1, len(args_list)//(workers*4) or 1))
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    arr = np.array(times, dtype=float)
    return float(np.median(arr)), float(np.mean(arr))


# ---------------- GNS CPU multiprocessing (optional) ----------------
_GLOBAL_GNS_MODEL = None
def _init_gns_worker(model_name, model_kwargs):
    global _GLOBAL_GNS_MODEL, _GNSMsg_EdgeSelfAttn
    cls = _GNSMsg_EdgeSelfAttn if model_name.startswith("GNSMsg_EdgeSelfAttn") else GNSMsg
    _GLOBAL_GNS_MODEL = cls(**model_kwargs).to("cpu").eval()
    try: torch.set_num_threads(1)
    except Exception: pass

def _gns_forward_worker(batch_np):
    global _GLOBAL_GNS_MODEL
    (bus_type, Lines, Y, Ys, Yc, Sstart, Vstart, sizes) = batch_np
    bt, lc, Yt, Ys_t = torch.from_numpy(bus_type), torch.from_numpy(Lines), torch.from_numpy(Y), torch.from_numpy(Ys)
    Yc_t, Ss, Vs, sz = torch.from_numpy(Yc), torch.from_numpy(Sstart), torch.from_numpy(Vstart), (None if sizes is None else torch.from_numpy(sizes))
    with torch.no_grad():
        out = _GLOBAL_GNS_MODEL(bt, lc, Yt, Ys_t, Yc_t, Ss, Vs, sz)
        if isinstance(out, tuple): _ = out[0]
    return 0

def tensors_to_numpy_batch(bt, lc, Y, Ys, Yc, Ss, Vs, sz):
    def _np(x): return np.ascontiguousarray(x.detach().cpu().numpy())
    return (_np(bt), _np(lc), _np(Y), _np(Ys), _np(Yc), _np(Ss), _np(Vs), (None if sz is None else _np(sz)))

def split_numpy_batches(batch_np, n_chunks):
    (bus_type, Lines, Y, Ys, Yc, Sstart, Vstart, sizes) = batch_np
    def _split(a): return [a] if a is None else np.array_split(a, n_chunks, axis=0)
    return list(zip(_split(bus_type), _split(Lines), _split(Y), _split(Ys), _split(Yc), _split(Sstart), _split(Vstart), _split(sizes)))


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Benchmark inference with batching & multiprocessing")

    ap.add_argument("--models", nargs="+",
                    default=["GNSMsg_EdgeSelfAttn_armijo"],
                    choices=["GNSMsg", "GNSMsg_armijo", "GNSMsg_EdgeSelfAttn", "GNSMsg_EdgeSelfAttn_armijo"],
                    help="Which GNS models to benchmark")
    # argparse
    ap.add_argument("--fill_strategy", choices=["cycle", "first", "random"], default="cycle",
                    help="How to fill when not enough distinct samples for a given N")

    ap.add_argument("--PARQUET", nargs="+", default=["./data/HVN_15000_NR_plain_4_to_32_buses.parquet"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--startN", type=int, default=4)
    ap.add_argument("--endN", type=int, default=32)

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

    # Benchmark controls
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--batch_mode", type=str, default="replicate", choices=["replicate", "distinct"])
    ap.add_argument("--batch_style", type=str, default="blockdiag", choices=["blockdiag", "plain"],
                    help="blockdiag uses collate_blockdiag (sets sizes); plain stacks per-item and passes sizes=None")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--per_unit", action="store_true")
    ap.add_argument("--pinn_inference", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--step", type=int, default=1, help="Step size for N range")

    # Multiprocessing knobs
    ap.add_argument("--nr_workers", type=int, default=-1)
    ap.add_argument("--gns_workers", type=int, default=1)

    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")
    print(args.PARQUET)
    full_ds = ChanghunDataset(args.PARQUET, per_unit=args.per_unit, device=device)
    print(f"Loaded dataset with {len(full_ds)} samples.")

    Ns = list(range(args.startN, args.endN + 1, args.step))

    idx_by_N = get_one_index_per_N(full_ds, set(Ns))
    missing = [N for N in Ns if N not in idx_by_N]
    if missing: print(f"WARNING: No sample for N={missing} (skipped).")

    def model_kwargs_common():
        return dict(d=args.d, d_hi=args.d_hi, K=args.K, pinn=args.pinn_inference, gamma=args.gamma, v_limit=args.vlimit)

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

    if args.compile :
        models = {}
        for name, (clsname, kwargs) in model_specs.items():
            try:
                m = (_GNSMsg_EdgeSelfAttn if clsname == "GNSMsg_EdgeSelfAttn" else GNSMsg)(**kwargs).to(device).eval()
                try:
                    m = torch.compile(m, mode="reduce-overhead", fullgraph=True)
                except Exception as e:
                    print(f"Warning: Could not compile model {name}: {e}")
                models[name] = (clsname, kwargs, m)
                print(f"[OK] {name:28s} params={count_parameters(m):,}")
            except Exception as e:
                print(f"[SKIP] {name}: {e}")
    else :
        models = {}
        for name, (clsname, kwargs) in model_specs.items():
            try:
                m = (_GNSMsg_EdgeSelfAttn if clsname == "GNSMsg_EdgeSelfAttn" else GNSMsg)(**kwargs).to(device).eval()
                models[name] = (clsname, kwargs, m)
                print(f"[OK] {name:28s} params={count_parameters(m):,}")
            except Exception as e:
                print(f"[SKIP] {name}: {e}")



    if not models:
        print("No models to benchmark.");  sys.exit(1)

    rows = []

    for N in Ns:
        if N not in idx_by_N: continue

        # Build B samples for this N
        if args.batch_mode == "distinct":
            idxs = get_indices_for_N(full_ds, N, args.batch_size, strategy=args.fill_strategy)
            if not idxs:
                # fallback: replicate the single available sample for this N
                base = full_ds[idx_by_N[N]]
                samples = [base] * args.batch_size
            else:
                samples = [full_ds[i] for i in idxs]
        else:
            base = full_ds[idx_by_N[N]]
            samples = [base] * args.batch_size

        # ---------- Prepare GNS batch inputs ----------
        if args.batch_style == "blockdiag":
            batch = collate_blockdiag(samples)  # returns dict with batch-dim=1; sizes & offsets w/o batch
            # push to chosen device (tensors likely already on device)
            def g(k): return batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k]
            bus_type = g("bus_type")        # (1, M)
            Lines    = g("Lines_connected") # (1, E_total)
            Y        = g("Ybus")            # (1, M, M)
            Ys       = g("Y_Lines")         # (1, E_total)
            Yc       = g("Y_C_Lines")       # (1, E_total)
            Sstart   = g("S_start")         # (1, M)
            Vstart   = g("V_start")         # (1, M, 2)
            sizes    = g("sizes")           # (num_grids,)
        else:
            # plain per-item batching: DO NOT pass sizes (avoid block-diag path)
            def stack_key(key): return torch.stack([s[key] for s in samples], dim=0).to(device)
            bus_type = stack_key("bus_type")          # (B, N)
            Lines    = stack_key("Lines_connected")   # (B, E)
            Y        = stack_key("Ybus")              # (B, N, N)
            Ys       = stack_key("Y_Lines")           # (B, E)
            Yc       = stack_key("Y_C_Lines")         # (B, E)
            Sstart   = stack_key("S_start")           # (B, N)
            Vstart   = stack_key("V_start")           # (B, N, 2)
            sizes    = None

        # If your model expects float edges:
        # Lines = Lines.float()

        # ---------- Newton–Raphson timing (parallel across batch) ----------
        nr_args_list = make_nr_batch_args(samples)
        try:
            nr_med_ms, nr_mean_ms = time_nr_batch(nr_args_list, workers=args.nr_workers,
                                                  warmup=args.warmup, repeat=args.repeat)
        except Exception as e:
            print(f"N={N}: Newton failed: {e}")
            nr_med_ms, nr_mean_ms = float("nan"), float("nan")

        # ---------- GNS timing ----------
        for name, (clsname, kwargs, model) in models.items():
            B = args.batch_size
            if device.type == "cuda" or args.gns_workers <= 1:
                def _call():
                    with torch.no_grad():
                        out = model(bus_type, Lines, Y, Ys, Yc, Sstart, Vstart, sizes)
                        if isinstance(out, tuple) and len(out) == 2: _ = out[0]
                try:
                    med_ms, mean_ms = time_callable_ms(_call, device=device,
                                                       warmup=args.warmup, repeat=args.repeat)
                except Exception as e:
                    print(f"N={N}: model {name} failed: {e}")
                    med_ms, mean_ms = float("nan"), float("nan")
            else:
                # CPU multiprocessing across batch
                # Convert to numpy and split into worker chunks
                batch_np = tensors_to_numpy_batch(bus_type, Lines, Y, Ys, Yc, Sstart, Vstart,
                                                  (sizes if sizes is None else sizes.unsqueeze(0).repeat(1,1)).squeeze(0).numpy()
                                                  if (sizes is not None and isinstance(sizes, torch.Tensor)) else None)
                chunks = split_numpy_batches(batch_np, args.gns_workers)
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=args.gns_workers,
                              initializer=_init_gns_worker,
                              initargs=(clsname if "EdgeSelfAttn" in name else "GNSMsg", kwargs)) as pool:
                    for _ in range(args.warmup):
                        pool.map(_gns_forward_worker, chunks)
                    times = []
                    for _ in range(args.repeat):
                        t0 = time.perf_counter()
                        pool.map(_gns_forward_worker, chunks)
                        t1 = time.perf_counter()
                        times.append((t1 - t0) * 1000.0)
                arr = np.array(times, dtype=float)
                med_ms, mean_ms = float(np.median(arr)), float(np.mean(arr))

            speed = (nr_med_ms / med_ms) if (med_ms > 0 and math.isfinite(nr_med_ms)) else float("nan")
            rows.append({
                "N": N, "batch_size": B, "batch_mode": args.batch_mode, "batch_style": args.batch_style,
                "method": name, "device": device.type,
                "median_ms_batch": round(med_ms, 3), "mean_ms_batch": round(mean_ms, 3),
                "median_ms_per_item": round(med_ms / (1 if args.batch_style == "blockdiag" else B), 3)
                                       if med_ms == med_ms else float("nan"),
                "newton_ms_batch": round(nr_med_ms, 3),
                "newton_ms_per_item": round(nr_med_ms / B, 3) if nr_med_ms == nr_med_ms else float("nan"),
                "speedup_vs_newton_batch": round(speed, 3) if math.isfinite(speed) else speed,
                "speedup_vs_newton_per_item":
                    round((nr_med_ms / B) / (med_ms / (1 if args.batch_style == "blockdiag" else B)), 3)
                    if (med_ms > 0 and math.isfinite(nr_med_ms)) else float("nan"),
            })

            print(
                f"N={N:4d} B={B:3d} [{args.batch_style}] | {name:28s} "
                f"batch med {med_ms:8.3f} ms | "
                f"NR batch {nr_med_ms:8.3f} ms | "
                f"x{speed if math.isfinite(speed) else float('nan'):>6.2f}"
            )

    # ------------- Save & Summary -------------
    os.makedirs("./results/bench", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"./results/bench/inference_bench_{stamp}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "N","batch_size","batch_mode","batch_style","method","device",
            "median_ms_batch","mean_ms_batch","median_ms_per_item",
            "newton_ms_batch","newton_ms_per_item",
            "speedup_vs_newton_batch","speedup_vs_newton_per_item",
        ])
        writer.writeheader();  writer.writerows(rows)

    print("\n=== Summary (median ms per batch; higher is faster speedup) ===")
    byN = defaultdict(list)
    for r in rows: byN[r["N"]].append(r)
    for N in sorted(byN):
        print(f"\nN={N}")
        for r in sorted(byN[N], key=lambda x: x["median_ms_batch"] if math.isfinite(x["median_ms_batch"]) else 1e18):
            print(f"  B={r['batch_size']:3d} [{r['batch_style']}] {r['method']:28s} "
                  f"batch {r['median_ms_batch']:8} ms  "
                  f"NR {r['newton_ms_batch']:8} ms  x{r['speedup_vs_newton_batch']}")

    print(f"\nSaved CSV → {out_csv}")


if __name__ == "__main__":
    main()
