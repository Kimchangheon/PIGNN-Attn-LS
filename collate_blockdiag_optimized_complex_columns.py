from typing import Dict, List

import torch

# --------------------------------------------------------------------------- #
#   Fields present in each per-grid sample produced by the Dataset class      #
# --------------------------------------------------------------------------- #
# Per-bus vectors (length N). Complex vectors are fine here.
_VECTOR_FIELDS = (
    "bus_type U_start U_newton S_start S_newton I_newton".split()
)  # shape: (N,)

# Two-channel per-bus tensors (mag, angle)
_TWO_CHANNEL_FIELDS = ("V_start", "V_newton")  # shape: (N, 2)

# Per-edge tensors (variable length across samples). Includes complex Y_Lines and currents.
_EDGE_FIELDS = (
    "Lines_connected Y_Lines Y_C_Lines".split()
)  # variable length per-edge tensors

# Square matrix fields (N×N). Ybus is complex.
_MATRIX_FIELDS = ("Ybus",)

# --------------------------------------------------------------------------- #
#                               helpers                                        #
# --------------------------------------------------------------------------- #

def _concat(samples: List[Dict[str, torch.Tensor]], fields, dim: int = 0):
    """Concatenate *fields* extracted from *samples* along *dim*."""
    return {f: torch.cat([s[f] for s in samples], dim=dim) for f in fields}

# --------------------------------------------------------------------------- #
#                                main API                                      #
# --------------------------------------------------------------------------- #

def collate_blockdiag(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a list of per-grid samples into **one** block-diagonal mega-grid.

    The resulting batch has logical batch-size **1** (all individual grids
    are merged into a single disconnected graph). This keeps downstream code
    simple, while still allowing variable-sized power grids per sample.

    The function is intentionally *device-agnostic*: tensors keep whatever
    device/dtype they already live on, so you can load data directly to GPU
    during Dataset creation and avoid extra copies here.
    """
    out: Dict[str, torch.Tensor] = {}

    # ── 1. concatenate per-bus / per-edge tensors ──────────────────────────
    out.update(_concat(samples, _VECTOR_FIELDS, dim=0))         # ⇒ (M,)
    out.update(_concat(samples, _TWO_CHANNEL_FIELDS, dim=0))    # ⇒ (M, 2)
    out.update(_concat(samples, _EDGE_FIELDS, dim=0))           # ⇒ (E_total, ...)

    # ── 2. build block-diagonal admittance matrices (complex) ──────────────
    # Only one matrix field: Ybus
    out["Ybus"] = torch.block_diag(*[s["Ybus"] for s in samples])  # ⇒ (M, M)

    # ── 3. bookkeeping: where each grid starts inside the mega-grid ────────
    sizes = torch.tensor([s["bus_type"].numel() for s in samples], device=out["bus_type"].device)
    offsets = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]))

    # ── 4. add explicit batch dimension (B = 1) ────────────────────────────
    for k, v in out.items():
        if k in _MATRIX_FIELDS:            # matrices (M, M)
            out[k] = v.unsqueeze(0)        # ⇒ (1, M, M)
        elif k in _TWO_CHANNEL_FIELDS:     # (M, 2)
            out[k] = v.unsqueeze(0)        # ⇒ (1, M, 2)
        else:
            out[k] = v.unsqueeze(0)        # ⇒ (1, M) or (1, E, *)

    # keep raw tensors for optional use (no batch dim)
    out["offsets"] = offsets               # (num_grids,)
    out["sizes"] = sizes                   # (num_grids,)

    return out
