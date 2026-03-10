from typing import Dict, List, Union
import torch


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #

_VECTOR_FIELDS = [
    "bus_type",
    "vn_kv",
    "V_base_bus",
    "Y_shunt_bus",
    "U_start",
    "U_newton",
    "S_start",
    "S_newton",
]

_TWO_CHANNEL_FIELDS = [
    "V_start",
    "V_newton",
]

_BRANCH_FIELDS = [
    "Branch_status",
    "Branch_tau",
    "Branch_shift_deg",
    "Branch_y_series_from",
    "Branch_y_series_to",
    "Branch_y_series_ft",
    "Branch_y_shunt_from",
    "Branch_y_shunt_to",
    "Is_trafo",
    "Branch_hv_is_f",
    "Branch_n",
    "Y_Lines",
    "Y_C_Lines",
]

_BRANCH_INDEX_FIELDS = [
    "Branch_f_bus",
    "Branch_t_bus",
]

_MATRIX_FIELDS = [
    "Ybus",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _concat_present(samples, fields, dim=0):
    out = {}
    for f in fields:
        vals = [s.get(f, None) for s in samples]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[f] = torch.cat(vals, dim=dim)
    return out


# --------------------------------------------------------------------------- #
# main API
# --------------------------------------------------------------------------- #

def collate_blockdiag(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Collate per-sample graphs into one block-diagonal mega-graph.

    Important:
      - bus-index branch fields are offset to global node numbering
      - Ybus is block-diagonalized
      - everything gets an explicit batch dimension B=1
    """
    out: Dict[str, Union[torch.Tensor, None]] = {}

    device_for_meta = None
    for s in samples:
        for v in s.values():
            if isinstance(v, torch.Tensor):
                device_for_meta = v.device
                break
        if device_for_meta is not None:
            break
    if device_for_meta is None:
        device_for_meta = torch.device("cpu")

    sizes = torch.tensor([int(s["N"]) for s in samples], device=device_for_meta, dtype=torch.long)
    branch_sizes = torch.tensor([int(s["nl"]) for s in samples], device=device_for_meta, dtype=torch.long)
    offsets = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]))

    out.update(_concat_present(samples, _VECTOR_FIELDS, dim=0))
    out.update(_concat_present(samples, _TWO_CHANNEL_FIELDS, dim=0))
    out.update(_concat_present(samples, _BRANCH_FIELDS, dim=0))

    for name in _BRANCH_INDEX_FIELDS:
        vals = []
        for s, off in zip(samples, offsets):
            vals.append(s[name] + off)
        out[name] = torch.cat(vals, dim=0)

    have_all_y = all(isinstance(s.get("Ybus", None), torch.Tensor) for s in samples)
    if have_all_y and len(samples) > 0:
        out["Ybus"] = torch.block_diag(*[s["Ybus"] for s in samples])
    else:
        out["Ybus"] = None

    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)

    out["sizes"] = sizes
    out["branch_sizes"] = branch_sizes
    out["offsets"] = offsets

    return out