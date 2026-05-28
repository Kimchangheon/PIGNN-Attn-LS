"""EdgeSelfAttn model package.

Public entrypoint:
    from models.edge_selfattn import GNSMsg_EdgeSelfAttn

Importing this package also imports `builder`, which self-registers
`GNSMsg_EdgeSelfAttn` into `models.registry.MODEL_REGISTRY`.
"""

from .model import GNSMsg_EdgeSelfAttn
from . import builder  # noqa: F401  -- side-effect: registers the model

__all__ = ["GNSMsg_EdgeSelfAttn"]
