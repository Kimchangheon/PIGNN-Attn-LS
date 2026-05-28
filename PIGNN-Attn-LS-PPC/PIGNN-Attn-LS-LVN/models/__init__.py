"""Model architectures (standalone PIGNN-Attn-LS / LVN bundle).

Only the `edge_selfattn` package is bundled here. Importing this manifest
runs its builder, which self-registers:
  - GNSMsg_EdgeSelfAttn         (PIGNN-Attn-LS, single-voltage HVN/MVN)
  - GNSMsg_EdgeSelfAttn_VnFeat  (PIGNN-Attn-LS + vn_log per-bus feature;
                                 the LVN-targeted variant)
  - GNSMsg_EdgeSelfAttn_Tied    (weight-tied K-iteration baseline)
"""

from . import edge_selfattn  # noqa: F401  -- side-effect: registers the models
