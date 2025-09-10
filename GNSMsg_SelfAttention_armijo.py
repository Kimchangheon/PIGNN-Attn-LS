# GNSMsg_EdgeSelfAttn.py (optimized)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max

# --------------------------- utils ---------------------------

@torch.no_grad()
def _segmented_softmax(logits_b_e_h: torch.Tensor, dst_e: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Vectorized softmax over incoming edges per (batch, head, node).
    logits_b_e_h: (B, E, H)
    dst_e:        (E,) destination node index in [0..N)
    returns:      (B, E, H) normalized weights
    """
    B, E, H = logits_b_e_h.shape
    device = logits_b_e_h.device
    N = num_nodes
    # Build a single segment-id that encodes (batch, head, dst)
    b_ids = torch.arange(B, device=device).view(B, 1, 1)
    h_ids = torch.arange(H, device=device).view(1, 1, H)
    dst = dst_e.view(1, E, 1)
    seg = (b_ids * (N * H)) + (h_ids * N) + dst              # (B,E,H)
    seg = seg.reshape(-1)                                    # (B*E*H,)

    src = logits_b_e_h.reshape(-1)                           # (B*E*H,)

    # subtract segment-wise max for stability
    max_per_seg, _ = scatter_max(src, seg, dim=0, dim_size=B * N * H)
    max_g = max_per_seg.index_select(0, seg)                 # gather
    x = torch.exp(src - max_g)
    denom = scatter_add(x, seg, dim=0, dim_size=B * N * H)
    denom_g = denom.index_select(0, seg)
    alpha = (x / (denom_g + 1e-12)).reshape(B, E, H)         # (B,E,H)
    return alpha

def _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask):
    """
    Compute ∞-norm of power mismatch for (possibly) batched candidate voltages.
    Y:      (B?, N, N) or (1, N, N)
    v, th:  (..., N)
    Returns: scalar tensor
    """
    # Broadcast Y across leading alpha/batch dims
    # Vc: (..., N) complex
    Vc = v * torch.exp(1j * th)
    # align Y to match ... dims
    Yb = Y
    while Yb.dim() < Vc.dim() + 1:  # want Yb.shape == (..., N, N)
        Yb = Yb.unsqueeze(0)
    Ic = torch.matmul(Yb, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()
    DP = (P_set - Sc.real)
    DQ = (Q_set - Sc.imag)
    # broadcast masks
    DP = DP.masked_fill(slack_mask, 0.0)
    DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)
    DP_max = DP.abs().amax(dim=-1)
    DQ_max = DQ.abs().amax(dim=-1)
    return torch.maximum(DP_max, DQ_max).amax()

# --------------------- attention block -----------------------

class EdgeSelfAttnBlock(nn.Module):
    """
    Sparse graph self-attention with edge bias.
    Uses segmented-softmax (vectorized) over incoming edges per (batch, head, node).
    """
    def __init__(self, d_model: int, n_heads: int, edge_feat_dim: int, ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feat_dim, max(8, edge_feat_dim * 2)),
            nn.LeakyReLU(0.1),
            nn.Linear(max(8, edge_feat_dim * 2), self.h)  # per-head bias
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        hid = ffn_hidden or (4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_dir: torch.Tensor, edge_feat_dir: torch.Tensor):
        """
        x:             (B, N, D)
        edge_index_dir:(E, 2) directed (src=j, dst=i)
        edge_feat_dir: (E, F)
        """
        B, N, D = x.shape
        device = x.device
        src = edge_index_dir[:, 0]
        dst = edge_index_dir[:, 1]

        y = self.ln1(x)
        Q = self.q(y).view(B, N, self.h, self.dh)   # (B,N,H,dh)
        K = self.k(y).view(B, N, self.h, self.dh)
        V = self.v(y).view(B, N, self.h, self.dh)

        Qi = Q[:, dst, :, :]                        # (B,E,H,dh)
        Kj = K[:, src, :, :]                        # (B,E,H,dh)
        Vj = V[:, src, :, :]                        # (B,E,H,dh)

        logits = (Qi * Kj).sum(dim=-1) / math.sqrt(self.dh)    # (B,E,H)
        bias = self.edge_bias(edge_feat_dir).unsqueeze(0)      # (1,E,H)
        logits = logits + bias

        # segmented softmax over incoming edges per (batch, head, node)
        alpha = _segmented_softmax(logits, dst, N)             # (B,E,H)
        attn_msg = alpha.unsqueeze(-1) * Vj                    # (B,E,H,dh)

        # aggregate to dst nodes
        out = torch.zeros(B, N, self.h, self.dh, device=device, dtype=x.dtype)
        out.index_add_(1, dst, attn_msg)                       # (B,N,H,dh)
        out = self.drop(self.out(out.reshape(B, N, D)))
        x = x + out

        z = self.ln2(x)
        z = self.drop(self.ffn(z))
        return x + z

# ------------------ main model ------------------

class GNSMsg_EdgeSelfAttn(nn.Module):
    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
        d_model: int = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.K, self.d, self.d_hi = K, d, d_hi
        self.pinn, self.gamma, self.v_limit, self.use_armijo = pinn, gamma, v_limit, use_armijo

        self.d_model = d_model if d_model is not None else d_hi
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.num_attn_layers = num_attn_layers

        self.bus_feat_dim = 4 + d      # [v, θ, ΔP, ΔQ] + m
        self.edge_feat_dim = 3         # [Ysr, Ysi, Yc]

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(self.d_model, self.n_heads, self.edge_feat_dim,
                              ffn_hidden=4 * self.d_model, dropout=attn_dropout)
            for _ in range(self.num_attn_layers)
        ])

        # per-iteration heads
        self.theta_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.v_head     = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.m_head     = nn.ModuleList([nn.Linear(self.d_model, d) for _ in range(K)])

        for k in range(K):
            nn.init.zeros_(self.theta_head[k].weight); nn.init.zeros_(self.theta_head[k].bias)
            nn.init.zeros_(self.v_head[k].weight);     nn.init.zeros_(self.v_head[k].bias)
            nn.init.zeros_(self.m_head[k].weight);     nn.init.zeros_(self.m_head[k].bias)

        # cache for undirected pair indices per graph size
        self._pair_cache = {}  # n -> (pairs_n on device)

    @torch.no_grad()
    def _pairs_for_n(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Return upper-triangular undirected pairs for n nodes: shape (n*(n-1)//2, 2)
        Cached per (n, device).
        """
        key = (n, device)
        if key in self._pair_cache:
            return self._pair_cache[key]
        # vectorized triu
        iu = torch.triu_indices(n, n, offset=1, device=device)
        pairs = iu.t().contiguous()  # (e_all, 2)
        self._pair_cache[key] = pairs
        return pairs

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph):
        device = bus_type.device
        B, N = bus_type.shape  # for blockdiag batching, B==1 and N==sum of subgraphs

        Ysr, Ysi = Ys.real, Ys.imag
        P_set, Q_set = S.real, S.imag

        v  = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m  = torch.zeros(B, N, self.d, device=device)

        # -------- build edge lists once (outside K loop) --------
        if n_nodes_per_graph is not None:
            Line = Line.squeeze(0) if Line.dim() == 2 else Line
            Ysr, Ysi, Yc = Ysr.squeeze(0), Ysi.squeeze(0), Yc.squeeze(0)

            edge_index_parts, edge_feat_parts = [], []
            ptr = 0
            offset = 0
            for n in n_nodes_per_graph:
                e_all = n * (n - 1) // 2
                mask_g = Line[ptr:ptr + e_all]
                if mask_g.any():
                    pairs_g = self._pairs_for_n(int(n), device)              # (e_all,2)
                    e_idx_g = pairs_g[mask_g] + offset
                    edge_index_parts.append(e_idx_g)
                    feat_g = torch.stack([Ysr[ptr:ptr+e_all][mask_g],
                                          Ysi[ptr:ptr+e_all][mask_g],
                                          Yc[ptr:ptr+e_all][mask_g]], dim=-1)
                    edge_feat_parts.append(feat_g)
                ptr += e_all
                offset += int(n)

            if edge_index_parts:
                undirected = torch.cat(edge_index_parts, dim=0)             # (E,2)
                edge_feat  = torch.cat(edge_feat_parts, dim=0)              # (E,3)
            else:
                undirected = torch.empty(0, 2, dtype=torch.long, device=device)
                edge_feat  = torch.empty(0, 3, dtype=Ysr.dtype, device=device)

            # directed duplication for attention
            edge_index_dir = torch.cat([undirected, undirected[:, [1, 0]]], dim=0)  # (2E,2)
            edge_feat_dir  = torch.cat([edge_feat,  edge_feat], dim=0)              # (2E,3)
        else:
            # plain batching (fallback): build per-graph directed edges
            pairs = self._pairs_for_n(N, device)  # assume all items share N
            edge_index_dir_list, edge_feat_dir_list = [], []
            for b in range(B):
                mask = Line[b]
                e = pairs[mask]
                feat_b = torch.stack([Ysr[b, mask], Ysi[b, mask], Yc[b, mask]], dim=-1)
                edge_index_dir_list.append(torch.cat([e, e[:, [1, 0]]], dim=0))
                edge_feat_dir_list.append(torch.cat([feat_b, feat_b], dim=0))

        slack_mask = (bus_type == 1)
        pv_mask    = (bus_type == 2)

        if self.pinn:
            phys_loss = torch.zeros(1, device=device)

        # ------------------------- K iterations -------------------------
        for k in range(self.K):
            # power mismatches
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()
            DP = (P_set - Sc.real)
            DQ = (Q_set - Sc.imag)
            DP = DP.masked_fill(slack_mask, 0.0)
            DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)

            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)   # (B,N,4)
            ctx = self.in_proj(torch.cat([bus_feat, m], dim=-1))

            if n_nodes_per_graph is not None:
                x = ctx
                for blk in self.blocks:
                    x = blk(x, edge_index_dir, edge_feat_dir)  # (B,N,D)
            else:
                x = ctx.clone()
                for b in range(B):
                    if edge_index_dir_list[b].numel() == 0:
                        continue
                    xb = x[b:b+1]
                    e_b, ef_b = edge_index_dir_list[b], edge_feat_dir_list[b]
                    for blk in self.blocks:
                        xb = blk(xb, e_b, ef_b)
                    x[b:b+1] = xb

            dth = self.theta_head[k](x).squeeze(-1)
            dv  = self.v_head[k](x).squeeze(-1)
            dm  = torch.tanh(self.m_head[k](x))
            dm  = F.layer_norm(dm, dm.shape[-1:])

            # constraints
            dth = dth.clone(); dv = dv.clone()
            dth = dth.masked_fill(slack_mask, 0.0)
            dv  = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                dtheta_max = 0.30
                dvm_frac   = 0.10
                v_abs = v.abs()
                dth = torch.clamp(dth, -dtheta_max, dtheta_max)
                dv  = torch.clamp(dv, -dvm_frac * v_abs, dvm_frac * v_abs)

            # -------- Armijo (vectorized tries) --------
            if self.use_armijo:
                v_min, v_max = 0.8, 1.2
                F0 = _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask)

                # Try several alphas in parallel (one batched matmul)
                # tune depth as needed (trade accuracy vs speed)
                alphas = v.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625])
                T = alphas.numel()

                v_try = torch.clamp(v.unsqueeze(0) + alphas.view(T, 1, 1) * dv.unsqueeze(0), v_min, v_max)
                th_try = th.unsqueeze(0) + alphas.view(T, 1, 1) * dth.unsqueeze(0)
                th_try = (th_try + math.pi) % (2 * math.pi) - math.pi

                # Evaluate F for all alphas at once
                F_all = []
                for t in range(T):
                    F_all.append(_batched_mismatch_inf_norm(Y, v_try[t], th_try[t], P_set, Q_set, slack_mask, pv_mask))
                F_all = torch.stack(F_all)  # (T,)

                c1 = 1e-4
                cond = F_all <= (1.0 - c1 * alphas) * F0
                if cond.any():
                    t_sel = int(torch.nonzero(cond, as_tuple=False)[0].item())
                    a = float(alphas[t_sel])
                    v = v_try[t_sel]
                    th = th_try[t_sel]
                    m = m + a * dm
                else:
                    # tiny nudge if it helps
                    a = float(alphas[-1])
                    v2 = torch.clamp(v + a * dv, v_min, v_max)
                    th2 = th + a * dth
                    th2 = (th2 + math.pi) % (2 * math.pi) - math.pi
                    if _batched_mismatch_inf_norm(Y, v2, th2, P_set, Q_set, slack_mask, pv_mask) < F0:
                        v, th, m = v2, th2, m + a * dm
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v  = torch.clamp(v + dv, 0.8, 1.2)
                m  = m + dm

            if self.pinn:
                phys_loss = phys_loss + (self.gamma ** (self.K - 1 - k)) * ((DP**2 + DQ**2).mean())

        out = torch.stack([v, th], dim=-1)
        return (out, phys_loss) if self.pinn else out
