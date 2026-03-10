import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max


# --------------------------- utils ---------------------------

def _segmented_softmax(logits_b_e_h: torch.Tensor, dst_e: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax over incoming edges per (batch, head, destination-node).
    logits_b_e_h: (B, E, H)
    dst_e:        (E,)
    """
    B, E, H = logits_b_e_h.shape
    device = logits_b_e_h.device
    N = num_nodes

    b_ids = torch.arange(B, device=device).view(B, 1, 1)
    h_ids = torch.arange(H, device=device).view(1, 1, H)
    dst = dst_e.view(1, E, 1)

    seg = (b_ids * (N * H)) + (h_ids * N) + dst
    seg = seg.reshape(-1)
    src = logits_b_e_h.reshape(-1)

    max_per_seg, _ = scatter_max(src, seg, dim=0, dim_size=B * N * H)
    max_g = max_per_seg.index_select(0, seg)
    x = torch.exp(src - max_g)

    denom = scatter_add(x, seg, dim=0, dim_size=B * N * H)
    denom_g = denom.index_select(0, seg)

    alpha = (x / (denom_g + 1e-12)).reshape(B, E, H)
    return alpha


def _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask):
    Vc = v * torch.exp(1j * th)
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)
    Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()

    DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
    DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

    return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1)).amax()


def _build_dense_Y_from_branchrows_single(
    N: int,
    Branch_f_bus: torch.Tensor,
    Branch_t_bus: torch.Tensor,
    Branch_status: torch.Tensor,
    Branch_tau: torch.Tensor,
    Branch_shift_deg: torch.Tensor,
    Branch_y_series_from: torch.Tensor,
    Branch_y_series_to: torch.Tensor,
    Branch_y_series_ft: torch.Tensor,
    Branch_y_shunt_from: torch.Tensor,
    Branch_y_shunt_to: torch.Tensor,
    Y_shunt_bus: torch.Tensor,
) -> torch.Tensor:
    """
    Direct-SI dense Ybus reconstruction from one metadata row per PPC branch row.
    """
    device = Branch_f_bus.device
    dtype = Branch_y_series_ft.dtype

    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(Y_shunt_bus.to(dtype))

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    tau = Branch_tau[mask].to(dtype.real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(dtype.real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to   = Branch_y_series_to[mask].to(dtype)
    y_ft   = Branch_y_series_ft[mask].to(dtype)
    ysh_f  = Branch_y_shunt_from[mask].to(dtype)
    ysh_t  = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = (y_to   + ysh_t / 2.0)
    Yft = -y_ft / torch.conj(a)
    Ytf = -y_ft / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)

    return Y


def _build_directed_edges_single(
    Branch_f_bus: torch.Tensor,
    Branch_t_bus: torch.Tensor,
    Branch_status: torch.Tensor,
    Branch_tau: torch.Tensor,
    Branch_shift_deg: torch.Tensor,
    Branch_y_series_ft: torch.Tensor,
    Branch_y_shunt_from: torch.Tensor,
    Branch_y_shunt_to: torch.Tensor,
    Is_trafo: torch.Tensor,
):
    """
    Build directed sparse edge list and direction-aware edge features from branch rows.

    Edge feature for f->t:
      [Re(Yft_dir), Im(Yft_dir),
       Re(ysh_f),   Im(ysh_f),
       Re(ysh_t),   Im(ysh_t),
       tau, theta_rad, is_trafo]

    Edge feature for t->f swaps the shunts and flips theta sign.
    """
    device = Branch_f_bus.device
    ctype = Branch_y_series_ft.dtype
    rtype = Branch_tau.dtype

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, 9, dtype=rtype, device=device)
        )

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    tau = Branch_tau[mask].to(rtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(rtype))
    is_tr = Is_trafo[mask].to(rtype)

    a = tau.to(ctype) * torch.exp(1j * theta.to(ctype))

    y_ft = Branch_y_series_ft[mask].to(ctype)
    ysh_f = Branch_y_shunt_from[mask].to(ctype)
    ysh_t = Branch_y_shunt_to[mask].to(ctype)

    ydir_ft = -y_ft / torch.conj(a)  # f -> t
    ydir_tf = -y_ft / a              # t -> f

    feat_ft = torch.stack([
        ydir_ft.real.to(rtype), ydir_ft.imag.to(rtype),
        ysh_f.real.to(rtype),   ysh_f.imag.to(rtype),
        ysh_t.real.to(rtype),   ysh_t.imag.to(rtype),
        tau, theta, is_tr
    ], dim=-1)

    feat_tf = torch.stack([
        ydir_tf.real.to(rtype), ydir_tf.imag.to(rtype),
        ysh_t.real.to(rtype),   ysh_t.imag.to(rtype),
        ysh_f.real.to(rtype),   ysh_f.imag.to(rtype),
        tau, -theta, is_tr
    ], dim=-1)

    edge_index = torch.cat([
        torch.stack([f, t], dim=1),
        torch.stack([t, f], dim=1),
    ], dim=0)

    edge_feat = torch.cat([feat_ft, feat_tf], dim=0)
    return edge_index, edge_feat


# --------------------- attention block -----------------------

class EdgeSelfAttnBlock(nn.Module):
    """
    Sparse graph self-attention with edge bias.
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
            nn.Linear(edge_feat_dim, max(16, 2 * edge_feat_dim)),
            nn.LeakyReLU(0.1),
            nn.Linear(max(16, 2 * edge_feat_dim), self.h)
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
        edge_index_dir:(E, 2)
        edge_feat_dir: (E, F)
        """
        B, N, D = x.shape
        device = x.device
        src = edge_index_dir[:, 0]
        dst = edge_index_dir[:, 1]

        y = self.ln1(x)
        Q = self.q(y).view(B, N, self.h, self.dh)
        K = self.k(y).view(B, N, self.h, self.dh)
        V = self.v(y).view(B, N, self.h, self.dh)

        Qi = Q[:, dst, :, :]
        Kj = K[:, src, :, :]
        Vj = V[:, src, :, :]

        logits = (Qi * Kj).sum(dim=-1) / math.sqrt(self.dh)
        bias = self.edge_bias(edge_feat_dir).unsqueeze(0)
        logits = logits + bias

        alpha = _segmented_softmax(logits, dst, N)
        attn_msg = alpha.unsqueeze(-1) * Vj

        out = torch.zeros(B, N, self.h, self.dh, device=device, dtype=x.dtype)
        out.index_add_(1, dst, attn_msg)
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
        self.K = K
        self.d = d
        self.d_hi = d_hi
        self.pinn = pinn
        self.gamma = gamma
        self.v_limit = v_limit
        self.use_armijo = use_armijo

        self.d_model = d_model if d_model is not None else d_hi
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        self.num_attn_layers = num_attn_layers

        self.bus_feat_dim = 4 + d
        self.edge_feat_dim = 9   # <-- now includes tau + theta + is_trafo

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(
                self.d_model,
                self.n_heads,
                self.edge_feat_dim,
                ffn_hidden=4 * self.d_model,
                dropout=attn_dropout
            )
            for _ in range(self.num_attn_layers)
        ])

        self.theta_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.v_head     = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.m_head     = nn.ModuleList([nn.Linear(self.d_model, d) for _ in range(K)])

        for k in range(K):
            nn.init.zeros_(self.theta_head[k].weight); nn.init.zeros_(self.theta_head[k].bias)
            nn.init.zeros_(self.v_head[k].weight);     nn.init.zeros_(self.v_head[k].bias)
            nn.init.zeros_(self.m_head[k].weight);     nn.init.zeros_(self.m_head[k].bias)

    def forward(
        self,
        bus_type,
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
        Is_trafo,
        Y,
        S,
        V0,
        n_nodes_per_graph=None,
        Y_shunt_bus=None,
    ):
        """
        Works for:
          - blockdiag batching (recommended): B=1, global node indices in branch rows
          - plain batching fallback: B>1, each batch item has local branch rows
        """
        device = bus_type.device
        B, N = bus_type.shape

        # reconstruct Y if needed
        if Y is None:
            if Y_shunt_bus is None:
                raise ValueError("Y is None and Y_shunt_bus is also None; cannot reconstruct Y.")
            if B == 1:
                Y = _build_dense_Y_from_branchrows_single(
                    N,
                    Branch_f_bus.squeeze(0),
                    Branch_t_bus.squeeze(0),
                    Branch_status.squeeze(0),
                    Branch_tau.squeeze(0),
                    Branch_shift_deg.squeeze(0),
                    Branch_y_series_from.squeeze(0),
                    Branch_y_series_to.squeeze(0),
                    Branch_y_series_ft.squeeze(0),
                    Branch_y_shunt_from.squeeze(0),
                    Branch_y_shunt_to.squeeze(0),
                    Y_shunt_bus.squeeze(0),
                ).unsqueeze(0)
            else:
                Ys = []
                for b in range(B):
                    Ys.append(_build_dense_Y_from_branchrows_single(
                        N,
                        Branch_f_bus[b], Branch_t_bus[b], Branch_status[b],
                        Branch_tau[b], Branch_shift_deg[b],
                        Branch_y_series_from[b], Branch_y_series_to[b], Branch_y_series_ft[b],
                        Branch_y_shunt_from[b], Branch_y_shunt_to[b],
                        Y_shunt_bus[b],
                    ))
                Y = torch.stack(Ys, dim=0)
        else:
            if Y.dim() == 2:
                Y = Y.unsqueeze(0)

        # -------- build sparse attention graph --------
        if B == 1:
            edge_index_dir, edge_feat_dir = _build_directed_edges_single(
                Branch_f_bus.squeeze(0),
                Branch_t_bus.squeeze(0),
                Branch_status.squeeze(0),
                Branch_tau.squeeze(0),
                Branch_shift_deg.squeeze(0),
                Branch_y_series_ft.squeeze(0),
                Branch_y_shunt_from.squeeze(0),
                Branch_y_shunt_to.squeeze(0),
                Is_trafo.squeeze(0),
            )
            edge_index_dir_list = None
            edge_feat_dir_list = None
        else:
            edge_index_dir = None
            edge_feat_dir = None
            edge_index_dir_list = []
            edge_feat_dir_list = []
            for b in range(B):
                ei, ef = _build_directed_edges_single(
                    Branch_f_bus[b],
                    Branch_t_bus[b],
                    Branch_status[b],
                    Branch_tau[b],
                    Branch_shift_deg[b],
                    Branch_y_series_ft[b],
                    Branch_y_shunt_from[b],
                    Branch_y_shunt_to[b],
                    Is_trafo[b],
                )
                edge_index_dir_list.append(ei)
                edge_feat_dir_list.append(ef)

        # -------- initialize states --------
        P_set, Q_set = S.real, S.imag
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = torch.zeros(B, N, self.d, device=device, dtype=V0.dtype)

        slack_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)

        phys_terms = []

        # ------------------------- K iterations -------------------------
        for k in range(self.K):
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()

            DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
            x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

            # sparse attention message passing
            if B == 1:
                for blk in self.blocks:
                    x = blk(x, edge_index_dir, edge_feat_dir)
            else:
                x_new = x.clone()
                for b in range(B):
                    xb = x[b:b + 1]
                    ei = edge_index_dir_list[b]
                    ef = edge_feat_dir_list[b]
                    if ei.numel() == 0:
                        x_new[b:b + 1] = xb
                        continue
                    for blk in self.blocks:
                        xb = blk(xb, ei, ef)
                    x_new[b:b + 1] = xb
                x = x_new

            dth = self.theta_head[k](x).squeeze(-1)
            dv  = self.v_head[k](x).squeeze(-1)
            dm  = torch.tanh(self.m_head[k](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            dth = dth.masked_fill(slack_mask, 0.0)
            dv  = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                dtheta_max = 0.30
                dvm_frac = 0.10
                v_abs = v.abs()
                dth = torch.clamp(dth, -dtheta_max, dtheta_max)
                dv = torch.clamp(dv, -dvm_frac * v_abs, dvm_frac * v_abs)

            if self.use_armijo:
                v_min, v_max = 0.75, 1.20

                with torch.no_grad():
                    F0 = _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask)

                alphas = v.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625])
                T = int(alphas.numel())

                v_try = torch.clamp(v.unsqueeze(0) + alphas.view(T, 1, 1) * dv.unsqueeze(0), v_min, v_max)
                th_try = (th.unsqueeze(0) + alphas.view(T, 1, 1) * dth.unsqueeze(0) + math.pi) % (2 * math.pi) - math.pi

                with torch.no_grad():
                    F_all = torch.stack([
                        _batched_mismatch_inf_norm(Y, v_try[t], th_try[t], P_set, Q_set, slack_mask, pv_mask)
                        for t in range(T)
                    ])
                    c1 = 1e-4
                    cond = F_all <= (1.0 - c1 * alphas) * F0
                    found = bool(cond.any())

                if found:
                    t_sel = int(torch.nonzero(cond, as_tuple=False)[0].item())
                    a = float(alphas[t_sel])
                    v = v_try[t_sel]
                    th = th_try[t_sel]
                    m = m + a * dm
                else:
                    a = float(alphas[-1])
                    v2 = torch.clamp(v + a * dv, v_min, v_max)
                    th2 = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    with torch.no_grad():
                        ok = _batched_mismatch_inf_norm(Y, v2, th2, P_set, Q_set, slack_mask, pv_mask) < F0
                    if ok:
                        v, th, m = v2, th2, m + a * dm
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, 0.75, 1.20)
                m = m + dm

            if self.pinn:
                term = (self.gamma ** (self.K - 1 - k)) * ((DP ** 2 + DQ ** 2).mean())
                phys_terms.append(term)

        out = torch.stack([v, th], dim=-1)

        if self.pinn:
            phys_loss = torch.sum(torch.stack(phys_terms))
            return out, phys_loss
        else:
            return out