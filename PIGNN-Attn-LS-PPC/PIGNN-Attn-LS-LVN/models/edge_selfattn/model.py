from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .admittance import build_dense_Y, build_edges_blockdiag, build_edges_plain
from .attention import EdgeSelfAttnBlock
from .mismatch import mismatch_inf_norm_per_candidate


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
        d_model: int | None = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0,
        dtheta_max: float = 0.30,
        dvm_frac: float = 0.10,
        v_min: float = 0.75,
        v_max: float = 1.20,
        tied_heads: bool = False,
        bus_feat_extra_dim: int = 0,
    ):
        super().__init__()
        self.K, self.d, self.d_hi = int(K), int(d), int(d_hi)
        self.pinn = bool(pinn)
        self.gamma = float(gamma)
        self.v_limit = bool(v_limit)
        self.use_armijo = bool(use_armijo)

        self.dtheta_max = float(dtheta_max)
        self.dvm_frac = float(dvm_frac)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.d_model = int(d_model if d_model is not None else d_hi)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_attn_layers = int(num_attn_layers)

        # bus_feat_extra_dim: extra per-bus scalar features appended to
        # [v, theta, dP, dQ]. Currently used for vn_log (per-bus voltage
        # class) on LVN data. HVN/default keeps it at 0 for backward compat
        # with all existing checkpoints.
        self.bus_feat_extra_dim = int(bus_feat_extra_dim)
        self.bus_feat_dim = 4 + self.bus_feat_extra_dim + self.d
        self.edge_feat_dim = 4  # [Ysr, Ysi, Yc_real, Yc_imag]

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [
                EdgeSelfAttnBlock(
                    self.d_model,
                    self.n_heads,
                    self.edge_feat_dim,
                    ffn_hidden=4 * self.d_model,
                    dropout=attn_dropout,
                )
                for _ in range(self.num_attn_layers)
            ]
        )

        # tied_heads: share a single (theta/v/m) head set across all K iterations.
        # Untied (default) matches PIGNN-Attn-LS; tied gives a fair-parameter
        # comparison against the weight-shared PE_DEQ_PF operator.
        self.tied_heads = bool(tied_heads)
        n_heads_sets = 1 if self.tied_heads else self.K
        self.theta_head = nn.ModuleList(
            [nn.Linear(self.d_model, 1) for _ in range(n_heads_sets)]
        )
        self.v_head = nn.ModuleList(
            [nn.Linear(self.d_model, 1) for _ in range(n_heads_sets)]
        )
        self.m_head = nn.ModuleList(
            [nn.Linear(self.d_model, self.d) for _ in range(n_heads_sets)]
        )

        for j in range(n_heads_sets):
            nn.init.zeros_(self.theta_head[j].weight)
            nn.init.zeros_(self.theta_head[j].bias)
            nn.init.zeros_(self.v_head[j].weight)
            nn.init.zeros_(self.v_head[j].bias)
            nn.init.zeros_(self.m_head[j].weight)
            nn.init.zeros_(self.m_head[j].bias)

        self._pair_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    @torch.no_grad()
    def _pairs_for_n(self, n: int, device: torch.device) -> torch.Tensor:
        key = (int(n), device)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        iu = torch.triu_indices(int(n), int(n), offset=1, device=device)
        pairs = iu.t().contiguous()
        self._pair_cache[key] = pairs
        return pairs

    def _apply_blocks(self, x: torch.Tensor, edge_index_dir: torch.Tensor, edge_feat_dir: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, edge_index_dir, edge_feat_dir)
        return x

    def _apply_constraints(
        self,
        *,
        v: torch.Tensor,
        dth: torch.Tensor,
        dv: torch.Tensor,
        slack_mask: torch.Tensor,
        pv_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dth = dth.masked_fill(slack_mask, 0.0)
        dv = dv.masked_fill(slack_mask | pv_mask, 0.0)

        if self.v_limit:
            v_abs = v.abs()
            dth = torch.clamp(dth, -self.dtheta_max, self.dtheta_max)
            dv = torch.clamp(dv, -self.dvm_frac * v_abs, self.dvm_frac * v_abs)

        return dth, dv

    def _wrap_theta(self, th: torch.Tensor) -> torch.Tensor:
        return (th + math.pi) % (2 * math.pi) - math.pi

    def _armijo_alpha(
        self,
        *,
        Y: torch.Tensor,
        v: torch.Tensor,
        th: torch.Tensor,
        dv: torch.Tensor,
        dth: torch.Tensor,
        P_set: torch.Tensor,
        Q_set: torch.Tensor,
        slack_mask: torch.Tensor,
        pv_mask: torch.Tensor,
    ) -> torch.Tensor:
        alphas = v.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625])

        with torch.no_grad():
            v_det = v.detach()
            th_det = th.detach()
            dv_det = dv.detach()
            dth_det = dth.detach()

            F0 = mismatch_inf_norm_per_candidate(Y, v_det, th_det, P_set, Q_set, slack_mask, pv_mask).amax()

            v_try = torch.clamp(
                v_det.unsqueeze(0) + alphas.view(-1, 1, 1) * dv_det.unsqueeze(0),
                self.v_min,
                self.v_max,
            )
            th_try = self._wrap_theta(th_det.unsqueeze(0) + alphas.view(-1, 1, 1) * dth_det.unsqueeze(0))

            F_tb = mismatch_inf_norm_per_candidate(Y, v_try, th_try, P_set, Q_set, slack_mask, pv_mask)
            F_all = F_tb.amax(dim=-1)  # (T,) max over batch

            c1 = 1e-4
            cond = F_all <= (1.0 - c1 * alphas) * F0
            if bool(cond.any()):
                t_sel = int(torch.nonzero(cond, as_tuple=False)[0].item())
                a_sel = float(alphas[t_sel].item())
            else:
                a_sel = float(alphas[-1].item())

        return v.new_tensor(a_sel)

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph, *, vn_log=None, **_unused):
        device = bus_type.device
        B, N = bus_type.shape

        P_set, Q_set = S.real, S.imag
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = v.new_zeros(B, N, self.d)

        # -------- Build edges + (optional) Y --------
        edge_index_dir = None
        edge_feat_dir = None
        edge_index_dir_list = None
        edge_feat_dir_list = None

        if n_nodes_per_graph is not None:
            Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys_1d = Ys.squeeze(0)
            Yc_1d = Yc.squeeze(0)

            undirected, _, edge_index_dir, edge_feat_dir, ys_edge, yc_edge = build_edges_blockdiag(
                line_mask_1d=Line_1d,
                Ys_1d=Ys_1d,
                Yc_1d=Yc_1d,
                n_nodes_per_graph=n_nodes_per_graph,
                edge_feat_dim=self.edge_feat_dim,
                pairs_for_n=self._pairs_for_n,
                device=device,
            )

            if Y is None:
                Y = build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)

        else:
            pairs = self._pairs_for_n(N, device)
            edge_index_dir_list, edge_feat_dir_list, undirected_list, mask_list = build_edges_plain(
                Line=Line,
                Ys=Ys,
                Yc=Yc,
                N=N,
                edge_feat_dim=self.edge_feat_dim,
                pairs=pairs,
                device=device,
            )

            if Y is None:
                Y_list = []
                for b in range(B):
                    mask = mask_list[b]
                    undirected = undirected_list[b]
                    ys_edge_b = Ys[b][mask]
                    yc_edge_b = Yc[b][mask]
                    Y_list.append(build_dense_Y(N, undirected, ys_edge_b, yc_edge_b, device=device))
                Y = torch.stack(Y_list, dim=0)

        slack_mask = bus_type == 1
        pv_mask = bus_type == 2

        phys_loss = v.new_zeros(1) if self.pinn else None

        for k in range(self.K):
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()
            DP = (P_set - Sc.real)
            DQ = (Q_set - Sc.imag)
            DP = DP.masked_fill(slack_mask, 0.0)
            DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)

            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
            if self.bus_feat_extra_dim > 0:
                # Append per-bus extra features (e.g. vn_log for LVN voltage
                # classes). Shape (B, N) -> (B, N, 1); concat to (B, N, 4+1).
                if vn_log is None:
                    extra = bus_feat.new_zeros(bus_feat.shape[:-1] + (self.bus_feat_extra_dim,))
                else:
                    extra = vn_log.unsqueeze(-1)
                    if extra.shape[-1] != self.bus_feat_extra_dim:
                        raise ValueError(
                            f"vn_log last dim {extra.shape[-1]} != bus_feat_extra_dim "
                            f"{self.bus_feat_extra_dim}"
                        )
                bus_feat = torch.cat([bus_feat, extra], dim=-1)
            x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

            if n_nodes_per_graph is not None:
                x = self._apply_blocks(x, edge_index_dir, edge_feat_dir)
            else:
                # Per-graph sparse attention (graphs are independent)
                x_out = x.clone()
                for b in range(B):
                    e_b = edge_index_dir_list[b]
                    if e_b.numel() == 0:
                        continue
                    xb = self._apply_blocks(x[b : b + 1], e_b, edge_feat_dir_list[b])
                    x_out[b : b + 1] = xb
                x = x_out

            head_idx = 0 if self.tied_heads else k
            dth = self.theta_head[head_idx](x).squeeze(-1)
            dv = self.v_head[head_idx](x).squeeze(-1)
            dm = torch.tanh(self.m_head[head_idx](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            dth, dv = self._apply_constraints(v=v, dth=dth, dv=dv, slack_mask=slack_mask, pv_mask=pv_mask)

            if self.use_armijo:
                a = self._armijo_alpha(
                    Y=Y,
                    v=v,
                    th=th,
                    dv=dv,
                    dth=dth,
                    P_set=P_set,
                    Q_set=Q_set,
                    slack_mask=slack_mask,
                    pv_mask=pv_mask,
                )
                th = self._wrap_theta(th + a * dth)
                v = torch.clamp(v + a * dv, self.v_min, self.v_max)
                m = m + a * dm
            else:
                th = self._wrap_theta(th + dth)
                v = torch.clamp(v + dv, self.v_min, self.v_max)
                m = m + dm

            if self.pinn and phys_loss is not None:
                phys_loss = phys_loss + (self.gamma ** (self.K - 1 - k)) * ((DP**2 + DQ**2).mean())

        out = torch.stack([v, th], dim=-1)
        return (out, phys_loss) if self.pinn else out
