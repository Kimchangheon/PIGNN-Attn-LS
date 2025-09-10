import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

# -----------------------------------------------------------------------
# Re-usable MLP block (unchanged API)
# -----------------------------------------------------------------------

class LearningBlock(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.lin1  = nn.Linear(dim_in,  hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lin2  = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.lin3  = nn.Linear(hidden_dim, dim_out)
        self.act   = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.act(self.norm1(self.lin1(x)))
        x = self.act(self.norm2(self.lin2(x)))
        return self.lin3(x)  # keep last layer raw (or tanh)


# -----------------------------------------------------------------------
# Optimized GNSMsg with block-diagonal Y·V, edge precompute, and cached pairs
# Mirrors the performance-minded structure used in GNSMsg_EdgeSelfAttn,
# while keeping the original MLP message-passing semantics.
# -----------------------------------------------------------------------

class GNSMsg(nn.Module):
    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
    ):
        super().__init__()
        self.K, self.d, self.d_hi = K, d, d_hi
        self.pinn, self.gamma, self.v_limit, self.use_armijo = pinn, gamma, v_limit, use_armijo

        # Edge MLPs per-iteration (same semantics as original GNSMsg)
        # input: [m_j, Ysr, Ysi, Yc] → d
        edge_in_dim = d + 3
        self.edge_mlp = nn.ModuleList([LearningBlock(edge_in_dim, d_hi, d) for _ in range(K)])

        # Per-iteration node-update heads (same shapes as original)
        in_dim = 4 + d + d  # [v, θ, ΔP, ΔQ] + m_i + Σφ
        self.theta_upd = nn.ModuleList([LearningBlock(in_dim, d_hi, 1) for _ in range(K)])
        self.v_upd     = nn.ModuleList([LearningBlock(in_dim, d_hi, 1) for _ in range(K)])
        self.m_upd     = nn.ModuleList([LearningBlock(in_dim, d_hi, d) for _ in range(K)])

        # caches for pair generation and block groups (to cut Python overhead)
        self._pair_cache = {}         # (n, device) -> pairs
        self._groups_key = None       # cache key per-forward for Y groups
        self._groups = None
        self._sizes = None

    # ---------------------- helpers: edge pairs & block groups ----------------------

    @torch.no_grad()
    def _pairs_for_n(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, device)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        iu = torch.triu_indices(n, n, offset=1, device=device)
        pairs = iu.t().contiguous()  # (e_all, 2)
        self._pair_cache[key] = pairs
        return pairs

    @torch.no_grad()
    def _build_block_groups(self, Y: torch.Tensor, sizes: torch.Tensor):
        """
        Y:     (1, M, M) complex block-diagonal matrix
        sizes: (G,) number of buses per subgraph (on CPU or GPU)
        Returns groups dict mapping block size n -> { 'Y': (K, n, n), 'ofs': [offsets] }
        along with (sizes, offsets, M).
        """
        assert Y.dim() == 3 and Y.shape[0] == 1
        device = Y.device
        sizes = sizes.to(torch.long)
        offsets = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]))
        M = int(sizes.sum().item())

        uniq = torch.unique(sizes).tolist()
        groups = {}
        for n in uniq:
            sel = (sizes == n)
            ofs = offsets[sel].tolist()
            if not ofs:
                continue
            blocks = [Y[0, o:o+n, o:o+n].contiguous() for o in ofs]
            groups[int(n)] = {'Y': torch.stack(blocks, dim=0), 'ofs': ofs}
        return groups, sizes, offsets, M

    def _bdiag_mv_groups(self, groups: dict, sizes: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Multiply block-diagonal Y (via groups) by v (shape (1, M), complex) -> (1, M)."""
        assert v.dim() == 2 and v.shape[0] == 1
        out = torch.empty_like(v)
        for n, info in groups.items():
            ofs_list = info['ofs']
            if not ofs_list:
                continue
            Vg = torch.stack([v[0, o:o+n] for o in ofs_list], dim=0)          # (K, n)
            Ig = torch.bmm(info['Y'], Vg.unsqueeze(-1)).squeeze(-1)           # (K, n)
            for t, o in enumerate(ofs_list):
                out[0, o:o+n] = Ig[t]
        return out

    def _mismatch_inf_norm_block(self, groups, sizes, v, th, P_set, Q_set, slack_mask, pv_mask):
        Vc = v * torch.exp(1j * th)                                 # (1,M)
        Ic = self._bdiag_mv_groups(groups, sizes, Vc)               # (1,M)
        Sc = Vc * Ic.conj()
        DP = (P_set - Sc.real).clone()
        DQ = (Q_set - Sc.imag).clone()
        DP.masked_fill_(slack_mask, 0.0)
        DQ.masked_fill_(slack_mask | pv_mask, 0.0)
        DP_max = DP.abs().amax(dim=-1)
        DQ_max = DQ.abs().amax(dim=-1)
        return torch.maximum(DP_max, DQ_max).amax()

    # -------------------------------- forward --------------------------------

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph):
        device = bus_type.device
        B, N = bus_type.shape

        Ysr, Ysi = Ys.real, Ys.imag
        P_set, Q_set = S.real, S.imag

        v  = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m  = torch.zeros(B, N, self.d, device=device)

        # ------------------ build edges & (optional) block groups once ------------------
        if n_nodes_per_graph is not None:
            # Block-diagonal batching expected: B == 1, N == sum(sizes)
            sizes = n_nodes_per_graph.to(torch.long)
            Line = Line.squeeze(0) if Line.dim() == 2 else Line
            Ysr, Ysi, Yc = Ysr.squeeze(0), Ysi.squeeze(0), Yc.squeeze(0)

            edge_index_parts, edge_feat_parts = [], []
            ptr = 0
            offset = 0
            for n_i in sizes.tolist():
                e_all = n_i * (n_i - 1) // 2
                mask_g = Line[ptr:ptr + e_all]
                if mask_g.any():
                    pairs_g = self._pairs_for_n(int(n_i), device)            # (e_all, 2)
                    e_idx_g = pairs_g[mask_g] + offset
                    edge_index_parts.append(e_idx_g)
                    feat_g = torch.stack([Ysr[ptr:ptr+e_all][mask_g],
                                          Ysi[ptr:ptr+e_all][mask_g],
                                          Yc[ptr:ptr+e_all][mask_g]], dim=-1).to(v.dtype)
                    edge_feat_parts.append(feat_g)
                ptr += e_all
                offset += int(n_i)

            if edge_index_parts:
                edge_index = torch.cat(edge_index_parts, dim=0)               # (E,2)
                edge_feat  = torch.cat(edge_feat_parts, dim=0)                # (E,3)
            else:
                edge_index = torch.empty(0, 2, dtype=torch.long, device=device)
                edge_feat  = torch.empty(0, 3, dtype=v.dtype, device=device)

            # degree (undirected): add 1 to both endpoints per edge
            N_total = int(sizes.sum().item())
            deg = torch.zeros(N_total, device=device, dtype=v.dtype)
            if edge_index.numel() > 0:
                ones = torch.ones(edge_index.size(0), device=device, dtype=v.dtype)
                deg.index_add_(0, edge_index[:, 0], ones)
                deg.index_add_(0, edge_index[:, 1], ones)
            A = deg.clamp_min_(1.0).reciprocal()                               # (N_total,)

            # Build & cache block groups for fast Y·V
            groups_key = (int(N), tuple(sizes.tolist()))
            if self._groups_key != groups_key:
                self._groups, self._sizes, _, _ = self._build_block_groups(Y, sizes)
                self._groups_key = groups_key
            groups = self._groups
            sizes  = self._sizes
            use_blockdiag = True
        else:
            # Fallback: per-item batching
            pairs = self._pairs_for_n(N, device)  # assume shared N across batch
            edge_index_list, edge_feat_list, A_list = [], [], []
            for b in range(B):
                mask = Line[b]
                e = pairs[mask]
                edge_index_list.append(e)
                feat_b = torch.stack([Ysr[b, mask], Ysi[b, mask], Yc[b, mask]], dim=-1).to(v.dtype)
                edge_feat_list.append(feat_b)

                deg_b = torch.zeros(N, device=device, dtype=v.dtype)
                if e.numel() > 0:
                    ones = torch.ones(e.size(0), device=device, dtype=v.dtype)
                    deg_b.index_add_(0, e[:, 0], ones)
                    deg_b.index_add_(0, e[:, 1], ones)
                A_list.append(deg_b.clamp_min_(1.0).reciprocal())
            use_blockdiag = False

        slack_mask = (bus_type == 1)
        pv_mask    = (bus_type == 2)

        if self.pinn:
            phys_loss = torch.zeros(1, device=device)

        # --------------------------------- K iterations ---------------------------------
        for k in range(self.K):
            # --- power mismatches (block-sparse Y·V when possible) ---
            Vc = v * torch.exp(1j * th)
            if use_blockdiag:
                Ic = self._bdiag_mv_groups(groups, sizes, Vc)                 # (1,N)
            else:
                Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)            # (B,N)

            Sc = Vc * Ic.conj()
            DP = (P_set - Sc.real)
            DQ = (Q_set - Sc.imag)
            DP = DP.masked_fill(slack_mask, 0.0)
            DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)

            # --- build Σ_j φ(m_j, line_ij) with one pass over edges ---
            if use_blockdiag:
                N_total = v.size(1)
                M_neigh = torch.zeros(N_total, self.d, device=device, dtype=v.dtype)
                if edge_index.numel() > 0:
                    m_j = m[0, edge_index[:, 1], :]                           # (E,d)
                    phi_in = torch.cat([m_j, edge_feat], dim=-1)              # (E,d+3)
                    phi = self.edge_mlp[k](phi_in)                            # (E,d)
                    M_neigh.index_add_(0, edge_index[:, 0], phi)
                    M_neigh.index_add_(0, edge_index[:, 1], phi)
                    M_neigh = M_neigh * A.unsqueeze(-1)                       # degree-norm
                M_neigh = M_neigh.unsqueeze(0)                                 # (1,N,d)
            else:
                M_neigh = torch.zeros(B, N, self.d, device=device, dtype=v.dtype)
                for b in range(B):
                    e_idx = edge_index_list[b]
                    if e_idx.numel() == 0:
                        continue
                    m_j = m[b, e_idx[:, 1], :]
                    phi_in = torch.cat([m_j, edge_feat_list[b]], dim=-1)
                    phi = self.edge_mlp[k](phi_in)
                    M_neigh[b].index_add_(0, e_idx[:, 0], phi)
                    M_neigh[b].index_add_(0, e_idx[:, 1], phi)
                    M_neigh[b] = M_neigh[b] * A_list[b].unsqueeze(-1)

            # --- node-level update heads ---
            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)                   # (B,N,4)
            feats = torch.cat([bus_feat, m, M_neigh], dim=-1)                 # (B,N,4+2d)

            dth = self.theta_upd[k](feats).squeeze(-1)
            dv  = self.v_upd[k](feats).squeeze(-1)
            dm  = torch.tanh(self.m_upd[k](feats))
            dm  = F.layer_norm(dm, dm.shape[-1:])

            # constraints
            dth = dth.masked_fill(slack_mask, 0.0)
            dv  = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                dtheta_max = 0.30
                dvm_frac   = 0.10
                v_abs = v.abs()
                dth = torch.clamp(dth, -dtheta_max, dtheta_max)
                dv  = torch.clamp(dv, -dvm_frac * v_abs, dvm_frac * v_abs)

            # --- Armijo backtracking (few fixed trials) ---
            if self.use_armijo:
                v_min, v_max = 0.8, 1.2
                if use_blockdiag:
                    F0 = self._mismatch_inf_norm_block(groups, sizes, v, th, P_set, Q_set, slack_mask, pv_mask)
                else:
                    # Dense fallback (B graphs)
                    Vc0 = v * torch.exp(1j * th)
                    Ic0 = torch.matmul(Y, Vc0.unsqueeze(-1)).squeeze(-1)
                    Sc0 = Vc0 * Ic0.conj()
                    DP0 = (P_set - Sc0.real).masked_fill(slack_mask, 0.0)
                    DQ0 = (Q_set - Sc0.imag).masked_fill(slack_mask | pv_mask, 0.0)
                    F0  = torch.maximum(DP0.abs().amax(dim=-1), DQ0.abs().amax(dim=-1)).amax()

                alphas = v.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625])
                accepted = False
                for a in alphas:
                    v_try = torch.clamp(v + a * dv, v_min, v_max)
                    th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    if use_blockdiag:
                        F1 = self._mismatch_inf_norm_block(groups, sizes, v_try, th_try, P_set, Q_set, slack_mask, pv_mask)
                    else:
                        Vc1 = v_try * torch.exp(1j * th_try)
                        Ic1 = torch.matmul(Y, Vc1.unsqueeze(-1)).squeeze(-1)
                        Sc1 = Vc1 * Ic1.conj()
                        DP1 = (P_set - Sc1.real).masked_fill(slack_mask, 0.0)
                        DQ1 = (Q_set - Sc1.imag).masked_fill(slack_mask | pv_mask, 0.0)
                        F1  = torch.maximum(DP1.abs().amax(dim=-1), DQ1.abs().amax(dim=-1)).amax()
                    if F1 <= (1.0 - 1e-4 * a) * F0:
                        v, th, m = v_try, th_try, m + a * dm
                        accepted = True
                        break
                if not accepted:
                    a = alphas[-1]
                    v2 = torch.clamp(v + a * dv, v_min, v_max)
                    th2 = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    if use_blockdiag:
                        ok = self._mismatch_inf_norm_block(groups, sizes, v2, th2, P_set, Q_set, slack_mask, pv_mask) < F0
                    else:
                        Vc2 = v2 * torch.exp(1j * th2)
                        Ic2 = torch.matmul(Y, Vc2.unsqueeze(-1)).squeeze(-1)
                        Sc2 = Vc2 * Ic2.conj()
                        DP2 = (P_set - Sc2.real).masked_fill(slack_mask, 0.0)
                        DQ2 = (Q_set - Sc2.imag).masked_fill(slack_mask | pv_mask, 0.0)
                        ok  = torch.maximum(DP2.abs().amax(dim=-1), DQ2.abs().amax(dim=-1)).amax() < F0
                    if ok:
                        v, th, m = v2, th2, m + a * dm
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v  = torch.clamp(v + dv, 0.8, 1.2)
                m  = m + dm

            # physics loss (discounted)
            if self.pinn:
                phys_loss = phys_loss + (self.gamma ** (self.K - 1 - k)) * ((DP**2 + DQ**2).mean())

        out = torch.stack([v, th], dim=-1)
        return (out, phys_loss) if self.pinn else out
