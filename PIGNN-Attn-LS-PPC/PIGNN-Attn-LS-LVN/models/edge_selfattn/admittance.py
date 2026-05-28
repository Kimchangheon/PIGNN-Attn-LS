from __future__ import annotations

import torch


def split_yc_parts(yc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (real, imag) for Yc, treating real tensors as imag=0."""

    if torch.is_complex(yc):
        return yc.real, yc.imag
    return yc, torch.zeros_like(yc)


def build_dense_Y(
    num_nodes: int,
    edges_undirected: torch.Tensor,
    ys_edge: torch.Tensor,
    yc_edge: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build dense Ybus from undirected edges with series admittance Ys and shunt Yc.

    Args:
        num_nodes: N
        edges_undirected: (E,2) with i<j indices
        ys_edge: (E,) complex (or will be cast)
        yc_edge: (E,) complex or real

    Returns:
        (N,N) complex tensor
    """

    N = int(num_nodes)
    if edges_undirected.numel() == 0:
        dtype = ys_edge.dtype if ys_edge.numel() else torch.complex64
        return torch.zeros(N, N, dtype=dtype, device=device)

    i = edges_undirected[:, 0]
    j = edges_undirected[:, 1]

    if not torch.is_complex(ys_edge):
        ys_edge = ys_edge.to(torch.complex64)
    if not torch.is_complex(yc_edge):
        yc_edge = yc_edge.to(ys_edge.dtype)

    Yloc = torch.zeros(N, N, dtype=ys_edge.dtype, device=device)

    Yloc.index_put_((i, j), -ys_edge, accumulate=True)
    Yloc.index_put_((j, i), -ys_edge, accumulate=True)

    diag = torch.zeros(N, dtype=ys_edge.dtype, device=device)
    diag.index_add_(0, i, ys_edge)
    diag.index_add_(0, j, ys_edge)
    diag.index_add_(0, i, yc_edge)
    diag.index_add_(0, j, yc_edge)

    Yloc.diagonal().add_(diag)
    return Yloc


def build_edges_blockdiag(
    *,
    line_mask_1d: torch.Tensor,
    Ys_1d: torch.Tensor,
    Yc_1d: torch.Tensor,
    n_nodes_per_graph: list[int] | torch.Tensor,
    edge_feat_dim: int,
    pairs_for_n,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build directed edges/features for block-diag batching (B==1).

    Returns:
        undirected (E,2), edge_feat (E,F), edge_index_dir (2E,2), edge_feat_dir (2E,F),
        plus ys_edge/y c_edge aligned with undirected edges for Y building.
    """

    Ysr, Ysi = Ys_1d.real, Ys_1d.imag
    Yc_real, Yc_imag = split_yc_parts(Yc_1d)

    edge_index_parts: list[torch.Tensor] = []
    edge_feat_parts: list[torch.Tensor] = []
    ys_parts: list[torch.Tensor] = []
    yc_parts: list[torch.Tensor] = []

    ptr = 0
    offset = 0
    for n in n_nodes_per_graph:
        n = int(n)
        e_all = n * (n - 1) // 2
        mask_g = line_mask_1d[ptr : ptr + e_all].bool()
        if mask_g.any():
            pairs_g = pairs_for_n(n, device)
            e_idx_g = pairs_g[mask_g] + offset
            edge_index_parts.append(e_idx_g)

            if edge_feat_dim != 4:
                raise ValueError("edge_feat_dim != 4 is not supported by this model")

            feat_g = torch.stack(
                [
                    Ysr[ptr : ptr + e_all][mask_g],
                    Ysi[ptr : ptr + e_all][mask_g],
                    Yc_real[ptr : ptr + e_all][mask_g].to(Ysr.dtype),
                    Yc_imag[ptr : ptr + e_all][mask_g].to(Ysr.dtype),
                ],
                dim=-1,
            )
            edge_feat_parts.append(feat_g)

            ys_parts.append(Ys_1d[ptr : ptr + e_all][mask_g])
            yc_parts.append(Yc_1d[ptr : ptr + e_all][mask_g])

        ptr += e_all
        offset += n

    if edge_index_parts:
        undirected = torch.cat(edge_index_parts, dim=0)
        edge_feat = torch.cat(edge_feat_parts, dim=0)
        ys_edge = torch.cat(ys_parts, dim=0)
        yc_edge = torch.cat(yc_parts, dim=0)
    else:
        undirected = torch.empty(0, 2, dtype=torch.long, device=device)
        edge_feat = torch.empty(0, edge_feat_dim, dtype=Ysr.dtype, device=device)
        ys_edge = torch.empty(0, dtype=Ys_1d.dtype, device=device)
        yc_edge = torch.empty(0, dtype=Yc_1d.dtype, device=device)

    edge_index_dir = torch.cat([undirected, undirected[:, [1, 0]]], dim=0)
    edge_feat_dir = torch.cat([edge_feat, edge_feat], dim=0)
    return undirected, edge_feat, edge_index_dir, edge_feat_dir, ys_edge, yc_edge


def build_edges_plain(
    *,
    Line: torch.Tensor,
    Ys: torch.Tensor,
    Yc: torch.Tensor,
    N: int,
    edge_feat_dim: int,
    pairs: torch.Tensor,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Build directed edges/features for each batch element.

    Returns:
        edge_index_dir_list, edge_feat_dir_list, undirected_list, mask_list
    """

    B = int(Line.shape[0])
    edge_index_dir_list: list[torch.Tensor] = []
    edge_feat_dir_list: list[torch.Tensor] = []
    undirected_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []

    for b in range(B):
        mask = Line[b].bool()
        e_b = pairs[mask]
        undirected_list.append(e_b)
        mask_list.append(mask)

        Ysr_b, Ysi_b = Ys[b].real, Ys[b].imag
        Yc_real_b, Yc_imag_b = split_yc_parts(Yc[b])

        if e_b.numel() > 0:
            if edge_feat_dim != 4:
                raise ValueError("edge_feat_dim != 4 is not supported by this model")

            feat_b = torch.stack(
                [
                    Ysr_b[mask],
                    Ysi_b[mask],
                    Yc_real_b[mask].to(Ysr_b.dtype),
                    Yc_imag_b[mask].to(Ysr_b.dtype),
                ],
                dim=-1,
            )
            edge_index_dir_list.append(torch.cat([e_b, e_b[:, [1, 0]]], dim=0))
            edge_feat_dir_list.append(torch.cat([feat_b, feat_b], dim=0))
        else:
            edge_index_dir_list.append(torch.empty(0, 2, dtype=torch.long, device=device))
            edge_feat_dir_list.append(torch.empty(0, edge_feat_dim, dtype=Ysr_b.dtype, device=device))

    return edge_index_dir_list, edge_feat_dir_list, undirected_list, mask_list
