### MLP as Aggregator

At each **message** passing step $k$, the generic form is:

$$
\begin{aligned}
m_i^{(k+1)}
&= \mathrm{UPDATE}^{(k)}\!\Big(
m_i^{(k)},\,
\Phi_{\mathrm{phys}}^{(k)}\big(
\mathrm{AGGREGATE}^{(k)}(\{\,m_j^{(k)} \mid j \in \mathcal{N}(i)\,\}),\,
\mathrm{state}^{(k)}
\big)\Big).
\end{aligned}
$$

The detailed steps are as follows.

1. **M1 — Message aggregation (degree-normalized).**  
   For each undirected edge $j \to i$ with line attributes
   $\ell_{ij}=(g_{ij},\,b_{ij},\,b_{ij}^{\mathrm{sh}})$ (series conductance, susceptance, end-shunt),

    $$
    \phi_{ij}^{(k)} \;=\; \mathrm{MLP}_{\phi}^{(k)}\!\big(m_j^{(k)},\,\ell_{ij}\big),
    \qquad
    M_i^{(k)} \;=\; \frac{1}{\lvert \mathcal N(i)\rvert} \sum_{j\in\mathcal N(i)} \phi_{ij}^{(k)}.
    $$

2. **M2 — Physics-informed transformation $\Phi_{\mathrm{phys}}^{(k)}$.**  
   Given $Y$ (built from line attributes per the problem formulation), compute

    $$
    \Delta P_i^{(k)} = P_i^{\mathrm{set}} - P_i^{(k)}, \qquad
    \Delta Q_i^{(k)} = Q_i^{\mathrm{set}} - Q_i^{(k)}.
    $$

   Masks enforce $\Delta P_i{=}\Delta Q_i{=}0$ on **slack** buses and $\Delta Q_i{=}0$ on **PV** buses.

3. **M3 — Node update.**  
   Concatenate electrical and learned features,

    $$
    \mathrm{ctx}_i^{(k)}
    = \big[V_i^{(k)},\,\theta_i^{(k)},\,\Delta P_i^{(k)},\,\Delta Q_i^{(k)},\,m_i^{(k)},\,M_i^{(k)}\big]
    \in \mathbb R^{4+2d},
    $$

   predict increments,

    $$
    \begin{aligned}
      \Delta\theta_i^{(k)} &= L_\theta^{(k)}\!\big(\mathrm{ctx}_i^{(k)}\big), \\
      \Delta V_i^{(k)}     &= L_v^{(k)}\!\big(\mathrm{ctx}_i^{(k)}\big), \\
      \Delta m_i^{(k)}     &= \tanh\!\Big(L_m^{(k)}\!\big(\mathrm{ctx}_i^{(k)}\big)\Big),
    \end{aligned}
    $$

   and apply masked updates:

    $$
    \begin{aligned}
      \theta_i^{(k+1)} &= \theta_i^{(k)} + \Delta\theta_i^{(k)}, \\
      V_i^{(k+1)}      &= V_i^{(k)}      + \Delta V_i^{(k)}, \\
      m_i^{(k+1)}      &= m_i^{(k)}      + \Delta m_i^{(k)}.
    \end{aligned}
    $$

4. **M4 — Physics-informed loss (discounted).**

    $$
    \mathcal L_{\rm phys}
    \;=\; \sum_{k=0}^{K-1}
      \gamma^{\,K-1-k}\,
      \frac{1}{N}\sum_{i=1}^N \Big[(\Delta P_i^{(k)})^2+(\Delta Q_i^{(k)})^2\Big],
    \qquad \gamma\in(0,1].
    $$
---

### Self-Attention as Aggregator

For the attention variant, compute the physics features as in **M2** and then replace **M1** by an edge-conditioned self-attention that is sparse and state-dependent.

1. **A1 — Node embedding.** At step $k$,

    $$
    b_i^{(k)}=\big[V_i^{(k)},\,\theta_i^{(k)},\,\Delta P_i^{(k)},\,\Delta Q_i^{(k)},\,m_i^{(k)}\big]\in\mathbb{R}^{4+d}.
    $$

2. **A2 — Edge-wise multi-head attention ($O(EH)$).**  
   For each undirected edge $\{i,j\}$, score both directions $j{\to}i$ and $i{\to}j$.  
   For head $h=1,\dots,H$ with $d_h=d_{\mathrm{model}}/H$,

    $$
    q_i^{(h)}=W_Q^{(h)} b_i^{(k)},\quad
    k_j^{(h)}=W_K^{(h)} b_j^{(k)},\quad
    u_j^{(h)}=W_V^{(h)} b_j^{(k)}.
    $$

   Add a physics-based edge bias from $\ell_{ij}$,

    $$
    s_{ij}^{(h)}=\frac{\langle q_i^{(h)},k_j^{(h)}\rangle}{\sqrt{d_h}}+\beta_{ij}^{(h)},
    \qquad
    \beta_{ij}^{(h)}=f_{\mathrm{edge}}^{(h)}(\ell_{ij}),
    $$

   and normalize over incoming neighbors of $i$:

    $$
    \alpha_{ij}^{(h)}=\frac{\exp(s_{ij}^{(h)})}{\sum_{u\in\mathcal N(i)}\exp(s_{iu}^{(h)})}.
    $$

3. **A3 — Attention context (supplants raw concatenation).**

    $$
    \mathrm{ctx}_i^{(k)}
    = W_O\!\left[
    \sum_{j}\alpha_{ij}^{(1)}u_j^{(1)}\;\big\|\;\cdots\;\big\|\;\sum_{j}\alpha_{ij}^{(H)}u_j^{(H)}
    \right]\in\mathbb{R}^{d}.
    $$

   This replaces the entire concatenated feature: the **M3** update heads now take $\mathrm{ctx}_i^{(k)}$ alone, since $(V,\theta,\Delta P,\Delta Q,m)$ already condition the attention via $(q,k,v)$.

> **Notes.** Attention operates only on existing edges (no dense $N^2$ attention) with a per-destination softmax; when logits are uniform it reduces to a degree-normalized neighbor average.