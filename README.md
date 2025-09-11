
---

### MLP as Aggregator

At each **message** passing step \$k\$, the generic form is:

$$
\begin{aligned}
m_i^{(k+1)}
&= \operatorname{UPDATE}^{(k)}\!\Big(
m_i^{(k)},\,
\Phi_{\text{phys}}^{(k)}\big(
\operatorname{AGGREGATE}^{(k)}(\{\,m_j^{(k)} \mid j \in \mathcal{N}(i)\}),\,
\text{state}^{(k)}
\big)\Big).
\end{aligned}
$$

The detailed steps are as follows.

1. **M1 — Message aggregation (degree-normalized).**
   For each undirected edge \$j \to i\$ with line attributes
   \$\ell\_{ij}=(g\_{ij},,b\_{ij},,b\_{ij}^{\mathrm{sh}})\$ (series conductance, susceptance, end-shunt),

   $$
   \phi_{ij}^{(k)} \;=\; \mathrm{MLP}_{\phi}^{(k)}\!\big(m_j^{(k)},\,\ell_{ij}\big),
   \qquad
   M_i^{(k)} \;=\; \frac{1}{\deg(i)} \sum_{j\in\mathcal N(i)} \phi_{ij}^{(k)}.
   $$

2. **M2 — Physics-informed transformation \$\Phi\_{\text{phys}}^{(k)}\$.**
   Given \$Y\$ (built from line attributes per the problem formulation), compute

   $$
   \Delta P_i^{(k)} = P_i^{\mathrm{set}} - P_i^{(k)}, \qquad
   \Delta Q_i^{(k)} = Q_i^{\mathrm{set}} - Q_i^{(k)}.
   $$

   Masks enforce \$\Delta P\_i{=}\Delta Q\_i{=}0\$ on **slack** buses and \$\Delta Q\_i{=}0\$ on **PV** buses.

3. **M3 — Node update.**
   Concatenate electrical and learned features,

   $$
   \operatorname{ctx}_i^{(k)}
   = \big[V_i^{(k)},\,\theta_i^{(k)},\,\Delta P_i^{(k)},\,\Delta Q_i^{(k)},\,m_i^{(k)},\,M_i^{(k)}\big]
   \in \mathbb R^{4+2d},
   $$

   predict increments,

   $$
   \begin{aligned}
     \Delta\theta_i^{(k)} &= L_\theta^{(k)}\!\big(\operatorname{ctx}_i^{(k)}\big), \\
     \Delta V_i^{(k)}     &= L_v^{(k)}\!\big(\operatorname{ctx}_i^{(k)}\big), \\
     \Delta m_i^{(k)}     &= \tanh\!\Big(L_m^{(k)}\!\big(\operatorname{ctx}_i^{(k)}\big)\Big),
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

1. **A1 — Node embedding.**
   At step \$k\$,

   $$
   b_i^{(k)}=\big[V_i^{(k)},\,\theta_i^{(k)},\,\Delta P_i^{(k)},\,\Delta Q_i^{(k)},\,m_i^{(k)}\big]\in\mathbb{R}^{4+d}.
   $$

2. **A2 — Edge-wise multi-head attention (\$O(EH)\$).**
   For each undirected edge \${i,j}\$, score both directions \$j{\to}i\$ and \$i{\to}j\$.
   For head \$h=1,\dots,H\$ with \$d\_h=d\_{\text{model}}/H\$,

   $$
   q_i^{(h)}=W_Q^{(h)} b_i^{(k)},\quad
   k_j^{(h)}=W_K^{(h)} b_j^{(k)},\quad
   u_j^{(h)}=W_V^{(h)} b_j^{(k)}.
   $$

   Add a physics-based edge bias from \$\ell\_{ij}\$,

   $$
   s_{ij}^{(h)}=\frac{\langle q_i^{(h)},k_j^{(h)}\rangle}{\sqrt{d_h}}+\beta_{ij}^{(h)},
   \qquad
   \beta_{ij}^{(h)}=f_{\text{edge}}^{(h)}(\ell_{ij}),
   $$

   and normalize over incoming neighbors of \$i\$:

   $$
   \alpha_{ij}^{(h)}=\frac{\exp(s_{ij}^{(h)})}{\sum_{u\in\mathcal N(i)}\exp(s_{iu}^{(h)})}.
   $$

3. **A3 — Attention context (supplants raw concatenation).**

   $$
   \operatorname{ctx}_i^{(k)}
   = W_O\!\left[
   \sum_{j}\alpha_{ij}^{(1)}u_j^{(1)}\;\big\|\;\cdots\;\big\|\;\sum_{j}\alpha_{ij}^{(H)}u_j^{(H)}
   \right]\in\mathbb{R}^{d}.
   $$

   This replaces the entire concatenated feature: the **M3** update heads now take \$\operatorname{ctx}\_i^{(k)}\$ alone, since \$(V,\theta,\Delta P,\Delta Q,m)\$ already condition the attention via \$(q,k,v)\$.

> **Notes.** Attention operates only on existing edges (no dense \$N^2\$ attention) with a per-destination softmax; when logits are uniform it reduces to a degree-normalized neighbor average.

3.1 Sampling Pipeline

This repository synthesizes AC power-flow (PF) scenarios and computes reference solutions with Newton–Raphson (NR). NR is used only to generate benchmark states for evaluation; training remains physics-informed (no NR labels).

Steps (S1–S9)
	1.	S1 — Choose regime and size
Draw a voltage regime g ∈ {MVN, HVN} and a bus count N ∈ [4, 32]. Assign base quantities (U_base, S_base) by regime.
	2.	S2 — Sample line parameters and lengths
For each unordered bus pair (i, j), draw a line length L_ij and per-km series parameters (R′, X′) within operator-typical ranges. Form series impedance and admittance (engineering units):
$$
Z_{ij} = (R’ + jX’),L_{ij}, \qquad Y_{ij} = \frac{1}{Z_{ij}}.
$$
Ranges reflect realistic R/X (higher at MV; X > R at HV) and overhead-line statistics.
	3.	S3 — Random topology + connectivity
Sample an undirected edge set over N buses. Ensure the slack bus reaches all others via BFS; only connected draws proceed.
	4.	S4 — Shunt charging (π-model)
Draw per-km shunt capacitance C′ by regime, set ω = 2πf with f = 50 Hz, and assign half to each line end:
$$
b_{ij}^{\text{sh}} = \tfrac{1}{2},\omega,C’ , L_{ij}.
$$
Accumulate per-bus shunt: B_i^{sh} = Σ_j b_{ij}^{sh}.
	5.	S5 — Assemble nodal admittance
Build the bus admittance matrix Y \in \mathbb{C}^{N\times N} in π-model form:
$$
Y_{ij}=
\begin{cases}
-y_{ij}, & i\neq j \text{ and line } (i,j) \text{ exists},\
\sum\limits_{k\in \mathcal{N}(i)} y_{ik} + j,B_i^{\text{sh}}, & i = j,\
0, & \text{otherwise.}
\end{cases}
$$
	6.	S6 — Assign bus types
Fix bus 1 as Slack. Assign each remaining bus independently as PV or PQ to obtain a realistic Slack/PV/PQ mix for MV and HV.
	7.	S7 — Sample operating point
Draw bus injections (P_i, Q_i) by regime and type: Slack (0, 0), PV (P_i, 0), PQ (P_i, Q_i). Magnitudes follow the ranges in your parameter table. Form complex power S_i = P_i + jQ_i and the system vector S.
	8.	S8 — Initial voltages
Set complex initial guesses U^(0): Slack/PV magnitudes uniform in [0.9, 1.1] p.u., PQ at 1.0 p.u., zero phase. Convert using U_base.
	9.	S9 — Reference PF solution (NR)
Run NR for up to K iterations on (Y, S) under Slack/PV/PQ constraints to obtain (U*, S*). Mark non-convergent or disconnected cases; exclude them from metrics unless noted.

Notes
	•	Symbols: j is the engineering imaginary unit; p.u. denotes per-unit.
	•	MVN/HVN parameter ranges (e.g., R′, X′, C′, L) should be specified in your table or config for reproducibility.
