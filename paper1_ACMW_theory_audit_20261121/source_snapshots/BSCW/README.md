# BSCW: bounded-shift coupled watermark and task-risk frontier

Date: 2026-11-21  
Parent: \`92f5d40\` (LC-TCW review)

## Decision

**Retain temporal coupling as the watermark-design variable, but do not carry the full LC-TCW construction forward unchanged.** A simpler bounded-shift amplitude coupling (BSCW-\alpha) captures most of the two-block state-risk improvement with materially lower cross-time dependence. The first-coordinate coupling in full LC-TCW produces only a small additional gain against a block-fixed attacker, although it helps more against a watermark-aware adaptive attacker.

The current result is a **reduced two-sample, one-state-mode / two-attack-coordinate diagnostic**. It is not yet a theorem or experiment for the complete ten-step A0, unknown attack onset, or the original Yang consensus estimator.

## 1. Design family: BSCW-\alpha

Let
\[
Q(u)=\tfrac12\left[\Phi^{-1}\left(\frac{1+u}{2}\right)\right]^2,
\quad U\sim\mathrm{Unif}(0,1),\quad
S\sim\mathrm{Unif}[\alpha,1-\alpha],
\]
and set \(U_1=(U+S)\bmod 1\), with \(0\leq\alpha<1/2\). Independently draw signs \(\epsilon_0,\epsilon_1\in\{-1,+1\}\) and
\[
b_0=\epsilon_0\sqrt{Q(U)},\qquad b_1=\epsilon_1\sqrt{Q(U_1)}.
\]
Draw \(a_0,a_1\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1/2)\), independently of the b-coordinates, and define
\[
\beta_t=(a_t,b_t)^\top,\qquad M_t=\tfrac12I+\beta_t\beta_t^\top.
\]

Because both \(U\) and \(U_1\) are uniform and the signs are independent, every \(\beta_t\sim\mathcal N(0,\tfrac12I)\). Thus every single-time matrix \(M_t\) has **exactly the same marginal law** as the original independent Gaussian multiplicative watermark. The only change is the two-time joint law.

### Exact dependence–separation identities

For \(0<\alpha<1/2\), the conditional density of \(U_1\) given \(U_0\) is uniform on a circular arc of length \(w=1-2\alpha\). Therefore
\[
I(U_0;U_1)=\log\frac1{1-2\alpha},\qquad
\|P_{U_0,U_1}-P_{\mathrm{iid}}\|_{\mathrm{TV}}=2\alpha.
\]
The matrix pair reveals \(b_t^2=Q(U_t)\); the independent a-coordinates and independent signs add no further cross-time dependence. Hence the same mutual information and TV values hold for the matrix pair in this BSCW family.

Also,
\[
|M_{1,22}-M_{0,22}|=|Q(U_1)-Q(U_0)|\geq Q(\alpha).
\]
Proof: \(Q\) is increasing and convex. If the shift does not wrap, \(Q(u+s)-Q(u)\geq Q(s)\geq Q(\alpha)\). If it wraps, write \(v=u+s-1\); then \(Q(v+1-s)-Q(v)\geq Q(1-s)\geq Q(\alpha)\). Equality is approached at the arc endpoints. This is a deterministic separation of the selected watermark coefficient, not a uniform lower bound on the full two-dimensional observability matrix.

For \(\alpha=0\), \(S\) is uniform on the full unit circle, \(U_1\) is independent of \(U_0\), and the scheme reduces to the iid watermark.

| \alpha | \(I(M_0;M_1)\), nats | TV from iid pair | guaranteed \(|\Delta M_{22}|\) |
|---:|---:|---:|---:|
| 0.25 | 0.693147 | 0.50 | 0.050768 |
| 0.30 | 0.916291 | 0.60 | 0.074238 |
| 0.35 | 1.203973 | 0.70 | 0.102952 |
| 0.40 | 1.609438 | 0.80 | 0.137499 |
| 0.45 | 2.302585 | 0.90 | 0.178659 |

These are dependence measures, **not cryptographic secrecy guarantees**. Even the preferred \(\alpha=0.35\) pair is readily distinguishable from an iid pair (TV 0.70). The next protocol-level task must define what watermark history and decoded traffic the attacker actually observes.

## 2. Same-output, health-budgeted estimator

Use the archived two-sample block matrix
\[
B(M_0,M_1)=
\begin{bmatrix}
0&p_0&q_0\\
1&q_0&r_0\\
0&p_1&\rho q_1\\
1&q_1/\rho&r_1
\end{bmatrix},
\quad p_t=\lambda_w+a_t^2,\ q_t=a_tb_t,\ r_t=\lambda_w+b_t^2,
\quad \lambda_w=0.5,\ \rho=14/13.
\]
Write \(B=[b_x\ A]\), where \(b_x=(0,1,0,1)^\top\) is the target state-mode column and \(A\in\mathbb R^{4\times2}\) contains the two attack columns. Let \(W=\Sigma_\nu^{-1}\), and define
\[
P=b_x^\top Wb_x,\quad
a=\frac{b_x^\top WA}{P},\quad
C=A^\top WA-\frac{A^\top Wb_xb_x^\top WA}{P}\succeq0,
\quad \kappa=\frac{P_xP}{1+P_xP}.
\]
Here \(C\) is the Schur-complement / state-orthogonal attack information matrix. It is the same object as the nuisance information matrix proposed in the earlier TCW-TSE design, now used in an explicit estimator and risk calculation.

For observation \(Y=b_xx+A\xi+\nu\), use
\[
z=\frac{b_x^\top WY}{P},\qquad
\widehat\xi_\lambda=(C+\lambda I)^{-1}A^\top W
\left(I-\frac{b_xb_x^\top W}{P}\right)Y,
\]
\[
\boxed{\widehat x_\lambda=\kappa(z-a\widehat\xi_\lambda).}
\]
The attack estimate is formed only from the component orthogonal to the state signature; the state estimate then uses the finite Gaussian prior \(x\sim\mathcal N(0,P_x)\). The regularizer \(\lambda\) is calibrated to a healthy-MSE budget.

### Exact conditional risk decomposition

For \(\nu\sim\mathcal N(0,\Sigma_\nu)\), the state-orthogonal residual noise is independent of the state projection noise. Under the healthy model \(\xi=0\),
\[
R_H(M,\lambda)=H_0+
\kappa^2a(C+\lambda I)^{-1}C(C+\lambda I)^{-1}a^\top,
\quad H_0=\frac{P_x}{1+P_xP}.
\]
The increment is nonincreasing in \(\lambda\), since in an eigenbasis of \(C\) it is
\[
\kappa^2\sum_i (a u_i)^2\frac{c_i}{(c_i+\lambda)^2}.
\]
Thus a common \(\lambda\) can be found by bisection to enforce
\[
\mathbb E_\Gamma R_H(M,\lambda)\leq (1+\epsilon_H)H_0.
\]

For a fixed block attack \(\xi\), the remaining conditional attack bias is
\[
g(M,\lambda)\xi,\qquad
g(M,\lambda)=\kappa\lambda a(C+\lambda I)^{-1}.
\]
Define \(S_\lambda=\mathbb E_\Gamma[g^\top g]\). For an attacker choosing one fixed \(\xi\), independently of the secret watermark, with \(\|\xi\|^2\leq E\), the exact worst-case risk in this estimator class is
\[
\boxed{R_{\mathrm{fixed}}=\mathbb E_\Gamma R_H+E\lambda_{\max}(S_\lambda).}
\]
For the stronger threat model in which the attacker may choose a different \(\xi(M)\) after seeing the current watermark pair,
\[
\boxed{R_{\mathrm{adaptive}}=\mathbb E_\Gamma R_H+E\operatorname{tr}(S_\lambda).}
\]
These are risks of the **same final state output** used in both healthy and attacked cases; no pre-projection / post-projection object switch is involved.

## 3. Numerical diagnostic

The script evaluates the above conditional risks analytically for each watermark pair; Monte Carlo is used only to integrate over watermark pairs. Settings:

- \(P_x=1\) (finite state prior variance).
- Healthy budget \(\epsilon_H=2\%\).
- Local two-coordinate block attack constraint \(\|\xi\|^2\leq E=10\). This is a **local block stress test**, not a claim that it equals the original ten-step coding-domain energy.
- \(\Sigma_\nu=\operatorname{diag}(0.5,0.5,0.5,0.58/1.12^2)\). The final entry retains the archived normalized second-mode variance. The third entry is set to 0.5 as an explicit completion assumption because the archived reduced-model note did not separately specify the first-coordinate second-time variance; it must be checked against the frozen full model before an A0 claim.
- 15 independent replications, 50,000 watermark pairs per replication and scheme.
- Reported \(\pm\) values are standard deviations across the 15 replication estimates, not confidence intervals.

The common healthy benchmark and target are
\[
H_0=0.1936948971,\qquad (1.02)H_0=0.1975687951.
\]

| Scheme | Mean \(\lambda\) | Fixed-attack risk (mean ± SD) | Watermark-adaptive risk (mean ± SD) |
|---|---:|---:|---:|
| iid Gaussian | 26.9631 | 6.1023 ± 0.0215 | 6.2850 ± 0.0221 |
| BSCW, \(\alpha=0.25\) | 26.9756 | 5.7633 ± 0.0248 | 5.9468 ± 0.0262 |
| BSCW, \(\alpha=0.30\) | 26.9334 | 5.7371 ± 0.0240 | 5.9209 ± 0.0255 |
| BSCW, \(\alpha=0.35\) | 26.9259 | 5.7164 ± 0.0184 | 5.8995 ± 0.0195 |
| BSCW, \(\alpha=0.40\) | 26.9484 | 5.7031 ± 0.0232 | 5.8873 ± 0.0243 |
| BSCW, \(\alpha=0.45\) | 26.9391 | 5.6907 ± 0.0260 | 5.8751 ± 0.0278 |
| full LC-TCW | 24.4185 | 5.6752 ± 0.0168 | 5.7331 ± 0.0170 |

At \(\alpha=0.35\), the risk falls by about **6.32%** against the fixed block attacker and **6.13%** against the watermark-adaptive attacker, relative to iid, at the same 2% healthy budget. The \(\alpha=0.35\) versus \(\alpha=0.40\) risk differences are not resolved by the current replication variability, while \(\alpha=0.35\) has 25% less mutual information. The \(\alpha=0.35\) versus \(\alpha=0.30\) improvement is resolved by the current independent-replication comparison. Thus \(\alpha=0.35\) is the present **candidate knee**, not a proven global optimum.

Full LC-TCW gives a further small fixed-attacker improvement (about 0.72% relative to BSCW-0.35) and a larger watermark-adaptive improvement (about 2.82%), but its matrix-pair mutual information is 5.672677 nats and its TV distance from iid is 0.994859. BSCW-0.35 uses only 1.203973 nats and TV 0.70. This makes the first-coordinate coupling a conditional design choice: potentially useful for the stronger adaptive threat, but not obviously worth its additional temporal dependence under the fixed-secret-blind threat.

A separate 100,000-pair information-matrix diagnostic gives the 5th percentile of \(\lambda_{\min}(C)\) as 0.00358 (iid), 0.05517 (BSCW-0.35), 0.06637 (BSCW-0.40), and 0.09421 (full LC-TCW). This confirms that the coupling lifts the lower tail of the state-orthogonal attack information, but \(\lambda_{\min}(C)\) alone is not the task objective: the state-risk functional also weights the eigendirections through \(a\).

## 4. Interpretation and limits

1. **A genuine quantitative step:** the same single-time Gaussian marginal can be kept while a tunable two-time copula changes both a deterministic attack-coefficient separation and the health-budgeted state risk.
2. **A more economical candidate:** BSCW-0.35 captures most of the measured fixed-attacker benefit without the extra first-coordinate coupling of full LC-TCW.
3. **No secrecy claim:** finite mutual information is not equivalent to a secure key or conditional secrecy. The TV values remain large. The attacker's observation set and key access must be modeled explicitly.
4. **No full A0 claim:** this experiment assumes two consecutive attack-internal samples and a constant two-coordinate block nuisance. It does not include unknown onset, ten-tap template shape, the original BF stopping event, attack boundaries, or the recursive Yang consensus state error.
5. **No exact energy equivalence claim:** the local \(\|\xi\|^2\leq10\) constraint is not automatically the same as the archived ten-step coding-domain energy 10.
6. **Noise completion must be checked:** the third variance in \(\Sigma_\nu\) is an explicit provisional assumption.

## 5. Next proof / experiment gates

- Rebuild \(\Sigma_\nu\), \(B\), and the attack parameterization directly from the frozen region-model implementation; do not infer the missing channel variance.
- Extend BSCW-\(\alpha\) to the full ten-step attack template with unknown onset. Preserve the original BF and compare the exact same final Yang/Co-G output at identical delay.
- Evaluate the likelihood of the actual watermark process under every candidate onset. A receiver must not continue using an iid watermark likelihood after changing the generator.
- Define an attacker information filtration: which encoded packets, past watermarks, public state estimates, and current watermark components are visible before the attack is selected.
- Compare iid, BSCW-\(0.35\), BSCW-\(0.40\), and full LC-TCW using the same estimator class, same healthy budget, and same attack set. Include an oracle lower bound only as an offline diagnostic.
- Conduct a full novelty review against switching multiplicative watermark design and the latest DSE watermarking papers before using “new watermark design” as a final contribution claim.

## 6. Literature positioning (initial, not exhaustive)

- Zhou et al., “Watermarking-Based Protection Strategy Against Stealthy Integrity Attack on Distributed State Estimation,” *IEEE TAC*, 2023. Uses multiplicative watermarking and a KL-based detector for stealthy integrity attacks. DOI: https://doi.org/10.1109/TAC.2022.3171422.
- Gallo et al., “Switching multiplicative watermark design against covert attacks,” *Automatica*, 2025. Optimizes switching multiplicative filter parameters against a bounded-energy matched covert attacker; this is a direct novelty near-neighbor. DOI: https://doi.org/10.1016/j.automatica.2025.112301.
- “Watermarking-Based Attack Detection for Sensor Networks with Intermittent Observation Under Stealthy Attacks,” *Journal of Systems Science and Complexity*, online 2026. A recent DSE watermarking paper addressing intermittent observations and sensor-to-estimator / estimator-to-estimator attacks. DOI: https://doi.org/10.1007/s11424-026-5060-8.
- Zhu et al., “Double-Layer Detection-Based Distributed Secure State Estimation Under False Data Injection Attacks,” *IEEE Transactions on Power Systems*, 2026. Includes a dynamic-watermarking-based second-layer detector and replacement; DOI: https://doi.org/10.1109/TPWRS.2025.3586788.

The plausible distinction is not “using a multiplicative watermark” or “switching watermarks.” It is the combination of (i) exactly preserved per-step Gaussian matrix marginals, (ii) an explicit dependence/separation trade-off, and (iii) a task-weighted, health-budgeted state-risk decoder. That distinction remains a **candidate** until the full papers and full A0 experiment are checked.
