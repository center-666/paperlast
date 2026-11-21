# 第一篇论文修改：多维角度耦合乘性水印

日期：2026-11-21  
定位：对现有稿件 *Secure Distributed State Estimation Against Stealthy False Data Injection Attacks via Multiplicative Watermarking* 的方法与理论修改，不是第二篇论文。

## 0. 与现有稿件对齐

现有稿件的模型为
\[
x_{k+1}=Ax_k+\omega_k,\qquad y_k^j=H_jx_k+v_k^j,
\]
其中 \(x_k\in\mathbb R^n\)，单个传感器发送的测量包 \(y_k^j\in\mathbb R^m\)。乘性水印作用于**测量包空间**：
\[
M_k=\lambda I_m+\beta_k\beta_k^\top,\qquad \beta_k\in\mathbb R^m.
\]
因此，多维推广的关键维数是测量包维数 \(m\)，不是把状态维数 \(n\) 直接代入水印向量长度。\(n\) 通过扩展可观测矩阵和奇偶空间影响检测窗口及残差维数。

原稿的 iid Gaussian 水印已经允许一般 \(m\)，但其时间联合分布是独立的。此前 BSCW 的坐标分位数耦合以二维坐标为基础、依赖所选坐标轴，不能直接作为本文的多维主方法。此前标量有限先验重构器的风险实验也不属于本文奇偶空间检测模型，不能作为本文的核心性能证据。本文件改为直接使用原稿的奇偶残差、严格隐蔽攻击和 KL 指标。

## 1. 新水印：带独立刷新机制的各向同性角度耦合水印

设 \(m\ge2\)。令 \(\omega_m\) 为单位球面 \(\mathbb S^{m-1}\) 上的归一化均匀测度。首先定义
\[
Z_m(\kappa)=\int_{\mathbb S^{m-1}}
\exp[-\kappa(u^\top v)^2]\,d\omega_m(v),
\quad \kappa\ge0.
\]
由旋转不变性，\(Z_m(\kappa)\) 与 \(u\) 无关，且
\[
Z_m(\kappa)={}_1F_1\left(\frac12;\frac m2;-\kappa\right).
\]

定义方向转移核
\[
g_{\kappa,\rho}(v\mid u)
=(1-\rho)+\rho\frac{\exp[-\kappa(u^\top v)^2]}{Z_m(\kappa)},
\quad 0\le\rho\le1,
\]
其中密度相对于 \(\omega_m\)。\(\rho\) 是执行角度排斥转移的概率；以 \(1-\rho\) 的概率进行独立方向刷新。这样可减轻纯反平行/正交转移在偶数滞后产生的周期性相关。

生成过程：
1. \(u_0\sim\omega_m\)；
2. \(u_{k+1}\mid u_k\sim g_{\kappa,\rho}(\cdot\mid u_k)\)；
3. 独立生成 \(R_k^2\sim\sigma_\beta^2\chi_m^2\)，且半径序列与方向序列独立；
4. 令
\[
\beta_k=R_ku_k,\qquad
\boxed{M_k=\lambda I_m+\beta_k\beta_k^\top.}
\]

### 性质1：单拍完整边缘分布严格不变

转移核对称且旋转不变，故均匀球面测度是其平稳分布。由独立半径和均匀方向的 Gaussian 极坐标分解，
\[
\boxed{\beta_k\sim\mathcal N(0,\sigma_\beta^2I_m)}
\]
对每个 \(k\) 严格成立。因此 \(M_k\) 的每拍完整边缘分布与原稿 iid Gaussian 水印完全一致：
\[
E[M_k]=(\lambda+\sigma_\beta^2)I_m.
\]
无攻击时，接收端仍执行 \(M_kM_k^{-1}y_k^j=y_k^j\)，所以理想算术下健康估计器输入逐拍完全不变。且
\[
M_k^{-1}=\frac1\lambda\left(I_m-
\frac{\beta_k\beta_k^\top}{\lambda+\|\beta_k\|^2}\right),
\]
计算仍为 \(O(m^2)\)。

### 性质2：有限、可调的跨拍统计依赖

令
\[
C=(u_k^\top u_{k+1})^2,\qquad
C\sim\operatorname{Beta}\left(\frac12,\frac{m-1}{2}\right)
\]
作为独立方向参考下的基准变量。定义
\[
g(c)=(1-\rho)+\rho e^{-\kappa c}/Z_m(\kappa).
\]
则相邻方向的互信息为
\[
I_m(\kappa,\rho)=
\int_0^1 f_{\rm Beta}(c)g(c)\log g(c)\,dc<\infty.
\]
由于 \(M_k\) 只保留方向的投影 \(u_ku_k^\top\) 和半径，且转移核对 \(u\mapsto-u\) 不变，
\[
I(M_k;M_{k+1})= I_m(\kappa,\rho).
\]
矩阵序列本身是平稳 Markov 过程，因此长度 \(L\) 窗口相对 iid 边缘乘积分布的总相关为
\[
D_{\rm KL}\!\left(P_{M_{0:L-1}}\middle\|\mu_M^{\otimes L}\right)
=(L-1)I(M_0;M_1)
\le(L-1)I_m(\kappa,\rho).
\]
这不是密码学安全证明；它是对水印时间统计依赖的可计算约束。

相邻方向的平均平方余弦为
\[
c_1=E[(u_k^\top u_{k+1})^2]
=\frac{1-\rho}{m}+\rho\left[-\frac{\partial}{\partial\kappa}
\log Z_m(\kappa)\right].
\]
iid 时 \(c_1=1/m\)，\(\kappa>0,\rho>0\) 时 \(c_1<1/m\)，即相邻水印方向更少重复。

## 2. 奇偶残差中的精确跨时间协方差

保持原稿的奇偶空间与攻击模型。令 \(L=s+1\)，
\[
Z_s=[Z_{s,0}\ \cdots\ Z_{s,s}],\qquad
\Delta Y=\operatorname{col}(\delta_0,\ldots,\delta_s).
\]
严格隐蔽攻击满足 \(Z_s\Delta Y=0\)。水印解码后的攻击残差偏移为
\[
\Delta r_s
=Z_s\mathcal M_s\Delta Y
=\sum_{\ell=0}^{s}Z_{s,\ell}
(\beta_\ell^\top\delta_\ell)\beta_\ell,
\]
因为 \(\lambda Z_s\Delta Y=0\)。

记 \(X_\ell=(\beta_\ell^\top\delta_\ell)\beta_\ell\)。单拍边缘保持 Gaussian，因此
\[
E[X_\ell]=\sigma_\beta^2\delta_\ell,
\quad
\operatorname{Cov}(X_\ell)
=\sigma_\beta^4(\delta_\ell\delta_\ell^\top+
\|\delta_\ell\|^2I_m).
\]
故对严格隐蔽的确定性攻击，\(E[\Delta r_s]=0\)，每个对角协方差块与原稿完全相同。

角度耦合只改变跨时间协方差。定义
\[
c_h=E[(u_\ell^\top u_{\ell+h})^2],\qquad
b_h=\frac{c_h-1/m}{(m-1)(m+2)}.
\]
令
\[
\zeta_2=\frac{c_1-1/m}{1-1/m}.
\]
由于转移核是旋转不变的 zonal kernel，二阶球谐子空间上的 Markov 算子特征值为 \(\zeta_2\)，所以
\[
c_h=\frac1m+\left(1-\frac1m\right)\zeta_2^h.
\]
这使任意滞后的跨拍四阶矩均可由一个标量计算。

对 \(\ell\ne r\)，令 \(h=|r-\ell|\)，则
\[
\boxed{
\begin{aligned}
C_{\ell r}
&=\operatorname{Cov}(X_\ell,X_r)\\
&=\sigma_\beta^4m^2b_h\left[
(\delta_\ell^\top\delta_r)I_m+
\delta_r\delta_\ell^\top-
\frac2m\delta_\ell\delta_r^\top
\right].
\end{aligned}}
\]
于是完整残差协方差为
\[
\boxed{
\begin{aligned}
C_s(\kappa,\rho)
={}&\sum_{\ell=0}^{s}Z_{s,\ell}C_{\ell\ell}Z_{s,\ell}^\top\\
&+\sum_{0\le\ell<r\le s}
\left(Z_{s,\ell}C_{\ell r}Z_{s,r}^\top+
Z_{s,r}C_{\ell r}^\top Z_{s,\ell}^\top\right),
\end{aligned}}
\]
其中 \(C_{\ell\ell}\) 为上述单拍协方差。

当 \(\rho=0\) 或 \(\kappa=0\) 时，\(\zeta_2=0\)、所有 \(b_h=0\)，精确退化为原稿的独立水印协方差公式。新机制的有限窗口作用因此可清楚定位：**单拍边缘和单拍协方差不变，只有跨拍协方差块改变。**

## 3. 对原稿 Theorem 2 / Theorem 3 的可保留保证

角度耦合不应破坏原稿关于 KL 发散和隐蔽攻击能量抑制的结论。这里可以得到一个不依赖 \((\kappa,\rho)\) 的保守下界。

给定全部方向 \(U=(u_0,\ldots,u_s)\)，半径 \(R_\ell^2\) 相互独立，且
\[
\operatorname{Var}(R_\ell^2)=2m\sigma_\beta^4.
\]
利用全协方差公式，
\[
\operatorname{Cov}(\Delta r_s)
\succeq E[\operatorname{Cov}(\Delta r_s\mid U)].
\]
对均匀球面方向，
\[
E[(u^\top\delta)^2uu^\top]
=\frac{\|\delta\|^2I_m+2\delta\delta^\top}{m(m+2)}.
\]
因此
\[
\boxed{
C_s(\kappa,\rho)\succeq
\frac{2\sigma_\beta^4}{m+2}
\sum_{\ell=0}^{s}Z_{s,\ell}
(\|\delta_\ell\|^2I_m+2\delta_\ell\delta_\ell^\top)
Z_{s,\ell}^\top.}
\]
这一步只使用每拍方向均匀、半径跨拍独立，不要求方向跨拍独立。

沿用原稿的
\[
c_Z=\min_{0\le\ell\le s}
\operatorname{tr}(\Theta_s^{-1}Z_{s,\ell}Z_{s,\ell}^\top)>0,
\]
可得
\[
\operatorname{tr}(\Theta_s^{-1}C_s)
\ge \frac{2c_Z\sigma_\beta^4}{m+2}
E\|\Delta Y\|^2.
\]
结合原稿 Gaussian 最大熵下界和
\(f(t)=t-\log(1+t)\)，如果攻击仍满足 KL 阈值 \(\theta\)，则
\[
\boxed{
E\|\Delta Y\|^2
\le
\frac{m+2}{2c_Z\sigma_\beta^4}
\,M_\theta,\quad
M_\theta=s_r f^{-1}(2\theta/s_r),
}
\]
其中 \(s_r=\dim(r_s)\)。所以仍有 \(O(\sigma_\beta^{-4})\) 的隐蔽攻击能量抑制。对任意固定非零严格隐蔽攻击，KL 下界随 \(\sigma_\beta\to\infty\) 发散。

注意：这个统一下界比原 iid 情况的常数更保守，不能宣称理论下界本身更强。新水印的有限强度优势来自精确跨拍协方差项 \(C_{\ell r}\)，需要通过有限窗口 KL 优化与实验体现。

## 4. 有限窗口设计目标

原稿的检测目标是 KL 阈值。可以把水印设计明确写成
\[
\begin{aligned}
\max_{\kappa,\rho}\quad&
\min_{\Delta Y\in\mathcal A_E,\ Z_s\Delta Y=0}
D_G(\Theta_s+C_s(\kappa,\rho,\Delta Y)\,\|\,\Theta_s)\\
\text{s.t.}\quad&
(L-1)I_m(\kappa,\rho)\le \ell_{\max},\\
&\lambda,\sigma_\beta^2\text{ 固定，单拍边缘与通信包长不变}.
\end{aligned}
\]
这里
\[
D_G=\frac12\left[
\operatorname{tr}(\Theta_s^{-1}C_s)
-\log\det(I+\Theta_s^{-1/2}C_s\Theta_s^{-1/2})
\right]
\]
是非高斯攻击残差相对名义 Gaussian 残差 KL 的 Gaussian 等协方差下界。可选择最坏攻击方向，也可先按原稿 Proposition 1 的辅助系统攻击族进行平均设计。

## 5. 多维解析复算

用精确跨时间协方差公式，不用 Monte Carlo 估计水印协方差。每个 \((n,m)\) 组合生成 100 个随机系统/攻击实例：
- \(L=5\)，攻击能量 \(\|\Delta Y\|^2=10\)；
- \(A=Q\operatorname{diag}(1.03,\ldots,1.09)Q^\top\)，\(Q\) 为随机正交矩阵；
- \(H\in\mathbb R^{m\times n}\) 为随机 Gaussian 矩阵，仅保留扩展可观测矩阵满列秩的样本；
- \(Q_{\rm process}=I_n\)，\(R_{\rm measurement}=I_m\)；
- 攻击为 \(\Delta Y=H_{0,s}\xi\)，因此严格满足 \(Z_s\Delta Y=0\)，再缩放到能量 10；
- 固定刷新概率 \(\rho=0.7\)，比较 \(\kappa=0,0.5,1,2,4\)。

### 平均 Gaussian-equivalent KL

| 状态维数 \(n\) | 测量维数 \(m\) | iid \(\kappa=0\) | \(\kappa=2\) | 增幅 |
|---:|---:|---:|---:|---:|
| 2 | 2 | 0.32247 | 0.36465 | 13.08% |
| 4 | 2 | 0.16349 | 0.19465 | 19.05% |
| 4 | 3 | 0.25221 | 0.28115 | 11.47% |
| 4 | 4 | 0.35759 | 0.38303 | 7.11% |
| 6 | 2 | 0.08073 | 0.10097 | 25.08% |
| 6 | 3 | 0.15319 | 0.17473 | 14.06% |
| 6 | 4 | 0.20415 | 0.22316 | 9.31% |

每个维数组合均为 100 个配对系统/攻击实例。所有实例中 \(\kappa=2\) 的 Gaussian-equivalent KL 均高于 iid 基线。这里的 95% 区间见 results.json，它们是跨随机系统/攻击实例的配对区间，不是对所有系统的理论保证。

### \(\kappa=2,\rho=0.7\) 的泄露量与方向多样性

| \(m\) | iid \(E[(u_k^\top u_{k+1})^2]\) | 耦合后 | 每步互信息（nats） | \(L=5\) 窗口总相关（nats） |
|---:|---:|---:|---:|---:|
| 2 | 0.5000 | 0.3438 | 0.10083 | 0.40333 |
| 3 | 0.3333 | 0.2354 | 0.05944 | 0.23776 |
| 4 | 0.2500 | 0.1830 | 0.04020 | 0.16081 |

这些结果支持将 \(\kappa=2,\rho=0.7\) 作为下一轮原网络仿真的起始点，而不是最终最优参数。

## 6. 论文修改建议

1. **第4.1节整体替换**：保留 \(M_k=\lambda I_m+\beta_k\beta_k^\top\) 和 encode/decode 架构，将 iid Gaussian 生成器替换为上述角度耦合方向过程。
2. **定理2证明**：保留固定非零严格隐蔽攻击的 KL 发散结论，但使用新的协方差下界，而不是假设跨拍独立。
3. **定理3证明**：原稿式(49)的协方差等式不能继续使用；替换为精确跨时间协方差式和统一下界。
4. **增加有限窗口设计命题**：给出 \(c_h\)、\(C_{\ell r}\) 的闭式表达，并将 \((\kappa,\rho)\) 与有限窗口 KL 联系起来。
5. **增加多维实验**：至少加入 \((n,m)=(4,2),(4,3),(4,4)\) 和 \((6,2),(6,3),(6,4)\)，同时保留原 50 节点网络与 \(n=m=2\) 作为基准。
6. **攻击者模型边界**：当前协方差定理仍采用原稿的“攻击序列与窗口内实际水印实现统计独立”假设。互信息只量化历史水印之间的统计依赖，不自动证明面对能观察历史水印并自适应选择攻击的攻击者仍安全。若要覆盖这种更强攻击者，需要另立因果信息过滤模型和序贯检测证明。
7. **多链路部署**：每条有向链路使用独立伪随机种子/独立方向链，避免一个链路的水印状态泄露给其他链路。

## 7. 文献边界

- Zhou et al., *IEEE Transactions on Automatic Control*, 2023：在分布式估计中使用 iid Gaussian rank-one multiplicative watermark \(M_k=\lambda I+\beta_k\beta_k^\top\)，是本文直接基线。
- Ding et al., *International Journal of Robust and Nonlinear Control*, 2024：使用历史编码数据的两阶段周期动态编码机制，作用对象与本文测量包乘性水印不同，但说明“利用历史编码结构”本身不能作为唯一创新点。
- Gallo et al., *Automatica*, 2025：研究切换乘性水印参数的最优设计，是必须正面比较的近邻。本文候选差异在于保持原单拍 Gaussian 矩阵边缘不变，以各向同性角度 Markov copula 控制跨拍四阶矩，并直接推导奇偶残差的跨时间协方差与泄露—KL 折中。
- 2026 年 zonotopic set-membership watermarking 工作是相邻方向；需要在最终投稿前补充系统性查新。

## 8. 当前结论

这项设计已从“二维坐标耦合”升级为适用于任意 \(m\ge2\) 的旋转不变水印过程；状态维数 \(n\) 与测量维数 \(m\) 可以不同。它保留原稿单拍水印分布和无攻击估计输入，同时在严格隐蔽攻击下改变有限窗口跨拍协方差。现有解析复算显示多组 \((n,m)\) 下 Gaussian-equivalent KL 提高，但完整 50 节点闭环 RMSE、实际检测率和更强自适应攻击者仍待验证。
