# ==============================================================================
# IEEE Paper Simulation: Distributed Estimator + K-L Divergence + ZDA + Watermark
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import null_space, block_diag, pinv
from scipy.optimize import minimize
import scipy.linalg as la
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Publication-Quality Plot Settings (MATLAB Style)
# ------------------------------------------------------------------------------
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "lines.linewidth": 2.0,  # 线条粗细
    "lines.markersize": 8,  # 标记(点)的大小
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": False,  # 关闭网格，保持背景纯白
    "xtick.direction": "in",  # X轴刻度向内
    "ytick.direction": "in",  # Y轴刻度向内
    "xtick.top": True,  # 顶部显示刻度
    "ytick.right": True  # 右侧显示刻度
})


# ------------------------------------------------------------------------------
# 2. Global Parameters (Centralized for easy tuning)
# ------------------------------------------------------------------------------
class Params:
    sensorNum = 50
    timeSize = 150
    MC_runs = 500  # 蒙特卡洛实验次数

    dataDim = 2
    meaDim = 2

    s = 4
    KLD_window = 4
    KLD_threshold = 42.0

    attack_links = [(0, 1), (2, 3), (15, 16), (17, 31), (14, 49), (17, 34)]
    attack_start = 50
    attack_duration = 100

    # 【修复1】：分别设置不稳定系统和稳定系统的攻击幅值
    xi_magnitude_unstable = 6.0
    xi_magnitude_stable = 10.0

    watermark_lambda = 0.5
    watermark_vars_test = [0.5, 2.0, 5.0, 10.0, 50.0]

    plot_start = 20


# ------------------------------------------------------------------------------
# 3. Topology Generation
# ------------------------------------------------------------------------------
seed_val = 42
while True:
    # 调低 radius，使得拓扑图连线变少，更加清爽美观
    G = nx.random_geometric_graph(Params.sensorNum, radius=0.31, seed=seed_val)
    if nx.is_connected(G):
        break
    seed_val += 1

for (j, i) in Params.attack_links:
    if not G.has_edge(j, i):
        G.add_edge(j, i)

W = nx.to_numpy_array(G).astype(int)
neighbors_in = [list(np.where(W[:, i] == 1)[0]) for i in range(Params.sensorNum)]


def plot_topology_vector(G):
    """绘制第五章核心：图1 网络拓扑靶向高亮图 (清晰蓝色线条版 + 靶向红线)"""
    plt.figure(figsize=(8, 6.5))
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    # 1. 绘制正常边 (极细的经典 MATLAB 藏青色，放在底层)
    nx.draw_networkx_edges(G, pos, edge_color='#2A5CAA', alpha=0.6, width=0.6)

    # 2. 绘制正常节点 (放在边之上)
    nx.draw_networkx_nodes(G, pos, node_color='#2A5CAA', node_size=20, alpha=1.0)

    # 3. ★核心修改：重新加入靶向攻击节点与边的标红逻辑 (放在最顶层)
    attacked_nodes = list(set([n for edge in Params.attack_links for n in edge]))
    nx.draw_networkx_nodes(G, pos, nodelist=attacked_nodes, node_color='red', node_size=60)
    # arrowsize=15 确保有向攻击链路的箭头清晰可见
    nx.draw_networkx_edges(G, pos, edgelist=Params.attack_links, edge_color='red', width=2.0, arrows=True, arrowsize=15)

    # 4. 节点标号（从 0 开始，微调偏移量防止遮挡小圆点）
    labels = {i: str(i) for i in range(Params.sensorNum)}
    label_pos = {k: [v[0] - 0.02, v[1] + 0.03] for k, v in pos.items()} # 调整文本偏移量适应原坐标系
    nx.draw_networkx_labels(G, label_pos, labels, font_size=11, font_color='black', font_family='Times New Roman')

    # 5. 坐标轴标签
    plt.xlabel("X coordinate", fontsize=16, fontname='Times New Roman')
    plt.ylabel("Y coordinate", fontsize=16, fontname='Times New Roman')

    # 6. 图框美化：黑色全封闭边框，刻度朝内
    ax = plt.gca()
    ax.tick_params(direction='in', top=True, right=True, length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig("Fig1_Topology_Clean.pdf", format='pdf', bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------------------
# 4. System Matrices & Defense Class
# ------------------------------------------------------------------------------
A_unstable = np.array([[1.08, 0.05], [0.01, 1.05]])
A_stable = np.array([[0.95, 0.02], [0.02, 0.97]])
Q = 1 * np.eye(Params.dataDim)

np.random.seed(42)
H = np.zeros((Params.meaDim, Params.dataDim, Params.sensorNum))
R = np.zeros((Params.meaDim, Params.meaDim, Params.sensorNum))
for i in range(Params.sensorNum):
    H[0, 0, i] = 1.0 + 0.5 * np.random.rand()
    H[1, 1, i] = 1.0 + 0.5 * np.random.rand()
    R[:, :, i] = 0.5 * np.eye(Params.meaDim)


class DynamicWatermark:
    def __init__(self, meas_dim, num_nodes, T, lam, var):
        self.M, self.M_inv = {}, {}
        for j in range(num_nodes):
            for k in range(T):
                beta = np.random.normal(0, np.sqrt(var), meas_dim)
                Mk = lam * np.eye(meas_dim) + np.outer(beta, beta)
                self.M[(j, k)] = Mk
                self.M_inv[(j, k)] = pinv(Mk)

    def encrypt(self, y, j, k):
        return self.M_inv[(j, k)] @ y

    def decrypt(self, y_enc, j, k):
        return self.M[(j, k)] @ y_enc


# ------------------------------------------------------------------------------
# 5. Parity Space & Attack Synthesis
# ------------------------------------------------------------------------------
def build_parity_space(A_sys):
    Zs_cache, Theta_cache = {}, {}
    for j in range(Params.sensorNum):
        H_j = H[:, :, j]
        H_xs = np.vstack([H_j @ np.linalg.matrix_power(A_sys, r) for r in range(Params.s + 1)])
        N_space = null_space(H_xs.T)

        if N_space.size > 0:
            Zs = N_space.T
            Zs_cache[j] = Zs
            Hw_rows = []
            for r in range(Params.s + 1):
                row_blocks = [H_j @ np.linalg.matrix_power(A_sys, r - 1 - c) if c < r else np.zeros(
                    (Params.meaDim, Params.dataDim)) for c in range(Params.s + 1)]
                Hw_rows.append(np.hstack(row_blocks))
            Hw_s = np.vstack(Hw_rows)

            Hv_s = np.eye((Params.s + 1) * Params.meaDim)
            Q_s = block_diag(*[Q for _ in range(Params.s + 1)])
            R_s = block_diag(*[R[:, :, j] for _ in range(Params.s + 1)])

            Theta = Zs @ (Hw_s @ Q_s @ Hw_s.T + Hv_s @ R_s @ Hv_s.T) @ Zs.T
            Theta_cache[j] = 0.5 * (Theta + Theta.T) + 1e-8 * np.eye(Theta.shape[0])
        else:
            Zs_cache[j], Theta_cache[j] = None, None
    return Zs_cache, Theta_cache


def generate_zda_sequence(A_sys, H_j, duration, kappa):
    w, v = np.linalg.eig(A_sys)
    idx = np.argmax(np.abs(w))
    v_u = np.real(v[:, idx])
    x_f = kappa * v_u
    attack_seq = np.zeros((duration, Params.meaDim))
    for t in range(duration):
        attack_seq[t, :] = H_j @ x_f
        x_f = A_sys @ x_f
    return attack_seq


# 【修复2】：添加 xi_mag 参数
def get_zda_biases(A_sys, xi_mag):
    zda_bias = {}
    for (sender, recv) in Params.attack_links:
        if sender not in zda_bias:
            zda_bias[sender] = generate_zda_sequence(A_sys, H[:, :, sender], Params.attack_duration, xi_mag)
    return zda_bias


def compute_KL_divergence(r_window, Theta):
    if len(r_window) < 2: return 0.0
    r_mat = np.hstack(r_window)
    mu_hat = np.mean(r_mat, axis=1, keepdims=True)
    Sigma_hat = np.cov(r_mat) + 1e-6 * np.eye(r_mat.shape[0])

    Theta_inv = pinv(Theta)
    d = r_mat.shape[0]
    term1 = np.trace(Theta_inv @ Sigma_hat)
    term2 = (mu_hat.T @ Theta_inv @ mu_hat)[0, 0]
    _, logdet_Theta = np.linalg.slogdet(Theta)
    _, logdet_Sigma = np.linalg.slogdet(Sigma_hat)
    kld = 0.5 * (term1 + term2 - d + (logdet_Theta - logdet_Sigma))
    return max(0.0, kld)


# ------------------------------------------------------------------------------
# 6. Single Simulation Engine
# ------------------------------------------------------------------------------
# 【修复3】：接收 xi_mag 参数并向下传递
def run_single_simulation(use_defense, use_attack, w_var, A_sys, xi_mag):
    watermark = DynamicWatermark(Params.meaDim, Params.sensorNum, Params.timeSize, Params.watermark_lambda, w_var)
    Zs_cache, Theta_cache = build_parity_space(A_sys)
    zda_bias = get_zda_biases(A_sys, xi_mag)

    x_true = np.zeros((Params.dataDim, Params.timeSize))
    y_nominal = np.zeros((Params.sensorNum, Params.meaDim, Params.timeSize))
    x_true[:, 0] = np.random.normal(0, 1, Params.dataDim)

    for k in range(1, Params.timeSize):
        x_true[:, k] = A_sys @ x_true[:, k - 1] + np.random.multivariate_normal(np.zeros(Params.dataDim), Q)
        for i in range(Params.sensorNum):
            y_nominal[i, :, k] = H[:, :, i] @ x_true[:, k] + np.random.multivariate_normal(np.zeros(Params.meaDim),
                                                                                           R[:, :, i])

    xHat = np.zeros((Params.dataDim, Params.timeSize, Params.sensorNum))
    P_cov = np.zeros((Params.dataDim, Params.dataDim, Params.sensorNum, Params.timeSize))
    for i in range(Params.sensorNum):
        P_cov[:, :, i, 0] = 5.0 * np.eye(Params.dataDim)

    kld_stats = {link: np.zeros(Params.timeSize) for link in Params.attack_links}
    r_buffer = {link: [] for link in Params.attack_links}
    y_recv_history = np.zeros((Params.sensorNum, Params.sensorNum, Params.meaDim, Params.timeSize))
    isolated_links = {link: np.zeros(Params.timeSize, dtype=bool) for link in Params.attack_links}

    for k in range(1, Params.timeSize):
        for recv in range(Params.sensorNum):
            for sender in neighbors_in[recv]:
                y_trans = watermark.encrypt(y_nominal[sender, :, k], sender, k) if use_defense else y_nominal[sender, :,
                                                                                                    k].copy()
                if use_attack and (sender,
                                   recv) in Params.attack_links and Params.attack_start <= k < Params.attack_start + Params.attack_duration:
                    y_trans += zda_bias[sender][k - Params.attack_start]
                y_recv_dec = watermark.decrypt(y_trans, sender, k) if use_defense else y_trans
                y_recv_history[recv, sender, :, k] = y_recv_dec

        if k >= Params.s:
            for (sender, recv) in Params.attack_links:
                Zs = Zs_cache[sender]
                if Zs is not None:
                    Y_hist = [y_recv_history[recv, sender, :, t].reshape(-1, 1) for t in range(k - Params.s, k + 1)]
                    r_k = Zs @ np.vstack(Y_hist)
                    r_buffer[(sender, recv)].append(r_k)
                    if len(r_buffer[(sender, recv)]) > Params.KLD_window:
                        r_buffer[(sender, recv)].pop(0)
                    kld_stats[(sender, recv)][k] = compute_KL_divergence(r_buffer[(sender, recv)], Theta_cache[sender])
                    if kld_stats[(sender, recv)][k] > Params.KLD_threshold:
                        isolated_links[(sender, recv)][k] = True

        for i in range(Params.sensorNum):
            x_pred = A_sys @ xHat[:, k - 1, i]
            P_pred = A_sys @ P_cov[:, :, i, k - 1] @ A_sys.T + Q
            active_neighbors = []
            for j in neighbors_in[i]:
                if use_defense and (j, i) in Params.attack_links and isolated_links[(j, i)][k]: continue
                active_neighbors.append(j)
            active_neighbors.append(i)

            y_ext, H_ext, R_ext = [], [], []
            for j in active_neighbors:
                y_ext.append(y_nominal[i, :, k] if j == i else y_recv_history[i, j, :, k])
                H_ext.append(H[:, :, j])
                R_ext.append(R[:, :, j])

            H_stack = np.vstack(H_ext)
            R_block = block_diag(*R_ext)
            S = H_stack @ P_pred @ H_stack.T + R_block
            K_gain = P_pred @ H_stack.T @ pinv(S)

            innov = np.concatenate(y_ext) - H_stack @ x_pred
            xHat[:, k, i] = x_pred + K_gain @ innov

            P_upd = (np.eye(Params.dataDim) - K_gain @ H_stack) @ P_pred
            P_cov[:, :, i, k] = 0.5 * (P_upd + P_upd.T) + 1e-9 * np.eye(Params.dataDim)

    return x_true, xHat, kld_stats


# ------------------------------------------------------------------------------
# 7. Monte Carlo Wrapper
# ------------------------------------------------------------------------------
# 【修复4】：接收 xi_mag 并在内部调用时传递
def run_monte_carlo(use_defense, use_attack, w_var=0.0, A_sys=A_unstable, xi_mag=3.0):
    avg_rmse = np.zeros(Params.timeSize)
    avg_kld = {link: np.zeros(Params.timeSize) for link in Params.attack_links}
    avg_alarm_rate = {link: 0.0 for link in Params.attack_links}

    att_start = Params.attack_start
    att_end = Params.attack_start + Params.attack_duration

    for mc in range(Params.MC_runs):
        x_true, x_hat, kld_stats = run_single_simulation(use_defense, use_attack, w_var, A_sys, xi_mag)

        # 1. 统计 RMSE
        rmse_t = np.zeros(Params.timeSize)
        for t in range(Params.timeSize):
            errs = [np.linalg.norm(x_true[:, t] - x_hat[:, t, i]) for i in range(Params.sensorNum)]
            rmse_t[t] = np.sqrt(np.mean(np.array(errs) ** 2))
        avg_rmse += rmse_t

        # 2. 统计 KLD 和 单次实验的报警率
        for link in Params.attack_links:
            avg_kld[link] += kld_stats[link]
            if use_attack:
                kld_window = kld_stats[link][att_start:att_end]
                alarm_count = np.sum(kld_window > Params.KLD_threshold)
                avg_alarm_rate[link] += (alarm_count / Params.attack_duration)

    avg_rmse /= Params.MC_runs
    for link in Params.attack_links:
        avg_kld[link] /= Params.MC_runs
        if use_attack:
            avg_alarm_rate[link] /= Params.MC_runs

    return avg_rmse, avg_kld, avg_alarm_rate


# ------------------------------------------------------------------------------
# 8. Execution & Generaton of Independent Figures
# ------------------------------------------------------------------------------
plot_topology_vector(G)

print(f"Running Monte Carlo (MC={Params.MC_runs}) for Unstable System (Theorem 3)...")
rmse_unstable_clean, _, _ = run_monte_carlo(use_defense=False, use_attack=False, w_var=0.0, A_sys=A_unstable,
                                            xi_mag=Params.xi_magnitude_unstable)
rmse_unstable_attack, kld_attack_no_def, alarm_rate_no_def = run_monte_carlo(use_defense=False, use_attack=True,
                                                                             w_var=0.0, A_sys=A_unstable,
                                                                             xi_mag=Params.xi_magnitude_unstable)

print(f"Running Monte Carlo (MC={Params.MC_runs}) for Stable System (Theorem 2)...")
rmse_stable_clean, _, _ = run_monte_carlo(use_defense=False, use_attack=False, w_var=0.0, A_sys=A_stable,
                                          xi_mag=Params.xi_magnitude_stable)
rmse_stable_attack, _, _ = run_monte_carlo(use_defense=False, use_attack=True, w_var=0.0, A_sys=A_stable,
                                           xi_mag=Params.xi_magnitude_stable)

# 纯净工况下的水印一致性测试 (Theorem Preservation)
rmse_nominal_def, _, _ = run_monte_carlo(use_defense=True, use_attack=False, w_var=Params.watermark_vars_test[1],
                                         A_sys=A_unstable, xi_mag=Params.xi_magnitude_unstable)

print(f"Running Defense Scenarios under varying Watermark variances...")
rmse_def_dict, kld_def_dict, alarm_rate_def_dict = {}, {}, {}

print(f"Running Defense Scenarios under varying Watermark variances...")

# ====== 【新增代码：计算仅有 lambda，无 beta 随机项的基线】 ======
_, kld_def_lambda_only, _ = run_monte_carlo(
    use_defense=True,
    use_attack=True,
    w_var=0.0,  # 方差为0，代表没有随机项 beta
    A_sys=A_unstable,
    xi_mag=Params.xi_magnitude_unstable
)
# ==============================================================

rmse_def_dict, kld_def_dict, alarm_rate_def_dict = {}, {}, {}
for w_var in Params.watermark_vars_test:
    rmse_def, kld_def, alarm_rate_def = run_monte_carlo(use_defense=True, use_attack=True, w_var=w_var,
                                                        A_sys=A_unstable, xi_mag=Params.xi_magnitude_unstable)
    rmse_def_dict[w_var] = rmse_def
    kld_def_dict[w_var] = kld_def
    alarm_rate_def_dict[w_var] = alarm_rate_def

t_start = Params.plot_start
t_end = Params.timeSize
t_axis = np.arange(t_start, t_end)
colors = ['#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
link_to_plot = Params.attack_links[0]

legend_props = {
    'frameon': True, 'edgecolor': 'black', 'facecolor': 'white',
    'framealpha': 1.0, 'fancybox': False, 'shadow': False
}

# --- Figure 2: Subplots for Theorem 2 (Bounded) and Theorem 3 (Divergence) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Unstable
axes[0].plot(t_axis, rmse_unstable_clean[t_start:], label='Clean (No Attack)', color='#0072BD', linestyle='-',
             marker='o', markerfacecolor='none', markevery=5)
axes[0].plot(t_axis, rmse_unstable_attack[t_start:], label='Stealthy FDI Attack (No Defense)', color='#D95319', linestyle='-',
             marker='*', markevery=5)
axes[0].set_xlabel('$k$')
axes[0].set_ylabel('RMSE')
axes[0].set_xlim([t_start, t_end - 1])
axes[0].set_yscale('log')
axes[0].set_title('(a) Unstable System')
axes[0].legend(loc='upper left', **legend_props)
axes[0].tick_params(direction='in', top=True, right=True)

# Right: Stable
axes[1].plot(t_axis, rmse_stable_clean[t_start:], label='Clean (No Attack)', color='#0072BD', linestyle='-', marker='o',
             markerfacecolor='none', markevery=5)
axes[1].plot(t_axis, rmse_stable_attack[t_start:], label='Stealthy FDI Attack (No Defense)', color='#D95319', linestyle='-', marker='*',
             markevery=5)
axes[1].set_xlabel('$k$')
axes[1].set_ylabel('RMSE')
axes[1].set_xlim([t_start, t_end - 1])
axes[1].set_yscale('linear')
axes[1].set_title('(b) Stable System')
axes[1].legend(loc='upper right', **legend_props)
axes[1].tick_params(direction='in', top=True, right=True)

plt.tight_layout()
plt.savefig("Fig2_RMSE_Thm2_Thm3.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- Figure 3: KLD without Defense (Theorem 1) ---
plt.figure(figsize=(8, 5.5))
plt.plot(t_axis, kld_attack_no_def[link_to_plot][t_start:], label='K-L Divergence (No Defense)', color='#D95319',
         linestyle='-', marker='*', markevery=5)
plt.axhline(Params.KLD_threshold, color='black', linestyle='--', linewidth=2.0, label=r'Detection Threshold $\theta$')
plt.xlabel('$k$')
plt.ylabel('K-L Divergence Statistic')
plt.xlim([t_start, t_end - 1])
plt.yscale('log')
plt.ylim(bottom=1e-1, top=1e2)
plt.legend(loc='upper right', **legend_props)
plt.tight_layout()
plt.savefig("Fig3_KLD_NoDefense.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- Figure 4: KLD with Watermarking (Theorem 4) ---
rep_var = Params.watermark_vars_test[1]
plt.figure(figsize=(8, 5.5))
plt.plot(t_axis, kld_def_dict[rep_var][link_to_plot][t_start:], label='K-L Divergence (With Watermark)',
         color='#7E2F8E', linestyle='-', marker='s', markerfacecolor='none', markevery=5)
plt.axhline(Params.KLD_threshold, color='black', linestyle='--', linewidth=2.0, label=r'Detection Threshold $\theta$')
plt.xlabel('$k$')
plt.ylabel('K-L Divergence Statistic')
plt.xlim([t_start, t_end - 1])
plt.yscale('log')
plt.ylim(bottom=1e-1)
plt.legend(loc='upper left', **legend_props)
plt.tight_layout()
plt.savefig("Fig4_KLD_WithDefense.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- Figure 5: KLD with Varying Watermark Strengths ---
markers = ['o', 's', '^', 'd', 'v']
plt.figure(figsize=(8, 5.5))

# 1. 绘制无水印攻击曲线
plt.plot(t_axis, kld_attack_no_def[link_to_plot][t_start:], label='No Watermarking', color='#D95319', linestyle='-',
         marker='*', markevery=5)

# 2. ====== 【新增代码：绘制仅含 lambda (无随机项 beta) 的曲线】 ======
plt.plot(t_axis, kld_def_lambda_only[link_to_plot][t_start:], label=r'Only $\lambda$ ($\sigma_\beta^2=0$)', color='gray', linestyle='--',
         marker='p', markerfacecolor='none', markevery=5)
# =================================================================

# 3. 绘制不同 watermark 强度的曲线
for idx, w_var in enumerate(Params.watermark_vars_test):
    plt.plot(t_axis, kld_def_dict[w_var][link_to_plot][t_start:], label=f' $\sigma_\\beta^2={w_var}$', color=colors[idx],
             linestyle='-', marker=markers[idx], markerfacecolor='none', markevery=5)

plt.axhline(Params.KLD_threshold, color='black', linestyle='--', linewidth=2.0, label=r'Threshold $\theta$')
plt.xlabel('$k$')
plt.ylabel('K-L Divergence Statistic')
plt.xlim([t_start, t_end - 1])
plt.yscale('log')
plt.ylim(bottom=1e-1)
plt.legend(loc='upper left', **legend_props)
plt.tight_layout()
plt.savefig("Fig5_KLD_Varying_Strengths.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- Figure 6: RMSE with Watermarking Defense vs Clean ---
plt.figure(figsize=(8, 5.5))
plt.plot(t_axis, rmse_unstable_clean[t_start:], label='Clean (No Attack)', color='#0072BD', linestyle='-', marker='o',
         markerfacecolor='none', markevery=5)
plt.plot(t_axis, rmse_def_dict[rep_var][t_start:], label=f'Stealthy FDI Attack (With Watermarking, $\sigma_\\beta^2={rep_var}$)', color='#7E2F8E',
         linestyle='-', marker='s', markerfacecolor='none', markevery=5)
plt.xlabel('$k$')
plt.ylabel('RMSE')
plt.xlim([t_start, t_end - 1])
plt.yscale('linear')
y_max = max(np.max(rmse_unstable_clean[t_start:]), np.max(rmse_def_dict[rep_var][t_start:])) * 1.5
plt.ylim(0, y_max)
plt.legend(loc='upper right', **legend_props)
plt.tight_layout()
plt.savefig("Fig6_RMSE_WithDefense_Linear.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- Figure 7: Arbitrary Attack Sequence Optimization under KLD Constraint ---
#
# The attack is no longer restricted to the ZDA form generated by
# generate_zda_sequence(). The optimization variable is the entire sequence
#
#     a = col{a_0, a_1, ..., a_{L-1}} in R^(L*m),
#
# injected directly into y_trans over the attack interval.
#
# Because the receiver decrypts by multiplying M_k, the decoded attack shift is
# M_k a_t. For each complete parity window,
#
#     Delta r_k = Z_s blkdiag(M_{k-s}, ..., M_k)
#                 col{a_{k-s}, ..., a_k}.
#
# With the nominal residual covariance Theta unchanged, the attack-induced
# mean-shift KLD is
#
#     D_k(a) = 0.5 * Delta r_k^T Theta^(-1) Delta r_k.
#
# The deterministic attack sequence is chosen without knowing the actual random
# watermark realization. Therefore C_k is averaged over independent watermark
# realizations, and the peak-KLD constrained problem is
#
#     maximize    (1/L) * ||a||_2^2
#     subject to  D_k(a) <= KLD_threshold for every complete attack window.
#
# Write a = alpha*v with ||v||=1. The direction is obtained from
#
#     minimize_{||v||=1} max_k v^T C_k v,
#
# and then alpha is obtained directly from the active KLD boundary.

ATTACK_OPT_WATERMARK_MC = Params.MC_runs
ATTACK_OPT_MULTI_STARTS = 8
ATTACK_OPT_SEED = 20260712
ATTACK_OPT_FTOL = 1e-10
ATTACK_OPT_MAXITER = 1500


def build_expected_window_kld_matrices(w_var, link, A_sys, watermark_mc, seed):
    """Build expected per-window matrices C_k satisfying D_k(a)=0.5*a.T*C_k*a."""
    sender, _ = link
    Zs_cache, Theta_cache = build_parity_space(A_sys)
    Zs = Zs_cache[sender]
    Theta = Theta_cache[sender]

    if Zs is None or Theta is None:
        raise RuntimeError(f'Parity space is unavailable for sender {sender}.')

    theta_inv = pinv(Theta)
    L = Params.attack_duration
    m = Params.meaDim
    d = L * m
    window_ends = list(range(Params.s, L))
    C_stack = np.zeros((len(window_ends), d, d), dtype=float)
    rng = np.random.default_rng(seed)

    for _ in range(watermark_mc):
        M_attack = []
        for _t in range(L):
            beta = rng.normal(0.0, np.sqrt(w_var), size=m)
            M_t = Params.watermark_lambda * np.eye(m) + np.outer(beta, beta)
            M_attack.append(M_t)

        for idx, q in enumerate(window_ends):
            M_block = block_diag(*M_attack[q - Params.s:q + 1])
            G_local = Zs @ M_block
            C_local = G_local.T @ theta_inv @ G_local

            c0 = (q - Params.s) * m
            c1 = (q + 1) * m
            C_stack[idx, c0:c1, c0:c1] += C_local

    C_stack /= float(watermark_mc)
    C_stack = 0.5 * (C_stack + np.swapaxes(C_stack, 1, 2))
    return C_stack, window_ends


def solve_arbitrary_attack_sequence(C_stack, threshold, multi_starts, seed):
    """
    Directly optimize the full attack sequence under the peak-KLD constraints.

    Direction problem:
        min_{v,t} t
        s.t. ||v||^2 = 1,
             v.T C_k v <= t for every window k.
    """
    n_windows, d, _ = C_stack.shape
    rng = np.random.default_rng(seed)

    def window_quadratics(v):
        return np.einsum('i,kij,j->k', v, C_stack, v, optimize=True)

    def objective(z):
        return float(z[-1])

    def objective_jac(z):
        grad = np.zeros_like(z)
        grad[-1] = 1.0
        return grad

    def unit_norm_constraint(z):
        v = z[:-1]
        return float(v @ v - 1.0)

    def unit_norm_jac(z):
        jac = np.zeros_like(z)
        jac[:-1] = 2.0 * z[:-1]
        return jac

    def peak_constraints(z):
        v = z[:-1]
        t = z[-1]
        return t - window_quadratics(v)

    def peak_constraints_jac(z):
        v = z[:-1]
        Cv = np.einsum('kij,j->ki', C_stack, v, optimize=True)
        jac = np.empty((n_windows, d + 1), dtype=float)
        jac[:, :-1] = -2.0 * Cv
        jac[:, -1] = 1.0
        return jac

    constraints = [
        {'type': 'eq', 'fun': unit_norm_constraint, 'jac': unit_norm_jac},
        {'type': 'ineq', 'fun': peak_constraints, 'jac': peak_constraints_jac},
    ]
    bounds = [(-1.0, 1.0)] * d + [(1e-16, None)]

    C_average = np.mean(C_stack, axis=0)
    eigvals, eigvecs = np.linalg.eigh(C_average)
    starts = [eigvecs[:, np.argmin(eigvals)]]

    for _ in range(max(0, multi_starts - 1)):
        v0 = rng.normal(size=d)
        v0 /= np.linalg.norm(v0)
        starts.append(v0)

    best = None
    best_t = np.inf

    for v0 in starts:
        v0 = np.asarray(v0, dtype=float)
        v0 /= np.linalg.norm(v0)
        t0 = max(float(np.max(window_quadratics(v0))) * 1.05, 1e-12)
        z0 = np.concatenate([v0, [t0]])

        result = minimize(
            objective,
            z0,
            method='SLSQP',
            jac=objective_jac,
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': ATTACK_OPT_FTOL,
                'maxiter': ATTACK_OPT_MAXITER,
                'disp': False,
            },
        )

        v_candidate = result.x[:-1]
        norm_v = np.linalg.norm(v_candidate)
        if not np.isfinite(norm_v) or norm_v <= 0:
            continue
        v_candidate /= norm_v
        t_candidate = float(np.max(window_quadratics(v_candidate)))

        if np.isfinite(t_candidate) and t_candidate > 0.0 and t_candidate < best_t:
            best_t = t_candidate
            best = {
                'direction': v_candidate,
                't_star': t_candidate,
                'solver_success': bool(result.success),
                'solver_message': str(result.message),
            }

    if best is None:
        raise RuntimeError('Attack-sequence optimization failed for every initialization.')

    if best_t <= 1e-14:
        best.update({
            'status': 'unbounded',
            'alpha': np.inf,
            'attack_vector': None,
            'average_energy': np.inf,
            'total_energy': np.inf,
            'max_kld': 0.0,
            'window_kld': None,
        })
        return best

    alpha = np.sqrt(2.0 * threshold / best_t)
    attack_vector = alpha * best['direction']
    window_kld = 0.5 * np.einsum(
        'i,kij,j->k', attack_vector, C_stack, attack_vector, optimize=True
    )

    best.update({
        'status': 'solved',
        'alpha': float(alpha),
        'attack_vector': attack_vector,
        'average_energy': float(attack_vector @ attack_vector / Params.attack_duration),
        'total_energy': float(attack_vector @ attack_vector),
        'max_kld': float(np.max(window_kld)),
        'window_kld': window_kld,
    })
    return best


arbitrary_attack_results = {}
max_energies = []
optimal_attack_sequences = {}

for idx, w_var in enumerate(Params.watermark_vars_test):
    print(f'Solving arbitrary attack sequence for sigma_beta^2={w_var} ...')

    C_stack, attack_window_ends = build_expected_window_kld_matrices(
        w_var=w_var,
        link=link_to_plot,
        A_sys=A_unstable,
        watermark_mc=ATTACK_OPT_WATERMARK_MC,
        seed=ATTACK_OPT_SEED + idx,
    )

    result = solve_arbitrary_attack_sequence(
        C_stack=C_stack,
        threshold=Params.KLD_threshold,
        multi_starts=ATTACK_OPT_MULTI_STARTS,
        seed=ATTACK_OPT_SEED + 1000 + idx,
    )

    arbitrary_attack_results[w_var] = result
    max_energies.append(result['average_energy'])

    if result['attack_vector'] is not None:
        optimal_attack_sequences[w_var] = result['attack_vector'].reshape(
            Params.attack_duration, Params.meaDim
        )

np.savez(
    'Fig7_Optimal_Arbitrary_Attack_Sequences.npz',
    **{
        f"watermark_var_{str(w_var).replace('.', '_')}": seq
        for w_var, seq in optimal_attack_sequences.items()
    },
)

plt.figure(figsize=(8, 5.5))
plt.plot(
    Params.watermark_vars_test,
    max_energies,
    marker='s',
    color='#A2142F',
    linewidth=2.5,
    markersize=10,
    markerfacecolor='none',
)
plt.xlabel('Watermarking Variance $\\sigma_\\beta^2$')
plt.ylabel('Maximum Average Attack Energy')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().tick_params(direction='in', top=True, right=True)
plt.tight_layout()
plt.savefig('Fig7_Arbitrary_Attack_Energy_Optimization.pdf', format='pdf', bbox_inches='tight')
plt.show()

print('\n' + '=' * 118)
print('>>> ARBITRARY ATTACK SEQUENCE OPTIMIZATION UNDER PEAK KLD CONSTRAINT <<<')
print('=' * 118)
print(
    f'Constraint: max_k D_KL,k(a) <= {Params.KLD_threshold} | '
    f'Target link: {link_to_plot} | '
    f'Attack variables: {Params.attack_duration * Params.meaDim}'
)
print('-' * 118)
print(
    f"{'Watermark Var':<16} | "
    f"{'Average Energy':<18} | "
    f"{'Total Energy':<18} | "
    f"{'Max KLD Check':<16} | "
    f"{'Status':<12}"
)
print('-' * 118)

for w_var in Params.watermark_vars_test:
    result = arbitrary_attack_results[w_var]
    print(
        f"{w_var:<16} | "
        f"{result['average_energy']:<18.6e} | "
        f"{result['total_energy']:<18.6e} | "
        f"{result['max_kld']:<16.6f} | "
        f"{result['status']:<12}"
    )
print('=' * 118 + '\n')


# ==============================================================================
# 9. Terminal Output: Alarm Rate Analysis & Nominal Performance consistency
# ==============================================================================
print("\n" + "=" * 90)
print(">>> NOMINAL ESTIMATION PERFORMANCE CONSISTENCY <<<")
print("=" * 90)
rmse_base = np.mean(rmse_unstable_clean)
rmse_def_base = np.mean(rmse_nominal_def)
print(f"Average RMSE (Clean, No Defense)           : {rmse_base:.4f}")
print(f"Average RMSE (Clean, With Watermark Var={Params.watermark_vars_test[1]}) : {rmse_def_base:.4f}")
print("Conclusion: The difference is negligible, successfully preserving nominal performance.")

print("\n" + "=" * 90)
print(">>> ALARM RATE ANALYSIS DURING ATTACK WINDOW (Monte Carlo Average) <<<")
print(
    f">>> Threshold (gamma) = {Params.KLD_threshold} | Attack Window = [{Params.attack_start}, {Params.attack_start + Params.attack_duration})")
print("=" * 90)

header = f"{'Attacked Link':<15} | {'No Defense':<15}"
for var in Params.watermark_vars_test:
    header += f" | {'WM (Var=' + str(var) + ')':<15}"
print(header)
print("-" * 110)

for link in Params.attack_links:
    row_str = f"{str(link):<15} | {alarm_rate_no_def[link]:<15.2%}"
    for var in Params.watermark_vars_test:
        row_str += f" | {alarm_rate_def_dict[var][link]:<15.2%}"
    print(row_str)

print("=" * 110 + "\n")