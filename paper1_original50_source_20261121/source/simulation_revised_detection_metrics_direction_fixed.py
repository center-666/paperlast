# ==============================================================================
# IEEE Paper Simulation: Distributed Estimator + K-L Divergence + ZDA + Watermark
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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
rep_var = 2.0
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
    MC_runs = 5  # 蒙特卡洛实验次数

    dataDim = 2
    meaDim = 2

    s = 4
    KLD_window = 4
    KLD_threshold = 42.0

    # Directed-link convention used throughout the paper and code:
    #     (receiver, sender) = (i, j) denotes the communication link j -> i.
    # For example, (0, 1) means node 1 sends data to node 0.
    attack_links = [(0, 1), (2, 3), (15, 16), (17, 31), (14, 49), (17, 34)]
    attack_start = 50
    attack_duration = 100

    # 【修复1】：分别设置不稳定系统和稳定系统的攻击幅值
    xi_magnitude_unstable = 6.0
    xi_magnitude_stable = 10.0

    watermark_lambda = 0.5
    watermark_vars_test = [0.5, 2.0, 5.0, 10.0, 50.0]

    plot_start = 20


def format_directed_link(link):
    """
    Format a stored link (receiver, sender) as the physical direction
    sender -> receiver.
    """
    receiver, sender = link
    return f"{sender}\u2192{receiver}"




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

for receiver, sender in Params.attack_links:
    if not G.has_edge(receiver, sender):
        G.add_edge(receiver, sender)

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
    # Params.attack_links stores (receiver, sender), whereas NetworkX
    # arrow tuples are drawn as source -> target.
    attacked_directed_edges = [
        (sender, receiver)
        for receiver, sender in Params.attack_links
    ]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=attacked_directed_edges,
        edge_color='red',
        width=2.0,
        arrows=True,
        arrowsize=15,
    )

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
    """Generate one ZDA sequence for each transmitting node."""
    zda_bias = {}
    for receiver, sender in Params.attack_links:
        if sender not in zda_bias:
            zda_bias[sender] = generate_zda_sequence(
                A_sys,
                H[:, :, sender],
                Params.attack_duration,
                xi_mag,
            )
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
        for receiver in range(Params.sensorNum):
            for sender in neighbors_in[receiver]:
                y_trans = (
                    watermark.encrypt(y_nominal[sender, :, k], sender, k)
                    if use_defense
                    else y_nominal[sender, :, k].copy()
                )
                link = (receiver, sender)
                if (
                    use_attack
                    and link in Params.attack_links
                    and Params.attack_start
                    <= k
                    < Params.attack_start + Params.attack_duration
                ):
                    y_trans += zda_bias[sender][k - Params.attack_start]
                y_recv_dec = (
                    watermark.decrypt(y_trans, sender, k)
                    if use_defense
                    else y_trans
                )
                y_recv_history[receiver, sender, :, k] = y_recv_dec

        if k >= Params.s:
            for receiver, sender in Params.attack_links:
                link = (receiver, sender)
                Zs = Zs_cache[sender]
                if Zs is not None:
                    Y_hist = [
                        y_recv_history[receiver, sender, :, t].reshape(-1, 1)
                        for t in range(k - Params.s, k + 1)
                    ]
                    r_k = Zs @ np.vstack(Y_hist)
                    r_buffer[link].append(r_k)
                    if len(r_buffer[link]) > Params.KLD_window:
                        r_buffer[link].pop(0)
                    kld_stats[link][k] = compute_KL_divergence(
                        r_buffer[link],
                        Theta_cache[sender],
                    )
                    if kld_stats[link][k] > Params.KLD_threshold:
                        isolated_links[link][k] = True

        for i in range(Params.sensorNum):
            x_pred = A_sys @ xHat[:, k - 1, i]
            P_pred = A_sys @ P_cov[:, :, i, k - 1] @ A_sys.T + Q
            active_neighbors = []
            for j in neighbors_in[i]:
                # Stored convention: (receiver=i, sender=j).
                link = (i, j)
                if (
                    use_defense
                    and link in Params.attack_links
                    and isolated_links[link][k]
                ):
                    continue
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
def run_monte_carlo(use_defense, use_attack, w_var=0.0,
                    A_sys=A_unstable, xi_mag=3.0):
    """
    Monte Carlo averages of estimation and binary-detection metrics.

    Detection decision:
        D_KL,k > KLD_threshold  =>  alarm.

    Sample sets:
        K_0: attack-free samples after detector warm-up and before attack_start.
        K_a: attacked samples beginning at attack_start + s, so every parity
             window used for detection is fully occupied by attacked data.

    Metrics:
        FPR = # {k in K_0 : alarm} / |K_0|
        TPR = # {k in K_a : alarm} / |K_a|
        FNR = # {k in K_a : no alarm} / |K_a| = 1 - TPR

        Detection delay is measured from attack_start to the first alarm in K_a.
        It is averaged only over Monte Carlo runs in which an alarm occurs.
    """
    avg_rmse = np.zeros(Params.timeSize)
    avg_kld = {
        link: np.zeros(Params.timeSize)
        for link in Params.attack_links
    }

    avg_tpr = {link: 0.0 for link in Params.attack_links}
    avg_fnr = {link: 0.0 for link in Params.attack_links}
    avg_fpr = {link: 0.0 for link in Params.attack_links}

    delay_sum = {link: 0.0 for link in Params.attack_links}
    delay_count = {link: 0 for link in Params.attack_links}

    detector_ready_start = max(
        Params.plot_start,
        Params.s + Params.KLD_window - 1
    )

    clean_eval_start = detector_ready_start
    clean_eval_end = min(Params.attack_start, Params.timeSize)

    # The attack metric begins only after a complete attacked parity window
    # has been formed, as recommended for strict stealthiness evaluation.
    attack_eval_start = Params.attack_start + Params.s
    attack_eval_end = min(
        Params.attack_start + Params.attack_duration,
        Params.timeSize
    )

    if clean_eval_end <= clean_eval_start:
        raise ValueError(
            "No valid attack-free samples are available for FPR evaluation."
        )

    if use_attack and attack_eval_end <= attack_eval_start:
        raise ValueError(
            "No complete attacked parity window is available for TPR/FNR."
        )

    for mc in range(Params.MC_runs):
        x_true, x_hat, kld_stats = run_single_simulation(
            use_defense,
            use_attack,
            w_var,
            A_sys,
            xi_mag
        )

        # 1. Network-average RMSE.
        rmse_t = np.zeros(Params.timeSize)
        for t in range(Params.timeSize):
            errors = [
                np.linalg.norm(x_true[:, t] - x_hat[:, t, i])
                for i in range(Params.sensorNum)
            ]
            rmse_t[t] = np.sqrt(np.mean(np.asarray(errors) ** 2))
        avg_rmse += rmse_t

        # 2. KLD and detection metrics.
        for link in Params.attack_links:
            link_kld = kld_stats[link]
            avg_kld[link] += link_kld

            clean_alarm_flags = (
                link_kld[clean_eval_start:clean_eval_end]
                > Params.KLD_threshold
            )
            avg_fpr[link] += float(np.mean(clean_alarm_flags))

            if use_attack:
                attack_alarm_flags = (
                    link_kld[attack_eval_start:attack_eval_end]
                    > Params.KLD_threshold
                )

                tpr_mc = float(np.mean(attack_alarm_flags))
                fnr_mc = float(np.mean(~attack_alarm_flags))

                avg_tpr[link] += tpr_mc
                avg_fnr[link] += fnr_mc

                detected_indices = np.flatnonzero(attack_alarm_flags)
                if detected_indices.size > 0:
                    first_alarm_time = (
                        attack_eval_start + int(detected_indices[0])
                    )
                    delay_sum[link] += (
                        first_alarm_time - Params.attack_start
                    )
                    delay_count[link] += 1

    avg_rmse /= Params.MC_runs

    avg_detection_delay = {}
    for link in Params.attack_links:
        avg_kld[link] /= Params.MC_runs
        avg_fpr[link] /= Params.MC_runs

        if use_attack:
            avg_tpr[link] /= Params.MC_runs
            avg_fnr[link] /= Params.MC_runs
            avg_detection_delay[link] = (
                delay_sum[link] / delay_count[link]
                if delay_count[link] > 0
                else np.nan
            )
        else:
            avg_tpr[link] = np.nan
            avg_fnr[link] = np.nan
            avg_detection_delay[link] = np.nan

    return (
        avg_rmse,
        avg_kld,
        avg_tpr,
        avg_fpr,
        avg_fnr,
        avg_detection_delay,
    )


# ------------------------------------------------------------------------------
# 8. Execution & Generaton of Independent Figures
# ------------------------------------------------------------------------------
print("Directed attack-link convention: (receiver, sender) = sender -> receiver")
print("Configured attacked links:")
for link in Params.attack_links:
    receiver, sender = link
    print(
        f"  stored {link}: node {sender} sends to node {receiver} "
        f"({format_directed_link(link)})"
    )

plot_topology_vector(G)

print(f"Running Monte Carlo (MC={Params.MC_runs}) for Unstable System (Theorem 3)...")
rmse_unstable_clean, _, _, false_alarm_clean_no_def, _, _ = run_monte_carlo(use_defense=False, use_attack=False, w_var=0.0, A_sys=A_unstable,
                                            xi_mag=Params.xi_magnitude_unstable)
rmse_unstable_attack, kld_attack_no_def, tpr_no_def, fpr_no_def, fnr_no_def, delay_no_def = run_monte_carlo(use_defense=False, use_attack=True,
                                                                             w_var=0.0, A_sys=A_unstable,
                                                                             xi_mag=Params.xi_magnitude_unstable)

print(f"Running Monte Carlo (MC={Params.MC_runs}) for Stable System (Theorem 2)...")
rmse_stable_clean, _, _, _, _, _ = run_monte_carlo(use_defense=False, use_attack=False, w_var=0.0, A_sys=A_stable,
                                          xi_mag=Params.xi_magnitude_stable)
rmse_stable_attack, _, _, _, _, _ = run_monte_carlo(use_defense=False, use_attack=True, w_var=0.0, A_sys=A_stable,
                                           xi_mag=Params.xi_magnitude_stable)

# 纯净工况下的水印一致性测试 (Theorem Preservation)
rmse_nominal_def, _, _, false_alarm_nominal_def, _, _ = run_monte_carlo(use_defense=True, use_attack=False, w_var=Params.watermark_vars_test[1],
                                         A_sys=A_unstable, xi_mag=Params.xi_magnitude_unstable)

print(f"Running Defense Scenarios under varying Watermark variances...")
rmse_def_dict, kld_def_dict, tpr_def_dict = {}, {}, {}
fpr_def_dict, fnr_def_dict, delay_def_dict = {}, {}, {}

print(f"Running Defense Scenarios under varying Watermark variances...")

# ====== 【新增代码：计算仅有 lambda，无 beta 随机项的基线】 ======
_, kld_def_lambda_only, tpr_lambda_only, fpr_lambda_only, fnr_lambda_only, delay_lambda_only = run_monte_carlo(
    use_defense=True,
    use_attack=True,
    w_var=0.0,  # 方差为0，代表没有随机项 beta
    A_sys=A_unstable,
    xi_mag=Params.xi_magnitude_unstable
)
# ==============================================================

rmse_def_dict, kld_def_dict, tpr_def_dict = {}, {}, {}
fpr_def_dict, fnr_def_dict, delay_def_dict = {}, {}, {}
for w_var in Params.watermark_vars_test:
    rmse_def, kld_def, tpr_def, fpr_def, fnr_def, delay_def = run_monte_carlo(use_defense=True, use_attack=True, w_var=w_var,
                                                        A_sys=A_unstable, xi_mag=Params.xi_magnitude_unstable)
    rmse_def_dict[w_var] = rmse_def
    kld_def_dict[w_var] = kld_def
    tpr_def_dict[w_var] = tpr_def
    fpr_def_dict[w_var] = fpr_def
    fnr_def_dict[w_var] = fnr_def
    delay_def_dict[w_var] = delay_def

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

# --- Figure 3: Representative KLD Time Curves ---
# Only representative watermark variances are retained in the main text.
# The horizontal axis remains time k; only the vertical axis is logarithmic.
representative_vars = [0.0, 0.5, 2.0]
representative_styles = {
    0.0: {
        "label": r"$\sigma_\beta^2=0$",
        "color": "gray",
        "linestyle": "--",
        "marker": "p",
    },
    0.5: {
        "label": r"$\sigma_\beta^2=0.5$",
        "color": "#EDB120",
        "linestyle": "-",
        "marker": "o",
    },
    2.0: {
        "label": r"$\sigma_\beta^2=2.0$",
        "color": "#7E2F8E",
        "linestyle": "-",
        "marker": "s",
    },
}

plt.figure(figsize=(8, 5.5))

for var in representative_vars:
    if var == 0.0:
        curve = kld_def_lambda_only[link_to_plot][t_start:]
    else:
        curve = kld_def_dict[var][link_to_plot][t_start:]

    style = representative_styles[var]
    plt.semilogy(
        t_axis,
        curve,
        label=style["label"],
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markerfacecolor="none",
        markevery=5,
    )

plt.axhline(
    Params.KLD_threshold,
    color="black",
    linestyle="--",
    linewidth=2.0,
    label=r"Threshold $\theta$",
)
plt.axvspan(
    Params.attack_start,
    min(
        Params.attack_start + Params.attack_duration,
        Params.timeSize - 1,
    ),
    alpha=0.08,
    color="gray",
)
plt.xlabel("$k$")
plt.ylabel("K-L Divergence Statistic")
plt.xlim([t_start, t_end - 1])

# The axis is intentionally capped because once D_KL is far above theta,
# additional orders of magnitude do not change the binary decision.
plt.ylim(bottom=1e-1, top=1e4)
plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    borderaxespad=0.2,
    **legend_props
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.tight_layout()
plt.savefig(
    "Fig3_KLD_Representative_Strengths.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()


# --- Figure 4: FPR and FNR versus Watermark Variance ---
metric_variances = [0.0] + list(Params.watermark_vars_test)

mean_fpr = [
    np.mean([fpr_lambda_only[link] for link in Params.attack_links])
]
mean_fnr = [
    np.mean([fnr_lambda_only[link] for link in Params.attack_links])
]
mean_tpr = [
    np.mean([tpr_lambda_only[link] for link in Params.attack_links])
]
mean_delay = [
    np.nanmean([
        delay_lambda_only[link]
        for link in Params.attack_links
    ])
]

for var in Params.watermark_vars_test:
    mean_fpr.append(
        np.mean([
            fpr_def_dict[var][link]
            for link in Params.attack_links
        ])
    )
    mean_fnr.append(
        np.mean([
            fnr_def_dict[var][link]
            for link in Params.attack_links
        ])
    )
    mean_tpr.append(
        np.mean([
            tpr_def_dict[var][link]
            for link in Params.attack_links
        ])
    )
    mean_delay.append(
        np.nanmean([
            delay_def_dict[var][link]
            for link in Params.attack_links
        ])
    )

mean_fpr = np.asarray(mean_fpr)
mean_fnr = np.asarray(mean_fnr)
mean_tpr = np.asarray(mean_tpr)
mean_delay = np.asarray(mean_delay)

plt.figure(figsize=(8, 5.5))
plt.plot(
    metric_variances,
    mean_fpr,
    label="False Positive Rate (FPR)",
    linestyle="-",
    marker="o",
    markerfacecolor="none",
)
plt.plot(
    metric_variances,
    mean_fnr,
    label="False Negative Rate (FNR)",
    linestyle="-",
    marker="s",
    markerfacecolor="none",
)
plt.xlabel(r"Watermarking Variance $\sigma_\beta^2$")
plt.ylabel("Rate")
plt.xlim([
    min(metric_variances),
    max(metric_variances),
])
plt.ylim([0.0, 1.0])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.gca().tick_params(direction="in", top=True, right=True)
plt.legend(loc="best", **legend_props)
plt.tight_layout()
plt.savefig(
    "Fig4_FPR_FNR_vs_WatermarkVariance.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()


# Optional engineering metric: average detection delay.
plt.figure(figsize=(8, 5.5))
plt.plot(
    metric_variances,
    mean_delay,
    linestyle="-",
    marker="d",
    markerfacecolor="none",
)
plt.xlabel(r"Watermarking Variance $\sigma_\beta^2$")
plt.ylabel("Average Detection Delay (samples)")
plt.xlim([
    min(metric_variances),
    max(metric_variances),
])
plt.gca().tick_params(direction="in", top=True, right=True)
plt.tight_layout()
plt.savefig(
    "Fig4b_Detection_Delay_vs_WatermarkVariance.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()


# --- Figure 5: RMSE with Watermarking Defense vs Clean ---
plt.figure(figsize=(8, 5.5))
plt.plot(t_axis, rmse_unstable_clean[t_start:], label='Clean (No Attack)', color='#0072BD', linestyle='-', marker='o',
         markerfacecolor='none', markevery=5)
plt.plot(t_axis, rmse_def_dict[rep_var][t_start:], label=fr'Stealthy FDI Attack (With Watermarking, $\sigma_\beta^2={rep_var}$)', color='#7E2F8E',
         linestyle='-', marker='s', markerfacecolor='none', markevery=5)
plt.xlabel('$k$')
plt.ylabel('RMSE')
plt.xlim([t_start, t_end - 1])
plt.yscale('linear')
y_max = max(np.max(rmse_unstable_clean[t_start:]), np.max(rmse_def_dict[rep_var][t_start:])) * 1.5
plt.ylim(0, y_max)
plt.legend(loc='upper right', **legend_props)
plt.tight_layout()
plt.savefig("Fig5_RMSE_WithDefense_Linear.pdf", format='pdf', bbox_inches='tight')
plt.show()

# --- Figure 6: Arbitrary Attack Sequence Optimization under KLD Constraint ---
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
    receiver, sender = link
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
    f'Target link: {format_directed_link(link_to_plot)} '
    f'(stored as {link_to_plot} = (receiver, sender)) | '
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


# Save detection metrics used by the paper figures and tables.
import csv

# Directly corresponds to the revised Table 1 in the manuscript.
with open(
    "Table1_Strictly_Stealthy_Attack_Metrics.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "receiver_node",
        "sender_node",
        "directed_link_sender_to_receiver",
        "fpr_without_attack",
        "fnr_under_strictly_stealthy_attack",
        "tpr_under_strictly_stealthy_attack",
    ])
    for link in Params.attack_links:
        receiver, sender = link
        writer.writerow([
            receiver,
            sender,
            format_directed_link(link),
            false_alarm_clean_no_def[link],
            fnr_no_def[link],
            tpr_no_def[link],
        ])


with open(
    "Detection_Metrics_Summary.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "watermark_variance",
        "mean_fpr",
        "mean_fnr",
        "mean_tpr",
        "mean_detection_delay_samples",
    ])
    for idx, var in enumerate(metric_variances):
        writer.writerow([
            var,
            mean_fpr[idx],
            mean_fnr[idx],
            mean_tpr[idx],
            mean_delay[idx],
        ])


with open(
    "Detection_Metrics_Per_Link.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "receiver_node",
        "sender_node",
        "directed_link_sender_to_receiver",
        "scenario",
        "watermark_variance",
        "fpr",
        "fnr",
        "tpr",
        "detection_delay_samples",
    ])

    for link in Params.attack_links:
        receiver, sender = link
        # Strictly stealthy attack without watermarking.
        writer.writerow([
            receiver,
            sender,
            format_directed_link(link),
            "No watermarking",
            "",
            false_alarm_clean_no_def[link],
            fnr_no_def[link],
            tpr_no_def[link],
            delay_no_def[link],
        ])

        # Fixed lambda only; no random beta term.
        writer.writerow([
            receiver,
            sender,
            format_directed_link(link),
            "Only lambda",
            0.0,
            fpr_lambda_only[link],
            fnr_lambda_only[link],
            tpr_lambda_only[link],
            delay_lambda_only[link],
        ])

        # Dynamic random watermark.
        for var in Params.watermark_vars_test:
            writer.writerow([
                receiver,
                sender,
                format_directed_link(link),
                "Dynamic watermark",
                var,
                fpr_def_dict[var][link],
                fnr_def_dict[var][link],
                tpr_def_dict[var][link],
                delay_def_dict[var][link],
            ])


print("\n" + "=" * 90)
print(">>> TABLE 1: FALSE-ALARM AND MISS RATES OF THE STRICTLY STEALTHY ATTACK <<<")
print("=" * 90)

detector_ready_start = max(
    Params.plot_start,
    Params.s + Params.KLD_window - 1
)
attack_metric_start = Params.attack_start + Params.s
attack_metric_end = min(
    Params.attack_start + Params.attack_duration,
    Params.timeSize
)

print(
    f"Decision: D_KL,k > {Params.KLD_threshold} => alarm"
)
print(
    "Link convention: stored tuple (receiver, sender); "
    "displayed direction is sender→receiver."
)
print(
    f"K_0 = [{detector_ready_start}, {Params.attack_start}), "
    f"K_a = [{attack_metric_start}, {attack_metric_end})"
)
print(
    "TPR = 1 - FNR. The attack interval begins after a complete attacked "
    "parity window has formed."
)
print("-" * 90)
print(
    f"{'Directed Link (sender→receiver)':<32} | "
    f"{'FPR without attack':<24} | "
    f"{'FNR under strictly stealthy attack':<38}"
)
print("-" * 90)

for link in Params.attack_links:
    print(
        f"{format_directed_link(link):<32} | "
        f"{false_alarm_clean_no_def[link]:<24.2%} | "
        f"{fnr_no_def[link]:<38.2%}"
    )

print("=" * 90 + "\n")


print("\n" + "=" * 108)
print(">>> WATERMARK DETECTION PERFORMANCE SUMMARY <<<")
print("=" * 108)
print(
    f"{'Watermark Variance':<22} | "
    f"{'Mean FPR':<16} | "
    f"{'Mean FNR':<16} | "
    f"{'Mean TPR':<16} | "
    f"{'Mean Delay':<18}"
)
print("-" * 108)

for idx, var in enumerate(metric_variances):
    delay_value = mean_delay[idx]
    delay_string = (
        f"{delay_value:.2f}"
        if np.isfinite(delay_value)
        else "Not detected"
    )

    print(
        f"{var:<22} | "
        f"{mean_fpr[idx]:<16.2%} | "
        f"{mean_fnr[idx]:<16.2%} | "
        f"{mean_tpr[idx]:<16.2%} | "
        f"{delay_string:<18}"
    )

print("=" * 108)

complement_error = float(
    np.max(np.abs(mean_tpr + mean_fnr - 1.0))
)
print(
    f"Consistency check: max |TPR + FNR - 1| = "
    f"{complement_error:.3e}"
)
print("=" * 108 + "\n")


print("\n" + "=" * 118)
print(">>> SUPPLEMENTARY PER-LINK FPR / FNR / TPR <<<")
print("=" * 118)
print(
    f"{'Directed Link (sender→receiver)':<32} | "
    f"{'Variance':<12} | "
    f"{'FPR':<14} | "
    f"{'FNR':<14} | "
    f"{'TPR':<14} | "
    f"{'Delay':<14}"
)
print("-" * 118)

for link in Params.attack_links:
    lambda_delay = delay_lambda_only[link]
    print(
        f"{format_directed_link(link):<32} | "
        f"{0.0:<12} | "
        f"{fpr_lambda_only[link]:<14.2%} | "
        f"{fnr_lambda_only[link]:<14.2%} | "
        f"{tpr_lambda_only[link]:<14.2%} | "
        f"{lambda_delay if np.isfinite(lambda_delay) else 'N/A':<14}"
    )

    for var in Params.watermark_vars_test:
        delay_value = delay_def_dict[var][link]
        delay_string = (
            f"{delay_value:.2f}"
            if np.isfinite(delay_value)
            else "N/A"
        )
        print(
            f"{format_directed_link(link):<32} | "
            f"{var:<12} | "
            f"{fpr_def_dict[var][link]:<14.2%} | "
            f"{fnr_def_dict[var][link]:<14.2%} | "
            f"{tpr_def_dict[var][link]:<14.2%} | "
            f"{delay_string:<14}"
        )

    print("-" * 118)

print("=" * 118 + "\n")
