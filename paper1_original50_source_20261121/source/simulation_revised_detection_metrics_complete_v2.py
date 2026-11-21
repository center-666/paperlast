# ==============================================================================
# IEEE Paper Simulation: Distributed Estimator + K-L Divergence + ZDA + Watermark
# Revised version (aligned with the paper discussion):
#   1) finite attack interval for K-L attack/end/recovery visualization;
#   2) continuous attack for stable/unstable impact figure;
#   3) FPR is estimated from separate attack-free Monte Carlo trials;
#   4) FNR/TPR are estimated from fully attacked detector windows only;
#   5) every directed link is printed and exported as CSV and LaTeX tables;
#   6) representative K-L curves use sigma_beta^2 = 0, 0.5, and 2.0;
#   7) paper comparison uses sigma_beta^2 in {0, 0.5, 2, 5, 10, 50};
#   8) representative RMSE variance is explicitly defined (fixes rep_var error).
# ==============================================================================

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.linalg import block_diag, null_space, pinv
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# 1. Publication-quality plot settings
# ------------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# ------------------------------------------------------------------------------
# 2. Global parameters
# ------------------------------------------------------------------------------
class Params:
    sensorNum = 50

    # Leave enough samples after the attack to show K-L recovery.
    timeSize = 150

    # Use environment variable MC_RUNS for quick debugging, e.g. MC_RUNS=5.
    # Restore 300/500 for final paper results.
    MC_runs = int(os.environ.get("MC_RUNS", "1000"))

    dataDim = 2
    meaDim = 2

    # Parity order s => parity stack contains s+1 measurements.
    s = 4

    # Number of parity residuals used to estimate current residual statistics.
    KLD_window = 4
    KLD_threshold = 42.0

    # Stored convention: (receiver, sender) = (i, j), physically j -> i.
    attack_links = [
        (0, 1),
        
        (2, 3),
        (15, 16),
        (17, 31),
        (14, 49),
        (17, 34),
    ]

    attack_start = 50

    # Finite attack interval used by K-L/FPR/FNR/defense experiments.
    attack_duration = 60

    # Continuous-through-end attack used only by the impact figure, so that the
    # stable-system recovery is caused by A being Schur-stable, not attack stop.
    impact_attack_duration = timeSize - attack_start

    xi_magnitude_unstable = 6.0
    xi_magnitude_stable = 10.0

    watermark_lambda = 0.5

    # Watermark variances used in the manuscript comparison.
    # sigma_beta^2 = 0 is added separately as the deterministic-scaling baseline.
    watermark_vars_test = [0.50, 2.00, 5.00, 10.00, 50.00]

    # Representative K-L curves retained in the main paper.
    representative_kld_vars = [0.0, 0.50, 2.00]
    representative_rmse_var = 0.50

    plot_start = 20

    output_dir = Path(os.environ.get("SIM_OUTPUT_DIR", "simulation_outputs"))

    # The arbitrary attack optimization is computationally expensive.
    run_attack_optimization = True
    attack_opt_watermark_mc = MC_runs
    attack_opt_multi_starts = 8
    attack_opt_seed = 20260712
    attack_opt_ftol = 1e-10
    attack_opt_maxiter = 1500


Params.output_dir.mkdir(parents=True, exist_ok=True)


def output_path(filename: str) -> Path:
    return Params.output_dir / filename


def format_directed_link(link: tuple[int, int]) -> str:
    """Format stored (receiver, sender) as physical sender -> receiver."""
    receiver, sender = link
    return f"{sender}\u2192{receiver}"


def format_directed_link_latex(link: tuple[int, int]) -> str:
    """LaTeX form of stored (receiver, sender), i.e. sender \to receiver."""
    receiver, sender = link
    return rf"{sender}\to {receiver}"


def nearest_key(mapping: dict[float, object], target: float) -> float:
    """Return the numeric dictionary key nearest to target."""
    if not mapping:
        raise ValueError("Cannot select a representative key from an empty mapping.")
    return min(mapping.keys(), key=lambda value: abs(float(value) - target))


def safe_nanmean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else np.nan


# ------------------------------------------------------------------------------
# 3. Topology generation
# ------------------------------------------------------------------------------
seed_val = 42
while True:
    G = nx.random_geometric_graph(
        Params.sensorNum,
        radius=0.31,
        seed=seed_val,
    )
    if nx.is_connected(G):
        break
    seed_val += 1

for receiver, sender in Params.attack_links:
    if not G.has_edge(receiver, sender):
        G.add_edge(receiver, sender)

W = nx.to_numpy_array(G).astype(int)
neighbors_in = [
    list(np.where(W[receiver, :] == 1)[0])
    for receiver in range(Params.sensorNum)
]


def plot_topology_vector(graph: nx.Graph) -> None:
    plt.figure(figsize=(8, 6.5))
    pos = nx.get_node_attributes(graph, "pos")
    if not pos:
        pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#2A5CAA",
        alpha=0.6,
        width=0.6,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#2A5CAA",
        node_size=20,
        alpha=1.0,
    )

    attacked_nodes = sorted({node for edge in Params.attack_links for node in edge})
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=attacked_nodes,
        node_color="red",
        node_size=60,
    )

    attacked_directed_edges = [
        (sender, receiver)
        for receiver, sender in Params.attack_links
    ]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=attacked_directed_edges,
        edge_color="red",
        width=2.0,
        arrows=True,
        arrowsize=15,
    )

    labels = {i: str(i) for i in range(Params.sensorNum)}
    label_pos = {
        key: [value[0] - 0.02, value[1] + 0.03]
        for key, value in pos.items()
    }
    nx.draw_networkx_labels(
        graph,
        label_pos,
        labels,
        font_size=11,
        font_color="black",
        font_family="DejaVu Serif",
    )

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    ax = plt.gca()
    ax.tick_params(direction="in", top=True, right=True, length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(output_path("Fig1_Topology_Clean.pdf"), bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------------------
# 4. System matrices and watermark
# ------------------------------------------------------------------------------
A_unstable = np.array([[1.08, 0.05], [0.01, 1.05]], dtype=float)
A_stable = np.array([[0.95, 0.02], [0.02, 0.97]], dtype=float)
Q = np.eye(Params.dataDim)

np.random.seed(42)
H = np.zeros((Params.meaDim, Params.dataDim, Params.sensorNum))
R = np.zeros((Params.meaDim, Params.meaDim, Params.sensorNum))
for i in range(Params.sensorNum):
    H[0, 0, i] = 1.0 + 0.5 * np.random.rand()
    H[1, 1, i] = 1.0 + 0.5 * np.random.rand()
    R[:, :, i] = 0.5 * np.eye(Params.meaDim)


class DynamicWatermark:
    def __init__(
        self,
        meas_dim: int,
        num_nodes: int,
        horizon: int,
        lam: float,
        var: float,
    ) -> None:
        self.M: dict[tuple[int, int], np.ndarray] = {}
        self.M_inv: dict[tuple[int, int], np.ndarray] = {}

        for node in range(num_nodes):
            for k in range(horizon):
                beta = np.random.normal(0.0, np.sqrt(var), meas_dim)
                matrix = lam * np.eye(meas_dim) + np.outer(beta, beta)
                self.M[(node, k)] = matrix
                self.M_inv[(node, k)] = pinv(matrix)

    def encrypt(self, measurement: np.ndarray, node: int, k: int) -> np.ndarray:
        return self.M_inv[(node, k)] @ measurement

    def decrypt(self, encoded: np.ndarray, node: int, k: int) -> np.ndarray:
        return self.M[(node, k)] @ encoded


# ------------------------------------------------------------------------------
# 5. Parity space and attack synthesis
# ------------------------------------------------------------------------------
def build_parity_space(
    A_sys: np.ndarray,
) -> tuple[dict[int, np.ndarray | None], dict[int, np.ndarray | None]]:
    Zs_cache: dict[int, np.ndarray | None] = {}
    Theta_cache: dict[int, np.ndarray | None] = {}

    for sender in range(Params.sensorNum):
        H_sender = H[:, :, sender]
        H_xs = np.vstack([
            H_sender @ np.linalg.matrix_power(A_sys, power)
            for power in range(Params.s + 1)
        ])
        left_null_basis = null_space(H_xs.T)

        if left_null_basis.size == 0:
            Zs_cache[sender] = None
            Theta_cache[sender] = None
            continue

        Zs = left_null_basis.T
        Zs_cache[sender] = Zs

        # Keep the original s+1-noise-block representation for compatibility.
        Hw_rows = []
        for row in range(Params.s + 1):
            row_blocks = [
                H_sender @ np.linalg.matrix_power(A_sys, row - 1 - col)
                if col < row
                else np.zeros((Params.meaDim, Params.dataDim))
                for col in range(Params.s + 1)
            ]
            Hw_rows.append(np.hstack(row_blocks))
        Hw_s = np.vstack(Hw_rows)

        Hv_s = np.eye((Params.s + 1) * Params.meaDim)
        Q_s = block_diag(*[Q for _ in range(Params.s + 1)])
        R_s = block_diag(*[
            R[:, :, sender]
            for _ in range(Params.s + 1)
        ])

        Theta = Zs @ (
            Hw_s @ Q_s @ Hw_s.T
            + Hv_s @ R_s @ Hv_s.T
        ) @ Zs.T

        Theta_cache[sender] = (
            0.5 * (Theta + Theta.T)
            + 1e-8 * np.eye(Theta.shape[0])
        )

    return Zs_cache, Theta_cache


def generate_zda_sequence(
    A_sys: np.ndarray,
    H_sender: np.ndarray,
    duration: int,
    magnitude: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(A_sys)
    dominant_index = int(np.argmax(np.abs(eigenvalues)))
    dominant_vector = np.real(eigenvectors[:, dominant_index])
    dominant_vector /= max(np.linalg.norm(dominant_vector), 1e-12)

    virtual_state = magnitude * dominant_vector
    sequence = np.zeros((duration, Params.meaDim))

    for t in range(duration):
        sequence[t, :] = H_sender @ virtual_state
        virtual_state = A_sys @ virtual_state

    return sequence


def get_zda_biases(
    A_sys: np.ndarray,
    xi_mag: float,
    attack_duration: int,
) -> dict[int, np.ndarray]:
    """Generate one attack sequence for every transmitting node."""
    zda_bias: dict[int, np.ndarray] = {}
    for receiver, sender in Params.attack_links:
        del receiver
        if sender not in zda_bias:
            zda_bias[sender] = generate_zda_sequence(
                A_sys,
                H[:, :, sender],
                attack_duration,
                xi_mag,
            )
    return zda_bias


def compute_KL_divergence(
    residual_window: list[np.ndarray],
    Theta: np.ndarray,
) -> float:
    if len(residual_window) < 2:
        return 0.0

    residual_matrix = np.hstack(residual_window)
    mean_hat = np.mean(residual_matrix, axis=1, keepdims=True)
    covariance_hat = np.atleast_2d(np.cov(residual_matrix))
    covariance_hat = (
        0.5 * (covariance_hat + covariance_hat.T)
        + 1e-6 * np.eye(residual_matrix.shape[0])
    )

    theta_inv = pinv(Theta)
    dimension = residual_matrix.shape[0]

    trace_term = np.trace(theta_inv @ covariance_hat)
    mean_term = float((mean_hat.T @ theta_inv @ mean_hat)[0, 0])

    sign_theta, logdet_theta = np.linalg.slogdet(Theta)
    sign_cov, logdet_cov = np.linalg.slogdet(covariance_hat)
    if sign_theta <= 0 or sign_cov <= 0:
        return 0.0

    kld = 0.5 * (
        trace_term
        + mean_term
        - dimension
        + logdet_theta
        - logdet_cov
    )
    return max(0.0, float(kld))


# ------------------------------------------------------------------------------
# 6. Single simulation engine
# ------------------------------------------------------------------------------
def run_single_simulation(
    use_defense: bool,
    use_attack: bool,
    w_var: float,
    A_sys: np.ndarray,
    xi_mag: float,
    attack_duration: int,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]]:
    watermark = DynamicWatermark(
        Params.meaDim,
        Params.sensorNum,
        Params.timeSize,
        Params.watermark_lambda,
        w_var,
    )
    Zs_cache, Theta_cache = build_parity_space(A_sys)
    zda_bias = get_zda_biases(A_sys, xi_mag, attack_duration)

    attack_end = min(
        Params.attack_start + attack_duration,
        Params.timeSize,
    )

    x_true = np.zeros((Params.dataDim, Params.timeSize))
    y_nominal = np.zeros((Params.sensorNum, Params.meaDim, Params.timeSize))
    x_true[:, 0] = np.random.normal(0.0, 1.0, Params.dataDim)

    y_nominal[:, :, 0] = np.stack([
        H[:, :, node] @ x_true[:, 0]
        + np.random.multivariate_normal(
            np.zeros(Params.meaDim),
            R[:, :, node],
        )
        for node in range(Params.sensorNum)
    ])

    for k in range(1, Params.timeSize):
        x_true[:, k] = (
            A_sys @ x_true[:, k - 1]
            + np.random.multivariate_normal(
                np.zeros(Params.dataDim),
                Q,
            )
        )
        for node in range(Params.sensorNum):
            y_nominal[node, :, k] = (
                H[:, :, node] @ x_true[:, k]
                + np.random.multivariate_normal(
                    np.zeros(Params.meaDim),
                    R[:, :, node],
                )
            )

    x_hat = np.zeros((Params.dataDim, Params.timeSize, Params.sensorNum))
    P_cov = np.zeros((
        Params.dataDim,
        Params.dataDim,
        Params.sensorNum,
        Params.timeSize,
    ))
    for node in range(Params.sensorNum):
        P_cov[:, :, node, 0] = 5.0 * np.eye(Params.dataDim)

    kld_stats = {
        link: np.zeros(Params.timeSize)
        for link in Params.attack_links
    }
    residual_buffers: dict[tuple[int, int], list[np.ndarray]] = {
        link: []
        for link in Params.attack_links
    }

    y_received = np.zeros((
        Params.sensorNum,
        Params.sensorNum,
        Params.meaDim,
        Params.timeSize,
    ))

    isolated_links = {
        link: np.zeros(Params.timeSize, dtype=bool)
        for link in Params.attack_links
    }

    for k in range(1, Params.timeSize):
        # Communication, optional attack, and optional watermark decoding.
        for receiver in range(Params.sensorNum):
            for sender in neighbors_in[receiver]:
                transmitted = (
                    watermark.encrypt(y_nominal[sender, :, k], sender, k)
                    if use_defense
                    else y_nominal[sender, :, k].copy()
                )

                link = (receiver, sender)
                attack_active = (
                    use_attack
                    and link in Params.attack_links
                    and Params.attack_start <= k < attack_end
                )
                if attack_active:
                    transmitted = transmitted + zda_bias[sender][
                        k - Params.attack_start
                    ]

                decoded = (
                    watermark.decrypt(transmitted, sender, k)
                    if use_defense
                    else transmitted
                )
                y_received[receiver, sender, :, k] = decoded

        # Link-wise parity detector.
        if k >= Params.s:
            for receiver, sender in Params.attack_links:
                link = (receiver, sender)
                Zs = Zs_cache[sender]
                Theta = Theta_cache[sender]
                if Zs is None or Theta is None:
                    continue

                stacked_measurements = np.vstack([
                    y_received[receiver, sender, :, time_index].reshape(-1, 1)
                    for time_index in range(k - Params.s, k + 1)
                ])
                residual = Zs @ stacked_measurements

                residual_buffers[link].append(residual)
                if len(residual_buffers[link]) > Params.KLD_window:
                    residual_buffers[link].pop(0)

                kld_stats[link][k] = compute_KL_divergence(
                    residual_buffers[link],
                    Theta,
                )
                isolated_links[link][k] = (
                    kld_stats[link][k] > Params.KLD_threshold
                )

        # Distributed measurement-sharing estimator.
        for receiver in range(Params.sensorNum):
            x_pred = A_sys @ x_hat[:, k - 1, receiver]
            P_pred = (
                A_sys @ P_cov[:, :, receiver, k - 1] @ A_sys.T
                + Q
            )

            active_senders: list[int] = []
            for sender in neighbors_in[receiver]:
                link = (receiver, sender)
                if (
                    use_defense
                    and link in Params.attack_links
                    and isolated_links[link][k]
                ):
                    continue
                active_senders.append(sender)
            active_senders.append(receiver)

            y_ext: list[np.ndarray] = []
            H_ext: list[np.ndarray] = []
            R_ext: list[np.ndarray] = []

            for sender in active_senders:
                measurement = (
                    y_nominal[receiver, :, k]
                    if sender == receiver
                    else y_received[receiver, sender, :, k]
                )
                y_ext.append(measurement)
                H_ext.append(H[:, :, sender])
                R_ext.append(R[:, :, sender])

            H_stack = np.vstack(H_ext)
            R_block = block_diag(*R_ext)
            innovation_covariance = H_stack @ P_pred @ H_stack.T + R_block
            gain = P_pred @ H_stack.T @ pinv(innovation_covariance)

            innovation = np.concatenate(y_ext) - H_stack @ x_pred
            x_hat[:, k, receiver] = x_pred + gain @ innovation

            P_updated = (
                np.eye(Params.dataDim) - gain @ H_stack
            ) @ P_pred
            P_cov[:, :, receiver, k] = (
                0.5 * (P_updated + P_updated.T)
                + 1e-9 * np.eye(Params.dataDim)
            )

    return x_true, x_hat, kld_stats


# ------------------------------------------------------------------------------
# 7. Monte Carlo wrapper and detection metrics
# ------------------------------------------------------------------------------
def metric_intervals(
    use_attack: bool,
    attack_duration: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Return clean indices, attacked indices, attack end, first fully attacked
    K-L time, and first fully recovered K-L time.

    A K-L statistic uses:
      - a parity window of s+1 measurements, and
      - a buffer of KLD_window parity residuals.

    Therefore the first fully attacked K-L statistic occurs after
    s + KLD_window - 1 samples, and the same guard is used after attack end.
    """
    detector_ready_start = max(
        Params.plot_start,
        Params.s + Params.KLD_window - 1,
    )

    attack_end = min(
        Params.attack_start + attack_duration,
        Params.timeSize,
    )
    detector_guard = Params.s + Params.KLD_window - 1
    first_fully_attacked = Params.attack_start + detector_guard
    first_fully_recovered = attack_end + detector_guard

    if not use_attack:
        clean_indices = np.arange(
            detector_ready_start,
            Params.timeSize,
            dtype=int,
        )
        attacked_indices = np.asarray([], dtype=int)
        return (
            clean_indices,
            attacked_indices,
            attack_end,
            first_fully_attacked,
            first_fully_recovered,
        )

    pre_attack_clean = np.arange(
        detector_ready_start,
        min(Params.attack_start, Params.timeSize),
        dtype=int,
    )
    post_attack_clean = np.arange(
        min(first_fully_recovered, Params.timeSize),
        Params.timeSize,
        dtype=int,
    )
    clean_indices = np.concatenate([
        pre_attack_clean,
        post_attack_clean,
    ])

    attacked_indices = np.arange(
        min(first_fully_attacked, Params.timeSize),
        min(attack_end, Params.timeSize),
        dtype=int,
    )

    return (
        clean_indices,
        attacked_indices,
        attack_end,
        first_fully_attacked,
        first_fully_recovered,
    )


def run_monte_carlo(
    use_defense: bool,
    use_attack: bool,
    w_var: float = 0.0,
    A_sys: np.ndarray = A_unstable,
    xi_mag: float = 3.0,
    attack_duration: int = Params.attack_duration,
) -> tuple[
    np.ndarray,
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """
    Metrics:
      FPR = P(alarm | no attack)
      TPR = P(alarm | attack)
      FNR = P(no alarm | attack) = 1 - TPR

    FPR is evaluated on valid pre-attack and post-recovery samples.
    FNR/TPR are evaluated only on K-L windows fully occupied by attacks.
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

    (
        clean_indices,
        attacked_indices,
        _attack_end,
        first_fully_attacked,
        _first_fully_recovered,
    ) = metric_intervals(use_attack, attack_duration)

    if clean_indices.size == 0:
        raise ValueError(
            "No valid attack-free samples are available for FPR evaluation."
        )
    if use_attack and attacked_indices.size == 0:
        raise ValueError(
            "No fully attacked K-L samples are available for FNR/TPR."
        )

    for _mc in range(Params.MC_runs):
        x_true, x_hat, kld_stats = run_single_simulation(
            use_defense,
            use_attack,
            w_var,
            A_sys,
            xi_mag,
            attack_duration,
        )

        rmse_time = np.zeros(Params.timeSize)
        for k in range(Params.timeSize):
            errors = np.asarray([
                np.linalg.norm(x_true[:, k] - x_hat[:, k, node])
                for node in range(Params.sensorNum)
            ])
            rmse_time[k] = np.sqrt(np.mean(errors ** 2))
        avg_rmse += rmse_time

        for link in Params.attack_links:
            link_kld = kld_stats[link]
            avg_kld[link] += link_kld

            clean_alarms = link_kld[clean_indices] > Params.KLD_threshold
            avg_fpr[link] += float(np.mean(clean_alarms))

            if use_attack:
                attack_alarms = (
                    link_kld[attacked_indices] > Params.KLD_threshold
                )
                avg_tpr[link] += float(np.mean(attack_alarms))
                avg_fnr[link] += float(np.mean(~attack_alarms))

                detected_positions = np.flatnonzero(attack_alarms)
                if detected_positions.size > 0:
                    first_alarm_time = int(
                        attacked_indices[int(detected_positions[0])]
                    )
                    delay_sum[link] += (
                        first_alarm_time - Params.attack_start
                    )
                    delay_count[link] += 1

    avg_rmse /= Params.MC_runs

    avg_delay: dict[tuple[int, int], float] = {}
    for link in Params.attack_links:
        avg_kld[link] /= Params.MC_runs
        avg_fpr[link] /= Params.MC_runs

        if use_attack:
            avg_tpr[link] /= Params.MC_runs
            avg_fnr[link] /= Params.MC_runs
            avg_delay[link] = (
                delay_sum[link] / delay_count[link]
                if delay_count[link] > 0
                else np.nan
            )
        else:
            avg_tpr[link] = np.nan
            avg_fnr[link] = np.nan
            avg_delay[link] = np.nan

    return (
        avg_rmse,
        avg_kld,
        avg_tpr,
        avg_fpr,
        avg_fnr,
        avg_delay,
    )


# ------------------------------------------------------------------------------
# 8. Table and CSV helpers
# ------------------------------------------------------------------------------
def print_no_watermark_table(
    clean_fpr: dict[tuple[int, int], float],
    attack_fnr: dict[tuple[int, int], float],
    attack_tpr: dict[tuple[int, int], float],
) -> None:
    width = 104
    print("\n" + "=" * width)
    print(">>> PER-LINK DETECTION METRICS WITHOUT RANDOM WATERMARKING <<<")
    print("=" * width)
    print(
        f"{'Directed link':<22} | "
        f"{'FPR (clean)':<18} | "
        f"{'FNR (attack)':<18} | "
        f"{'TPR (attack)':<18}"
    )
    print("-" * width)

    for link in Params.attack_links:
        print(
            f"{format_directed_link(link):<22} | "
            f"{clean_fpr[link]:<18.2%} | "
            f"{attack_fnr[link]:<18.2%} | "
            f"{attack_tpr[link]:<18.2%}"
        )

    mean_fpr = np.mean([clean_fpr[link] for link in Params.attack_links])
    mean_fnr = np.mean([attack_fnr[link] for link in Params.attack_links])
    mean_tpr = np.mean([attack_tpr[link] for link in Params.attack_links])
    print("-" * width)
    print(
        f"{'Average':<22} | "
        f"{mean_fpr:<18.2%} | "
        f"{mean_fnr:<18.2%} | "
        f"{mean_tpr:<18.2%}"
    )
    print("=" * width)


def print_watermark_summary_table(
    variances: list[float],
    fpr_by_var: dict[float, dict[tuple[int, int], float]],
    fnr_by_var: dict[float, dict[tuple[int, int], float]],
    tpr_by_var: dict[float, dict[tuple[int, int], float]],
    delay_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    width = 110
    print("\n" + "=" * width)
    print(">>> MEAN WATERMARK DETECTION PERFORMANCE <<<")
    print("=" * width)
    print(
        f"{'Variance':<14} | "
        f"{'Mean FPR':<16} | "
        f"{'Mean FNR':<16} | "
        f"{'Mean TPR':<16} | "
        f"{'Mean delay':<18}"
    )
    print("-" * width)

    for variance in variances:
        mean_fpr = np.mean([
            fpr_by_var[variance][link]
            for link in Params.attack_links
        ])
        mean_fnr = np.mean([
            fnr_by_var[variance][link]
            for link in Params.attack_links
        ])
        mean_tpr = np.mean([
            tpr_by_var[variance][link]
            for link in Params.attack_links
        ])
        mean_delay = safe_nanmean([
            delay_by_var[variance][link]
            for link in Params.attack_links
        ])
        delay_text = (
            f"{mean_delay:.2f}"
            if np.isfinite(mean_delay)
            else "Not detected"
        )

        print(
            f"{variance:<14g} | "
            f"{mean_fpr:<16.2%} | "
            f"{mean_fnr:<16.2%} | "
            f"{mean_tpr:<16.2%} | "
            f"{delay_text:<18}"
        )

    print("=" * width)


def print_per_link_watermark_tables(
    variances: list[float],
    fpr_by_var: dict[float, dict[tuple[int, int], float]],
    fnr_by_var: dict[float, dict[tuple[int, int], float]],
    tpr_by_var: dict[float, dict[tuple[int, int], float]],
    delay_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    width = 116
    print("\n" + "=" * width)
    print(">>> COMPLETE PER-LINK FPR / FNR / TPR TABLES <<<")
    print("=" * width)

    for variance in variances:
        print(f"\nWatermarking variance sigma_beta^2 = {variance:g}")
        print("-" * width)
        print(
            f"{'Directed link':<22} | "
            f"{'FPR':<14} | "
            f"{'FNR':<14} | "
            f"{'TPR':<14} | "
            f"{'Delay':<16}"
        )
        print("-" * width)

        for link in Params.attack_links:
            delay = delay_by_var[variance][link]
            delay_text = f"{delay:.2f}" if np.isfinite(delay) else "N/A"
            print(
                f"{format_directed_link(link):<22} | "
                f"{fpr_by_var[variance][link]:<14.2%} | "
                f"{fnr_by_var[variance][link]:<14.2%} | "
                f"{tpr_by_var[variance][link]:<14.2%} | "
                f"{delay_text:<16}"
            )

        print("-" * width)
        print(
            f"{'Average':<22} | "
            f"{np.mean([fpr_by_var[variance][link] for link in Params.attack_links]):<14.2%} | "
            f"{np.mean([fnr_by_var[variance][link] for link in Params.attack_links]):<14.2%} | "
            f"{np.mean([tpr_by_var[variance][link] for link in Params.attack_links]):<14.2%} | "
            f"{safe_nanmean([delay_by_var[variance][link] for link in Params.attack_links]):<16.2f}"
        )

    print("=" * width + "\n")


def save_detection_metric_csvs(
    clean_no_watermark_fpr: dict[tuple[int, int], float],
    no_watermark_fnr: dict[tuple[int, int], float],
    no_watermark_tpr: dict[tuple[int, int], float],
    variances: list[float],
    fpr_by_var: dict[float, dict[tuple[int, int], float]],
    fnr_by_var: dict[float, dict[tuple[int, int], float]],
    tpr_by_var: dict[float, dict[tuple[int, int], float]],
    delay_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    with output_path("Table1_No_Watermark_Per_Link_FPR_FNR.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "receiver_node",
            "sender_node",
            "directed_link_sender_to_receiver",
            "fpr_clean",
            "fnr_strictly_stealthy_attack",
            "tpr_strictly_stealthy_attack",
        ])
        for link in Params.attack_links:
            receiver, sender = link
            writer.writerow([
                receiver,
                sender,
                format_directed_link(link),
                clean_no_watermark_fpr[link],
                no_watermark_fnr[link],
                no_watermark_tpr[link],
            ])

    with output_path("Table2_Watermark_Per_Link_FPR_FNR.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "watermark_variance",
            "receiver_node",
            "sender_node",
            "directed_link_sender_to_receiver",
            "fpr",
            "fnr",
            "tpr",
            "mean_detection_delay_samples",
        ])
        for variance in variances:
            for link in Params.attack_links:
                receiver, sender = link
                writer.writerow([
                    variance,
                    receiver,
                    sender,
                    format_directed_link(link),
                    fpr_by_var[variance][link],
                    fnr_by_var[variance][link],
                    tpr_by_var[variance][link],
                    delay_by_var[variance][link],
                ])

    with output_path("Table3_Watermark_Mean_FPR_FNR.csv").open(
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
        for variance in variances:
            writer.writerow([
                variance,
                np.mean([
                    fpr_by_var[variance][link]
                    for link in Params.attack_links
                ]),
                np.mean([
                    fnr_by_var[variance][link]
                    for link in Params.attack_links
                ]),
                np.mean([
                    tpr_by_var[variance][link]
                    for link in Params.attack_links
                ]),
                safe_nanmean([
                    delay_by_var[variance][link]
                    for link in Params.attack_links
                ]),
            ])


def print_latex_table1_rows(
    clean_fpr: dict[tuple[int, int], float],
    attack_fnr: dict[tuple[int, int], float],
) -> None:
    print("\nLaTeX rows for Table 1 (without watermarking):")
    for link in Params.attack_links:
        print(
            rf"${format_directed_link_latex(link)}$ & "
            rf"{100.0 * clean_fpr[link]:.2f}\% & "
            rf"{100.0 * attack_fnr[link]:.2f}\% \\"
        )


def print_latex_watermark_rows(
    variances: list[float],
    fpr_by_var: dict[float, dict[tuple[int, int], float]],
    fnr_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    print("\nLaTeX rows for the complete per-link watermark table:")
    for variance in variances:
        for row_index, link in enumerate(Params.attack_links):
            variance_text = rf"{variance:g}" if row_index == 0 else ""
            print(
                rf"{variance_text} & "
                rf"${format_directed_link_latex(link)}$ & "
                rf"{100.0 * fpr_by_var[variance][link]:.2f}\% & "
                rf"{100.0 * fnr_by_var[variance][link]:.2f}\% \\"
            )
        print(r"\hline")


def save_latex_detection_tables(
    clean_no_watermark_fpr: dict[tuple[int, int], float],
    no_watermark_fnr: dict[tuple[int, int], float],
    variances: list[float],
    fpr_by_var: dict[float, dict[tuple[int, int], float]],
    fnr_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    """Save directly reusable LaTeX table environments."""
    table1_lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\caption{FPR and FNR of the K-L Divergence Detector Without Watermarking}",
        r"\label{tab:fpr_fnr_no_watermark}",
        r"\begin{tabular}{c|cc}",
        r"\hline",
        r"Targeted Link & FPR & FNR \\",
        r"\hline",
    ]
    for link in Params.attack_links:
        table1_lines.append(
            rf"${format_directed_link_latex(link)}$ & "
            rf"{100.0 * clean_no_watermark_fpr[link]:.2f}\% & "
            rf"{100.0 * no_watermark_fnr[link]:.2f}\% \\"
        )
    table1_lines.extend([
        r"\hline",
        rf"Average & {100.0 * np.mean(list(clean_no_watermark_fpr.values())):.2f}\% & "
        rf"{100.0 * np.mean(list(no_watermark_fnr.values())):.2f}\% \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    output_path("Table1_NoWatermark_FPR_FNR.tex").write_text(
        "\n".join(table1_lines),
        encoding="utf-8",
    )

    table2_lines = [
        r"\begin{table*}[htb]",
        r"\centering",
        r"\caption{Per-Link FPR and FNR under Different Watermarking Intensities}",
        r"\label{tab:fpr_fnr_watermark_per_link}",
        r"\begin{tabular}{c|c|cc}",
        r"\hline",
        r"$\sigma_\beta^2$ & Targeted Link & FPR & FNR \\",
        r"\hline",
    ]
    for variance in variances:
        for row_index, link in enumerate(Params.attack_links):
            variance_text = rf"{variance:g}" if row_index == 0 else ""
            table2_lines.append(
                rf"{variance_text} & ${format_directed_link_latex(link)}$ & "
                rf"{100.0 * fpr_by_var[variance][link]:.2f}\% & "
                rf"{100.0 * fnr_by_var[variance][link]:.2f}\% \\"
            )
        table2_lines.append(r"\hline")
    table2_lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    output_path("Table2_Watermark_PerLink_FPR_FNR.tex").write_text(
        "\n".join(table2_lines),
        encoding="utf-8",
    )



# ------------------------------------------------------------------------------
# 9. Plot helpers
# ------------------------------------------------------------------------------
legend_props = {
    "frameon": True,
    "edgecolor": "black",
    "facecolor": "white",
    "framealpha": 1.0,
    "fancybox": False,
    "shadow": False,
}


def plot_impact_rmse(
    rmse_unstable_clean: np.ndarray,
    rmse_unstable_attack: np.ndarray,
    rmse_stable_clean: np.ndarray,
    rmse_stable_attack: np.ndarray,
) -> None:
    t_start = Params.plot_start
    t_axis = np.arange(t_start, Params.timeSize)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(
        t_axis,
        rmse_unstable_clean[t_start:],
        label="Clean (No Attack)",
        color="#0072BD",
        marker="o",
        markerfacecolor="none",
        markevery=6,
    )
    axes[0].plot(
        t_axis,
        rmse_unstable_attack[t_start:],
        label="Strictly Stealthy FDI Attack",
        color="#D95319",
        marker="*",
        markevery=6,
    )
    axes[0].axvline(Params.attack_start, color="black", linestyle=":")
    axes[0].set_xlabel("$k$")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xlim(t_start, Params.timeSize - 1)
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Unstable System")
    axes[0].legend(loc="upper left", **legend_props)

    axes[1].plot(
        t_axis,
        rmse_stable_clean[t_start:],
        label="Clean (No Attack)",
        color="#0072BD",
        marker="o",
        markerfacecolor="none",
        markevery=6,
    )
    axes[1].plot(
        t_axis,
        rmse_stable_attack[t_start:],
        label="Strictly Stealthy FDI Attack",
        color="#D95319",
        marker="*",
        markevery=6,
    )
    axes[1].axvline(Params.attack_start, color="black", linestyle=":")
    axes[1].set_xlabel("$k$")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlim(t_start, Params.timeSize - 1)
    axes[1].set_title("(b) Stable System")
    axes[1].legend(loc="upper right", **legend_props)

    plt.tight_layout()
    plt.savefig(output_path("Fig2_RMSE_Impact_Stable_Unstable.pdf"), bbox_inches="tight")
    plt.close()


def _representative_kld_curves(
    kld_lambda_only: dict[tuple[int, int], np.ndarray],
    kld_by_var: dict[float, dict[tuple[int, int], np.ndarray]],
    link_to_plot: tuple[int, int],
) -> list[tuple[float, np.ndarray]]:
    """Collect the three manuscript K-L curves: 0, 0.5, and 2.0."""
    curves: list[tuple[float, np.ndarray]] = []
    for requested_variance in Params.representative_kld_vars:
        if requested_variance == 0.0:
            curves.append((0.0, kld_lambda_only[link_to_plot]))
        else:
            actual_variance = nearest_key(kld_by_var, requested_variance)
            curves.append((actual_variance, kld_by_var[actual_variance][link_to_plot]))
    return curves


def plot_representative_kld(
    kld_lambda_only: dict[tuple[int, int], np.ndarray],
    kld_by_var: dict[float, dict[tuple[int, int], np.ndarray]],
    link_to_plot: tuple[int, int],
) -> None:
    """
    Save two complementary K-L figures.

    Fig3a keeps the full logarithmic magnitude. Fig3b focuses on the threshold
    neighborhood so the post-attack return below theta is visually clear even
    when the attack-period K-L value is extremely large.
    """
    t_start = Params.plot_start
    time_axis = np.arange(t_start, Params.timeSize)

    attack_end = min(
        Params.attack_start + Params.attack_duration,
        Params.timeSize,
    )
    detector_guard = Params.s + Params.KLD_window - 1
    first_fully_recovered = min(
        attack_end + detector_guard,
        Params.timeSize - 1,
    )

    curves = _representative_kld_curves(
        kld_lambda_only,
        kld_by_var,
        link_to_plot,
    )

    style_map = {
        0.0: ("--", "p"),
        0.5: ("-", "o"),
        2.0: ("-", "s"),
    }

    # ------------------------------------------------------------------
    # Full logarithmic-scale figure: no artificial upper clipping.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for variance, full_curve in curves:
        linestyle, marker = style_map.get(float(variance), ("-", "o"))
        ax.semilogy(
            time_axis,
            np.maximum(full_curve[t_start:], 1e-8),
            label=rf"$\sigma_\beta^2={variance:g}$",
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="none",
            markevery=7,
        )

    ax.axhline(
        Params.KLD_threshold,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=r"Threshold $\theta$",
    )
    ax.axvspan(
        Params.attack_start,
        attack_end,
        alpha=0.10,
        color="gray",
        label="Attack interval",
    )
    # ax.axvline(attack_end, color="black", linestyle=":", linewidth=1.2)
    # ax.axvline(
    #     first_fully_recovered,
    #     color="black",
    #     linestyle="-.",
    #     linewidth=1.2,
    #     label="First fully attack-free detector window",
    # )
    ax.set_xlabel("$k$")
    ax.set_ylabel("K-L Divergence Statistic")
    ax.set_xlim(t_start, Params.timeSize - 1)
    ax.set_ylim(bottom=1e-1)
    ax.legend(loc="upper left", ncol=2, **legend_props)
    fig.tight_layout()
    fig.savefig(
        output_path("Fig3a_KLD_Full_LogScale_Attack_End_Recovery.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Threshold-focused figure: values above the display ceiling are capped
    # only for visualization. The original values remain unchanged in data.
    # ------------------------------------------------------------------
    display_ceiling = 8.0 * Params.KLD_threshold
    display_floor = max(1e-1, 0.25 * Params.KLD_threshold)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for variance, full_curve in curves:
        linestyle, marker = style_map.get(float(variance), ("-", "o"))
        original = np.maximum(full_curve[t_start:], 1e-8)
        displayed = np.minimum(original, display_ceiling)
        ax.semilogy(
            time_axis,
            displayed,
            label=rf"$\sigma_\beta^2={variance:g}$",
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="none",
            markevery=7,
        )

    ax.axhline(
        Params.KLD_threshold,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=r"Threshold $\theta$",
    )
    ax.axvspan(
        Params.attack_start,
        attack_end,
        alpha=0.10,
        color="gray",
        label="Attack interval",
    )
    # ax.axvline(attack_end, color="black", linestyle=":", linewidth=1.2)
    # ax.axvline(
    #     first_fully_recovered,
    #     color="black",
    #     linestyle="-.",
    #     linewidth=1.2,
    #     label="First fully attack-free detector window",
    # )
    ax.set_xlabel("$k$")
    ax.set_ylabel("K-L Divergence Statistic")
    ax.set_xlim(t_start, Params.timeSize - 1)
    ax.set_ylim(display_floor, display_ceiling)
    # ax.text(
    #     0.99,
    #     0.98,
    #     rf"Values above ${display_ceiling:g}$ are capped for display only",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=10,
    # )
    ax.legend(loc="upper left", ncol=2, **legend_props)
    
    fig.tight_layout()
    fig.savefig(
        output_path("Fig3b_KLD_Threshold_Focused_Recovery.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_fpr_fnr(
    variances: list[float],
    fpr_by_var: dict[float, dict[tuple[int, int], float]],
    fnr_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    """
    Plot mean FPR and FNR in separate panels. The shaded band gives the
    minimum-to-maximum range across attacked directed links.

    Categorical x positions are used so the manuscript points
    {0, 0.5, 2, 5, 10, 50} remain visually separated.
    """
    x_positions = np.arange(len(variances), dtype=float)
    x_labels = [f"{value:g}" for value in variances]

    fpr_matrix = np.asarray([
        [fpr_by_var[var][link] for link in Params.attack_links]
        for var in variances
    ])
    fnr_matrix = np.asarray([
        [fnr_by_var[var][link] for link in Params.attack_links]
        for var in variances
    ])

    mean_fpr = np.mean(fpr_matrix, axis=1)
    min_fpr = np.min(fpr_matrix, axis=1)
    max_fpr = np.max(fpr_matrix, axis=1)

    mean_fnr = np.mean(fnr_matrix, axis=1)
    min_fnr = np.min(fnr_matrix, axis=1)
    max_fnr = np.max(fnr_matrix, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    axes[0].plot(
        x_positions,
        mean_fpr,
        marker="o",
        markerfacecolor="none",
        label="Mean FPR",
    )
    axes[0].fill_between(
        x_positions,
        min_fpr,
        max_fpr,
        alpha=0.15,
        label="Per-link range",
    )
    axes[0].set_xlabel(r"Watermarking Variance $\sigma_\beta^2$")
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_xticks(x_positions, x_labels)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    fpr_upper = max(0.05, 1.25 * float(np.max(max_fpr)))
    axes[0].set_ylim(0.0, min(1.0, fpr_upper))
    axes[0].set_title("(a) False Positive Rate")
    axes[0].legend(loc="best", **legend_props)

    axes[1].plot(
        x_positions,
        mean_fnr,
        marker="s",
        markerfacecolor="none",
        label="Mean FNR",
    )
    axes[1].fill_between(
        x_positions,
        min_fnr,
        max_fnr,
        alpha=0.15,
        label="Per-link range",
    )
    axes[1].set_xlabel(r"Watermarking Variance $\sigma_\beta^2$")
    axes[1].set_ylabel("False Negative Rate")
    axes[1].set_xticks(x_positions, x_labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("(b) False Negative Rate")
    axes[1].legend(loc="best", **legend_props)

    fig.tight_layout()
    fig.savefig(
        output_path("Fig4_FPR_FNR_vs_WatermarkVariance.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_detection_delay(
    variances: list[float],
    delay_by_var: dict[float, dict[tuple[int, int], float]],
) -> None:
    """Optional diagnostic figure; undetected cases remain NaN."""
    mean_delay = np.asarray([
        safe_nanmean([
            delay_by_var[var][link]
            for link in Params.attack_links
        ])
        for var in variances
    ])

    if not np.any(np.isfinite(mean_delay)):
        return

    x_positions = np.arange(len(variances), dtype=float)
    x_labels = [f"{value:g}" for value in variances]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        x_positions,
        mean_delay,
        marker="d",
        markerfacecolor="none",
    )
    ax.set_xlabel(r"Watermarking Variance $\sigma_\beta^2$")
    ax.set_ylabel("Average Detection Delay (samples)")
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlim(-0.25, len(variances) - 0.75)
    fig.tight_layout()
    fig.savefig(
        output_path("Fig4b_Detection_Delay_vs_WatermarkVariance.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_rmse_with_defense(
    rmse_clean: np.ndarray,
    rmse_by_var: dict[float, np.ndarray],
) -> None:
    rep_var = nearest_key(
        rmse_by_var,
        Params.representative_rmse_var,
    )

    t_start = Params.plot_start
    time_axis = np.arange(t_start, Params.timeSize)
    attack_end = min(
        Params.attack_start + Params.attack_duration,
        Params.timeSize,
    )

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        time_axis,
        rmse_clean[t_start:],
        label="Clean (No Attack)",
        color="#0072BD",
        marker="o",
        markerfacecolor="none",
        markevery=7,
    )
    plt.plot(
        time_axis,
        rmse_by_var[rep_var][t_start:],
        label=(
            rf"Stealthy FDI Attack "
            rf"(With Watermarking, $\sigma_\beta^2={rep_var:g}$)"
        ),
        color="#7E2F8E",
        marker="s",
        markerfacecolor="none",
        markevery=7,
    )
    plt.axvspan(
        Params.attack_start,
        attack_end,
        alpha=0.10,
        color="gray",
        label="Attack interval",
    )
    plt.xlabel("$k$")
    plt.ylabel("RMSE")
    plt.xlim(t_start, Params.timeSize - 1)
    y_max = max(
        np.max(rmse_clean[t_start:]),
        np.max(rmse_by_var[rep_var][t_start:]),
    ) * 1.35
    plt.ylim(0.0, y_max)
    plt.legend(loc="upper right", **legend_props)
    plt.tight_layout()
    plt.savefig(
        output_path("Fig5_RMSE_WithDefense_Linear.pdf"),
        bbox_inches="tight",
    )
    plt.close()


# ------------------------------------------------------------------------------
# 10. Optional arbitrary attack sequence optimization (Theorem 3 figure)
# ------------------------------------------------------------------------------
def build_expected_window_kld_matrices(
    w_var: float,
    link: tuple[int, int],
    A_sys: np.ndarray,
    watermark_mc: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    receiver, sender = link
    del receiver

    Zs_cache, Theta_cache = build_parity_space(A_sys)
    Zs = Zs_cache[sender]
    Theta = Theta_cache[sender]
    if Zs is None or Theta is None:
        raise RuntimeError(f"Parity space is unavailable for sender {sender}.")

    theta_inv = pinv(Theta)
    length = Params.attack_duration
    m = Params.meaDim
    dimension = length * m
    window_ends = list(range(Params.s, length))
    C_stack = np.zeros((len(window_ends), dimension, dimension))
    rng = np.random.default_rng(seed)

    for _ in range(watermark_mc):
        watermark_matrices = []
        for _time_index in range(length):
            beta = rng.normal(0.0, np.sqrt(w_var), size=m)
            matrix = (
                Params.watermark_lambda * np.eye(m)
                + np.outer(beta, beta)
            )
            watermark_matrices.append(matrix)

        for index, window_end in enumerate(window_ends):
            M_block = block_diag(*watermark_matrices[
                window_end - Params.s:window_end + 1
            ])
            local_mapping = Zs @ M_block
            local_matrix = local_mapping.T @ theta_inv @ local_mapping

            column_start = (window_end - Params.s) * m
            column_end = (window_end + 1) * m
            C_stack[
                index,
                column_start:column_end,
                column_start:column_end,
            ] += local_matrix

    C_stack /= float(watermark_mc)
    C_stack = 0.5 * (
        C_stack + np.swapaxes(C_stack, 1, 2)
    )
    return C_stack, window_ends


def solve_arbitrary_attack_sequence(
    C_stack: np.ndarray,
    threshold: float,
    multi_starts: int,
    seed: int,
) -> dict[str, object]:
    n_windows, dimension, _ = C_stack.shape
    rng = np.random.default_rng(seed)

    def window_quadratics(vector: np.ndarray) -> np.ndarray:
        return np.einsum(
            "i,kij,j->k",
            vector,
            C_stack,
            vector,
            optimize=True,
        )

    def objective(z: np.ndarray) -> float:
        return float(z[-1])

    def objective_jac(z: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(z)
        gradient[-1] = 1.0
        return gradient

    def unit_norm_constraint(z: np.ndarray) -> float:
        vector = z[:-1]
        return float(vector @ vector - 1.0)

    def unit_norm_jac(z: np.ndarray) -> np.ndarray:
        jacobian = np.zeros_like(z)
        jacobian[:-1] = 2.0 * z[:-1]
        return jacobian

    def peak_constraints(z: np.ndarray) -> np.ndarray:
        vector = z[:-1]
        auxiliary = z[-1]
        return auxiliary - window_quadratics(vector)

    def peak_constraints_jac(z: np.ndarray) -> np.ndarray:
        vector = z[:-1]
        C_vector = np.einsum(
            "kij,j->ki",
            C_stack,
            vector,
            optimize=True,
        )
        jacobian = np.empty((n_windows, dimension + 1))
        jacobian[:, :-1] = -2.0 * C_vector
        jacobian[:, -1] = 1.0
        return jacobian

    constraints = [
        {
            "type": "eq",
            "fun": unit_norm_constraint,
            "jac": unit_norm_jac,
        },
        {
            "type": "ineq",
            "fun": peak_constraints,
            "jac": peak_constraints_jac,
        },
    ]
    bounds = [(-1.0, 1.0)] * dimension + [(1e-16, None)]

    average_matrix = np.mean(C_stack, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(average_matrix)
    starts = [eigenvectors[:, int(np.argmin(eigenvalues))]]

    for _ in range(max(0, multi_starts - 1)):
        initial = rng.normal(size=dimension)
        initial /= np.linalg.norm(initial)
        starts.append(initial)

    best: dict[str, object] | None = None
    best_value = np.inf

    for initial in starts:
        initial = np.asarray(initial, dtype=float)
        initial /= np.linalg.norm(initial)
        auxiliary_initial = max(
            float(np.max(window_quadratics(initial))) * 1.05,
            1e-12,
        )
        z0 = np.concatenate([initial, [auxiliary_initial]])

        result = minimize(
            objective,
            z0,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": Params.attack_opt_ftol,
                "maxiter": Params.attack_opt_maxiter,
                "disp": False,
            },
        )

        candidate = result.x[:-1]
        norm_candidate = np.linalg.norm(candidate)
        if not np.isfinite(norm_candidate) or norm_candidate <= 0:
            continue
        candidate /= norm_candidate
        candidate_value = float(np.max(window_quadratics(candidate)))

        if (
            np.isfinite(candidate_value)
            and candidate_value > 0.0
            and candidate_value < best_value
        ):
            best_value = candidate_value
            best = {
                "direction": candidate,
                "t_star": candidate_value,
                "solver_success": bool(result.success),
                "solver_message": str(result.message),
            }

    if best is None:
        raise RuntimeError(
            "Attack-sequence optimization failed for every initialization."
        )

    if best_value <= 1e-14:
        best.update({
            "status": "unbounded",
            "alpha": np.inf,
            "attack_vector": None,
            "average_energy": np.inf,
            "total_energy": np.inf,
            "max_kld": 0.0,
            "window_kld": None,
        })
        return best

    alpha = np.sqrt(2.0 * threshold / best_value)
    attack_vector = alpha * np.asarray(best["direction"])
    window_kld = 0.5 * np.einsum(
        "i,kij,j->k",
        attack_vector,
        C_stack,
        attack_vector,
        optimize=True,
    )

    best.update({
        "status": "solved",
        "alpha": float(alpha),
        "attack_vector": attack_vector,
        "average_energy": float(
            attack_vector @ attack_vector / Params.attack_duration
        ),
        "total_energy": float(attack_vector @ attack_vector),
        "max_kld": float(np.max(window_kld)),
        "window_kld": window_kld,
    })
    return best


def run_arbitrary_attack_optimization(
    link_to_plot: tuple[int, int],
) -> None:
    results: dict[float, dict[str, object]] = {}
    max_energies: list[float] = []
    optimal_sequences: dict[float, np.ndarray] = {}

    for index, variance in enumerate(Params.watermark_vars_test):
        print(
            f"Solving arbitrary attack sequence for "
            f"sigma_beta^2={variance:g} ..."
        )

        C_stack, _window_ends = build_expected_window_kld_matrices(
            w_var=variance,
            link=link_to_plot,
            A_sys=A_unstable,
            watermark_mc=Params.attack_opt_watermark_mc,
            seed=Params.attack_opt_seed + index,
        )

        result = solve_arbitrary_attack_sequence(
            C_stack=C_stack,
            threshold=Params.KLD_threshold,
            multi_starts=Params.attack_opt_multi_starts,
            seed=Params.attack_opt_seed + 1000 + index,
        )

        results[variance] = result
        max_energies.append(float(result["average_energy"]))

        attack_vector = result["attack_vector"]
        if attack_vector is not None:
            optimal_sequences[variance] = np.asarray(attack_vector).reshape(
                Params.attack_duration,
                Params.meaDim,
            )

    np.savez(
        output_path("Fig6_Optimal_Arbitrary_Attack_Sequences.npz"),
        **{
            f"watermark_var_{str(var).replace('.', '_')}": sequence
            for var, sequence in optimal_sequences.items()
        },
    )

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        Params.watermark_vars_test,
        max_energies,
        marker="s",
        color="#A2142F",
        linewidth=2.5,
        markersize=9,
        markerfacecolor="none",
    )
    plt.xlabel(r"Watermarking Variance $\sigma_\beta^2$")
    plt.ylabel("Maximum Average Attack Energy")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(
        output_path("Fig6_Arbitrary_Attack_Energy_Optimization.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 112)
    print(">>> ARBITRARY ATTACK OPTIMIZATION UNDER PEAK KLD CONSTRAINT <<<")
    print("=" * 112)
    print(
        f"Constraint: max_k D_KL,k(a) <= {Params.KLD_threshold} | "
        f"Target: {format_directed_link(link_to_plot)}"
    )
    print("-" * 112)
    print(
        f"{'Variance':<14} | "
        f"{'Average energy':<20} | "
        f"{'Total energy':<20} | "
        f"{'Max KLD':<16} | "
        f"{'Status':<12}"
    )
    print("-" * 112)
    for variance in Params.watermark_vars_test:
        result = results[variance]
        print(
            f"{variance:<14g} | "
            f"{float(result['average_energy']):<20.6e} | "
            f"{float(result['total_energy']):<20.6e} | "
            f"{float(result['max_kld']):<16.6f} | "
            f"{str(result['status']):<12}"
        )
    print("=" * 112 + "\n")


# ------------------------------------------------------------------------------
# 11. Main execution
# ------------------------------------------------------------------------------
def main() -> None:
    print("Directed-link convention: stored (receiver, sender), displayed sender->receiver")
    print("Configured attacked links:")
    for link in Params.attack_links:
        print(f"  {link} = {format_directed_link(link)}")

    finite_attack_end = min(
        Params.attack_start + Params.attack_duration,
        Params.timeSize,
    )
    detector_guard = Params.s + Params.KLD_window - 1
    print(
        f"Finite detection attack interval: "
        f"[{Params.attack_start}, {finite_attack_end})"
    )
    print(
        f"First fully attacked K-L sample: "
        f"k={Params.attack_start + detector_guard}"
    )
    print(
        f"First fully recovered K-L sample: "
        f"k={finite_attack_end + detector_guard}"
    )

    plot_topology_vector(G)

    # ------------------------------------------------------------------
    # Impact experiment: attack remains active through the end.
    # ------------------------------------------------------------------
    print(
        f"Running continuous-impact experiments "
        f"(MC={Params.MC_runs}) ..."
    )
    rmse_unstable_clean, _, _, clean_no_def_fpr, _, _ = run_monte_carlo(
        use_defense=False,
        use_attack=False,
        w_var=0.0,
        A_sys=A_unstable,
        xi_mag=Params.xi_magnitude_unstable,
        attack_duration=Params.impact_attack_duration,
    )
    rmse_unstable_attack_continuous, _, _, _, _, _ = run_monte_carlo(
        use_defense=False,
        use_attack=True,
        w_var=0.0,
        A_sys=A_unstable,
        xi_mag=Params.xi_magnitude_unstable,
        attack_duration=Params.impact_attack_duration,
    )
    rmse_stable_clean, _, _, _, _, _ = run_monte_carlo(
        use_defense=False,
        use_attack=False,
        w_var=0.0,
        A_sys=A_stable,
        xi_mag=Params.xi_magnitude_stable,
        attack_duration=Params.impact_attack_duration,
    )
    rmse_stable_attack_continuous, _, _, _, _, _ = run_monte_carlo(
        use_defense=False,
        use_attack=True,
        w_var=0.0,
        A_sys=A_stable,
        xi_mag=Params.xi_magnitude_stable,
        attack_duration=Params.impact_attack_duration,
    )

    plot_impact_rmse(
        rmse_unstable_clean,
        rmse_unstable_attack_continuous,
        rmse_stable_clean,
        rmse_stable_attack_continuous,
    )

    # ------------------------------------------------------------------
    # Finite attack experiment: used for K-L recovery and detection metrics.
    # ------------------------------------------------------------------
    print("Running no-watermark finite-attack detection experiment ...")
    (
        _rmse_no_def_finite,
        _kld_no_def_finite,
        no_def_tpr,
        _no_def_fpr_from_attack_run,
        no_def_fnr,
        _no_def_delay,
    ) = run_monte_carlo(
        use_defense=False,
        use_attack=True,
        w_var=0.0,
        A_sys=A_unstable,
        xi_mag=Params.xi_magnitude_unstable,
        attack_duration=Params.attack_duration,
    )

    # Lambda-only baseline: encoded/decoded but no random beta term.
    # Use separate clean and attacked trials so FPR and FNR follow their
    # definitions exactly:
    #   FPR = P(alarm | attack-free trial)
    #   FNR = P(no alarm | fully attacked detector window)
    print("Running clean lambda-only baseline (sigma_beta^2=0) ...")
    (
        rmse_clean_lambda_only,
        _kld_clean_lambda_only,
        _tpr_clean_lambda_only,
        fpr_clean_lambda_only,
        _fnr_clean_lambda_only,
        _delay_clean_lambda_only,
    ) = run_monte_carlo(
        use_defense=True,
        use_attack=False,
        w_var=0.0,
        A_sys=A_unstable,
        xi_mag=Params.xi_magnitude_unstable,
        attack_duration=Params.attack_duration,
    )

    print("Running attacked lambda-only baseline (sigma_beta^2=0) ...")
    (
        rmse_lambda_only,
        kld_lambda_only,
        tpr_lambda_only,
        _fpr_lambda_attack_run,
        fnr_lambda_only,
        delay_lambda_only,
    ) = run_monte_carlo(
        use_defense=True,
        use_attack=True,
        w_var=0.0,
        A_sys=A_unstable,
        xi_mag=Params.xi_magnitude_unstable,
        attack_duration=Params.attack_duration,
    )

    rmse_by_var: dict[float, np.ndarray] = {0.0: rmse_lambda_only}
    clean_rmse_by_var: dict[float, np.ndarray] = {0.0: rmse_clean_lambda_only}
    kld_by_var: dict[float, dict[tuple[int, int], np.ndarray]] = {}
    tpr_by_var: dict[float, dict[tuple[int, int], float]] = {
        0.0: tpr_lambda_only,
    }
    fpr_by_var: dict[float, dict[tuple[int, int], float]] = {
        0.0: fpr_clean_lambda_only,
    }
    fnr_by_var: dict[float, dict[tuple[int, int], float]] = {
        0.0: fnr_lambda_only,
    }
    delay_by_var: dict[float, dict[tuple[int, int], float]] = {
        0.0: delay_lambda_only,
    }

    print("Running dynamic-watermark scenarios ...")
    for variance in Params.watermark_vars_test:
        print(f"  clean trial, sigma_beta^2={variance:g}")
        (
            rmse_clean,
            _kld_clean,
            _tpr_clean,
            fpr_clean,
            _fnr_clean,
            _delay_clean,
        ) = run_monte_carlo(
            use_defense=True,
            use_attack=False,
            w_var=variance,
            A_sys=A_unstable,
            xi_mag=Params.xi_magnitude_unstable,
            attack_duration=Params.attack_duration,
        )

        print(f"  attacked trial, sigma_beta^2={variance:g}")
        (
            rmse_attack,
            kld_attack,
            tpr_attack,
            _fpr_attack_run,
            fnr_attack,
            delay_attack,
        ) = run_monte_carlo(
            use_defense=True,
            use_attack=True,
            w_var=variance,
            A_sys=A_unstable,
            xi_mag=Params.xi_magnitude_unstable,
            attack_duration=Params.attack_duration,
        )

        clean_rmse_by_var[variance] = rmse_clean
        rmse_by_var[variance] = rmse_attack
        kld_by_var[variance] = kld_attack
        tpr_by_var[variance] = tpr_attack
        fpr_by_var[variance] = fpr_clean
        fnr_by_var[variance] = fnr_attack
        delay_by_var[variance] = delay_attack

    metric_variances = [0.0] + list(Params.watermark_vars_test)
    link_to_plot = Params.attack_links[0]

    # Reuse the clean trial at the representative variance for nominal
    # transparency comparison; no extra Monte Carlo run is needed.
    nominal_rep_var = nearest_key(
        clean_rmse_by_var,
        Params.representative_rmse_var,
    )
    rmse_nominal_watermarked = clean_rmse_by_var[nominal_rep_var]

    # Figures.
    plot_representative_kld(
        kld_lambda_only,
        kld_by_var,
        link_to_plot,
    )
    plot_fpr_fnr(
        metric_variances,
        fpr_by_var,
        fnr_by_var,
    )
    plot_detection_delay(
        metric_variances,
        delay_by_var,
    )
    plot_rmse_with_defense(
        rmse_unstable_clean,
        rmse_by_var,
    )

    # Terminal tables and CSV files.
    print_no_watermark_table(
        clean_no_def_fpr,
        no_def_fnr,
        no_def_tpr,
    )
    print_watermark_summary_table(
        metric_variances,
        fpr_by_var,
        fnr_by_var,
        tpr_by_var,
        delay_by_var,
    )
    print_per_link_watermark_tables(
        metric_variances,
        fpr_by_var,
        fnr_by_var,
        tpr_by_var,
        delay_by_var,
    )
    print_latex_table1_rows(
        clean_no_def_fpr,
        no_def_fnr,
    )
    print_latex_watermark_rows(
        metric_variances,
        fpr_by_var,
        fnr_by_var,
    )
    save_detection_metric_csvs(
        clean_no_def_fpr,
        no_def_fnr,
        no_def_tpr,
        metric_variances,
        fpr_by_var,
        fnr_by_var,
        tpr_by_var,
        delay_by_var,
    )
    save_latex_detection_tables(
        clean_no_def_fpr,
        no_def_fnr,
        metric_variances,
        fpr_by_var,
        fnr_by_var,
    )

    # Consistency checks.
    maximum_complement_error = max(
        abs(
            tpr_by_var[variance][link]
            + fnr_by_var[variance][link]
            - 1.0
        )
        for variance in metric_variances
        for link in Params.attack_links
    )
    print(
        f"\nConsistency check: max |TPR + FNR - 1| = "
        f"{maximum_complement_error:.3e}"
    )

    rmse_clean_mean = float(np.mean(rmse_unstable_clean))
    rmse_watermarked_mean = float(np.mean(rmse_nominal_watermarked))
    print("\nNominal estimation consistency:")
    print(f"  Clean, no defense mean RMSE      = {rmse_clean_mean:.6f}")
    print(
        f"  Clean, watermark var={nominal_rep_var:g} mean RMSE "
        f"= {rmse_watermarked_mean:.6f}"
    )
    print(
        f"  Absolute difference              = "
        f"{abs(rmse_clean_mean - rmse_watermarked_mean):.6e}"
    )

    if Params.run_attack_optimization:
        run_arbitrary_attack_optimization(link_to_plot)

    print(f"\nAll figures and tables were saved to: {Params.output_dir.resolve()}")


if __name__ == "__main__":
    main()
