"""Causal matched-key subspace GLRT for one registered directed raw link.

The score is twice the maximized Gaussian log-likelihood ratio for an unknown
two-dimensional virtual-state amplitude and a finite set of onset positions
inside the current eight-report buffer. Scores are then calibrated as a
pathwise maximum across six links and all legal decision times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BUFFER_REPORTS = 8
RESIDUAL_BLOCK = 5
EFFECTIVE_RANK = 14
SUPPORT_RELATIVE_EIGEN_TOL = 1e-10
TEMPLATE_RELATIVE_SINGULAR_TOL = 1e-10


@dataclass
class SenderGeometry:
    sender: int
    z: np.ndarray
    raw_to_residual: np.ndarray
    covariance_raw: np.ndarray
    covariance_residual: np.ndarray
    whitening_map: np.ndarray
    support_basis: np.ndarray
    support_eigenvalues: np.ndarray
    null_basis: np.ndarray
    rank: int
    attack_state_maps: np.ndarray
    diagnostics: dict


def raw_report_covariance(model, sender: int, report_count: int = BUFFER_REPORTS) -> np.ndarray:
    """Joint covariance of raw healthy sender reports in one local window."""
    h = np.asarray(model.H[sender], dtype=float)
    p = np.eye(2)
    state_covariance = [p.copy()]
    for _ in range(1, report_count):
        p = model.A @ p @ model.A.T + model.Q
        state_covariance.append(p.copy())
    covariance = np.zeros((2 * report_count, 2 * report_count), dtype=float)
    for row in range(report_count):
        for column in range(row, report_count):
            if row == column:
                block = h @ state_covariance[row] @ h.T + model.R[sender]
            else:
                transition = np.linalg.matrix_power(model.A, column - row)
                block = h @ state_covariance[row] @ transition.T @ h.T
            covariance[2 * row:2 * row + 2, 2 * column:2 * column + 2] = block
            covariance[2 * column:2 * column + 2, 2 * row:2 * row + 2] = block.T
    return covariance


def residual_overlap_map(z: np.ndarray, report_count: int = BUFFER_REPORTS) -> np.ndarray:
    """Map eight ordered 2-D reports to four overlapping 8-D parity residuals."""
    if z.shape != (8, 10):
        raise ValueError(f"Expected frozen Z shape (8,10), received {z.shape}")
    output = np.zeros(((report_count - 4) * 8, 2 * report_count), dtype=float)
    for residual_index in range(report_count-4):
        first_report = residual_index
        output[8 * residual_index:8 * (residual_index + 1),
               2 * first_report:2 * (first_report + 5)] = z
    return output


def build_sender_geometry(model, sender: int, zero_first_report=False, report_count=8) -> SenderGeometry:
    # All monitored edges from a given sender use the same source parity map.
    edge_ids = np.flatnonzero(model.links[:, 1] == sender)
    if len(edge_ids) == 0:
        raise ValueError(f"Sender {sender} is not in the frozen monitored-link list")
    z = np.asarray(model.Z[int(edge_ids[0])], dtype=float)
    attack_state_maps = np.zeros((report_count, report_count, 2, 2), dtype=float)
    h = np.asarray(model.H[sender], dtype=float)
    for onset in range(report_count):
        for report in range(onset, report_count):
            attack_state_maps[onset, report] = (
                h @ np.linalg.matrix_power(model.A, report - onset)
            )
    mapping = residual_overlap_map(z, report_count)
    observability = np.vstack([
        model.H[sender] @ np.linalg.matrix_power(model.A, tick)
        for tick in range(report_count)
    ])
    parity_annihilation = float(np.linalg.norm(mapping @ observability))
    if parity_annihilation > 1e-10:
        raise AssertionError(
            f"Sender {sender} parity map does not annihilate the eight-report signal model"
        )
    raw_covariance = raw_report_covariance(model, sender, report_count)
    if zero_first_report:
        raw_covariance[:2, :] = 0.0
        raw_covariance[:, :2] = 0.0
    residual_covariance = mapping @ raw_covariance @ mapping.T
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_covariance)
    maximum = float(eigenvalues[-1])
    keep = eigenvalues > maximum * SUPPORT_RELATIVE_EIGEN_TOL
    rank = int(np.count_nonzero(keep))
    if rank != 2 * report_count - 2:
        raise AssertionError(
            f"Sender {sender} residual support rank is {rank}, expected {2 * report_count - 2}"
        )
    support = eigenvectors[:, keep]
    null = eigenvectors[:, ~keep]
    whitening = (support / np.sqrt(eigenvalues[keep])).T @ mapping
    whitened_covariance = whitening @ raw_covariance @ whitening.T
    np.testing.assert_allclose(
        whitened_covariance, np.eye(rank), rtol=2e-9, atol=2e-9
    )
    diagnostics = {
        "zero_first_report_startup_geometry": bool(zero_first_report),
        "raw_report_dimension": int(mapping.shape[1]),
        "unprojected_residual_dimension": int(mapping.shape[0]),
        "effective_rank": rank,
        "parity_annihilation_fro": parity_annihilation,
        "support_relative_eigen_tolerance": SUPPORT_RELATIVE_EIGEN_TOL,
        "minimum_support_eigenvalue": float(eigenvalues[keep][0]),
        "maximum_support_eigenvalue": float(eigenvalues[keep][-1]),
        "minimum_retained_condition_number": float(eigenvalues[keep][-1] / eigenvalues[keep][0]),
        "maximum_discarded_eigenvalue": float(np.max(eigenvalues[~keep])) if (~keep).any() else 0.0,
        "whitened_covariance_max_abs_error": float(
            np.max(np.abs(whitened_covariance - np.eye(rank)))
        ),
    }
    return SenderGeometry(
        sender=sender,
        z=z,
        raw_to_residual=mapping,
        covariance_raw=raw_covariance,
        covariance_residual=residual_covariance,
        whitening_map=whitening,
        support_basis=support,
        support_eigenvalues=eigenvalues[keep],
        null_basis=null,
        rank=rank,
        attack_state_maps=attack_state_maps,
        diagnostics=diagnostics,
    )


def sender_template_maps(model, sender: int, matrices: np.ndarray,
                         attack_state_maps: np.ndarray | None = None) -> np.ndarray:
    """Return raw 8-report designs for onset offsets 0..7, shape [8,16,2].

    Onset zero is the unknown pre-active-at-window-start template: after onset,
    the arbitrary two-dimensional state is carried by A. Thus an attack active
    before the buffer is represented by a free state at its first report.
    """
    if matrices.ndim == 3:
        matrices = np.broadcast_to(matrices[None],
                                   (1,) + matrices.shape)
    if matrices.shape[1:] != (BUFFER_REPORTS, 2, 2):
        raise ValueError("Expected candidate M windows [path,8,2,2]")
    if attack_state_maps is None:
        h = np.asarray(model.H[sender], dtype=float)
        attack_state_maps = np.zeros((BUFFER_REPORTS, BUFFER_REPORTS, 2, 2), dtype=float)
        for onset in range(BUFFER_REPORTS):
            for report in range(onset, BUFFER_REPORTS):
                attack_state_maps[onset, report] = (
                    h @ np.linalg.matrix_power(model.A, report - onset)
                )
    designs = np.zeros((matrices.shape[0], BUFFER_REPORTS,
                        2 * BUFFER_REPORTS, 2), dtype=float)
    # axis 1 indexes candidate onset offset, axis 2 raw report coordinate
    for onset in range(BUFFER_REPORTS):
        for report in range(onset, BUFFER_REPORTS):
            block = matrices[:, report] @ attack_state_maps[onset, report]
            designs[:, onset, 2 * report:2 * report + 2] = block
    return designs


def template_subspace_score(whitened_records: np.ndarray,
                            whitened_design: np.ndarray) -> np.ndarray:
    """Stable least-squares projection score, independent of attack direction."""
    if whitened_records.ndim != 2 or whitened_design.ndim != 3:
        raise ValueError("Expected y[B,r] and X[B,r,2]")
    if whitened_design.shape[:2] != whitened_records.shape:
        raise ValueError("Template and record dimensions do not agree")
    left, singular, _ = np.linalg.svd(whitened_design, full_matrices=False)
    coordinates = np.einsum("bri,br->bi", left, whitened_records)
    keep = singular > singular[:, :1] * TEMPLATE_RELATIVE_SINGULAR_TOL
    return np.sum(np.where(keep, coordinates ** 2, 0.0), axis=1)


def template_subspace_score_batch(whitened_records: np.ndarray,
                                  whitened_design: np.ndarray) -> np.ndarray:
    """Vectorized least-squares projection over candidate onset templates."""
    if whitened_records.ndim != 2 or whitened_design.ndim != 4:
        raise ValueError("Expected y[B,r] and X[B,onset,r,2]")
    if (whitened_design.shape[0] != whitened_records.shape[0]
            or whitened_design.shape[2] != whitened_records.shape[1]):
        raise ValueError("Batched template and record dimensions do not agree")
    return np.stack([template_subspace_score(whitened_records, x)
                     for x in whitened_design.swapaxes(0, 1)], axis=1)


def template_condition_summary(model, matrices: np.ndarray,
                               geometry: dict[int, SenderGeometry]) -> dict:
    """Summarize effective rank and conditioning of causal onset templates."""
    output = {}
    for sender, record in geometry.items():
        sender_matrices = matrices[:, :, sender]
        if sender_matrices.shape[1] != BUFFER_REPORTS:
            sender_matrices = sender_matrices[:, :BUFFER_REPORTS]
        raw = sender_template_maps(
            model, sender, sender_matrices, record.attack_state_maps
        )
        white = np.einsum("ri,bsic->bsrc", record.whitening_map, raw)
        singular = np.linalg.svd(white, compute_uv=False)
        keep = singular > singular[..., :1] * TEMPLATE_RELATIVE_SINGULAR_TOL
        ranks = keep.sum(axis=-1)
        smallest = np.min(np.where(keep, singular, np.inf), axis=-1)
        largest = singular[..., 0]
        condition = np.divide(largest, smallest, out=np.full_like(largest, np.inf),
                              where=smallest > 0)
        finite = condition[np.isfinite(condition)]
        output[str(sender)] = {
            "template_count": int(ranks.size),
            "rank_counts": {str(rank): int(np.count_nonzero(ranks == rank))
                            for rank in range(3)},
            "condition_finite_fraction": float(np.isfinite(condition).mean()),
            "condition_quantiles_finite": (
                np.quantile(finite, [0., .5, .95, 1.]).tolist()
                if finite.size else None
            ),
            "max_singular_value": float(largest.max()),
            "minimum_retained_singular_value": float(smallest[np.isfinite(smallest)].min())
            if np.isfinite(smallest).any() else None,
            "relative_singular_tolerance": TEMPLATE_RELATIVE_SINGULAR_TOL,
        }
    return output


def matched_scores(model, physical, sender_packets, key_matrices,
                   geometry_cache: dict[int, SenderGeometry] | None = None,
                   startup_geometry_cache: dict[int, SenderGeometry] | None = None,
                   start_time: int = 5) -> np.ndarray:
    """Online causal scan score: max over eight onset templates per edge.

    At decision k, this function reads only raw sender reports k-7..k and the
    corresponding eight M matrices. It does not read attack onset, stop, raw
    injected delta separately, angle, or future key matrices.
    """
    from online_scores import matched, operators
    senders = model.links[:, 1]
    return matched(sender_packets[:, :, senders], key_matrices[:, :, senders],
                   operators(model), start_time)


def plugin_scores(model, physical, delta, sender_packets=None):
    """Exact frozen plugin scorer from the isolated legacy parity replay."""
    from replay import scores as legacy_scores
    from types import SimpleNamespace
    if sender_packets is None:
        sender_packets = physical.y
    return legacy_scores(
        model,
        SimpleNamespace(x=physical.x, y=sender_packets),
        delta,
    )
