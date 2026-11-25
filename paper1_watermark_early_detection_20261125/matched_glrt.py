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
    output = np.zeros((4 * 8, 2 * report_count), dtype=float)
    for residual_index in range(4):
        first_report = residual_index
        output[8 * residual_index:8 * (residual_index + 1),
               2 * first_report:2 * (first_report + 5)] = z
    return output


def build_sender_geometry(model, sender: int) -> SenderGeometry:
    # All monitored edges from a given sender use the same source parity map.
    edge_ids = np.flatnonzero(model.links[:, 1] == sender)
    if len(edge_ids) == 0:
        raise ValueError(f"Sender {sender} is not in the frozen monitored-link list")
    z = np.asarray(model.Z[int(edge_ids[0])], dtype=float)
    mapping = residual_overlap_map(z)
    observability = np.vstack([
        model.H[sender] @ np.linalg.matrix_power(model.A, tick)
        for tick in range(BUFFER_REPORTS)
    ])
    parity_annihilation = float(np.linalg.norm(mapping @ observability))
    if parity_annihilation > 1e-10:
        raise AssertionError(
            f"Sender {sender} parity map does not annihilate the eight-report signal model"
        )
    raw_covariance = raw_report_covariance(model, sender)
    residual_covariance = mapping @ raw_covariance @ mapping.T
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_covariance)
    maximum = float(eigenvalues[-1])
    keep = eigenvalues > maximum * SUPPORT_RELATIVE_EIGEN_TOL
    rank = int(np.count_nonzero(keep))
    if rank != EFFECTIVE_RANK:
        raise AssertionError(
            f"Sender {sender} residual support rank is {rank}, expected {EFFECTIVE_RANK}"
        )
    support = eigenvectors[:, keep]
    null = eigenvectors[:, ~keep]
    whitening = (support / np.sqrt(eigenvalues[keep])).T @ mapping
    whitened_covariance = whitening @ raw_covariance @ whitening.T
    np.testing.assert_allclose(
        whitened_covariance, np.eye(rank), rtol=2e-9, atol=2e-9
    )
    diagnostics = {
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
        diagnostics=diagnostics,
    )


def sender_template_maps(model, sender: int, matrices: np.ndarray) -> np.ndarray:
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
    h = np.asarray(model.H[sender], dtype=float)
    designs = np.zeros((matrices.shape[0], BUFFER_REPORTS,
                        2 * BUFFER_REPORTS, 2), dtype=float)
    # axis 1 indexes candidate onset offset, axis 2 raw report coordinate
    for onset in range(BUFFER_REPORTS):
        for report in range(onset, BUFFER_REPORTS):
            transition = np.linalg.matrix_power(model.A, report - onset)
            block = matrices[:, report] @ h @ transition
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
    gram = np.einsum("bsri,bsrj->bsij", whitened_design,
                     whitened_design, optimize=True)
    gram = 0.5 * (gram + gram.swapaxes(-1, -2))
    cross = np.einsum("bsri,br->bsi", whitened_design,
                      whitened_records, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    maximum = np.maximum(eigenvalues[..., -1:], 0.0)
    keep = eigenvalues > maximum * TEMPLATE_RELATIVE_SINGULAR_TOL ** 2
    rotated = np.einsum("bsji,bsj->bsi", eigenvectors, cross,
                         optimize=True)
    terms = np.divide(rotated ** 2, eigenvalues,
                      out=np.zeros_like(rotated), where=keep)
    return np.sum(terms, axis=-1)


def template_condition_summary(model, matrices: np.ndarray,
                               geometry: dict[int, SenderGeometry]) -> dict:
    """Summarize effective rank and conditioning of causal onset templates."""
    output = {}
    for sender, record in geometry.items():
        sender_matrices = matrices[:, :, sender]
        if sender_matrices.shape[1] != BUFFER_REPORTS:
            sender_matrices = sender_matrices[:, :BUFFER_REPORTS]
        raw = sender_template_maps(model, sender, sender_matrices)
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
                   start_time: int = BUFFER_REPORTS - 1) -> np.ndarray:
    """Online causal scan score: max over eight onset templates per edge.

    At decision k, this function reads only raw sender reports k-7..k and the
    corresponding eight M matrices. It does not read attack onset, stop, raw
    injected delta separately, angle, or future key matrices.
    """
    count = len(physical.x)
    if key_matrices.shape != (count, model.T, model.N, 2, 2):
        raise ValueError("Expected a full [path,time,sender,2,2] watermark key schedule")
    if sender_packets.shape != (count, model.T, model.N, 2):
        raise ValueError("Expected the received two-dimensional sender-packet history")
    geometry_cache = geometry_cache or {
        int(sender): build_sender_geometry(model, int(sender))
        for sender in np.unique(model.links[:, 1])
    }
    output = np.zeros((count, model.T, len(model.links)), dtype=float)
    for edge, (receiver, sender) in enumerate(model.links):
        sender = int(sender)
        geometry = geometry_cache[sender]
        for tick in range(start_time, model.T):
            first = tick - BUFFER_REPORTS + 1
            if first < 0:
                continue
            reports = sender_packets[:, first:tick + 1, sender]
            flat_reports = reports.reshape(count, -1)
            whitened = flat_reports @ geometry.whitening_map.T
            key_window = key_matrices[:, first:tick + 1, sender]
            raw_designs = sender_template_maps(model, sender, key_window)
            whitened_designs = np.einsum(
                "ri,bsic->bsrc", geometry.whitening_map, raw_designs
            )
            all_onsets = template_subspace_score_batch(whitened, whitened_designs)
            output[:, tick, edge] = np.max(all_onsets, axis=1)
    return output


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
