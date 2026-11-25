"""Deterministic interface/support checks before registered Stage-1 paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V3 = ROOT / "paper1_sscd_commonrandom_20261122"
THEORY = ROOT / "paper1_50node_tv_gate_theory_20261121"
SOURCE50 = ROOT / "paper1_common_g50_gain_acceptance_20261121"
sys.path[:0] = [str(V3), str(THEORY), str(SOURCE50), str(HERE)]

from run_crn_development import load_backend
from run_crn_final import measure
from watermark_keys import (
    attack_delta_for_schedule,
    beta_for_scheme,
    clean_encode_decode,
    make_key_draws,
    matrices_from_beta,
)
from matched_glrt import (
    BUFFER_REPORTS,
    build_sender_geometry,
    matched_scores,
    plugin_scores,
    raw_report_covariance,
    residual_overlap_map,
    sender_template_maps,
    template_subspace_score,
)


def explicit_history_map(model, sender):
    z = np.asarray(model.Z[np.flatnonzero(model.links[:, 1] == sender)[0]])
    output = np.zeros((32, 16))
    for residual_index in range(4):
        output[8 * residual_index:8 * (residual_index + 1),
               2 * residual_index:2 * (residual_index + 5)] = z
    return output


def covariance_monte_carlo(model, sender, geometry, count=8192, seed=831799):
    rng = np.random.default_rng(seed)
    x = np.empty((count, BUFFER_REPORTS, 2))
    x[:, 0] = rng.normal(size=(count, 2))
    for tick in range(1, BUFFER_REPORTS):
        x[:, tick] = x[:, tick - 1] @ model.A.T + rng.normal(size=(count, 2))
    y = np.einsum("ij,btj->bti", model.H[sender], x)
    y += rng.normal(size=y.shape) @ np.linalg.cholesky(model.R[sender]).T
    raw = y.reshape(count, -1)
    white = raw @ geometry.whitening_map.T
    covariance = np.cov(white, rowvar=False)
    return {
        "paths": count,
        "seed_namespace": [20261125, 8317, 99, seed],
        "whitened_covariance_max_abs_error": float(np.max(np.abs(covariance - np.eye(geometry.rank)))),
        "whitened_covariance_diagonal_range": [float(covariance.diagonal().min()),
                                                 float(covariance.diagonal().max())],
        "whitened_covariance_mean_offdiagonal": float(
            (covariance - np.diag(np.diag(covariance))).mean()
        ),
    }


def independent_lstsq_scores(y, x):
    scores = np.zeros(len(y))
    for index in range(len(y)):
        estimate, _, _, _ = np.linalg.lstsq(x[index], y[index], rcond=None)
        fitted = x[index] @ estimate
        scores[index] = fitted @ fitted
    return scores


def main():
    old, load_model, physical_paths, common_g, _ = load_backend()
    reports = []
    for model_name in ("unstable", "stable"):
        model = load_model(model_name)
        physical = physical_paths(model, 16, 20261125_8317_9900 + int(model_name == "stable"))
        draws = make_key_draws(
            16, 20261125_8317_9910 + int(model_name == "stable"),
            20261125_8317_9920 + int(model_name == "stable"),
            20261125_8317_9930 + int(model_name == "stable"),
            20261125_8317_9940 + int(model_name == "stable"),
        )
        beta = beta_for_scheme(draws, "negative_radial_bin")
        matrices = matrices_from_beta(beta)
        decoded = clean_encode_decode(model, physical, matrices)
        clean_roundtrip_error = float(np.max(np.abs(decoded - physical.y)))
        clean_roundtrip_relative_error = float(np.max(
            np.abs(decoded - physical.y) / np.maximum(1.0, np.abs(physical.y))
        ))
        if clean_roundtrip_relative_error > 1e-13:
            raise AssertionError(f"Clean encoding roundtrip failed for {model_name}")

        geometries = {
            int(sender): build_sender_geometry(model, int(sender))
            for sender in np.unique(model.links[:, 1])
        }
        geometry_reports = {}
        for sender, geometry in geometries.items():
            independently_built = explicit_history_map(model, sender)
            np.testing.assert_array_equal(geometry.raw_to_residual, independently_built)
            geometry_reports[str(sender)] = {
                **geometry.diagnostics,
                "monitored_edge_indices": np.flatnonzero(model.links[:, 1] == sender).astype(int).tolist(),
                "history_map_independent_build_max_abs_error": float(
                    np.max(np.abs(geometry.raw_to_residual - independently_built))
                ),
                "empirical_noise_whitening": covariance_monte_carlo(
                    model, sender, geometry, seed=20261125_8320 + sender
                    + 100 * int(model_name == "stable")
                ),
            }

        raw_attack, scale = old.rotated_raw_attack(model, 0.)
        delta = attack_delta_for_schedule(model, matrices, raw_attack, 50)
        observed_packets = physical.y.copy()
        for edge, sender_id in enumerate(model.links[:, 1].astype(int)):
            observed_packets[:, :, sender_id] += delta[:, :, edge]
        match = matched_scores(model, physical, observed_packets, matrices, geometries)
        plugin = plugin_scores(model, physical, delta, sender_packets=observed_packets)
        if match.shape != plugin.shape or not np.isfinite(match).all():
            raise AssertionError("Detector interface shape or finite-value check failed")
        if np.any(match[:, :7] != 0.0):
            raise AssertionError("Matched GLRT emitted a score before its 8-report buffer existed")

        sender = int(model.links[0, 1])
        edge = 0
        geometry = geometries[sender]
        tick = 7
        raw_reports = (physical.y[:, tick - 7:tick + 1, sender]
                       + delta[:, tick - 7:tick + 1, edge])
        whitened_record = raw_reports.reshape(16, -1) @ geometry.whitening_map.T
        key_window = matrices[:, tick - 7:tick + 1, sender]
        raw_design = sender_template_maps(model, sender, key_window)[:, 0]
        whitened_design = np.einsum("ri,bic->brc", geometry.whitening_map, raw_design)
        left, singular, _ = np.linalg.svd(whitened_design, full_matrices=False)
        coords = np.einsum("bri,br->bi", left, whitened_record)
        keep = singular > singular[:, :1] * 1e-10
        projection_score = np.sum(np.where(keep, coords ** 2, 0.0), axis=1)
        least_squares_score = independent_lstsq_scores(whitened_record, whitened_design)
        np.testing.assert_allclose(projection_score, least_squares_score,
                                   rtol=2e-10, atol=2e-10)

        changed_future_m = matrices.copy()
        changed_future_packets = physical.y.copy()
        changed_future_packets = observed_packets.copy()
        changed_future_m[:, tick + 1:] = 20.0 * np.eye(2)
        changed_future_packets[:, tick + 1:] += 100.0
        future_match = matched_scores(
            model, SimpleNamespace(x=physical.x, y=changed_future_packets),
            changed_future_packets, changed_future_m, geometries,
        )
        np.testing.assert_array_equal(match[:, tick], future_match[:, tick])

        direction_checks = []
        for angle in (0., -45., 45.):
            rotated_attack, angle_scale = old.rotated_raw_attack(model, angle)
            stacked_h = np.vstack([model.H[int(sender)] for _, sender in model.links])
            initial_state, _, rank, _ = np.linalg.lstsq(
                stacked_h, rotated_attack[50].reshape(-1), rcond=None
            )
            if rank != 2:
                raise AssertionError("The three registered attack directions lost a state dimension.")
            for onset, decision, expected_offset in ((50, 57, 0), (65, 69, 3)):
                shifted_delta = np.zeros_like(delta)
                shifted_delta[:, onset:onset + 60] = np.einsum(
                    "bteij,tej->btei",
                    matrices[:, onset:onset + 60, model.links[:, 1]],
                    rotated_attack[50:110],
                )
                first = decision - 7
                for edge, (_, sender_id) in enumerate(model.links):
                    edge_geometry = geometries[int(sender_id)]
                    raw_key_window = matrices[:, first:decision + 1, int(sender_id)]
                    candidate_design = sender_template_maps(
                        model, int(sender_id), raw_key_window
                    )[:, expected_offset]
                    predicted_delta = np.einsum(
                        "bic,c->bi", candidate_design, initial_state
                    )
                    observed_delta = shifted_delta[:, first:decision + 1, edge].reshape(
                        len(physical.x), -1
                    )
                    # Check the physical direction is represented by the
                    # nuisance-state subspace for the exact keyed reports.
                    template_error = float(np.max(np.abs(observed_delta[0] - predicted_delta[0])))
                    residual_signal = edge_geometry.raw_to_residual @ observed_delta[0]
                    null_projection = edge_geometry.null_basis.T @ residual_signal
                    scaled_null = float(
                        np.linalg.norm(null_projection)
                        / max(1.0, np.linalg.norm(residual_signal))
                    )
                    if scaled_null > 2e-8:
                        raise AssertionError("Attack mean falls outside the healthy residual support.")
                    if template_error > 2e-8:
                        raise AssertionError("A registered physical direction is not spanned by the GLRT template.")
                    direction_checks.append({
                        "angle_degrees": angle,
                        "onset": onset,
                        "decision_time": decision,
                        "candidate_onset_offset": expected_offset,
                        "receiver": int(model.links[edge, 0]),
                        "sender": int(sender_id),
                        "template_max_abs_error": template_error,
                        "standardized_null_support_projection": scaled_null,
                        "attack_energy_scale": float(angle_scale),
                    })

        reports.append({
            "model": model_name,
            "smoke_paths": len(physical.x),
            "clean_encode_decode_max_abs_error": clean_roundtrip_error,
            "clean_encode_decode_max_scaled_error": clean_roundtrip_relative_error,
            "rotated_zero_degree_attack_scale": float(scale),
            "matched_score_shape": list(match.shape),
            "plugin_score_shape": list(plugin.shape),
            "matched_first_score_time": 7,
            "matched_vs_independent_lstsq_max_abs_error": float(
                np.max(np.abs(projection_score - least_squares_score))
            ),
            "future_M_packet_invariance_at_time_7": True,
            "three_direction_and_two_onset_template_checks": direction_checks,
            "sender_covariance_and_support": geometry_reports,
        })
    output = {
        "status": "PASS_INTERFACE_AND_COVARIANCE_SMOKE",
        "baseline_commit": "8953e1dd9a5e2535ead2698acb6f345a85d4f532",
        "formal_paths_generated": False,
        "models": reports,
    }
    target = HERE / "INTERFACE_SMOKE.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"], "report": str(target),
                      "models": [row["model"] for row in reports]}, indent=2))


if __name__ == "__main__":
    main()
