"""Independent Riccati, noise, model-cost and calibration readback for stage one."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.stats import beta

import audit_frontend as frontend_audit
import frontend
import models_quality as quality

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csvout(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def independent_gain(A, Q, C, V, fallback):
    if fallback:
        O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(4)])
        _, singular, right = np.linalg.svd(O, full_matrices=True)
        null = right[(singular > 1e-10).sum():].T
        np.testing.assert_allclose(null @ null.T, np.diag([0, 0, 1, 1]), atol=1e-8)
        reduced = C[:, :2]
        prior = solve_discrete_are(A[:2, :2].T, reduced.T, Q[:2, :2], V)
        gain = np.vstack((np.linalg.solve(reduced @ prior @ reduced.T+V, reduced @ prior).T,
                          np.zeros((2, 4))))
    else:
        prior = solve_discrete_are(A.T, C.T, Q, V)
        gain = np.linalg.solve(C @ prior @ C.T+V, C @ prior).T
    F = (np.eye(4)-gain @ C) @ A
    noise = (np.eye(4)-gain @ C) @ Q @ (np.eye(4)-gain @ C).T+gain @ V @ gain.T
    return gain, F, noise


def main():
    models = read(HERE / "quality_models/MODELS.json")
    run = read(HERE / "frontend_health/RUN.json")
    legacy = read(HERE / "legacy_audit/TASK1.json")
    front_audit = read(HERE / "FRONTEND_AUDIT.json")
    assert legacy["grid_count"] == 6912 and legacy["grid_mismatch_count"] == 0
    assert legacy["actual_alarm_decisions"] == 223735
    assert front_audit["status"] == "PASS"
    for filename, digest in run["archives"].items():
        assert quality.sha(HERE / "frontend_health" / filename) == digest
    for filename, digest in run["identity"]["source_sha256"].items():
        assert quality.sha(HERE / filename) == digest
    budget_rows = []
    maximum_gain_error = 0.
    maximum_cost_error = 0.
    maximum_reference_covariance_error = 0.
    maximum_q_regression_error = 0.
    gain_checks = 0
    m7_checks = 0
    for row in models["configurations"]:
        name = row["name"]
        model = frontend_audit.load_model(row["configuration"])
        path = HERE / "quality_models" / f"{name}.npz"
        assert quality.sha(path) == row["archive_sha256"]
        with np.load(path) as archive:
            A, Q, H, R = (archive[field] for field in ("A", "Q", "H", "R"))
            # Independently re-design every distinct base local mode.
            cache = set()
            for mode in range(128):
                for receiver, neighbors in enumerate(model.neighbors):
                    active = [j for j in neighbors if not any(
                        mode & (1 << e) and i == receiver and sender == j
                        for e, (i, sender) in enumerate(model.links[:6]))]
                    key = (receiver, tuple(active))
                    if key in cache:
                        continue
                    cache.add(key)
                    C = np.vstack((H[receiver], H[active].sum(axis=0)))
                    V = block_diag(R[receiver], R[active].sum(axis=0))
                    gain, F, _ = independent_gain(A, Q, C, V, False)
                    saved = np.column_stack((archive["base_K"][mode, receiver],
                                             archive["base_G"][mode, receiver]))
                    error = float(np.max(np.abs(gain-saved)))
                    maximum_gain_error = max(maximum_gain_error, error)
                    np.testing.assert_allclose(gain, saved, rtol=2e-10, atol=2e-10)
                    np.testing.assert_allclose(F, archive["base_F"][mode, receiver], rtol=2e-10, atol=2e-10)
                    gain_checks += 1
            p0 = archive["target_Pplus"][0]
            f_full = archive["target_F"][1, -1]
            o_full = archive["target_Omega"][1, -1]
            costs = []
            fixed_costs = []
            for q_index, q in enumerate(quality.Q):
                C = archive["target_C"][1, q_index]
                V = archive["target_V"][1, q_index]
                gain, F, noise = independent_gain(A, Q, C, V, row["topology"] == "U" and q == 0)
                saved = np.column_stack((archive["target_K"][1, q_index],
                                         archive["target_G"][1, q_index]))
                np.testing.assert_allclose(gain, saved, rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(noise, archive["target_Omega"][1, q_index], rtol=2e-10, atol=2e-10)
                current = F @ p0 @ F.T+noise
                total = np.trace(current[2:, 2:])
                for _ in range(59):
                    current = f_full @ current @ f_full.T+o_full
                    total += np.trace(current[2:, 2:])
                costs.append(float(total/60))
                current = p0.copy()
                total = 0.
                for step in range(60):
                    current = F @ current @ F.T+noise
                    total += np.trace(current[2:, 2:])
                    np.testing.assert_allclose(current, archive["covariance_traces"][q_index, step+1],
                                               rtol=2e-10, atol=2e-10)
                fixed_costs.append(total)
                gain_checks += 1
                budget_rows.append({
                    "configuration": name, "q": q, "health_cost": costs[-1],
                    "epsilon_H": row["epsilon_original"], "epsilon_V": row["epsilon_updated"],
                    "margin_H": row["budget_margin_H"][q_index],
                    "margin_V": row["budget_margin_V"][q_index],
                    "health_feasible_H": int(q in row["health_feasible_H"]),
                    "health_feasible_V": int(q in row["health_feasible_V"]),
                    "fixed_T": row["fixed_T_q"],
                })
            np.testing.assert_allclose(costs, row["health_cost"], rtol=2e-10, atol=2e-10)
            maximum_cost_error = max(maximum_cost_error, float(np.max(abs(np.array(costs)-row["health_cost"]))))
            kappa = (fixed_costs[0]-fixed_costs[-1])/(fixed_costs[-1]+1e-12)
            np.testing.assert_allclose(kappa, row["kappa_L60"], rtol=2e-10, atol=2e-10)
            for edge, (_, sender) in enumerate(model.links):
                raw = frontend.geometry(model, int(sender), 5)["raw_covariance"]
                covariance = model.Z[edge] @ raw @ model.Z[edge].T
                # Frozen build_parity_space adds this registered numerical ridge.
                covariance = (covariance+covariance.T)/2+1e-8*np.eye(len(covariance))
                error = float(np.max(abs(covariance-model.Theta[edge])))
                maximum_reference_covariance_error = max(maximum_reference_covariance_error, error)
                np.testing.assert_allclose(covariance, model.Theta[edge], rtol=2e-10, atol=2e-10)
            # Independently check sampled seven-step products of the complete graph archive.
            graph_file = HERE / "quality_models" / name / f"{row['topology']}_{row['dynamics']}_target35_qwords.npz"
            assert quality.sha(graph_file) == row["M7_archive_sha256"]
            with np.load(graph_file) as graph_archive:
                saved_norms = graph_archive["norm"]
                saved_ranks = graph_archive["obs_rank"]
                admitted = graph_archive["legal"].ravel()
                ids = np.random.default_rng([20261126, 94017, row["configuration"], 716]).integers(
                    0, 7**7, 2048)
                for word_id in ids:
                    digits = (word_id // (7**np.arange(6, -1, -1))) % 7
                    product = np.eye(4)
                    for digit in digits:
                        product = archive["target_F"][1, digit] @ product
                    expected = np.linalg.norm(product, 2)
                    np.testing.assert_allclose(expected, saved_norms[word_id], rtol=2e-12, atol=2e-12)
                assert np.all(saved_norms[admitted] <= 1-1e-8)
                if row["dynamics"] == "unstable":
                    assert np.all(saved_ranks[admitted] == 4)
                m7_checks += len(ids)
        # Whole-path healthy q=0/q=1 regression through the frozen B1 entry.
        paths = quality.physical(model, 16, [20261126, 94017, row["configuration"], 1, 6])
        masks = np.zeros((16, 150, 7), dtype=bool)
        masks[:, 1:, 6] = np.arange(1, 150) % 3 == 0
        packed = np.sum(masks*(1 << np.arange(7)), axis=2).astype(np.uint8)
        optimized = quality.frozen.health.evaluate(model, paths, packed)
        actions = np.full((16, 4, 150), 6, dtype=np.int8)
        actions[:, 0] = np.where(masks[:, :, 6], 0, 6)
        literal = quality.frozen.common.literal_policy_replay(
            model, paths, np.zeros((16, 150, 7, 2)), masks, actions)
        for policy, q_index in ((0, 0), (1, 6)):
            difference = float(np.max(abs(literal[:, policy, 2]-optimized["missing"][:, q_index])))
            maximum_q_regression_error = max(maximum_q_regression_error, difference)
            np.testing.assert_allclose(literal[:, policy, 2], optimized["missing"][:, q_index],
                                       rtol=2e-9, atol=2e-8)
    csvout(HERE / "QUALITY_BUDGET_TABLE.csv", budget_rows)
    calibration_rows = []
    for result in run["results"]:
        index = result["configuration"]
        folder = HERE / "frontend_health"
        with np.load(folder / f"config{index:02}_purpose2.npz") as cal, \
                np.load(folder / f"config{index:02}_purpose3.npz") as healthy:
            for probe in ("physical_probe", "key_probe"):
                assert not np.array_equal(cal[probe], healthy[probe])
            for a, record in enumerate(result["rows"]):
                n = len(cal["path_id"])
                rank = int(np.ceil((n+1)*.95))
                threshold = float(np.partition(cal["maxima"][:, a], rank-1)[rank-1])
                count = int((healthy["maxima"][:, a] > threshold).sum())
                assert threshold == record["threshold"] and count == record["alarms"]
                calibration_rows.append({
                    "configuration": result["spec"][0], "family": record["family"],
                    "threshold": threshold, "alarms": count, "paths": n,
                    "FWER": count/n, "CP95_lower": record["CP95"][0], "CP95_upper": record["CP95"][1],
                })
    csvout(HERE / "FRONTEND_CALIBRATION_TABLE.csv", calibration_rows)
    output = {
        "status": "PASS", "source_sha256": quality.sha(Path(__file__)),
        "healthy_run_sha256": quality.sha(HERE / "frontend_health/RUN.json"),
        "quality_model_index_sha256": quality.sha(HERE / "quality_models/MODELS.json"),
        "independently_redesigned_local_gains": gain_checks,
        "maximum_gain_abs_error": maximum_gain_error,
        "maximum_cost_abs_error": maximum_cost_error,
        "maximum_H1_reference_noise_covariance_abs_error": maximum_reference_covariance_error,
        "H1_frozen_reference_regularization": "symmetrize and add 1e-8 I, as in frozen build_parity_space",
        "maximum_healthy_q0_q1_literal_loss_abs_error": maximum_q_regression_error,
        "independent_M7_product_checks": m7_checks,
        "healthy_archives_verified": len(run["archives"]),
        "calibration_families_recomputed": len(calibration_rows),
        "outputs_sha256": {
            "QUALITY_BUDGET_TABLE.csv": quality.sha(HERE / "QUALITY_BUDGET_TABLE.csv"),
            "FRONTEND_CALIBRATION_TABLE.csv": quality.sha(HERE / "FRONTEND_CALIBRATION_TABLE.csv"),
        },
    }
    quality.frozen.atomic_json(HERE / "INDEPENDENT_VERIFY.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
