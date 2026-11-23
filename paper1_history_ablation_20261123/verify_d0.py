"""Deterministic gain and update checks for the D0 continuous-weight sweep."""

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CRITICAL = ROOT / "paper1_critical_complement_20261123"
sys.path.insert(0, str(CRITICAL))
sys.path.insert(0, str(HERE))

from run_critical import PHYSICAL, critical_model
from sweep_link_weight import EDGE, RECEIVER, SENDER, WEIGHTS, design_weighted_gains, digest


EXPECTED_PHYSICAL_SHA256 = "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d"


def verify_weighted_updates(model, weight: float, K, G, seed: int) -> float:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(50, 4))
    measurements = rng.normal(size=(50, 2))
    attacks = rng.normal(size=(7, 2))
    links = model.links
    edge_by_pair = {tuple(map(int, edge)): index for index, edge in enumerate(links)}
    maximum = 0.0
    for mask in range(128):
        for receiver, neighbors in enumerate(model.cfg["graph"]["neighbors"]):
            prediction = model.A @ state[receiver]
            local = measurements[receiver] - model.H[receiver] @ prediction
            literal_neighbor = np.zeros(2)
            for sender in neighbors:
                edge = edge_by_pair.get((receiver, sender))
                coefficient = 1.0
                if edge is not None:
                    if mask & (1 << edge):
                        coefficient = 0.0
                    elif edge == EDGE:
                        coefficient = weight
                packet = measurements[sender]
                if edge is not None and coefficient > 0.0:
                    packet = packet + attacks[edge]
                literal_neighbor += coefficient * (packet - model.H[sender] @ prediction)

            explicit = prediction + K[mask, receiver] @ local + G[mask, receiver] @ literal_neighbor

            raw_neighbor = sum((measurements[sender] - model.H[sender] @ prediction
                                for sender in neighbors), start=np.zeros(2))
            weighted_neighbor = raw_neighbor.copy()
            for sender in neighbors:
                edge = edge_by_pair.get((receiver, sender))
                if edge is None:
                    continue
                coefficient = 0.0 if mask & (1 << edge) else (weight if edge == EDGE else 1.0)
                clean = measurements[sender] - model.H[sender] @ prediction
                weighted_neighbor += (coefficient - 1.0) * clean
                if coefficient > 0.0:
                    weighted_neighbor += coefficient * attacks[edge]
            aggregate = prediction + K[mask, receiver] @ local + G[mask, receiver] @ weighted_neighbor
            maximum = max(maximum, float(np.max(np.abs(explicit - aggregate))))
    return maximum


def main() -> None:
    if digest(PHYSICAL) != EXPECTED_PHYSICAL_SHA256:
        raise ValueError("Frozen physical configuration hash mismatch.")
    cfg = json.loads(PHYSICAL.read_text(encoding="utf-8"))
    report = {"status": "PASS", "physical_configuration_sha256": digest(PHYSICAL), "models": {}}
    for model_index, label in enumerate(("unstable", "stable")):
        model, _, _ = critical_model(label, cfg)
        max_update_error = 0.0
        min_positive_weight_observability = float("inf")
        for weight in WEIGHTS:
            K, G, F = design_weighted_gains(model, float(weight))
            if weight == 1.0:
                np.testing.assert_allclose(K, model.K, rtol=1e-8, atol=1e-9)
                np.testing.assert_allclose(G, model.G, rtol=1e-8, atol=1e-9)
                np.testing.assert_allclose(F, model.F, rtol=1e-8, atol=1e-9)
            elif weight == 0.0:
                for mask in range(64):
                    drop = mask | (1 << EDGE)
                    np.testing.assert_allclose(K[mask, RECEIVER], K[drop, RECEIVER], atol=0, rtol=0)
                    np.testing.assert_allclose(G[mask, RECEIVER], G[drop, RECEIVER], atol=0, rtol=0)
                    np.testing.assert_allclose(F[mask, RECEIVER], F[drop, RECEIVER], atol=0, rtol=0)
            if weight > 0.0:
                singular_values = np.linalg.svd(np.vstack((model.H[RECEIVER],
                    sum((model.H[j] * (float(weight) if j == SENDER else 1.0)
                         for j in model.cfg["graph"]["neighbors"][RECEIVER]),
                        start=np.zeros((2, model.n))))), compute_uv=False)
                min_positive_weight_observability = min(min_positive_weight_observability, float(singular_values[-1]))
            max_update_error = max(max_update_error,
                                   verify_weighted_updates(model, float(weight), K, G,
                                                           7000 + 100 * model_index + int(weight * 10)))
            if not all(np.all(np.isfinite(item)) for item in (K, G, F)):
                raise AssertionError(f"Non-finite gain table at {label}, omega={weight}.")
        if max_update_error > 2e-12:
            raise AssertionError(f"Weighted update equivalence failed: {max_update_error}.")
        report["models"][label] = {
            "weight_grid": WEIGHTS.tolist(),
            "maximum_literal_aggregate_update_difference": max_update_error,
            "minimum_weighted_observation_singular_value_over_positive_grid": min_positive_weight_observability,
            "zero_weight_uses_binary_drop_fallback": True,
            "unit_weight_matches_registered_critical_model": True,
        }
    output = HERE / "d0_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
