"""Deterministic pre-execution checks and finite-state blind-run probabilities."""
import json
from pathlib import Path

import numpy as np

import study_transfer as study
from verify_transfer import independent_inputs


def run_probability(transition, ticks=60, length=8):
    # Track probability mass by present state and designated run length.
    probability = np.zeros((4, length), dtype=np.longdouble)
    probability[0, 1] = probability[2, 1] = .25
    probability[1, 0] = probability[3, 0] = .25
    for _ in range(1, ticks):
        updated = np.zeros_like(probability)
        for previous in range(4):
            for next_state in range(4):
                for run in range(length):
                    new_length = (run+1 if previous == next_state else 1) if next_state in (0, 2) else 0
                    if new_length < length:
                        updated[next_state, new_length] += probability[previous, run] * transition[previous, next_state]
        probability = updated
    return float(1-probability.sum())


def audit():
    matrices = {}
    for label, offsets in (("iid", (0, 1, 2, 3)), ("no_repeat", (1, 2, 3)),
                           ("no_opposite", (0, 1, 3))):
        transition = np.zeros((4, 4))
        for state in range(4):
            transition[state, [(state+step) % 4 for step in offsets]] = 1/len(offsets)
        np.testing.assert_allclose(transition.sum(axis=0), 1)
        np.testing.assert_allclose(transition.sum(axis=1), 1)
        positive = transition[transition > 0]
        entropy = float(-np.sum(positive*np.log2(positive))/4)
        matrices[label] = {
            "transition": transition.tolist(), "conditional_entropy_bits": entropy,
            "adjacent_mutual_information_nat": float((2-entropy)*np.log(2)),
            "per_sender_60_report_identity_blind_run_probability": run_probability(transition),
            "six_senders_identity_blind_run_probability": 1-(1-run_probability(transition))**6,
        }
    count = 0
    maximum_difference = 0.
    transition_bounds = []
    for mi, model_name in enumerate(study.MODELS):
        original = study.inputs(99, mi, 0, 2, 0, 2)
        source, model, physical, estimator, ops, uniforms, keys = independent_inputs(99, mi, 0, 2, 0, 2)
        np.testing.assert_array_equal(original[4].y, physical.y)
        for label in study.SCHEMES[:3]:
            direct = study.exp.ROTATION[study.rotation_codes(uniforms, label)][:, :, model.links[:, 1]]
            np.testing.assert_array_equal(direct, keys[label])
        np.testing.assert_array_equal(original[-1], keys["negative"])
        raw = physical.y[:, :, model.links[:, 1]]
        fixture = source.rotated_raw_attack(model, 0.)[0][50:110]
        for label, key in keys.items():
            encoded = np.linalg.solve(key, raw[..., None])[..., 0]
            np.testing.assert_allclose((key @ encoded[..., None])[..., 0], raw, rtol=1e-12, atol=1e-12)
            if label != "negative":
                np.testing.assert_array_equal(key @ np.swapaxes(key, -1, -2),
                                              np.broadcast_to(np.eye(2), key.shape))
                np.testing.assert_allclose(np.linalg.norm(encoded, axis=-1),
                                           np.linalg.norm(raw, axis=-1), atol=1e-13)
            delta = np.zeros_like(raw)
            delta[:, 50:110] = (key[:, 50:110] @ fixture[None, ..., None])[..., 0]
            full = study.exp.receiver.matched(raw+delta, key, ops)
            healthy = study.exp.receiver.matched(raw, key, ops)
            future_packets = raw+delta
            future_packets[:, 73:] += 123
            future_keys = key.copy()
            future_keys[:, 73:] = np.eye(2)
            changed = study.exp.receiver.matched(future_packets, future_keys, ops)
            np.testing.assert_array_equal(changed[:, :73], full[:, :73])
            for _, support in study.SUPPORTS:
                mask = np.zeros(6, dtype=bool)
                mask[list(support)] = True
                mixed = delta*mask[None, None, :, None]
                direct = study.exp.receiver.matched(raw+mixed, key, ops)
                assembled = np.where(mask[None, None], full, healthy)
                error = float(np.max(np.abs(direct-assembled)))
                maximum_difference = max(maximum_difference, error)
                np.testing.assert_array_equal(direct, assembled)
                count += 1
        for edge, (_, sender) in enumerate(model.links):
            H = model.H[sender]
            D = H @ model.A @ np.linalg.inv(H)
            for label, definition in matrices.items():
                allowed = np.array(definition["transition"]) > 0
                singular = []
                for a, b in zip(*np.where(allowed)):
                    N = study.exp.ROTATION[b] @ D - D @ study.exp.ROTATION[a]
                    if np.linalg.matrix_rank(N, tol=1e-10) < 2:
                        singular.append([int(a), int(b)])
                transition_bounds.append({
                    "model": model_name, "edge": edge, "rule": label,
                    "rank_deficient_allowed_pairs": singular,
                })
    # Check disjoint population seeds before any registered path is generated.
    probes = []
    for population in (0, 1, 2):
        result = study.inputs(99, 0, 0, population, 8, 2)
        probes.append((result[4].y[:, 1, 0], result[5][:, 1, 0]))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not np.array_equal(probes[a][0], probes[b][0])
        assert not np.array_equal(probes[a][1], probes[b][1])
    # Full output shapes, indexing, endpoint packaging and direct independent score path.
    thresholds = {arm: {"threshold": 22. if arm.endswith("exact") else 60.}
                  for arm in study.ARMS}
    smoke = study.evaluate(99, 0, 0, 2, 0, 2, thresholds)
    assert smoke["values"].shape == (2, 9, 8, 17)
    assert np.isfinite(smoke["values"]).all()
    result = {
        "status": "PASS", "source_sha256": study.sha(Path(__file__)),
        "mixed_support_score_checks": count,
        "maximum_local_score_assembly_difference": maximum_difference,
        "future_packet_and_key_invariance": True, "codec_roundtrip": True,
        "population_seed_probe_checks": True,
        "smoke_shape": list(smoke["values"].shape),
        "transition_laws": matrices, "allowed_pair_rank_checks": transition_bounds,
    }
    study.atomic_json(study.HERE / "DESIGN_AUDIT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "mixed_support_score_checks", "smoke_shape")}))


if __name__ == "__main__":
    audit()
