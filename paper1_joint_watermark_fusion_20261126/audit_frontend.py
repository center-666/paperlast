"""Noiseless, causal and independent-LS audits for each four-dimensional sender."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

import frontend
import models_quality as quality

HERE = Path(__file__).resolve().parent


def configurations():
    rows = [(name, d, t, 35, factor) for d, t, _, factor in quality.SPECS
            for name in (f"{t}_{d}_R35x{factor:g}",)]
    rows.extend((f"R_{d}_target1", d, "R", 1, 1.) for d in ("unstable", "stable"))
    return rows


def load_model(index):
    name, d, t, sender, factor = configurations()[index]
    if sender == 35:
        return quality.construct(d, t, factor)[0]
    with np.load(quality.frozen.modes.OUTPUT / "mode_tables.npz") as gains:
        return quality.frozen.common.configure((d, t, sender, "negative"), gains)[0]


def main():
    rows = []
    score_errors = []
    template_errors = []
    for index, (name, d, t, target, factor) in enumerate(configurations()):
        model = load_model(index)
        ops = frontend.operators(model)
        packet_rng = np.random.default_rng([20261126, 94017, index, 0, 0, 2])
        initial = packet_rng.normal(size=(16, 4))
        for edge, (_, sender) in enumerate(model.links):
            H = model.H[sender]
            D = H @ model.A @ np.linalg.pinv(H, rcond=1e-10)
            closed = np.linalg.norm(H @ model.A-D @ H)/max(1e-30, np.linalg.norm(H @ model.A))
            assert closed < 1e-10
            deterministic = np.stack([initial @ np.linalg.matrix_power(model.A, k).T @ H.T
                                      for k in range(8)], axis=1)
            difference = deterministic[:, 1:]-deterministic[:, :-1] @ D.T
            np.testing.assert_allclose(difference, 0., atol=3e-12)
            vector = np.array([6., 0.])
            inherited_error = float(np.linalg.norm(vector-D @ vector)/np.linalg.norm(vector))
            V2 = H @ model.Q @ H.T+model.R[sender]+D @ model.R[sender] @ D.T
            actual_gamma = []
            hypothetical_gamma = []
            for a in range(4):
                for b in range(4):
                    if a == b:
                        continue
                    actual = frontend.ROTATIONS[b]-D @ frontend.ROTATIONS[a]
                    hypothetical = frontend.ROTATIONS[b] @ D-D @ frontend.ROTATIONS[a]
                    actual_gamma.append(float(eigh(actual.T @ np.linalg.solve(V2, actual),
                                                   2*np.eye(2), eigvals_only=True)[0]))
                    hypothetical_gamma.append(float(eigh(
                        hypothetical.T @ np.linalg.solve(V2, hypothetical),
                        np.eye(2)+D.T @ D, eigvals_only=True)[0]))
            rows.append({
                "configuration": name, "edge": edge, "receiver": int(model.links[edge, 0]),
                "sender": int(sender), "H_shape": list(H.shape),
                "H_rank": int(np.linalg.matrix_rank(H, tol=1e-10)),
                "D": D.tolist(), "relative_output_closure_error": float(closed),
                "trajectory_max_abs_error": float(np.max(abs(difference))),
                "old_W5_dynamic_template_relative_error_for_D2_constant_input": inherited_error,
                "D2_active_active_constant_template_gamma_min": min(actual_gamma),
                "W5_dynamic_hypothesis_gamma_min": min(hypothetical_gamma),
                "projection_state_ranks": [ops[w][1][edge]["state_rank"] for w in (5, 6, 7, 8)],
                "whitened_support_ranks": [ops[w][1][edge]["support_rank"] for w in (5, 6, 7, 8)],
                "burst_off_off_signal": "zero, both reports carry zero registered input",
            })
        paths = quality.physical(model, 16, [20261126, 94017, index, 1, 0])
        raw = paths.y[:, :, model.links[:, 1]]
        uniforms = packet_rng.random((16, 150, 50))
        for rule in (False, True):
            all_keys = frontend.keys(uniforms, rule)
            keys = all_keys[:, :, model.links[:, 1]]
            np.testing.assert_array_equal(keys @ np.swapaxes(keys, -1, -2),
                                          np.broadcast_to(np.eye(2), keys.shape))
            np.testing.assert_allclose((keys @ (np.swapaxes(keys, -1, -2) @ raw[..., None]))[..., 0],
                                       raw, rtol=0, atol=1e-12)
            for fixture in ("continuous", "burst"):
                delta = np.zeros_like(raw)
                active = np.ones(60) if fixture == "continuous" else (np.arange(60) % 10 < 6)
                delta[:, 50:110, 6] = (keys[:, 50:110, 6] @
                                      (active[:, None]*np.array([6., 0.]))[None, ..., None])[..., 0]
                actual = frontend.scores(raw+delta, keys, ops)
                changed_packets = (raw+delta).copy()
                changed_packets[:, 74:] += 123.
                changed_keys = keys.copy()
                changed_keys[:, 74:] = np.eye(2)
                np.testing.assert_array_equal(frontend.scores(changed_packets, changed_keys, ops)[:, :74],
                                              actual[:, :74])
                for tick in (5, 6, 7, 8, 52, 59, 60, 65, 109):
                    width = min(8, tick)
                    start = tick-width+1
                    for edge in (0, 6):
                        whitening = ops[width][1][edge]["whitening"]
                        expected = frontend.independent_score(
                            (raw+delta)[0, start:tick+1, edge],
                            keys[0, start:tick+1, edge], whitening)
                        error = abs(actual[0, tick, edge]-expected)
                        np.testing.assert_allclose(actual[0, tick, edge], expected, rtol=2e-10, atol=2e-7)
                        score_errors.append(error)
                    if tick >= 50 and tick <= 109:
                        whitening = ops[width][1][6]["whitening"]
                        signal = delta[0, start:tick+1, 6]
                        mean = whitening @ signal.ravel()
                        projected = frontend.independent_score(signal, keys[0, start:tick+1, 6], whitening)
                        residual = abs(projected-mean @ mean)/max(1., mean @ mean)
                        assert residual < 1e-10
                        template_errors.append(residual)
            # Shared sender 1 on two receiver arcs uses identical per-sender keys.
            if target == 1:
                np.testing.assert_array_equal(keys[:, :, 0], keys[:, :, 6])
    output = {
        "status": "PASS", "source_sha256": quality.sha(Path(__file__)),
        "frontend_source_sha256": quality.sha(HERE / "frontend.py"),
        "native_source_sha256": quality.sha(HERE / "projection.c"),
        "rows": rows, "independent_LS_score_cases": len(score_errors),
        "maximum_absolute_native_vs_LS_score_difference": max(score_errors),
        "true_constant_and_burst_mean_subspace_cases": len(template_errors),
        "maximum_relative_projection_energy_difference": max(template_errors),
        "future_packet_and_key_invariance": True, "codec_roundtrip": True,
        "same_sender_key_shared_across_outgoing_edges": True,
        "time": "first legitimate report k=1, scoring k=5, width=min(8,k)",
        "template": "D2 constant plus registered 6-on/4-off burst phases, unknown 2D amplitude and causal suffix start",
    }
    quality.frozen.atomic_json(HERE / "FRONTEND_AUDIT.json", output)
    print(json.dumps({k: output[k] for k in (
        "status", "independent_LS_score_cases", "maximum_absolute_native_vs_LS_score_difference",
        "true_constant_and_burst_mean_subspace_cases")}))


if __name__ == "__main__":
    main()
