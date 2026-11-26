"""Replay the exact high-loss quarter-turn path archived at W1."""
import json
from pathlib import Path

import numpy as np
import experiment as exp


def main():
    model_index, scene, path_id = 0, 0, 184
    batch = path_id // 32
    row = path_id % 32
    manifest = exp.read(exp.HERE / "confirm/RUN.json")
    ns = tuple(exp.namespace(2, model_index, scene, 2, batch))
    source, model, generate, estimator, ops = exp.old.backend("unstable")
    physical, draws, bits = exp.physical_and_keys(32, ns, model_index)
    raw = physical.y[:, :, model.links[:, 1]]
    fixture = source.rotated_raw_attack(model, 0.)[0][50:110]
    thresholds = manifest["thresholds"]["unstable"]
    onset = 50
    arms = ("quarter_exact", "diagonal_exact", "sign_exact", "iid_exact", "negative_exact",
            "quarter_plugin", "diagonal_plugin", "iid_plugin", "negative_plugin")
    paths = {}
    for scheme in ("quarter", "diagonal", "sign", "iid", "negative"):
        key = exp.schedule(scheme, draws, bits, 1.)
        delta = np.zeros_like(raw)
        delta[:, onset:onset+60] = (
            key[:, onset:onset+60] @ fixture[None, ..., None]
        )[..., 0]
        for mode in ("exact", "plugin"):
            tag = scheme + "_" + mode
            score = exp.scores(mode, model_index, physical, model, delta, raw, key)
            gate = score > thresholds[tag][1]
            world = estimator(model, physical, delta, sender_packets=physical.y,
                              gate_override=gate)
            error = np.sum((world.post - physical.x[:, :, None, :])**2, axis=-1)
            accepted = delta[row] * (~gate[row])[..., None]
            per_time_receiver = error[row][:, model.receiver_ids]
            paths[tag] = {
                "threshold": thresholds[tag][1],
                "gate": gate[row].astype(int).tolist(),
                "score": score[row].tolist(),
                "accepted_delta": accepted.tolist(),
                "node_error": error[row].tolist(),
                "receiver_error": per_time_receiver.tolist(),
                "attack_mse_receivers": float(per_time_receiver[50:110].mean()),
                "attack_mse_all50": float(error[row, 50:110].mean()),
                "recovery_mse_receivers": float(per_time_receiver[110:].mean()),
            }
    # The path identifier deterministically indexes 32-path chunk 5, row 24.
    output = {
        "source_commit": exp.read(exp.HERE / "confirm/RUN.json")["head"],
        "archive": "confirm/unstable_2_0.npz",
        "model": "unstable", "scene": scene, "path_id": path_id,
        "batch": batch, "batch_row": row, "seed_namespace": list(ns),
        "sender_order": model.links[:, 1].astype(int).tolist(),
        "edge_order_receiver_sender": model.links.astype(int).tolist(),
        "affected_receivers": model.receiver_ids.astype(int).tolist(),
        "arms": paths,
    }
    target = exp.HERE / "HIGH_LOSS_REPLAY.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "path_id": path_id,
        "mse": {tag: paths[tag]["attack_mse_receivers"] for tag in arms},
        "first_alarm": {
            tag: [int(x) for x in np.where(np.asarray(paths[tag]["gate"])[50:110].any(axis=1))[0][:1] + 50]
            for tag in arms
        },
        "record": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
