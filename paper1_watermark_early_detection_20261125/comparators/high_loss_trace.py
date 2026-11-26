"""Replay the exact high-loss quarter-turn path archived at W1."""
import json
import csv
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
    raw_arrays = {}
    trace = []
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
                "attack_mse_receivers": float(per_time_receiver[50:110].mean()),
                "attack_mse_all50": float(error[row, 50:110].mean()),
                "recovery_mse_receivers": float(per_time_receiver[110:].mean()),
                "first_alarm_tick_by_edge": [
                    (int(np.flatnonzero(gate[row, 50:110, e])[0] + 50)
                     if np.any(gate[row, 50:110, e]) else None)
                    for e in range(6)
                ],
            }
            raw_arrays[tag+"_score"] = score[row]
            raw_arrays[tag+"_gate"] = gate[row].astype(np.uint8)
            raw_arrays[tag+"_delta"] = delta[row]
            raw_arrays[tag+"_node_error"] = error[row]
            for tick in range(150):
                for e in range(6):
                    trace.append({
                        "arm":tag,"tick":tick,"edge":e,
                        "receiver":int(model.links[e,0]),"sender":int(model.links[e,1]),
                        "score":float(score[row,tick,e]),"threshold":float(thresholds[tag][1]),
                        "score_over_threshold":float(score[row,tick,e]/thresholds[tag][1]),
                        "alarm":int(gate[row,tick,e]),
                        "delta_norm":float(np.linalg.norm(delta[row,tick,e])),
                        "accepted_delta_norm":float(np.linalg.norm(accepted[tick,e])),
                        **{f"error_node{int(node)}":float(error[row,tick,node])
                           for node in model.receiver_ids},
                    })
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
    npz_path=exp.HERE/"HIGH_LOSS_PATH.npz"
    np.savez_compressed(npz_path,**raw_arrays)
    csv_path=exp.HERE/"HIGH_LOSS_TRACE.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trace[0]))
        writer.writeheader();writer.writerows(trace)
    print(json.dumps({
        "path_id": path_id,
        "mse": {tag: paths[tag]["attack_mse_receivers"] for tag in arms},
        "first_alarm": {
            tag: min((x for x in paths[tag]["first_alarm_tick_by_edge"] if x is not None),default=None)
            for tag in arms
        },
        "record": str(target),"trace":str(csv_path),"arrays":str(npz_path),
    }, indent=2))


if __name__ == "__main__":
    main()
