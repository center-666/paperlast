"""Regenerate task-3 purpose-5/6 paths and compare every fusion input."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("D2_D1_DIR",r"C:\Users\zhongxin\shixidoc\.d1_registration_worktree\paper1_D1_link_value_20261123")
import task3

HERE = Path(__file__).resolve().parent.parent
DEV = Path(r"C:\Users\zhongxin\shixidoc\paperlast-joint-audit\paper1_joint_watermark_fusion_20261126\task3_dev")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    run = json.loads((DEV/"RUN.json").read_text(encoding="utf-8"))
    records = []
    max_error = {name: 0. for name in ("x","y","delta","scores","matched","keys")}
    gates_equal = True
    for index, config in enumerate(task3.QUALITY_ORDER):
        path = DEV/f"config{index:02}.npz"
        assert sha(path) == run["archives"][path.name]
        model = task3.audit_frontend.load_model(index)
        thresholds = task3.frontend_thresholds(index)
        operators = task3.frontend.operators(model)
        with np.load(path) as archive:
            for fixture_index, fixture in enumerate(("continuous","burst")):
                data = task3.batch_data(model, operators, index, fixture_index, 5, 256, thresholds)
                for name, generated, saved in (
                    ("x", data["paths"].x, archive[fixture+"_x"]),
                    ("y", data["paths"].y, archive[fixture+"_y"]),
                    ("delta", data["delta"], archive[fixture+"_delta"]),
                    ("scores", data["scores"], archive[fixture+"_scores"]),
                    ("matched", data["matched"], archive[fixture+"_matched"]),
                    ("keys", data["keys"], archive[fixture+"_keys"]),
                ):
                    error = float(np.max(np.abs(generated-saved)))
                    max_error[name] = max(max_error[name], error)
                    np.testing.assert_array_equal(generated, saved)
                np.testing.assert_array_equal(data["gates"], archive[fixture+"_gates"])
            health = task3.batch_data(model, operators, index, 0, 6, 256, thresholds, first_batch=100)
            health["delta"].fill(0.)
            for name, generated, saved in (
                ("x", health["paths"].x, archive["health_x"]),
                ("y", health["paths"].y, archive["health_y"]),
                ("delta", health["delta"], archive["health_delta"]),
                ("scores", health["scores"], archive["health_scores"]),
                ("matched", health["matched"], archive["health_matched"]),
                ("keys", health["keys"], archive["health_keys"]),
            ):
                error = float(np.max(np.abs(generated-saved)))
                max_error[name] = max(max_error[name], error)
                np.testing.assert_array_equal(generated, saved)
            np.testing.assert_array_equal(health["gates"], archive["health_gates"])
        records.append(config)
        print("REGENERATED", config, flush=True)
    result = {
        "status": "PASS_all_evaluation_path_inputs_regenerated_bitwise",
        "configurations": records,
        "attack_physical_paths": 12*2*256,
        "health_physical_paths": 12*256,
        "maximum_absolute_difference": max_error,
        "gate_masks_identical": gates_equal,
        "task3_RUN_sha256": sha(DEV/"RUN.json"),
        "fusion_evaluation_receipt_sha256": sha(HERE/"class_weight_development/EVALUATION.json"),
        "information_boundary": "Purpose-5/6 inputs are disjoint from purpose-4 fitting streams. This is a reproducibility check of archived development paths, not a new random seed or independent confirmation.",
    }
    target = HERE/"class_weight_development/INPUT_REGENERATION.json"
    target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__=="__main__":
    main()
