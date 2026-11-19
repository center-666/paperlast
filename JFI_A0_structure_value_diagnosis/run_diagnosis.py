"""Audit frozen A0 results and run label diagnostics on a paired saved input set."""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASELINE = "67ef5aba1549bacf5852a54ba4bc162b4efd9aed"
PRIOR = "JFI_defensive_offline_validation"


def verify_dependencies(base):
    pin = json.loads((HERE / "dependency_lock.json").read_text(encoding="utf-8"))
    if pin["baseline"] != BASELINE:
        raise ValueError("Unexpected diagnostic baseline.")
    for relative, expected in pin["files"].items():
        raw = (base / relative).read_bytes().replace(b"\r\n", b"\n")
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen diagnostic dependency changed: {relative}")
    return len(pin["files"])


def evaluate(base=HERE.parent):
    base = Path(base).resolve()
    checked = verify_dependencies(base)
    paths = sys.path[:]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(base / PRIOR))
    try:
        previous = import_module("replay_evaluator")
        if Path(previous.__file__).resolve() != base / PRIOR / "replay_evaluator.py":
            raise ValueError("Offline evaluator was loaded from an unexpected checkout.")
        original = previous.verify_inputs(base)
        archive = import_module("archive_analysis")
        identification = import_module("identification_replay")
        receiver, common, evidence = previous.load_receiver_modules(base)
        spec = previous.read_json(base / previous.V1 / "registration.json")
        if (spec["horizon"], spec["template_duration"], spec["evaluation_onset"],
                spec["event_false_alarm"], spec["danger_squared"]) != (32, 10, 7, 0.05, 0.1):
            raise ValueError("The registered A0 protocol changed.")
        frozen = base / previous.FRONTIER / "results"
        comparison = previous.read_json(frozen / "comparison.json")["models"]
        models = {}
        for model in previous.MODELS:
            table = {entry["name"]: entry for entry in comparison[model]["method_table"]}
            models[model] = {
                "saved_outcomes": archive.analyze_archive(frozen / f"{model}_paired.npz",
                                                          comparison[model], 7, 32),
                "complete_record_replay": identification.inspect_complete_records(
                    base, model, table, receiver, common, evidence, spec
                ),
            }
        previous.verify_inputs(base)
        verify_dependencies(base)
    finally:
        sys.path[:] = paths
    return {
        "baseline": BASELINE,
        "original_frozen_inputs_verified": len(original["files"]),
        "offline_evaluator_dependencies_verified": checked,
        "physical_records_added": 0,
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent,
                        help="Root of the full frozen research tree")
    args = parser.parse_args()
    report = evaluate(args.baseline)
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    (destination / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    from figures import plot_all
    plot_all(report, destination)
    for name, row in report["models"].items():
        archive = row["saved_outcomes"]
        replay = row["complete_record_replay"]["closed_loop"]["methods"]
        counts = ", ".join(f"{label} {stats['event_counts']['before_alarm']}"
                           for label, stats in replay.items() if label in (
                               "V1a", "activity-label diagnostic", "true-onset diagnostic", "anchor@2%"
                           ))
        print(f"{name}: {archive['path_count']} archived outcomes; "
              f"strict pre-alarm counts in 12 complete replays: {counts}")


if __name__ == "__main__":
    main()
