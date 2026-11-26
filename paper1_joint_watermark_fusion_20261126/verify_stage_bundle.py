"""Final frozen-source and archive checks for the joint stage-one delivery."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import models_quality as quality


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    healthy = read(HERE / "frontend_health/RUN.json")
    independent = read(HERE / "INDEPENDENT_VERIFY.json")
    assert independent["status"] == "PASS"
    assert independent["source_sha256"] == quality.sha(HERE / "verify_joint.py")
    assert independent["healthy_run_sha256"] == quality.sha(HERE / "frontend_health/RUN.json")
    frozen_execution = subprocess.check_output(
        ["git", "rev-parse", "e951e8a"], cwd=HERE, text=True).strip()
    assert healthy["execution_commit"] == frozen_execution
    checks = 0
    for filename, digest in healthy["identity"]["source_sha256"].items():
        path = HERE / filename
        assert quality.sha(path) == digest
        blob = subprocess.check_output(
            ["git", "show", "e951e8a:" + path.relative_to(HERE.parent).as_posix()], cwd=HERE)
        assert hashlib.sha256(blob).hexdigest() == digest
        checks += 1
    for filename, digest in healthy["archives"].items():
        assert quality.sha(HERE / "frontend_health" / filename) == digest
        checks += 1
    models = read(HERE / "quality_models/MODELS.json")
    assert models["inputs"]["source_sha256"] == quality.sha(HERE / "models_quality.py")
    for row in models["configurations"]:
        archive = HERE / "quality_models" / (row["name"]+".npz")
        graph = HERE / "quality_models" / row["name"] / \
            f"{row['topology']}_{row['dynamics']}_target35_qwords.npz"
        assert quality.sha(archive) == row["archive_sha256"]
        assert quality.sha(graph) == row["M7_archive_sha256"]
        checks += 2
    for filename, digest in independent["outputs_sha256"].items():
        assert quality.sha(HERE / filename) == digest
    assert read(HERE / "FRONTEND_AUDIT.json")["frontend_source_sha256"] == quality.sha(HERE / "frontend.py")
    assert read(HERE / "PHYSICAL_REGRESSION.json")["source_sha256"] == quality.sha(HERE / "physical_regression.py")
    docs = {}
    for filename in ("PREEXEC_REGISTRATION.md", "INTERFACE_CONTRACT.md", "RESULTS_ZH.md", "REPRODUCE.md"):
        path = HERE / filename
        text = path.read_text(encoding="utf-8")
        assert all(ord(c) >= 32 or c in "\r\n\t" for c in text)
        assert text.count("$$") % 2 == 0
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if not link.startswith(("http", "#")):
                assert (HERE / link).exists(), (filename, link)
        docs[filename] = quality.sha(path)
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    d1_head = subprocess.check_output(
        ["git", "-C", str(quality.frozen.D1), "rev-parse", "HEAD"], text=True).strip()
    assert d1_head == "7e32a41b9802438f6645d9048ee46386caa53418"
    report = {
        "status": "PASS", "source_sha256": quality.sha(Path(__file__)),
        "exact_frozen_source_and_archive_checks": checks,
        "D1_dependency_HEAD": d1_head,
        "D2_parent": "d6d958348ec3bbc73fe42df0a9e9cf8a337a8d78",
        "calibration_execution_commit": healthy["execution_commit"],
        "task_original_sha256": quality.sha(HERE / "TASK_ORIGINAL.txt"),
        "report_sha256": docs,
        "scope": "Stage-one legacy replay, deterministic quality models, front-end numerical audit and independent healthy calibration.",
    }
    quality.frozen.atomic_json(HERE / "STAGE_BUNDLE_VERIFY.json", report)
    print(json.dumps({"status": "PASS", "frozen_checks": checks,
                      "healthy_paths_calibration_plus_holdout": 14*2048*2}))


if __name__ == "__main__":
    main()
