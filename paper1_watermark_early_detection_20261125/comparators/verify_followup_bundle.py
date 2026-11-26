"""Final identity, pairing, replay and report-integrity checks for W3-W5."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

import rotation_markov as markov


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    runs = {}
    archives_checked = 0
    source_checks = 0
    for label in ("tail_dev", "tail_confirm", "partial_dev", "partial_confirm",
                  "markov_dev", "markov_confirm"):
        folder = HERE / label
        run = read(folder / "RUN.json")
        for name, digest in run["archives"].items():
            assert sha(folder / name) == digest, (label, name)
            archives_checked += 1
        for name, digest in run["identity"]["source_sha256"].items():
            path = HERE / name
            if not path.exists():
                path = HERE.parent / name
            assert sha(path) == digest, (label, "source", name)
            git_bytes = subprocess.check_output(
                ["git", "show", ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
            assert hashlib.sha256(git_bytes).hexdigest() == digest, (label, "git_source", name)
            source_checks += 1
        runs[label] = {
            "run_sha256": sha(folder / "RUN.json"),
            "archives": len(run["archives"]),
            "namespace": run["identity"]["namespace"],
        }
    namespaces = [tuple(run["namespace"]) for run in runs.values()]
    assert len(set(namespaces)) == len(namespaces)

    lock_checks = []
    for name, development, confirmation, commit in (
        ("PARTIAL_LOCK.json", "partial_dev", "partial_confirm", "7f65ea1"),
        ("MARKOV_LOCK.json", "markov_dev", "markov_confirm", "b29145d"),
    ):
        lock = read(HERE / name)
        assert lock["development_manifest_sha256"] == runs[development]["run_sha256"]
        assert lock["development_analysis_sha256"] == sha(HERE / development / "ANALYSIS.json")
        historical = subprocess.check_output(
            ["git", "show", f"{commit}:{(HERE/name).relative_to(ROOT).as_posix()}"],
            cwd=ROOT,
        )
        assert hashlib.sha256(historical).hexdigest() == sha(HERE / name)
        confirmation_run = read(HERE / confirmation / "RUN.json")
        subprocess.run(["git", "merge-base", "--is-ancestor", commit,
                        confirmation_run["execution_commit"]], cwd=ROOT, check=True)
        lock_checks.append({"lock": name, "frozen_before_execution_commit": commit,
                            "sha256": sha(HERE / name)})

    for split in ("dev", "confirm"):
        folder = HERE / ("partial_" + split)
        audit = read(folder / "PAIRED_ANALYSIS_AUDIT.json")
        assert audit["run_sha256"] == sha(folder / "RUN.json")
        for filename in ("paired_absolute.csv", "paired_contrasts.csv"):
            assert audit[filename.replace(".csv", "_sha256")] == sha(folder / filename)
        assert audit["analysis_source_sha256"] == sha(HERE / "paired_localization_audit.py")

    for split in ("dev", "confirm"):
        folder = HERE / ("tail_" + split)
        audit = read(folder / "VERIFY_INDEPENDENT.json")
        manifest = read(folder / "RUN.json")
        for key, record in audit["scenes"].items():
            assert record["archive_sha256"] == manifest["archives"][key + ".npz"]

    restart = read(HERE / "MARKOV_RESTART_AUDIT.json")
    preserved = 0
    for label in ("markov_dev", "markov_confirm"):
        records = restart["records"][label]
        for name, digest in records.items():
            assert sha(HERE / "markov_superseded_health_reuse" / label / name) == digest
            preserved += 1
    assert sha(HERE / "markov_superseded_health_reuse" / "MARKOV_LOCK.json") == restart["records"]["MARKOV_LOCK.json"]
    preserved += 1

    folder = HERE / "markov_confirm"
    manifest = read(folder / "RUN.json")
    for model in ("unstable", "stable"):
        with np.load(folder / f"{model}_calibration.npz") as cal, \
                np.load(folder / f"{model}_health_0.npz") as healthy, \
                np.load(folder / f"{model}_attack_0.npz") as attacked:
            for probe in ("physical_probe", "key_probe"):
                for left, right in ((cal, healthy), (cal, attacked), (healthy, attacked)):
                    size = min(len(left[probe]), len(right[probe]))
                    assert not np.array_equal(left[probe][:size], right[probe][:size])

    replay_checks = []
    for model_index, scene in ((0, 0), (1, 1)):
        model = ("unstable", "stable")[model_index]
        actual = markov.path_outputs(
            16, model_index, scene, 0, 32, manifest["thresholds"][model], kind="attack")
        with np.load(folder / f"{model}_attack_{scene}.npz") as archived:
            for key, values in actual.items():
                count = 4 if key in (
                    "transition_count", "no_repeat_state_count", "iid_state_count") else 32
                np.testing.assert_array_equal(values, archived[key][:count], err_msg=key)
        replay_checks.append({"model": model, "scene": scene, "paths": 32,
                              "array_fields_identical": len(actual)})

    supplement = read(folder / "SUPPLEMENT_READOUT.json")
    assert supplement["run_sha256"] == sha(folder / "RUN.json")
    assert supplement["source_sha256"] == sha(HERE / "read_markov_results.py")
    for filename, digest in supplement["outputs"].items():
        assert sha(folder / filename) == digest
    with (folder / "supplement_absolute.csv").open(encoding="utf-8", newline="") as stream:
        absolute = list(csv.DictReader(stream))
    analysis = read(folder / "ANALYSIS.json")
    for record in absolute:
        group, arm, metric = (record[key] for key in ("group", "arm", "metric"))
        if group in analysis["groups"] and metric in ("early050", "attack_mse"):
            original_metric = "early3" if metric == "early050" else metric
            expected = analysis["groups"][group][arm][original_metric]["mean"]
            np.testing.assert_allclose(float(record["mean"]), expected, atol=1e-13, rtol=0)

    theory = read(HERE / "ROTATION_PAIR_BOUND.json")
    for record in theory["all_144_transition_checks"]:
        model = record["model"]
        sender = next(r for r in theory["sender_minima"]
                      if r["model"] == model and r["edge"] == record["edge"])
        D = np.array(sender["D"])
        a, b = record["previous_rotation"], record["next_rotation"]
        if a % 2 == b % 2 == 0:
            expected = 4*np.linalg.det(D)
        elif a % 2 != b % 2:
            expected = 2*np.linalg.det(D)
        else:
            expected = np.trace(D)**2 + (D[0, 1]-D[1, 0])**2
        np.testing.assert_allclose(record["det_mismatch"], expected, atol=2e-14)

    markdown = {}
    for name in ("FOLLOWUP_RESULTS_ZH.md", "FOLLOWUP_REPRODUCE.md",
                 "MECHANISM_AND_THEORY_ZH.md", "W3_RESULTS_ZH.md", "W4_RESULTS_ZH.md"):
        content = (HERE / name).read_text(encoding="utf-8")
        assert not any(ord(c) < 32 and c not in "\t\r\n" for c in content), name
        assert content.count(r"\[") == content.count(r"\]"), name
        assert content.count(r"\(") == content.count(r"\)"), name
        assert content.count("$$") % 2 == 0, name
        assert r"\n" not in content[-5:], name
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if not target.startswith(("http", "#")):
                assert (HERE / target).exists() or target == "FOLLOWUP_VERIFY.json", (name, target)
        markdown[name] = {"sha256": sha(HERE / name), "control_characters": 0,
                          "balanced_math_delimiters": True, "local_links_resolve": True}

    output = {
        "status": "PASS",
        "source_sha256": sha(Path(__file__)),
        "run_identities": runs,
        "archive_files_verified": archives_checked,
        "frozen_source_hash_checks": source_checks,
        "git_index_frozen_source_hash_checks": source_checks,
        "locks_verified": lock_checks,
        "superseded_files_preserved_and_verified": preserved,
        "calibration_holdout_attack_probes_distinct": True,
        "representative_exact_replays": replay_checks,
        "independent_determinant_identities_checked": len(theory["all_144_transition_checks"]),
        "supplement_points_match_registered_analysis": True,
        "w4_shared_path_analysis_hashes_verified": True,
        "markdown_checks": markdown,
        "markdown_scope": "Source-level control-character, delimiter and link checks. Original frozen NEXT_PROTOCOL.md formatting defects are documented in FOLLOWUP_REPRODUCE.md.",
    }
    (HERE / "FOLLOWUP_VERIFY.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"], "archives": archives_checked,
                      "source_checks": source_checks, "replayed_paths": 64,
                      "superseded_files_verified": preserved}))


if __name__ == "__main__":
    main()
