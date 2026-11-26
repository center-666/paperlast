"""Check frozen Git inputs, confirmation lock, information identities and report links."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import study_transfer as study

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    checked_archives = 0
    checked_sources = 0
    for split in ("dev", "confirm"):
        folder = HERE / split
        run = read(folder / "RUN.json")
        analysis = read(folder / "ANALYSIS.json")
        independent = read(folder / "VERIFY.json")
        assert independent["status"] == "PASS"
        assert independent["run_sha256"] == study.sha(folder / "RUN.json")
        assert analysis["run_sha256"] == study.sha(folder / "RUN.json")
        for name, digest in run["archives"].items():
            assert study.sha(folder / name) == digest
            checked_archives += 1
        for name, digest in run["identity"]["sources"].items():
            path = (HERE / name).resolve()
            assert study.sha(path) == digest
            blob = subprocess.check_output(
                ["git", "show", "HEAD:" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest() == digest
            checked_sources += 1
        for name, digest in analysis["table_sha256"].items():
            assert study.sha(folder / name) == digest
    lock = read(HERE / "CONFIRM_LOCK.json")
    assert lock["development_run_sha256"] == study.sha(HERE / "dev/RUN.json")
    assert lock["development_analysis_sha256"] == study.sha(HERE / "dev/ANALYSIS.json")
    frozen_lock = subprocess.check_output(
        ["git", "show", "b8c9bd9:"+(HERE/"CONFIRM_LOCK.json").relative_to(ROOT).as_posix()],
        cwd=ROOT)
    assert hashlib.sha256(frozen_lock).hexdigest() == study.sha(HERE / "CONFIRM_LOCK.json")
    assert read(HERE / "confirm/RUN.json")["identity"]["sources"] == lock["sources"]

    audit = read(HERE / "DESIGN_AUDIT.json")
    nr = np.array(audit["transition_laws"]["no_repeat"]["transition"])
    no = np.array(audit["transition_laws"]["no_opposite"]["transition"])
    opposite = np.roll(np.eye(4), 2, axis=1)
    for lag in range(1, 61):
        expected = np.linalg.matrix_power(nr, lag) @ np.linalg.matrix_power(opposite, lag)
        np.testing.assert_allclose(np.linalg.matrix_power(no, lag), expected, atol=1e-14)

    # Record imported research sources as a supplementary transitive dependency audit.
    study.exp.old.backend("stable")
    transitive = {}
    for module in tuple(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and path.suffix == ".py":
            relative = path.relative_to(ROOT).as_posix()
            transitive[relative] = study.sha(path)
    transitive.update({
        "frozen_physical_fixture_inputs": study.exp.old.base.source_hashes(),
    })

    markdown = {}
    for name in ("PROTOCOL.md", "THEORY.md", "METHOD_FREEZE.md", "REPRODUCE.md", "RESULTS_ZH.md"):
        path = HERE / name
        text = path.read_text(encoding="utf-8")
        assert all(ord(c) >= 32 or c in "\n\r\t" for c in text)
        assert text.count("$$") % 2 == 0
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("http", "#")):
                assert (HERE / target).exists() or target == "BUNDLE_VERIFY.json", (name, target)
        markdown[name] = study.sha(path)
    output = {
        "status": "PASS", "source_sha256": study.sha(Path(__file__)),
        "archives_verified": checked_archives, "source_and_git_hash_checks": checked_sources,
        "confirmation_lock_commit": "b8c9bd9",
        "equal_lag_information_matrix_identity_checked_lags": list(range(1, 61)),
        "supplementary_imported_sources": transitive, "report_sha256": markdown,
        "scope": "Frozen input and archive bytes, independent readout identity, lock chronology, information identity and source-level Markdown checks.",
    }
    study.atomic_json(HERE / "BUNDLE_VERIFY.json", output)
    print(json.dumps({"status": "PASS", "archives": checked_archives,
                      "frozen_source_checks": checked_sources, "reports": len(markdown)}))


if __name__ == "__main__":
    main()
