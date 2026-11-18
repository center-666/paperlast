"""Verify fixed source checksums, corrected paired traces, and nominal bounds."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from shared import DEFAULT_BASELINE, HERE, V11, digest, load_json, setup, verify_registration, write_json
from nominal_audit import bounded_mean_upper
from receiver_experiment import METHODS
from weight_experiment import first_flags


def check_delivery():
    reg = load_json(HERE / "registration.json")
    verify_registration(reg, DEFAULT_BASELINE)
    _, _, spec, _ = setup(DEFAULT_BASELINE)
    freeze_checks = load_json(V11 / "reproduction_checks.json")
    frozen_v11_results = {**freeze_checks["results_sha256"],
                          **{key: v["sha256"] for key, v in freeze_checks["trace_summary_checks"].items()}}
    for name, expected in frozen_v11_results.items():
        assert digest(V11 / name) == expected, name

    project = HERE.parents[2]
    stage_dir = HERE.parent
    index = load_json(stage_dir / "checks/修改前保护哈希.json")
    original = {name: value for name, value in index.items() if name.split("\\")[0] != "参考文献"}
    for name, expected in original.items():
        assert digest(project / name) == expected, name
    manifest = load_json(stage_dir / "minimum_contribution_gate_20261115/freeze_manifest.json")
    refs = manifest["local_reference_artifact_hashes"]
    for name, expected in refs.items():
        assert digest(stage_dir / "common_g_reference_revision_20261115" / name) == expected, name
    zipped = load_json(project / "研究规划/原稿归档/第一篇小论文_初始原稿_20261026_清单.json")
    with zipfile.ZipFile(project / zipped["archive"]) as old:
        for item in zipped["files"]:
            body = old.read(item["archive_member"])
            assert len(body) == item["bytes"] and sha256(body).hexdigest() == item["sha256"]
    main = load_json(stage_dir / "checks/攻击补强交付核查.json")
    old = load_json(stage_dir / "checks/交付核查.json")
    tex = [item for item in main["artifacts"] if item["file"].endswith(".tex")]
    for item in tex:
        assert digest(stage_dir / item["file"]) == item["sha256"]
    for item in (main, old):
        assert digest(Path(item["pdf"])) == item["pdf_sha256"]

    artifacts = {}
    for phase in ("development", "validation"):
        folder = HERE / "results" / phase
        results = load_json(folder / "summary.json")
        assert results["registration_sha256"] == digest(HERE / "registration.json")
        for model, rows in results["models"].items():
            file = folder / f"{model}_paths.npz"
            inputs = 0
            with np.load(file, allow_pickle=False) as z:
                for profile, values in rows["profiles"].items():
                    post = z[f"{profile}__post"]
                    healthy = z["healthy__post"]
                    onsets = z[f"{profile}__onset"]
                    oracle = METHODS.index("GLS-known")
                    if f"{profile}__paired_healthy_oracle_post" in z.files:
                        healthy[:, oracle] = z[f"{profile}__paired_healthy_oracle_post"]
                    p = np.sum((post-healthy)**2, axis=-1)
                    np.testing.assert_array_equal(z[f"{profile}__first_danger"],
                                                  first_flags(p>=spec["danger_squared"]))
                    observed = z[f"{profile}__common_alarm"]
                    np.testing.assert_array_equal(observed, z[f"{profile}__log_bf"]>=-np.log(spec["event_false_alarm"]))
                    b = first_flags(p>=spec["danger_squared"]) < first_flags(observed)[:, None]
                    np.testing.assert_array_equal(b, z[f"{profile}__before_alarm"])
                    assert np.all(p*(np.arange(p.shape[-1])[None, None, :]<onsets[:, None, None]) < 1e-24)
                    for j, name in enumerate(METHODS):
                        assert b[:, j].sum() == values["policies"][name]["danger_before_common"]
                    if phase == "validation":
                        same = sha256(np.delete(post, oracle, axis=1).tobytes()).hexdigest()
                        fixed = reg["pre_paired_reference_fix"]["validation_post_hashes_except_GLS_known"]
                        assert same == fixed[model][f"{profile}__post"]
                    inputs += len(post)
            artifacts[str(file.relative_to(HERE))] = {"paired_streams": inputs, "sha256": digest(file)}

    bounds = load_json(HERE / "results/nominal_audit.json")
    for model, bound in bounds.items():
        scores = np.load(HERE / f"results/{model}_bounded_health_scores.npy", allow_pickle=False)
        clip = reg["nominal_certificate"]["clip"]
        result = bounded_mean_upper(scores[:, 1], clip, bound["pointwise_error_probability"])
        upper = result+bound["analytic_tail_allowance"]+1e-9+np.mean(bound["conditional_cov_trace"])
        rel = upper/np.mean(bound["V0_exact_stage_trace"])-1
        np.testing.assert_allclose(rel, bound["policies"]["V1a"]["upper_relative"], atol=2e-14)
        assert rel <= .02
    test = subprocess.run([sys.executable, "-B", "-X", "utf8", "test_v12.py"], cwd=HERE,
                          capture_output=True, check=True, text=True)
    return {"validation_traces": artifacts, "paired_oracle_reference_corrected": True,
            "post_sha256_unchanged_for_nine_other_methods": True,
            "frozen_V1_1_result_files": len(frozen_v11_results),
            "frozen_V1_sources_and_shared_inputs_checked": True,
            "frozen_common_G_local_files": len(refs), "protected_original_and_chapter_files": len(original),
            "protected_stage_TeX_and_PDF": len(tex)+2, "archive_members": len(zipped["files"]),
            "deterministic_tests": test.stderr,
            "results_sha256": {str(f.relative_to(HERE)): digest(f)
                               for f in (HERE / "results").rglob("*.json")
                               if f.name != "delivery_checks.json"}}


if __name__ == "__main__":
    write_json(HERE / "results/delivery_checks.json", check_delivery())
    print("Verified the original files, all corrected paired traces, and the expected-health bounds.")
