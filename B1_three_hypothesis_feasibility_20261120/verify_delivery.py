"""Check data provenance, policy causality and conditional-model identities."""
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np

from experiment import HERE, BASE, frozen_digest, read, radial_angle_grid
from closed_loop import sampled_matrix


def main():
    reg = read(HERE / "registration.json")
    manifest = read(HERE / "input_manifest.json")
    for entry in manifest["attachments"]:
        path = HERE / entry["archived_file"]
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    provenance = read(HERE / "results/provenance.json")
    assert provenance["frozen_hashes_before"] == provenance["frozen_hashes_after"] == frozen_digest()
    assert provenance["registration_sha256"] == sha256((HERE / "registration.json").read_bytes()).hexdigest()
    assert provenance["script_sha256"] == sha256((HERE / "experiment.py").read_bytes()).hexdigest()
    assert set(reg["development_path_ids"]).isdisjoint(reg["evaluation_path_ids"])
    histories = read(HERE / "results/histories.json")
    assert len(histories) == 648
    for row in histories:
        stat = row["stats"]
        assert stat["dispersion_coefficient_error"] < 2e-11
        assert stat["I"] >= -1e-12 and stat["I"] <= stat["A"] + 5e-4
        for name, vector in row["vectors"].items():
            if name == "uniform":
                continue
            v = np.array(vector)
            assert np.linalg.norm(v) <= .6 + 1e-10
            assert abs(np.array(stat["gA"]) @ v) < 1e-10
            allowance = float(name.rsplit("_", 1)[-1]) * stat["common"][0]
            assert np.array(stat["g_common"])[0] @ v <= allowance + 1e-9
    validation = read(HERE / "results/validation.json")
    assert max(r["mean_error"] for r in validation["batch_gaussian_checks"]) < 3e-10
    assert max(r["covariance_error"] for r in validation["batch_gaussian_checks"]) < 3e-10

    controls = read(HERE / "results/causal_controls.json")
    control = {(r["path_id"], r["policy"], r["true_h_for_scoring"], r["tick"]): r for r in controls}
    for path_id in reg["evaluation_path_ids"]:
        for policy in ("uniform", "development_fixed", "adaptive_health_0", "adaptive_health_0.02"):
            # Onset 7 hasn't occurred when the k=7 matrix is selected.
            for h in (1, 2):
                np.testing.assert_allclose(control[path_id, policy, h, 7]["v"],
                                           control[path_id, policy, 0, 7]["v"], atol=0, rtol=0)
            # At selection of the onset-8 matrix, h2 and healthy have identical histories.
            np.testing.assert_allclose(control[path_id, policy, 2, 8]["v"],
                                       control[path_id, policy, 0, 8]["v"], atol=0, rtol=0)
    causal = read(HERE / "results/causal_summary.json")["results"]
    for name, row in causal["raw"]["0"].items():
        assert abs(row["full_mse"] - causal["raw"]["0"]["uniform"]["full_mse"]) < 1e-13
    rng = np.random.default_rng(20261120101)
    for _ in range(20):
        beta = rng.normal(size=2) * np.sqrt(.5)
        original = .5 * np.eye(2) + np.outer(beta, beta)
        angle = rng.uniform(0, 2 * np.pi)
        v = .6 * np.array([np.cos(angle), np.sin(angle)])
        actual = sampled_matrix(original, v)
        np.testing.assert_allclose(np.linalg.eigvalsh(actual), np.linalg.eigvalsh(original), atol=2e-13)
        sample = rng.normal(size=2)
        np.testing.assert_allclose(actual @ np.linalg.solve(actual, sample), sample, atol=2e-13)
    matrices, weights, f, _, _ = radial_angle_grid(8, 32, 7)
    for v in (np.array([.6, 0]), np.array([0, .6]), np.array([.3, -.4])):
        kernel = weights * (1 + f @ v)
        np.testing.assert_allclose(kernel.sum(), 1, atol=1e-13)
        np.testing.assert_allclose(np.einsum("b,bij->ij", kernel, matrices), np.eye(2), atol=2e-13)

    links = 0
    for file in HERE.glob("*.md"):
        for match in re.finditer(r"\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            target = match.group(1).split("#")[0]
            if target.startswith(("http://", "https://")) or not target:
                continue
            assert (file.parent / target).exists(), (str(file), target)
            links += 1
    result = dict(status="passed", original_attachments=len(manifest["attachments"]), histories=len(histories),
                  policy_causality="pre-onset decisions identical; no true candidate label in policy inputs",
                  preserved_files=len(provenance["frozen_hashes_before"]), relative_links=links,
                  scope="File/model/protocol checks; no claim of confirmed adaptive performance or originality")
    (HERE / "results/delivery_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
