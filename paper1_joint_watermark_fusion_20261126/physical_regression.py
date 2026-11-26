"""Nominal physical noise regression and quality-scaling checks on common normals."""
import json
from pathlib import Path

import numpy as np

import models_quality as quality

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    with np.load(quality.frozen.modes.OUTPUT / "mode_tables.npz") as gains:
        for index, (dynamics, topology) in enumerate(
                (d, t) for d in ("unstable", "stable") for t in ("U", "R")):
            nominal, _, _ = quality.frozen.common.configure((dynamics, topology, 35, "negative"), gains)
            seed = [20261126, 94017, index, 91]
            original = quality.frozen.health.physical(nominal, 16, np.random.SeedSequence(seed))
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            x = np.empty((16, 150, 4))
            x[:, 0] = rng.normal(size=(16, 4))
            cholesky = np.linalg.cholesky(nominal.Q)
            for tick in range(1, 150):
                x[:, tick] = x[:, tick-1] @ nominal.A.T + rng.normal(size=(16, 4)) @ cholesky.T
            noise = rng.normal(size=(16, 150, 50, 2))
            y = np.einsum("jmn,btn->btjm", nominal.H, x) + \
                noise*np.sqrt(np.diagonal(nominal.R, axis1=1, axis2=2))[None, None]
            np.testing.assert_array_equal(x, original.x)
            np.testing.assert_array_equal(y, original.y)
            base_error = 0.
            root = [20261126, 94017, index, 92, 0]
            shared = []
            for factor in (.5, 1., 2.):
                model, _, _, _ = quality.construct(dynamics, topology, factor)
                paths = quality.physical(model, 16, root)
                shared.append(paths)
                residual = paths.y-np.einsum("jmn,btn->btjm", model.H, paths.x)
                if factor == 1.:
                    for name in ("F", "K", "G"):
                        difference = float(np.max(abs(model.base[name]-nominal.base[name])))
                        base_error = max(base_error, difference)
                        np.testing.assert_allclose(model.base[name], nominal.base[name], rtol=2e-10, atol=2e-10)
            for paths in shared[1:]:
                np.testing.assert_array_equal(paths.x, shared[0].x)
                unaffected = np.delete(np.arange(50), 35)
                np.testing.assert_array_equal(paths.y[:, :, unaffected], shared[0].y[:, :, unaffected])
            signal = np.einsum("mn,btn->btm", nominal.H[35], shared[1].x)
            component = np.random.default_rng(np.random.SeedSequence([*root, 1])).normal(
                size=(16, 150, 50, 2))
            subtraction_error = 0.
            for factor, paths in zip((.5, 1., 2.), shared):
                expected = signal+component[:, :, 35]*np.sqrt(.5*factor)
                np.testing.assert_array_equal(paths.y[:, :, 35], expected)
                subtraction_error = max(subtraction_error, float(np.max(abs(
                    (paths.y[:, :, 35]-signal)/np.sqrt(factor)-
                    (shared[1].y[:, :, 35]-signal)))))
            rows.append({
                "dynamics": dynamics, "topology": topology, "same_stream_nominal_x_exact": True,
                "same_stream_nominal_y_exact": True, "nominal_base_gain_max_abs_error": base_error,
                "quality_scaling_common_normal_check": True,
                "max_abs_noise_recovery_subtraction_roundoff": subtraction_error,
            })
    report = {"status": "PASS", "source_sha256": quality.sha(Path(__file__)), "rows": rows,
              "formal_sampling": "Formal calibration uses independent process, measurement and key components. This diagnostic additionally couples quality configurations to verify the intended sensor-only noise scaling."}
    quality.frozen.atomic_json(HERE / "PHYSICAL_REGRESSION.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
