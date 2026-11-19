"""Recompute the finite Gaussian bounds and the paired prefix integral."""

import argparse
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import json
import sys

import numpy as np
import sympy as sp

from exact_bounds import START, LAST, certificate, fraction, matrix
from joint_integral import JointIntegral, prefix_events, quadrature


HERE = Path(__file__).resolve().parent
BASELINE = "74819cd7963417612e384be7974074c2380cdc13"
OFFLINE = "JFI_defensive_offline_validation"
MATCHING = Path("JFI_A0_early_identifiability_and_safety_theory/results/results.json")
REGISTRATION = Path("attack_gain_switch_audit_20261115/registration.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pinned_dependencies(base):
    manifest = read_json(HERE / "dependency_lock.json")
    if manifest["baseline"] != BASELINE:
        raise ValueError("Unexpected baseline pin.")
    for relative, expected in manifest["files"].items():
        raw = (base / relative).read_bytes().replace(b"\r\n", b"\n")
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen input differs: {relative}")
    return len(manifest["files"])


def recorded_template(base, old, name, settings):
    with np.load(base / old.OBSERVATIONS, allow_pickle=False) as archive:
        recorded = archive[f"{name}__original_Gaussian__ordinary__attack"]
    end = START+settings.template_duration
    np.testing.assert_array_equal(recorded[:START], 0.)
    np.testing.assert_array_equal(recorded[end:], 0.)
    np.testing.assert_allclose(np.sum(recorded**2), 10., atol=2e-11, rtol=0)
    # Algebraic cross-check of the frozen direction and its original energy normalization.
    model = json.loads((base / old.MODEL).read_text(encoding="utf-8"), parse_float=str)
    source = json.loads((base / REGISTRATION).read_text(encoding="utf-8"), parse_float=str)
    assert (source["attack_start"], source["attack_length"]) == (START, settings.template_duration)
    a = matrix(model["models"][name]["A"])
    h = matrix(model["H"][settings.target])
    direction = sp.Matrix([sp.Rational(x) for x in source["ordinary_direction"]])
    templates = [h*(a**lag)*direction for lag in range(settings.template_duration)]
    norm_sq = sum((v.T*v)[0] for v in templates)
    factor_sq = fraction(sp.Rational(source["energy_budget_coding_domain"])/norm_sq)
    check = np.sqrt(float(factor_sq))*np.asarray([list(v) for v in templates], dtype=float)
    error = float(np.max(np.abs(check-recorded[START:end])))
    np.testing.assert_allclose(check, recorded[START:end], rtol=1e-14, atol=2e-15)
    return recorded[START:end], templates[:LAST-START+1], factor_sq, error


def cross_check(base, old, name, receiver, prepared, spec, integrand):
    ids = spec["samples"]["physical_indices"]
    trusted, observed, key = old.saved_sensor_inputs(base, name, prepared.settings, ids)
    with np.load(base / old.V11 / f"phase1_results/{name}_step_logs.npz", allow_pickle=False) as logs:
        recorded_change = logs["A0__decoded_difference"]
    original = (
        receiver.Method("V0", "raw"),
        receiver.Method("V1a", "structure"),
        *(receiver.Method(name, "anchor", radius) for name, radius in integrand.radii),
    )
    hp = receiver.run_receiver(prepared, receiver.Observations(trusted, observed, key), original)
    ap = receiver.run_receiver(prepared, receiver.Observations(
        trusted, observed + recorded_change, key), original)
    u = integrand.archive_coordinates(trusted, observed)
    bf, offsets, posterior = integrand.evaluate(u, key[:, :LAST+1])
    old_difference = (ap.post-hp.post)[:, :, START:LAST+1]
    np.testing.assert_allclose(bf, ap.log_bf[:, :LAST+1], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(offsets, old_difference, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(prefix_events(bf, offsets)[1],
                                  prefix_events(ap.log_bf[:, :LAST+1], old_difference)[1])
    observer = receiver.BatchedObserver(prepared)
    attacked_input = receiver.Observations(trusted, observed + recorded_change, key)
    p_active = np.stack([observer.at(attacked_input,k).probability_active
                         for k in range(START,LAST+1)],axis=1)
    np.testing.assert_allclose(posterior[..., 0], p_active, atol=2e-11, rtol=2e-11)
    return {
        "saved_complete_inputs": len(ids),
        "maximum_absolute_log_bf_error": float(np.max(np.abs(bf-ap.log_bf[:, :LAST+1]))),
        "maximum_absolute_paired_output_error": float(np.max(np.abs(offsets-old_difference))),
        "maximum_absolute_active_posterior_error": float(np.max(np.abs(posterior[..., 0]-p_active))),
        "first_three_event_indicators_match": True,
    }


def evaluate_model(base, old, name, receiver, common, evidence, frontier, spec):
    settings = evidence.Settings.from_registration(spec)
    p = evidence.PreparedModel(common.load_model(name, base / old.MODEL), settings)
    assert (p.n, p.m) == (2, 2)
    saved, exact, factor_sq, max_profile_roundoff = recorded_template(base,old,name,settings)
    proof = certificate(base / old.MODEL, name, spec, exact, factor_sq)
    archive_proof = certificate(base / old.MODEL, name, spec, saved)
    for entry in ("saturation_upper_bound_all_radii", "saturation_ellipse_upper_bound",
                  "first_packet_probability_certificate", "v0_upper_bound"):
        assert proof[entry] == archive_proof[entry], entry
    matching = read_json(base / MATCHING)["models"][name]["nominal_matching"]
    radii = {"sat_nominal_matched": matching["matched_radius_for_point_estimate"]}
    for label, row in matching["frozen_points"].items():
        radii[label] = row["radius_frozen"]
    critical = float(np.sqrt(.1)/2)
    nominal_full, _, eigenvalues = frontier.covariances(p)
    tr = np.trace(nominal_full, axis1=1, axis2=2)
    v0 = float(p.nominal_mse().mean())
    cost_at = lambda r: float(frontier.rms_cost(r, tr, eigenvalues).mean())
    nominal = {label: {"radius": r, "nominal_mse": cost_at(r),
                       "pct_over_nominal_v0": 100*(cost_at(r)/v0-1)}
               for label, r in radii.items()}
    nominal["zero_risk_boundary"] = {"radius": critical, "nominal_mse": cost_at(critical),
                                     "pct_over_nominal_v0": 100*(cost_at(critical)/v0-1)}
    integrand = JointIntegral(p, saved, radii)
    return {
        "profile_verified_against_norm_10_formula_max_roundoff": max_profile_roundoff,
        "certificate_inputs": "Frozen model and registered direction decimals as exact rationals; normalization squared exact",
        "displayed_probability_bounds_identical_on_saved_dyadics": True,
        "rigorous_bounds": proof,
        "nominal_v0_mse": v0,
        "nominal_cost_numerical_evaluation": nominal,
        "v1a_match_mse_approximate_95_interval": matching["v1a_target_approximate_95_interval"],
        "matched_radius_approximate_95_range": matching["radius_range_from_target_approximate_95_interval"],
        "archive_regression": cross_check(base,old,name,receiver,p,spec,integrand),
        "numerical_integral": quadrature(integrand),
    }


def evaluate(base=HERE.parent):
    base = Path(base).resolve()
    extra = pinned_dependencies(base)
    paths = sys.path[:]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(base / OFFLINE))
    try:
        old = import_module("replay_evaluator")
        assert Path(old.__file__).resolve() == base / OFFLINE / "replay_evaluator.py"
        lock = old.verify_inputs(base)
        receiver, common, evidence = old.load_receiver_modules(base)
        frontier = import_module("frontier")
        spec = read_json(base / old.V1 / "registration.json")
        conditions = tuple(spec[key] for key in ("evaluation_onset","template_duration","horizon",
                                                 "no_event_prior","coded_energy_reference",
                                                 "event_false_alarm","danger_squared"))
        assert conditions == (START,10,32,.5,10,.05,.1)
        assert spec["watermark"] == {"lambda": .5, "q": .5}
        results = {name: evaluate_model(base,old,name,receiver,common,evidence,frontier,spec)
                   for name in old.MODELS}
        old.verify_inputs(base)
        pinned_dependencies(base)
    finally:
        sys.path[:] = paths
    return {"baseline": BASELINE, "protected_inputs": len(lock["files"])+extra,
            "onset_and_integration_cutoff": [START,LAST],
            "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent)
    args = parser.parse_args()
    result = evaluate(args.baseline)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                      encoding="utf-8",newline="\n")
    for name, model in result["models"].items():
        values = model["nominal_cost_numerical_evaluation"]
        runs = model["numerical_integral"]["replications"]["nodes_power_10"]
        print(f"{name}: critical cost increase "
              f"{values['zero_risk_boundary']['pct_over_nominal_v0']:.6f}%; "
              f"V0, V1a, matched saturation risk means by age "
              f"{np.round(np.mean([r['risk_by_method_by_age'][:3] for r in runs],axis=0),5).tolist()}")


if __name__ == "__main__":
    main()
