"""Read-only independent checks of the registered A0 paper evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag, cho_factor, cho_solve


HERE = Path(__file__).resolve().parent
BASELINE = "b1509be0265ecd0512befe999847fa5908087872"
CORE = Path("JFI_A0_safety_accuracy_theorem_review")
FRONTIER = Path("JFI_secure_estimation_accuracy_safety_frontier")
DIRECT = Path("JFI_A0_direct_joint_risk_feasibility")
MODEL = Path("common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json")
PROTOCOL = Path("JFI_reconstruction_attack_aware_soft_accommodation_V1/registration.json")
STORED_INPUT = Path("attack_gain_switch_audit_20261115/screen_arrays.npz")
POINTS = ("anchor@0.1%", "anchor@0.5%", "anchor@1%", "anchor@2%")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(base):
    lock = load(HERE / "dependency_lock.json")
    assert lock["baseline"] == BASELINE
    for relative, row in lock["files"].items():
        raw = (base / relative).read_bytes()
        if row["normalize_CRLF"]:
            raw = raw.replace(b"\r\n", b"\n")
        if sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Frozen dependency differs: {relative}")
    return len(lock["files"])


def path_covariance(a, q, horizon):
    """Form the full latent covariance from independent initial and process blocks."""
    n = len(a)
    powers = [np.linalg.matrix_power(a, i) for i in range(horizon)]
    transition = np.zeros((horizon*n, horizon*n))
    for k in range(horizon):
        for j in range(k+1):
            transition[k*n:(k+1)*n, j*n:(j+1)*n] = powers[k-j]
    return transition @ block_diag(np.eye(n), *[q]*(horizon-1)) @ transition.T


def schur(prefix, target, obs, noise):
    var = obs @ prefix @ obs.T + noise
    cross = prefix[target] @ obs.T
    return prefix[target, target] - cross @ cho_solve(cho_factor(var), cross.T)


def independent_moments(model, spec, name):
    """Batch Schur complements and finite convolution, independent of Riccati code."""
    n = model["n"]
    h, r, q = np.asarray(model["H"]), np.asarray(model["R"]), np.asarray(model["Q"])
    local, neighbor, target = [spec[key] for key in ("receiver", "healthy", "target")]
    a = np.asarray(model["models"][name]["A"])
    row = model["models"][name]["nodes"][local]
    k, g = (np.asarray(row[x]) for x in ("K", "G"))
    f = (np.eye(n)-k@h[local]-g@(h[neighbor]+h[target])) @ a
    h_trust = np.vstack(h[[local, neighbor]])
    r_trust = block_diag(*r[[local, neighbor]])
    h_all = np.vstack([h_trust, h[target]])
    r_all = block_diag(r_trust, r[target])
    cov = path_covariance(a, q, spec["horizon"])
    gain = np.hstack([k, g, g])
    full, gap, common = [], [], []
    first_innovation = None
    for t in range(spec["horizon"]):
        width = (t+1)*n
        state = slice(t*n, width)
        prefix = cov[:width, :width]
        rounds = np.eye(t+1)
        all_obs = np.kron(rounds, h_all)
        all_noise = np.kron(rounds, r_all)
        pf = schur(prefix, state, all_obs, all_noise)
        pt = schur(prefix, state, np.kron(rounds, h_trust), np.kron(rounds, r_trust))
        if t == spec["evaluation_onset"]:
            # Condition on the same old data, including the current trusted rows.
            keep = slice(0, len(all_obs)-len(h[target]))
            history = schur(prefix, state, all_obs[keep], all_noise[keep, keep])
            cross = history @ h[target].T
            innovation = h[target] @ cross + r[target]
            first_innovation = innovation, cross
        blocks = [np.linalg.matrix_power(f, t-j) @ gain for j in range(t+1)]
        coefficient = np.hstack([b @ h_all for b in blocks])
        coefficient[:, state] -= np.eye(n)
        v0 = coefficient @ prefix @ coefficient.T + sum(b @ r_all @ b.T for b in blocks)
        full.append(float(np.trace(pf)))
        gap.append(np.linalg.eigvalsh(pt-pf))
        common.append(float(np.trace(v0)))
    return (np.mean(full), np.asarray(gap), np.mean(common), np.linalg.det(g),
            np.linalg.eigvals(a), first_innovation)


def positive_integral_cost(radius, floor, gaps):
    """Independent nonnegative tail-integral evaluation of the moment."""
    def cost_per_stage(lam):
        lo, hi = lam
        def angular(angle):
            variance = lo+(hi-lo)*np.sin(angle)**2
            mass = quad(lambda u: 2*u*np.exp(-((radius+u)**2)/(2*variance)), 0, np.inf,
                        epsabs=2e-13, epsrel=2e-11)[0]
            return mass
        return (2/np.pi)*quad(angular, 0, np.pi/2, epsabs=2e-13, epsrel=2e-11)[0]
    return float(floor+np.mean([cost_per_stage(x) for x in gaps]))


def recount(base, name, record, horizon):
    methods = load(base / FRONTIER / "results/comparison.json")["models"][name]["method_table"]
    positions = {x["name"]:j for j, x in enumerate(methods)}
    counts = {}
    with np.load(base / FRONTIER / f"results/{name}_paired.npz", allow_pickle=False) as data:
        chosen = np.flatnonzero(
            (data["unknown_start__onset"] == 7) & (data["unknown_start__stop"] == 17)
        )
        alarms = data["unknown_start__first_alarm"][chosen]
        danger = data["unknown_start__first_danger"][chosen]
        base_column = positions["V1a"]
        for label in ("V1a", *POINTS):
            j = positions[label]
            rows = {}
            for limit, end in (("first", 7), ("third", 9)):
                flag = (danger[:,j] <= end) & (danger[:,j] < alarms)
                assert int(flag.sum()) == record[label][limit]["events"]
                assert len(chosen) == record[label][limit]["paths"]
                if label != "V1a":
                    comparator = (danger[:,base_column] <= end) & (danger[:,base_column] < alarms)
                    counts_pair = [int(np.count_nonzero(comparator & ~flag)),
                                   int(np.count_nonzero(flag & ~comparator))]
                    assert counts_pair == record[label][limit]["V1a_minus_saturation"]["discordant"]
                rows[limit] = int(flag.sum())
            rows["finite_same_round_pairs"] = int(np.count_nonzero((danger[:,j] == alarms) & (alarms < horizon)))
            counts[label] = rows
    return {"selected_paths":len(chosen), "method_event_counts":counts}


def evaluate(base):
    verified = verify(base)
    data, spec = load(base / MODEL), load(base / PROTOCOL)
    horizon = spec["horizon"]
    assert (horizon, spec["evaluation_onset"], spec["template_duration"], spec["danger_squared"],
            spec["coded_energy_reference"], spec["event_false_alarm"]) == (32, 7, 10, .1, 10, .05)
    reference = load(base / CORE / "results/results.json")["models"]
    saved = load(base / DIRECT / "results/results.json")["models"]
    rows = {}
    for name in ("unstable", "stable"):
        floor, gaps, v0, det_g, spectrum, first_innovation = independent_moments(data, spec, name)
        assert det_g != 0
        assert np.min(gaps) > 0
        original = reference[name]
        with np.load(base / STORED_INPUT, allow_pickle=False) as archive:
            first = archive[f"{name}__original_Gaussian__ordinary__attack"][spec["evaluation_onset"]]
        innovation, cross = first_innovation
        solved = np.linalg.solve(innovation, first)
        width = float(first @ solved)
        gain_norm_sq = float(np.linalg.norm(cross @ solved)**2)
        envelope = -8*np.log(spec["event_false_alarm"])
        floor_width = spec["watermark"]["lambda"]**2*width
        danger_ceiling = envelope*gain_norm_sq/width
        assert floor_width < envelope and danger_ceiling > spec["danger_squared"]
        reachability = original["parametric_reachability"]
        np.testing.assert_allclose(floor_width, np.mean(reachability["minimum_base_whitened_square"]),
                                   rtol=1e-11)
        np.testing.assert_allclose(danger_ceiling,
                                   np.mean(reachability["sufficient_strict_inequality_D_ceiling"]),
                                   rtol=1e-11)
        low, high = original["v0_mse"]
        assert low-1e-11 <= v0 <= high+1e-11
        rstar = np.sqrt(.1)/2
        jstar = positive_integral_cost(rstar, floor, gaps)
        lo, hi = original["critical"]["mse_interval"]
        assert lo-1e-11 <= jstar <= hi+1e-11
        op = {}
        for label, point in original["frozen_operating_points"].items():
            obtained = positive_integral_cost(point["radius"], floor, gaps)
            lo, hi = point["mse_interval"]
            assert lo-1e-11 <= obtained <= hi+1e-11
            op[label] = {"radius": point["radius"], "ordinary_float_audit": obtained,
                         "absolute_diff_from_saved_interval_midpoint": abs(obtained-(lo+hi)/2)}
        old = saved[name]["numerical_integral"]
        arrays = np.asarray([x["risk_by_method_by_age"] for x in old["replications"]["nodes_power_10"]])
        qmc_means = arrays.mean(axis=0)
        bounds = original["certified_first_packet"]["bounds"]
        consistent = {}
        for label, pair in bounds.items():
            mean = float(qmc_means[old["methods"].index(label),0])
            assert pair["joint_risk_lower"] < mean < pair["joint_risk_upper"]
            consistent[label] = mean
        rows[name] = {"v0_batched_covariance_mean": v0, "critical_mse_batched_covariance": jstar,
                      "critical_pct_check_only": 100*(jstar/v0-1),
                      "min_conditional_gap_eigenvalue": float(np.min(gaps)),
                      "local_common_g_det": float(det_g),
                      "plant_spectral_radius": float(np.max(np.abs(spectrum))),
                      "first_packet_bounds_from_batched_moments": {
                          "minimum_base_whitened_square": floor_width,
                          "sufficient_strict_inequality_D_ceiling": danger_ceiling,
                          "innovation_min_eigenvalue": float(np.linalg.eigvalsh(innovation)[0]),
                      },
                      "frozen_cost_points":op,
                      "saved_sobol_point_estimates_inside_first_packet_enclosures":consistent,
                      "holdout_recount":recount(base,name,original["archived_risk_intervals"],horizon)}
        print(f"{name}: {100*(jstar/v0-1):.12f}% independent check; "
              f"{rows[name]['holdout_recount']['selected_paths']} archived paths", flush=True)
    assert verify(base) == verified
    return {"baseline":BASELINE,"verified_inputs":verified,"new_synthetic_paths":0,"models":rows,
            "notes":"Batch covariance / positive tail integrals are independent floating-point checks."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=HERE.parent)
    opts = parser.parse_args()
    report = evaluate(opts.baseline.resolve())
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    (output / "independent_checks.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n"
    )
