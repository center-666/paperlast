"""Paper 1: fixed-watermark causal gate-swap ablation on the recovered 50-node common-G system.

Place this file at repo root under paper1_sscd_next_validation_20261122/, alongside
paper1_common_g50_gain_acceptance_20261121 and paper1_50node_tv_gate_theory_20261121.
It does not alter the estimator, six A0 links, parity score, or the gate rule.

No computations are started without --execute. Full evaluation needs exact frozen
repository contents; without them --dry-run checks only local script readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

THETA = {"iid": 0.0, "negative": -1.315236976827179}
ATTACK_WINDOW = (50, 110)
ELIGIBLE_WINDOW = (57, 110)
EVAL_WINDOW = (20, 150)
RECEIVERS = np.array([0, 2, 14, 15, 17], dtype=int)


def metric_paths(world, physical):
    """Return arrays with exactly one independent physical path per row."""
    squared = np.sum((world.post - physical.x[:, :, None])**2, axis=-1)
    net = squared.mean(axis=2)
    affected = squared[:, :, RECEIVERS].mean(axis=2)
    return {
        "network_post20": net[:, 20:].mean(axis=1),
        "network_attack": net[:, 50:110].mean(axis=1),
        "network_post_attack": net[:, 110:].mean(axis=1),
        "affected_post20": affected[:, 20:].mean(axis=1),
        "affected_attack": affected[:, 50:110].mean(axis=1),
        "affected_post_attack": affected[:, 110:].mean(axis=1),
    }


def describe(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    return {"mean": float(x.mean()), "paired_path_se": se, "paths": n,
            "normal_approx_95ci": [float(x.mean()-1.96*se), float(x.mean()+1.96*se)]}


def shapley_attribution(L_ii, L_ni, L_in, L_nn):
    """The two-factor 2x2 risk Shapley decomposition, per path and metric.

    First index: injected watermark attack family (i or negative n).
    Second index: gate schedule (computed from i or negative n).
    This is an intervention on the already-defined gate, NOT an independent
    cryptographic attacker or a new estimator.
    """
    direct = .5 * ((L_ni - L_ii) + (L_nn - L_in))
    gate = .5 * ((L_in - L_ii) + (L_nn - L_ni))
    total = L_nn - L_ii
    np.testing.assert_allclose(total, direct + gate, rtol=1e-11, atol=1e-11)
    return {"total": total, "direct_watermark_at_fixed_gate": direct,
            "gate_schedule_effect": gate}


def score_metrics(scores, threshold):
    alarm = np.asarray(scores) > threshold
    elig = alarm[:, ELIGIBLE_WINDOW[0]:ELIGIBLE_WINDOW[1]]
    if not elig.shape[1]:
        raise RuntimeError("Empty eligible detection window")
    # Censor undetected links at window end: no post-selection bias.
    first = np.argmax(elig, axis=1)
    any_link = elig.any(axis=1)
    delay = np.where(any_link, first, elig.shape[1]).mean(axis=1)
    return {"eligible_tpr": elig.mean(axis=(1, 2)),
            "restricted_mean_first_alarm_lag": delay,
            "path_any_alarm": elig.any(axis=(1, 2)).astype(float)}


def run(*, n_calibration, n_evaluation, seed):
    for folder in ["paper1_common_g50_gain_acceptance_20261121",
                   "paper1_50node_tv_gate_theory_20261121",
                   "paper1_SSCD_review_20261121"]:
        path = ROOT / folder
        if not path.is_dir():
            raise FileNotFoundError(f"Expected frozen sibling repository directory: {path}")
        sys.path.insert(0, str(path))

    from run_common_g_50 import load_common_g, physical_paths, run_common_g
    from replay import attack_watermarks
    from calibration_rule import path_fwer_cutoff

    out = {"kind": "paper1_common_G_50node_counterfactual_gate_ablation",
           "source": "frozen 99ee54f or later compatible repository contents",
           "calibration_paths": n_calibration,
           "evaluation_paths": n_evaluation,
           "seed": seed,
           "estimator": "original lookup common-G, unchanged",
           "attack": "original six registered source/receiver FDI links, k in [50,110)",
           "gate": "raw parity s4 four-residual plugin-KL, source unpruned windows",
           "models": {}}
    path_archive = {}

    for model_i, label in enumerate(["unstable", "stable"]):
        model = load_common_g(label)
        calibration_physical = physical_paths(model, n_calibration, seed + model_i*100_000)
        cal_world = run_common_g(model, calibration_physical, gate_on=False)
        calibration_max = np.max(cal_world.scores[:, 4:], axis=(1, 2))
        cutoff = path_fwer_cutoff(calibration_max, .05)
        threshold = float(cutoff["threshold"])
        physical = physical_paths(model, n_evaluation, seed + model_i*100_000 + 1)

        # Independent seeded attack-watermark laws, matched physical paths,
        # identical within-law initial directions/radii and identical A0 attack.
        deltas = {}
        for policy_i, (policy, theta) in enumerate(THETA.items()):
            delta, _ = attack_watermarks(
                model, n_evaluation, theta,
                seed + model_i*100_000 + 10_000,
                seed + model_i*100_000 + 20_000,
                seed + model_i*100_000 + 30_000 + 100*policy_i,
            )
            deltas[policy] = delta

        # Natural worlds and policy-determined raw detection gates.
        natural = {
            p: run_common_g(model, physical, deltas[p], gate_on=True, threshold=threshold)
            for p in THETA
        }
        g_i = natural["iid"].gates.copy()
        g_n = natural["negative"].gates.copy()
        # Four frozen-gate counterfactuals: loss(delta_policy, gate_policy).
        worlds = {
            "ii": natural["iid"],
            "nn": natural["negative"],
            "ni": run_common_g(model, physical, deltas["negative"],
                               gate_override=g_i, threshold=threshold),
            "in": run_common_g(model, physical, deltas["iid"],
                               gate_override=g_n, threshold=threshold),
        }
        losses = {key: metric_paths(world, physical) for key, world in worlds.items()}
        for world_name, metrics in losses.items():
            for metric_name, values in metrics.items():
                path_archive[f"{label}_{world_name}__{metric_name}"] = np.asarray(values, dtype=float)
        for policy, world in natural.items():
            path_archive[f"{label}_{policy}__gates"] = np.asarray(world.gates, dtype=np.uint8)
            path_archive[f"{label}_{policy}__masks"] = np.asarray(world.masks, dtype=np.int64)
            path_archive[f"{label}_{policy}__scores"] = np.asarray(world.scores, dtype=float)
        attribution = {}
        for name in losses["ii"]:
            decomp = shapley_attribution(*(losses[key][name] for key in ["ii", "ni", "in", "nn"]))
            attribution[name] = {component: describe(arr) for component, arr in decomp.items()}

        # Oracle permanently discards six known suspicious edges; it is not
        # available to a defender who knows no candidates.
        permanent_gates = np.zeros((n_evaluation, 150, 6), dtype=bool)
        permanent_gates[:, 1:, :] = True
        static_world = run_common_g(model, physical, deltas["iid"],
                                    gate_override=permanent_gates,
                                    threshold=threshold)
        healthy = run_common_g(model, physical, gate_on=True, threshold=threshold)
        static_losses = metric_paths(static_world, physical)
        natural_metrics = {p: {name: describe(v) for name, v in metric_paths(world,physical).items()}
                           for p,world in natural.items()}
        out["models"][label] = {
            "cutoff": cutoff,
            "healthy_path_any_alarm": describe(healthy.gates[:, 4:].any(axis=(1,2))),
            "natural": natural_metrics,
            "static_known_six": {name: describe(v) for name,v in static_losses.items()},
            "static_minus_negative": {name: describe(static_losses[name] - metric_paths(natural["negative"],physical)[name])
                                      for name in static_losses},
            "detection": {p: {name: describe(v) for name,v in score_metrics(natural[p].scores,threshold).items()}
                          for p in THETA},
            "gate_fraction_attack_window": {p: describe(natural[p].gates[:, 50:110].mean(axis=(1,2))) for p in THETA},
            "shapley_counterfactual_attribution": attribution,
            "limitations": [
                "This is a gate-intervention decomposition for fixed realized attacks and physical paths, not a theorem about all attackers.",
                "The same random calibration cutoff is used for both watermark laws within each dynamics.",
                "Conditional 95% path intervals do not account for new calibration repetitions.",
                "The registered six known attack candidates make static-known-six deletion a strong oracle baseline.",
                "The on-policy SSCD parameter optimization is not repeated on these evaluation paths.",
            ],
        }
        print(label, "threshold",threshold,"total gate-on iid→negative",attribution["network_post20"]["total"]["mean"])

    output_path = HERE / "crossover_results.json"
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",encoding="utf-8")
    path_output = HERE / "crossover_paths.npz"
    np.savez_compressed(path_output, **path_archive)
    source_dir = ROOT / "paper1_common_g50_gain_acceptance_20261121"
    manifest = {
        "source_commit_reference": "99ee54f29d4d471d121ca1b80f6f083f46e7ff03 or compatible later local tree",
        "calibration_paths_per_model": n_calibration,
        "evaluation_paths_per_model": n_evaluation,
        "seed": seed,
        "path_array_file": path_output.name,
        "array_keys": sorted(path_archive),
        "input_hashes": {
            "gain_table.json": hashlib.sha256((source_dir / "gain_table.json").read_bytes()).hexdigest(),
            "physical_configuration.json": hashlib.sha256((source_dir / "physical_configuration.json").read_bytes()).hexdigest(),
            "run_common_g_50.py": hashlib.sha256((source_dir / "run_common_g_50.py").read_bytes()).hexdigest(),
            "causal_gate_ablation.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "gate_array_semantics": "uint8 [paths,time,registered_edges]; scores float [paths,time,registered_edges]; masks int [paths,time]",
        "audit_note": "Arrays permit independent reconstruction of all four world losses and pathwise Shapley identities without rerunning physical generation."
    }
    (HERE / "crossover_path_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Output:", output_path)
    print("Path arrays:", path_output)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",help="Explicitly run physical simulation with frozen repo")
    parser.add_argument("--calibration", type=int, default=512)
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026112317)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({"status":"dry-run", "calibration":args.calibration,"paths":args.paths,
                          "note":"Source-verified 50-node repo is required; no simulation was launched."},indent=2))
        return
    if args.calibration<100 or args.paths<10:
        raise ValueError("Use a meaningful independent calibration/evaluation sample.")
    run(n_calibration=args.calibration,n_evaluation=args.paths,seed=args.seed)


if __name__=="__main__":
    main()
