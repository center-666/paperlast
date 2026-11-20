"""Causal regional follow-up. Policy observes its own history, not true h."""
from time import perf_counter
import numpy as np

from experiment import (BASE, HERE, Bank, attack_templates, choose, draw_region,
                        frozen_digest, integrate, model_set, radial_angle_grid, read, write)


def sampled_matrix(original, v):
    values, vectors = np.linalg.eigh(original)
    X = values[1] - .5
    axis = vectors[:, 1]
    original_angle = np.arctan2(axis[1], axis[0]) % np.pi
    if np.linalg.norm(v) == 0:
        return original.copy()
    target = original_angle / np.pi
    low, high = 0., np.pi
    for _ in range(50):
        mid = (low + high) / 2
        cdf = mid / np.pi + (v[0] * np.sin(4 * mid) + v[1] * (1 - np.cos(4 * mid))) / (4 * np.pi)
        if cdf < target:
            low = mid
        else:
            high = mid
    theta = (low + high) / 2
    u = np.array([np.cos(theta), np.sin(theta)])
    return .5 * np.eye(2) + X * np.outer(u, u)


def main():
    reg, spec = read(HERE / "registration.json"), read(HERE / "closed_loop_registration.json")
    model = model_set(reg)[spec["model"]]
    attacks, _ = attack_templates(model, reg)
    fixed = np.array(read(HERE / "results/summary.json")["one_step"][model.name]["development_fixed_v"])
    q = reg["quadrature"]
    grid = radial_angle_grid(q["radial"], q["angles"], q["normal_1d"])
    before = frozen_digest()
    paths, controls = [], []
    for path_id in spec["physical_path_ids"]:
        truth, y, matrices = draw_region(model, reg, path_id)
        for policy in spec["policies"]:
            for h in range(3):
                bank = Bank(model, reg, attacks)
                outputs = {name: [] for name in ("bayes", "common", "raw")}
                coding_energy = []
                for k in range(reg["horizon"]):
                    state = bank.state()
                    v = np.zeros(2)
                    elapsed = 0.
                    if k in spec["active_design_ticks"]:
                        if policy == "development_fixed":
                            v = fixed
                        elif policy.startswith("adaptive"):
                            started = perf_counter()
                            stats = integrate(bank, state, grid)
                            budget = float(policy.rsplit("_", 1)[-1])
                            v = choose(stats["gA"], stats["gI"], stats["g_common"][0], budget * stats["common"][0],
                                       reg["watermark"]["rho"], reg["minimum_predicted_information_gain"])
                            elapsed = perf_counter() - started
                        controls.append(dict(path_id=path_id, policy=policy, true_h_for_scoring=h,
                                             tick=k, v=v.tolist(), seconds=elapsed))
                    matrix = sampled_matrix(matrices[k], v)
                    # Coding resource includes healthy sensor signal, before attack.
                    coding_energy.append(float(np.sum(np.linalg.solve(matrix, y[k, 0]) ** 2)))
                    z = np.r_[y[k, 1], y[k, 2], y[k, 0] + matrix @ attacks[h, k]]
                    output = bank.update(state, z, matrix)
                    for method in outputs:
                        outputs[method].append(output[method])
                onset = reg["onsets"][max(0, h - 1)]
                for method, values in outputs.items():
                    errors = (np.array(values) - truth) ** 2
                    paths.append(dict(path_id=path_id, policy=policy, true_h=h, method=method,
                        full_mse=float(np.mean(np.sum(errors, axis=1))),
                        support_mse=float(np.mean(np.sum(errors[onset:onset + reg["duration"]], axis=1))),
                        missing_mode_mse=float(np.mean(errors[:, 1])),
                        mean_coding_energy=float(np.mean(coding_energy))))
        print("causal path", path_id, "completed", flush=True)
    summary = {}
    for method in ("common", "bayes", "raw"):
        summary[method] = {}
        for h in range(3):
            summary[method][str(h)] = {}
            baseline = [r for r in paths if r["method"] == method and r["true_h"] == h and r["policy"] == "uniform"]
            for policy in spec["policies"]:
                chosen = [r for r in paths if r["method"] == method and r["true_h"] == h and r["policy"] == policy]
                row = {}
                for key in ("full_mse", "support_mse", "missing_mode_mse", "mean_coding_energy"):
                    a = np.array([r[key] for r in chosen])
                    b = np.array([r[key] for r in baseline])
                    row[key] = float(np.mean(a))
                    row[key + "_change_vs_uniform"] = float(np.mean(a - b))
                    row[key + "_change_paired_se"] = float(np.std(a - b, ddof=1) / np.sqrt(len(a)))
                summary[method][str(h)][policy] = row
    assert before == frozen_digest()
    write(HERE / "results/causal_paths.json", paths)
    write(HERE / "results/causal_controls.json", controls)
    write(HERE / "results/causal_summary.json", dict(results=summary,
        active_policy_seconds_median=float(np.median([r["seconds"] for r in controls if r["seconds"] > 0])),
        status="Exploratory known-waveform finite-bank causal run; no unknown-parameter or all-onset guarantee"))
    print("causal summary written")


if __name__ == "__main__":
    main()
