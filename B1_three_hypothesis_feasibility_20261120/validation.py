"""Independent batch Gaussian check and frozen V1a regional replay."""
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
import json
from pathlib import Path
import sys

import numpy as np

from experiment import Bank, HERE, BASE, read, write, model_set, attack_templates, draw_region, frozen_digest


def load_evidence():
    name = "frozen_v1a_evidence_validation"
    spec = spec_from_file_location(name, BASE / "JFI_reconstruction_attack_aware_soft_accommodation_V1/accommodation.py")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_batch(evidence, model, reg):
    settings = evidence.Settings(horizon=reg["horizon"], template_duration=reg["duration"],
        coded_energy_reference=10., receiver=1, target=0, healthy=2, no_event_prior=.5,
        nominal_one_step_allowance=.02, uncertainty_standard_deviations=1., event_false_alarm=.05)
    prepared = evidence.PreparedModel(model, settings)
    attacks, _ = attack_templates(model, reg)
    _, ys, Ms = draw_region(model, reg, 16)
    checks = []
    for true_h in range(3):
        bank = Bank(model, reg, attacks)
        joint_records, deltas = [], [[] for _ in range(3)]
        for k in range(10):
            state = bank.state()
            delta = attacks[:, k] @ Ms[k].T
            z = np.r_[ys[k, 1], ys[k, 2], ys[k, 0] + delta[true_h]]
            bank.update(state, z, Ms[k])
            joint_records.append(z)
            for i in range(3):
                deltas[i].append(np.r_[np.zeros(4), delta[i]])
            if k not in (0, 7, 8, 9):
                continue
            maps = np.concatenate([prepared.trusted_maps[:k + 1], prepared.target_maps[:k + 1]], axis=1).reshape(-1, prepared.state_maps.shape[-1])
            covariance = maps @ maps.T
            cross = prepared.state_maps[k] @ maps.T
            gain = np.linalg.solve(covariance, cross.T).T
            recorded = np.array(joint_records).reshape(-1)
            batch_means = np.array([gain @ (recorded - np.array(d).reshape(-1)) for d in deltas])
            batch_P = prepared.state_maps[k] @ prepared.state_maps[k].T - gain @ cross.T
            me = float(np.max(np.abs(batch_means - bank.means)))
            pe = float(np.max(np.abs(batch_P - bank.P)))
            assert me < 3e-10 and pe < 3e-10
            checks.append(dict(h=true_h, tick=k, mean_error=me, covariance_error=pe))
    return prepared, checks


def replay_v1a(evidence, model, reg, prepared):
    from common_g_reference import common_step
    attacks, _ = attack_templates(model, reg)
    records = []
    for path in reg["evaluation_path_ids"]:
        x, y, matrices = draw_region(model, reg, path)
        for h in range(3):
            observer = evidence.OnlineSoftCompensator(prepared, common_step)
            estimates = []
            for k in range(reg["horizon"]):
                target = y[k, 0] + matrices[k] @ attacks[h, k]
                output = observer.step(y[k, 1], y[k, 2], target, matrices[k])
                estimates.append(output.posterior.copy())
            error = (np.asarray(estimates) - x) ** 2
            records.append(dict(path_id=path, h=h, total_mse=float(np.mean(np.sum(error, axis=1))),
                                last_eight_missing_mode_mse=float(np.mean(error[-8:, 1]))))
    return records


def main():
    reg = read(HERE / "registration.json")
    model = model_set(reg)[reg["additional_model"]]
    evidence = load_evidence()
    before = frozen_digest()
    prepared, checks = compare_batch(evidence, model, reg)
    result = dict(batch_gaussian_checks=checks, frozen_hashes_unchanged=before == frozen_digest())
    # Resolve the class interface explicitly before running the frozen policy.
    classes = [name for name in dir(evidence) if isinstance(getattr(evidence, name), type)]
    result["frozen_classes"] = classes
    result["regional_V1a_paths"] = replay_v1a(evidence, model, reg, prepared)
    result["regional_V1a_summary"] = {str(h): {key: float(np.mean([r[key] for r in result["regional_V1a_paths"] if r["h"] == h]))
        for key in ("total_mse", "last_eight_missing_mode_mse")} for h in range(3)}
    assert before == frozen_digest()
    write(HERE / "results/validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
