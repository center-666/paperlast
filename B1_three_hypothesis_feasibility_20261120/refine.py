"""Higher-accuracy review of the nine predeclared refinement histories.

Triggered by coarse/refined integration discrepancies. No attack/model/policy
parameter is tuned. Reuses the same coarse-selected policy vectors.
"""
import json
from pathlib import Path
import numpy as np

from experiment import (BASE, HERE, Bank, attack_templates, frozen_digest, integrate,
                        model_set, observations, radial_angle_grid, read, write)


def main():
    reg = read(HERE / "registration.json")
    models = model_set(reg)
    coarse = read(HERE / "results/histories.json")
    records = []
    hashes = frozen_digest()
    with np.load(BASE / "attack_gain_switch_audit_20261115/screen_arrays.npz", allow_pickle=False) as archive:
        for name, model in models.items():
            attacks, _ = attack_templates(model, reg)
            _, y, matrices = observations(model, reg, 16, archive)
            for h in range(3):
                bank = Bank(model, reg, attacks)
                for k in range(8):
                    state = bank.state()
                    z = np.r_[y[k, 1], y[k, 2], y[k, 0] + matrices[k] @ attacks[h, k]]
                    bank.update(state, z, matrices[k])
                original = next(r for r in coarse if r["model"] == name and r["path_id"] == 16
                                and r["true_h"] == h and r["tick"] == 8)
                v = np.array(original["vectors"]["information_health_0.02"])
                stage = []
                for radial, angle, normal in ((16, 64, 31), (24, 96, 41)):
                    stat = integrate(bank, bank.state(), radial_angle_grid(radial, angle, normal))
                    stage.append(dict(grid=[radial, angle, normal], info_gain=float(np.array(stat["gI"]) @ v),
                        conditional_actual_bayes_change=float(np.array(stat["g_bayes"])[h] @ v),
                        conditional_actual_common_change=float(np.array(stat["g_common"])[h] @ v),
                        mixture_identity_sensitivity_error=float(np.linalg.norm(stat["variance_identity_sensitivity_error"])),
                        gI=stat["gI"]))
                records.append(dict(model=name, true_h=h, chosen_coarse_v=v.tolist(), stages=stage))
                print(name, h, stage[-1], flush=True)
    assert hashes == frozen_digest()
    write(HERE / "results/high_accuracy_refinement.json", records)


if __name__ == "__main__":
    main()
