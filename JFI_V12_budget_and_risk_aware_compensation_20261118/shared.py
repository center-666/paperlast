"""Read-only interface to the frozen models and receivers."""
from hashlib import sha256

from budget_policy import HERE, V11
from weight_experiment import (DEFAULT_BASELINE, REFERENCE, PreparedModel, V1,  # noqa: F401
                               load_json, setup, write_json)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_registration(registration, baseline):
    for name, expected in registration["frozen_local_hashes"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Frozen local input differs: {name}")
    reference = registration["archive_seed_map"]
    if digest(baseline / reference["path"]) != reference["sha256"]:
        raise ValueError("The original comparison seed map changed.")
    original = load_json(baseline / reference["path"])
    older = load_json(V11 / "phase2_registration.json")
    used = {seed for model in original.values() for key in ("physical", "watermark")
            for seed in model[key]}
    for model in older["samples"].values():
        used.update(model["physical"])
        used.update(model["watermark"])
        used.add(model["archive_draw_seed"])
    new = [seed for model in registration["samples"].values() for key in ("physical", "watermark")
           for seed in model[key]]
    new += [value for model in registration["samples"].values()
            for value in (model["profile_seed"], model["nominal_certificate_seed"])]
    if len(new) != len(set(new)) or used.intersection(new):
        raise ValueError("A held-out seed is duplicated or has already been used.")


def load_prepared(baseline, name, common, settings):
    model_path = baseline / REFERENCE / "inputs" / "five_node_aggregate_model.json"
    return PreparedModel(common.load_model(name, model_path), settings)
