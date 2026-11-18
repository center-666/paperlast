"""Compare receivers on existing joint observation/decoder replays."""
import argparse
from pathlib import Path

from shared import DEFAULT_BASELINE, HERE, load_json, load_prepared, setup, verify_registration, write_json
from receiver_experiment import METHODS, assess, exhaust_ledger, stack
from run_validation import gain_on_nominal_grid
from weight_experiment import archived_cases, make_parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    reg = load_json(HERE / "registration.json")
    verify_registration(reg, args.baseline)
    common, settings, spec, _ = setup(args.baseline)
    spec = dict(spec, samples={"physical_indices": reg["archive_pool"]})
    reports = {}
    for name in reg["samples"]:
        p = load_prepared(args.baseline, name, common, settings)
        parity = make_parity(common, p, spec)
        scale = gain_on_nominal_grid(p, spec)
        cases, _ = archived_cases(args.baseline, common, p, spec)
        start = [spec["evaluation_onset"]]*len(cases["healthy"])
        normal = stack(cases["healthy"], start, p, common, spec, scale, parity)
        rows = {"healthy": assess(normal, normal, start, spec, False, False)}
        for profile in ("A1_H_mismatch", "A1_C_mismatch"):
            data = stack(cases[profile], start, p, common, spec, scale, parity)
            rows[profile] = assess(data, normal, start, spec, False, True)
            rows[profile]["V12a_ledger"] = exhaust_ledger(data, p)
        reports[name] = rows
        print(name, "historical joint receiver replay verified", flush=True)
    write_json(HERE / "results/archived_joint.json", {
        "evidence": "Development-only paired replay preserving the original joint observations and watermarks.",
        "physical_indices": reg["archive_pool"], "methods": METHODS, "models": reports,
    })


if __name__ == "__main__":
    main()
