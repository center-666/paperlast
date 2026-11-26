"""Independent healthy calibration and holdout for H1 and the adapted receiver."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.stats import beta

import audit_frontend as audit
import frontend
import models_quality as quality

HERE = Path(__file__).resolve().parent
OUT = HERE / "frontend_health"
N = 2048
BATCH = 32
FAMILIES = ("H1", "iid_matched", "no_repeat_matched")


@lru_cache(None)
def cached_model(index):
    model = audit.load_model(index)
    return model, frontend.operators(model)


def worker(arguments):
    index, purpose, batch = arguments
    model, ops = cached_model(index)
    root = [20261126, 94017, index, purpose, batch]
    paths = quality.physical(model, BATCH, root)
    uniforms = np.random.default_rng(np.random.SeedSequence([*root, 2])).random(
        (BATCH, 150, 50))
    raw = paths.y[:, :, model.links[:, 1]]
    scores = [quality.frozen.common.h1_score(model, paths)]
    for no_repeat in (False, True):
        keys = frontend.keys(uniforms, no_repeat)[:, :, model.links[:, 1]]
        scores.append(frontend.scores(raw, keys, ops))
    return {
        "path_id": np.arange(batch*BATCH, (batch+1)*BATCH),
        "physical_probe": paths.y[:, 1, 0], "key_probe": uniforms[:, 1, 0],
        "maxima": np.stack([score[:, 5:].max(axis=(1, 2)) for score in scores], axis=1),
    }


def interval(count, size):
    return [0. if count == 0 else float(beta.ppf(.025, count, size-count+1)),
            1. if count == size else float(beta.ppf(.975, count+1, size-count))]


def main(workers):
    OUT.mkdir(exist_ok=True)
    sources = {p.name: quality.sha(p) for p in (
        HERE / "PREEXEC_REGISTRATION.md", HERE / "calibrate_frontend.py",
        HERE / "models_quality.py", HERE / "frontend.py", HERE / "projection.c",
        HERE / "audit_frontend.py",
    )}
    identity = {
        "namespace": [20261126, 94017], "calibration_purpose": 2, "holdout_purpose": 3,
        "paths_each_population_configuration": N, "batch": BATCH,
        "families": FAMILIES, "configurations": audit.configurations(),
        "source_sha256": sources,
        "D2_frozen": "d6d958348ec3bbc73fe42df0a9e9cf8a337a8d78",
        "D1_frozen": "7e32a41b9802438f6645d9048ee46386caa53418",
    }
    identity = json.loads(json.dumps(identity))
    complete = OUT / "RUN.json"
    progress = OUT / "PROGRESS.json"
    state = {
        "identity": identity, "archives": {}, "results": [],
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "workers": workers,
    }
    if complete.exists():
        previous = json.loads(complete.read_text(encoding="utf-8"))
        assert previous["identity"] == identity
        for name, digest in previous["archives"].items():
            assert quality.sha(OUT / name) == digest
        return
    if progress.exists():
        state = json.loads(progress.read_text(encoding="utf-8"))
        assert state["identity"] == identity
        for name, digest in state["archives"].items():
            assert quality.sha(OUT / name) == digest
    frontend.kernel()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, spec in enumerate(audit.configurations()):
            for purpose in (2, 3):
                filename = f"config{index:02}_purpose{purpose}.npz"
                if filename in state["archives"]:
                    continue
                parts = list(pool.map(worker, [(index, purpose, b) for b in range(N//BATCH)]))
                arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                np.testing.assert_array_equal(arrays["path_id"], np.arange(N))
                quality.frozen.atomic_npz(OUT / filename, arrays)
                state["archives"][filename] = quality.sha(OUT / filename)
                quality.frozen.atomic_json(progress, state)
            cal = np.load(OUT / f"config{index:02}_purpose2.npz")
            healthy = np.load(OUT / f"config{index:02}_purpose3.npz")
            assert not np.array_equal(cal["physical_probe"], healthy["physical_probe"])
            assert not np.array_equal(cal["key_probe"], healthy["key_probe"])
            rows = []
            for family, column in zip(FAMILIES, cal["maxima"].T):
                rank = int(np.ceil((N+1)*.95))
                threshold = float(np.sort(column)[rank-1])
                family_index = FAMILIES.index(family)
                alarms = int((healthy["maxima"][:, family_index] > threshold).sum())
                rows.append({"family": family, "threshold": threshold, "order_rank": rank,
                             "alarms": alarms, "paths": N, "FWER": alarms/N,
                             "CP95": interval(alarms, N)})
            cal.close()
            healthy.close()
            if len(state["results"]) == index:
                state["results"].append({"configuration": index, "spec": spec, "rows": rows})
            quality.frozen.atomic_json(progress, state)
            print("calibration", index, spec[0], flush=True)
    quality.frozen.atomic_json(complete, state)
    progress.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
