"""Independent audit for the frozen B2 attack replay.

The verifier deliberately does not import ``b2_attack`` or any of its replay
helpers.  It reconstructs the detector score and common-G update directly,
then compares a small prefix of every archived condition with the saved
machine-readable result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paper1_sscd_commonrandom_20261122"))
from watson_crn import beta_from_common, make_common_draws, watermark_attack_from_beta

import b1_health as health
import b1_mode_models as modes


REGISTRATION = HERE / "B2_REGISTRATION.json"
REGISTRATION_HASH = HERE / "B2_REGISTRATION_SHA256.txt"
POLICY = HERE / "b1_health_results" / "policy_lock.json"
OUTPUT = HERE / "b2_attack_results"
Q_GRID = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
BITS = 1 << np.arange(7)
N = 2048
T = 150
START, STOP = 50, 110
PREFIX = 4
FIXTURE_VECTOR = np.asarray([6.0, 0.0])
METRICS = (
    "missing_window_mse", "missing_post20_mse", "missing_tail_mse",
    "full_window_mse", "network_window_mse", "network_post20_mse",
    "target_alarm_delay", "target_alarm_run_max",
    "target_attack_accept_fraction", "target_window_accept_fraction",
    "accepted_target_forcing",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(scene_index: int, fixture_index: int, watermark_index: int,
         stage: int, batch: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([
        20261124, 2207, scene_index, fixture_index,
        watermark_index, stage, batch,
    ])


def attack_template(fixture: str) -> np.ndarray:
    active = np.zeros(60, dtype=bool)
    if fixture == "continuous":
        active[:] = True
    elif fixture == "burst":
        active[:] = (np.arange(60) % 10) < 6
    else:
        raise ValueError(fixture)
    template = np.zeros((60, 7, 2), dtype=float)
    template[:, 6] = active[:, None] * FIXTURE_VECTOR
    return template


def active_attack_ticks(fixture: str) -> np.ndarray:
    active = np.zeros(60, dtype=bool)
    if fixture == "continuous":
        active[:] = True
    elif fixture == "burst":
        active[:] = (np.arange(60) % 10) < 6
    else:
        raise ValueError(fixture)
    return active


def direct_delta(count: int, watermark: str, fixture: str, scene_index: int,
                 fixture_index: int, batch: int) -> np.ndarray:
    watermark_index = 0 if watermark == "negative" else 1
    theta = -1.3152369768257723 if watermark == "negative" else 0.0
    parent = seed(scene_index, fixture_index, watermark_index, 2, batch)
    child = parent.spawn(3)
    draws = make_common_draws(
        count, 60, 7,
        int(child[0].generate_state(1)[0]),
        int(child[1].generate_state(1)[0]),
        int(child[2].generate_state(1)[0]),
    )
    beta = beta_from_common(draws, theta)
    return watermark_attack_from_beta(
        attack_template(fixture), beta, T, START)


def direct_scores(model, paths, delta: np.ndarray) -> np.ndarray:
    packets = paths.y[:, :, model.links[:, 1]].copy()
    packets[:, 0] = 0.0
    packets += delta
    windows = np.lib.stride_tricks.sliding_window_view(
        packets, 5, axis=1)
    windows = windows.transpose(0, 1, 2, 4, 3).reshape(
        len(paths.x), T - 4, len(model.links), 10)
    residual = np.einsum("edm,btem->bted", model.Z, windows)
    out = np.zeros((len(paths.x), T, len(model.links)))
    current = residual[:, 1:]
    out[:, 5:] = 0.5 * np.einsum(
        "btei,eij,btej->bte", current, model.precision, current,
        optimize=True)
    return out


def matrix_vector(matrix, vector):
    return np.einsum("...ij,...j->...i", matrix, vector)


def q_index(q: float) -> int:
    matches = np.flatnonzero(np.isclose(Q_GRID, q, atol=1e-12, rtol=0))
    if len(matches) != 1:
        raise ValueError(f"unregistered weight {q}")
    return int(matches[0])


def direct_metrics(model, paths, delta, masks, weights, fixture):
    count = len(paths.x)
    states = {name: np.zeros((count, 50, 4)) for name in weights}
    target_states = {name: np.zeros((count, 4)) for name in weights}
    sums = {name: np.zeros((count, 10)) for name in weights}
    raw = np.einsum("ij,btjm->btim", model.adjacency, paths.y)
    target_sender = int(model.links[6, 1])
    for tick in range(1, T):
        bits = (masks[:, tick, None] & BITS) != 0
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]])
        target_alarm = bits[:, 6]
        packet_delta = delta[:, tick, 6]
        received[:, 26] += packet_delta
        for name, q in weights.items():
            qpos = q_index(q)
            base = (
                matrix_vector(model.base["F"][masks[:, tick]], states[name])
                + matrix_vector(model.base["K"][masks[:, tick]], y)
                + matrix_vector(model.base["G"][masks[:, tick]], received)
            )
            effective_q = np.where(target_alarm, q, 1.0)
            packet = received[:, 26] + (effective_q - 1.0)[:, None] * (
                y[:, target_sender] + packet_delta)
            alarm_index = target_alarm.astype(int)
            target = (
                matrix_vector(model.target["F"][alarm_index, qpos],
                              target_states[name])
                + matrix_vector(model.target["K"][alarm_index, qpos],
                                y[:, 26])
                + matrix_vector(model.target["G"][alarm_index, qpos],
                                packet)
            )
            states[name] = base
            states[name][:, 26] = target
            target_states[name] = target
            error = (states[name] - paths.x[:, tick, None, :]) ** 2
            if START <= tick < STOP:
                sums[name][:, 0] += error[:, 26, 2:].sum(axis=-1)
                sums[name][:, 3] += error[:, 26].sum(axis=-1)
                sums[name][:, 4] += error.sum(axis=(1, 2)) / 50.0
                effective_delta = effective_q[:, None] * packet_delta
                sums[name][:, 9] += np.sum(
                    matrix_vector(model.target["G"][alarm_index, qpos],
                                  effective_delta) ** 2, axis=-1)
            if 20 <= tick < T:
                sums[name][:, 1] += error[:, 26, 2:].sum(axis=-1)
                sums[name][:, 5] += error.sum(axis=(1, 2)) / 50.0
            if STOP <= tick < T:
                sums[name][:, 2] += error[:, 26, 2:].sum(axis=-1)
    alarm = ((masks[:, :, None] & BITS[None, None, :]) != 0)[:, :, 6]
    delay = np.where(alarm[:, START:STOP].any(axis=1),
                     np.argmax(alarm[:, START:STOP], axis=1), -1)
    run = np.zeros(count, dtype=int)
    peak = np.zeros(count, dtype=int)
    for column in alarm[:, 1:].T:
        run = (run + 1) * column
        peak = np.maximum(peak, run)
    active = active_attack_ticks(fixture)
    attack_accept = (~alarm[:, START:STOP][:, active]).mean(axis=1)
    window_accept = (~alarm[:, START:STOP]).mean(axis=1)
    result = {}
    for name in weights:
        result[name] = {
            "missing_window_mse": sums[name][:, 0] / 60.0,
            "missing_post20_mse": sums[name][:, 1] / 130.0,
            "missing_tail_mse": sums[name][:, 2] / 40.0,
            "full_window_mse": sums[name][:, 3] / 60.0,
            "network_window_mse": sums[name][:, 4] / 60.0,
            "network_post20_mse": sums[name][:, 5] / 130.0,
            "target_alarm_delay": delay.astype(float),
            "target_alarm_run_max": peak.astype(float),
            "target_attack_accept_fraction": attack_accept.astype(float),
            "target_window_accept_fraction": window_accept.astype(float),
            "accepted_target_forcing": sums[name][:, 9] / 60.0,
        }
    return result


def expected_weights(scene_index: int, policy: dict) -> dict[str, float]:
    row = policy["rows"][scene_index]
    return {"A": float(row["A"]), "BD": float(row["B"]),
            "C": float(row["C"])}


def check_archive(path: Path, scene_index: int, fixture: str,
                  watermark: str, model, policy, gains) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        expected = {"masks", "threshold"}
        for strategy in ("A", "BD", "C"):
            expected.update(f"{strategy}__{metric}" for metric in METRICS)
        if keys != expected:
            raise AssertionError(f"{path.name}: archive keys differ")
        masks = archive["masks"]
        if masks.shape != (N, T) or masks.dtype.kind not in "iu":
            raise AssertionError(f"{path.name}: invalid mask array")
        if np.any(masks >= 128):
            raise AssertionError(f"{path.name}: mask has an unknown bit")
        scene_file = health.OUT / f"scene_{scene_index:02}.npz"
        with np.load(scene_file, allow_pickle=False) as scene:
            if not np.isclose(float(archive["threshold"]),
                              float(scene["threshold"]), atol=0, rtol=0):
                raise AssertionError(f"{path.name}: threshold mismatch")
        for strategy in ("A", "BD", "C"):
            for metric in METRICS:
                values = archive[f"{strategy}__{metric}"]
                if values.shape != (N,) or not np.isfinite(values).all():
                    raise AssertionError(f"{path.name}: invalid {strategy}/{metric}")
        prefix_count = PREFIX
        # The registered runner generates batches of 64. Reconstruct a full
        # batch before slicing because a vectorized horizon simulation consumes
        # random numbers in a batch-size-dependent order across time.
        paths_batch = health.physical(
            model, 64,
            health_seed(scene_index, fixture, watermark, 0))
        paths = type(paths_batch)(x=paths_batch.x[:prefix_count],
                                  y=paths_batch.y[:prefix_count])
        delta_batch = direct_delta(64, watermark, fixture, scene_index,
                                   0 if fixture == "continuous" else 1, 0)
        delta = delta_batch[:prefix_count]
        scores = direct_scores(model, paths, delta)
        masks_direct = ((scores > float(archive["threshold"])) @ BITS).astype(np.uint8)
        np.testing.assert_array_equal(masks_direct, masks[:PREFIX])
        direct = direct_metrics(model, paths, delta, masks_direct,
                                expected_weights(scene_index, policy), fixture)
        for strategy in ("A", "BD", "C"):
            for metric in METRICS:
                np.testing.assert_allclose(
                    direct[strategy][metric],
                    archive[f"{strategy}__{metric}"][:PREFIX],
                    rtol=2e-11, atol=2e-10,
                    err_msg=f"{path.name}: {strategy}/{metric}")
    return {"file": path.name, "paths_checked": PREFIX,
            "strategies": ["A", "BD", "C"]}


def health_seed(scene_index: int, fixture: str, watermark: str, batch: int):
    fixture_index = 0 if fixture == "continuous" else 1
    watermark_index = 0 if watermark == "negative" else 1
    return seed(scene_index, fixture_index, watermark_index, 2, batch)


def main() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if digest(REGISTRATION) != REGISTRATION_HASH.read_text(encoding="utf-8").split()[0]:
        raise AssertionError("B2 registration hash mismatch")
    if registration["path_count_per_scenario_fixture_watermark"] != N:
        raise AssertionError("registered path count changed")
    if registration["attack_window"] != [START, STOP]:
        raise AssertionError("registered attack window changed")
    if registration["strategies"].keys() != {"A", "BD", "C"}:
        raise AssertionError("strategy set changed")
    b1_run = json.loads((HERE / "B1_HEALTH_RUN.json").read_text(encoding="utf-8"))
    if registration["seeds"]["namespace"] == b1_run["seed_namespace"]:
        raise AssertionError("B2 random stream namespace overlaps B1")
    for row in policy["rows"]:
        if float(row["B"]) != float(row["D"]):
            raise AssertionError("B and D are not equivalent in the frozen policy table")
        if any(float(row[name]) not in Q_GRID for name in ("A", "B", "C")):
            raise AssertionError("policy lock contains a weight outside the registered grid")
    if not OUTPUT.exists():
        raise FileNotFoundError(OUTPUT)
    files = []
    for scene_index, spec in enumerate(health.scenarios()):
        dynamics, topology, target, watermark = spec
        with np.load(modes.OUTPUT / "mode_tables.npz") as gains:
            model = health.setup(spec, gains)
            if tuple(model.links[6]) != (26, target):
                raise AssertionError(f"scene {scene_index}: target link mismatch")
            for fixture in ("continuous", "burst"):
                filename = f"scene_{scene_index:02}_{fixture}_{watermark}.npz"
                path = OUTPUT / filename
                if not path.exists():
                    raise FileNotFoundError(path)
                files.append(check_archive(path, scene_index, fixture,
                                           watermark, model, policy, gains))
    expected_names = {f"scene_{i:02}_{fixture}_{health.scenarios()[i][3]}.npz"
                      for i in range(12) for fixture in ("continuous", "burst")}
    actual_names = {p.name for p in OUTPUT.glob("scene_*.npz")}
    if actual_names != expected_names:
        raise AssertionError("unexpected B2 output files")
    result = {"status": "PASS", "archive_count": len(files),
              "independent_prefix_paths_per_archive": PREFIX,
              "checks": files,
              "registration_sha256": digest(REGISTRATION),
              "policy_sha256": digest(POLICY)}
    (OUTPUT / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
