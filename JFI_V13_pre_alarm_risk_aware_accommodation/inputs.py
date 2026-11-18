"""Frozen receiver inputs and offline replay. Outcomes cannot select disturbances."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V12 = HERE.parent / "JFI_V12_budget_and_risk_aware_compensation_20261118"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V12))

from shared import DEFAULT_BASELINE, V11, digest, load_json, load_prepared, setup, write_json  # noqa: E402
from receiver_experiment import fresh_cases  # noqa: E402
from validate_v11 import paired_packets, replay_bank  # noqa: E402

LEGACY = ("healthy", "A0", "A1_H_mismatch", "A1_C_mismatch", "A0_onset_replay")
FRESH = ("healthy", "unknown_start", "timing_mismatch", "A1_H_replay", "A1_C_replay")
MODELS = ("unstable", "stable")


@dataclass(frozen=True)
class Streams:
    state: np.ndarray
    trusted: np.ndarray
    target: np.ndarray
    decoder: np.ndarray
    offset: np.ndarray
    start: np.ndarray
    stop: np.ndarray

    def take(self, selection):
        return Streams(self.state[selection], self.trusted[selection], self.target[selection],
                       self.decoder[selection], self.offset[selection], self.start[selection],
                       self.stop[selection])

    def __len__(self):
        return len(self.state)


def pack(records, starts, duration):
    as_array = lambda key: np.stack([getattr(row, key) for row in records])
    t = records[0].states.shape[0]
    start = np.asarray(starts, dtype=int)
    return Streams(as_array("states"), np.concatenate((as_array("local"), as_array("healthy")), axis=2),
                   as_array("attacked_target"), as_array("decoders"), as_array("decoded_difference"),
                   start, np.minimum(t, start+duration))


def legacy_cases(base, prepared, spec, common):
    phase96 = load_json(V12 / "registration.json")
    phase48 = load_json(V11 / "phase2_registration.json")
    ordinary, bank = replay_bank(base, prepared.model.name, phase48)
    saved48, _ = paired_packets(spec, phase48, common, prepared.model, ordinary, bank)
    a = {p: pack(rows, [spec["evaluation_onset"]]*len(rows), spec["template_duration"])
         for p, rows in saved48.items() if p in LEGACY}
    saved96, starts, _ = fresh_cases(base, prepared, spec, common, phase96)
    b = {p: pack(saved96[p], starts[p], spec["template_duration"]) for p in LEGACY}
    return a, b


def joint_cases(base, prepared, spec, common):
    from weight_experiment import archived_cases
    indices = load_json(V12 / "registration.json")["archive_pool"]
    old, _ = archived_cases(base, common, prepared, dict(spec, samples={"physical_indices": indices}))
    return {p: pack(old[p], np.full(len(indices), spec["evaluation_onset"]), spec["template_duration"])
            for p in ("healthy", "A1_H_mismatch", "A1_C_mismatch")}


def physical_streams(prepared, spec, rng_physical, rng_watermark, count):
    p = prepared
    u = p.trusted_maps
    H, m, dimension = p.target_maps.shape
    noise = rng_physical.standard_normal((count, dimension))
    trusted = (noise @ u.reshape(-1, dimension).T).reshape(count, H, 2*m)
    target = (noise @ p.target_maps.reshape(-1, dimension).T).reshape(count, H, m)
    states = (noise @ p.state_maps.reshape(-1, dimension).T).reshape(count, H, p.n)
    beta = rng_watermark.standard_normal((count, H, m))*np.sqrt(spec["watermark"]["q"])
    decoder = spec["watermark"]["lambda"]*np.eye(m)+beta[..., None]*beta[..., None, :]
    return states, trusted, target, decoder


def fresh_streams(base, prepared, spec, seed, count, healthy_only=False):
    """Draw physical noise and select only pre-registered fixed impairment replays."""
    phys, water, source = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    x, trusted, target, decoder = physical_streams(prepared, spec, phys, water, count)
    horizon = prepared.settings.horizon
    start = spec["evaluation_onset"]
    length = spec["template_duration"]
    zero = np.zeros_like(target)
    null = np.full(count, horizon)
    samples = {"healthy": Streams(x, trusted, target, decoder, zero, null, null)}
    if healthy_only:
        return samples
    names = load_json(V12 / "registration.json")
    ordinary, bank = replay_bank(base, prepared.model.name, names)
    for case in FRESH[1:]:
        if case in ("unknown_start", "timing_mismatch"):
            starts = source.choice([3, 7, 11], size=count)
            duration = np.full(count, length if case == "unknown_start" else 8)
            coded = np.zeros_like(target)
            for j, (begin, size) in enumerate(zip(starts, duration)):
                coded[j, begin:begin+size] = ordinary[start:start+size]
        else:
            label = "A1_H_mismatch" if case == "A1_H_replay" else "A1_C_mismatch"
            choices = source.integers(len(bank[label]), size=count)
            coded = bank[label][choices]
            starts, duration = np.full(count, start), np.full(count, length)
        if np.max(np.sum(coded**2, axis=(1, 2))) > spec["coded_energy_reference"]+1e-9:
            raise ValueError("Recorded receiver test is outside the registered energy.")
        stop = starts+duration
        times = np.arange(horizon)[None, :]
        support = (times>=starts[:, None]) & (times<stop[:, None])
        np.testing.assert_allclose(coded*(~support)[..., None], 0, atol=1e-12)
        distortion = (decoder @ coded[..., None])[..., 0]
        samples[case] = Streams(x, trusted, target+distortion, decoder, distortion, starts, stop)
    return samples
