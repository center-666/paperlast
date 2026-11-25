"""Independent D2-B development screening for history-aware health costs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.stats import beta as beta_distribution

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D1 = Path(os.environ["D2_D1_DIR"]).resolve()
D1_ROOT = D1.parent
sys.path.insert(0, str(D1))
import b1_health as health
import b1_mode_models as modes

sys.path.insert(0, str(D1_ROOT / "paper1_sscd_commonrandom_20261122"))
from watson_crn import beta_from_common, make_common_draws, watermark_attack_from_beta

Q = np.asarray([0., .2, .4, .6, .8, .9, 1.])
BITS = 1 << np.arange(7)
N = 512
FIT = 256
EVAL = 256
EVALUATION = 256
BATCH = 64
T = 150
START, STOP = 50, 110
NAMESPACE = (20261124, 4401, 1)
SMOKE_NAMESPACE = (20261124, 4401, 99)
BOOTSTRAP_REPLICATES = 2048
BOOTSTRAP_SEED_SUFFIX = 991
STRATEGIES = ("R1", "RH_static", "RH_scalar", "RH_dynamic")
OUT = HERE / "b_results_corrected"
PROGRESS = OUT / "B_RUN_PROGRESS.json"
GRAPH_DIR = HERE / "experiment_A" / "graphs"
CONFIG = ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json"
D2_PROTOCOL = HERE / "D2_B_PROTOCOL.md"
D2_MODEL = HERE / "D2_MODEL_CONTRACT.md"
W = np.diag([0., 0., 1., 1.])
W_MISSING = W


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def atomic_npz(path, arrays):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def make_seed(scene, fixture, watermark, stage, batch, namespace=NAMESPACE):
    return np.random.SeedSequence([*namespace, scene, fixture, watermark, stage, batch])


def active_ticks(fixture):
    if fixture == "continuous":
        return np.ones(60, dtype=bool)
    if fixture == "burst":
        return np.arange(60) % 10 < 6
    raise ValueError(fixture)


def active_attack(fixture):
    return active_ticks(fixture)


def concatenate(parts):
    return SimpleNamespace(
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
    )


def make_attack_delta(count, spec, fixture, scene, batch, namespace=NAMESPACE):
    theta = -1.3152369768257723 if spec[3] == "negative" else 0.0
    fixture_index = int(fixture == "burst")
    watermark_index = int(spec[3] == "iid")
    streams = make_seed(scene, fixture_index, watermark_index, 2, batch, namespace).spawn(3)
    draws = make_common_draws(
        count, 60, 7,
        *[int(stream.generate_state(1)[0]) for stream in streams],
    )
    template = np.zeros((60, 7, 2), dtype=float)
    template[:, 6] = active_ticks(fixture)[:, None] * np.asarray([6., 0.])
    return watermark_attack_from_beta(
        template, beta_from_common(draws, theta), T, START
    )


def configure(spec, gains):
    model = health.setup(spec, gains)
    topology, dynamics, sender = spec[1], spec[0], spec[2]
    key = f"{topology}_{dynamics}_target{sender}_q"
    model.target["Pplus"] = gains[f"{key}1_Pplus"][[0, 64], 26]
    for name in ("F", "K", "G", "Omega"):
        np.testing.assert_allclose(
            model.target[name][0, 6], model.target[name][1, 6],
            rtol=2e-12, atol=2e-12,
            err_msg=f"The q=1 {name} mode must not depend on the H1 target alarm bit.",
        )
    graph_path = GRAPH_DIR / f"{topology}_{dynamics}_target{sender}_qwords.npz"
    with np.load(graph_path, allow_pickle=False) as archive:
        graph = archive["legal"].copy()
    base_norm = np.linalg.norm(model.base["F"], ord=2, axis=(-2, -1))
    if not float(base_norm.max()) < 1.0:
        raise ValueError("Frozen binary modes outside receiver 26 fail the norm screen.")
    return model, graph, graph_path


def h1_score(model, paths, delta=None):
    if delta is None:
        return health.scores(model, paths)
    residual = health.residuals(model, paths, delta)
    scores = np.zeros((len(paths.x), T, 7))
    scores[:, 5:] = .5 * np.einsum(
        "btei,eij,btej->bte", residual[:, 1:], model.precision,
        residual[:, 1:], optimize=True
    )
    return scores


def generate_paths(model, spec, scene, fixture, count, stage, threshold=None,
                   namespace=NAMESPACE):
    """Generate one contiguous batch-aligned stream and one frozen H1 mask."""
    assert count % BATCH == 0
    paths, deltas, score_parts, mask_parts = [], [], [], []
    for batch in range(count // BATCH):
        physical = health.physical(
            model, BATCH,
            make_seed(scene, int(fixture == "burst"), int(spec[3] == "iid"),
                      stage, batch, namespace)
        )
        if stage == 2:
            delta = make_attack_delta(BATCH, spec, fixture, scene, batch, namespace)
            score = h1_score(model, physical, delta)
            deltas.append(delta)
        else:
            delta = None
            score = h1_score(model, physical)
        paths.append(physical)
        score_parts.append(score)
        if threshold is not None:
            mask_parts.append(score > threshold)
    result = concatenate(paths)
    scores = np.concatenate(score_parts)
    delta = np.concatenate(deltas) if deltas else np.zeros((count, T, 7, 2))
    masks = np.concatenate(mask_parts) if mask_parts else None
    return result, delta, scores, masks


def generate_health(model, scene, threshold, stage, path_count,
                    namespace=NAMESPACE):
    assert path_count % BATCH == 0
    paths, scores, masks = [], [], []
    for batch in range(path_count // BATCH):
        physical = health.physical(
            model, BATCH, make_seed(scene, 0, 0, stage, batch, namespace)
        )
        score = health.scores(model, physical)
        paths.append(physical)
        scores.append(score)
        masks.append(score > threshold)
    return concatenate(paths), np.concatenate(scores), np.concatenate(masks)


def calibrate_threshold(model, scene, count=FIT, namespace=NAMESPACE):
    maxima = []
    for batch in range(count // BATCH):
        paths = health.physical(model, BATCH,
                                make_seed(scene, 0, 0, 0, batch, namespace))
        maxima.extend(h1_score(model, paths)[:, 4:].max(axis=(1, 2)))
    maxima = np.asarray(maxima)
    rank = int(np.ceil((count + 1) * .95))
    return float(np.partition(maxima, rank - 1)[rank - 1]), maxima


def alarm_runs(target_alarm):
    runs = np.zeros(target_alarm.shape, dtype=np.int32)
    for tick in range(1, target_alarm.shape[1]):
        runs[:, tick] = np.where(
            target_alarm[:, tick], runs[:, tick - 1] + 1, 0
        )
    return runs


def make_cell(score, run, bounds):
    score_bin = np.searchsorted(bounds, score, side="right")
    duration_bin = np.where(run == 1, 0, np.where(run <= 3, 1, 2))
    return 3 * score_bin + duration_bin


def cell_index(score, run, bounds):
    return make_cell(score, run, bounds)


def single_step_damage(model, paths, delta, masks):
    """A-reference prior at k; compare clean/attacked target updates at that prior."""
    count = len(paths.x)
    labels = np.zeros((count, T, 7), dtype=float)
    reference_post = np.zeros((count, 4), dtype=float)
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    sender = int(model.links[6, 1])
    other_h_sum = sum(model.H[j] for j in model.neighbors[26] if j != sender)
    reference_q_index = int(np.flatnonzero(np.isclose(Q, 0.8))[0])

    for tick in range(1, T):
        bits = masks[:, tick]
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]]
            )
        alarm = bits[:, 6]
        mode = alarm.astype(int)
        perturbation = delta[:, tick, 6]
        prior = reference_post @ model.A.T

        for qi, q in enumerate(Q):
            effective_q = np.where(alarm, q, 1.0)
            gain_index = np.where(alarm, qi, 6)
            attacked_neighbor_sum = received[:, 26] + perturbation + (
                effective_q - 1.0
            )[:, None] * (y[:, sender] + perturbation)
            clean_neighbor_sum = received[:, 26] + (
                effective_q - 1.0
            )[:, None] * y[:, sender]
            local_residual = y[:, 26] - prior @ model.H[26].T
            target_h_sum = other_h_sum[None] + effective_q[:, None, None] * model.H[sender][None]
            attacked_neighbor_innovation = attacked_neighbor_sum - np.einsum(
                "bi,bji->bj", prior, target_h_sum
            )
            clean_neighbor_innovation = clean_neighbor_sum - np.einsum(
                "bi,bji->bj", prior, target_h_sum
            )
            attacked_state = (
                prior
                + np.einsum(
                    "bij,bj->bi", model.target["K"][mode, gain_index], local_residual
                )
                + np.einsum(
                    "bij,bj->bi", model.target["G"][mode, gain_index],
                    attacked_neighbor_innovation,
                )
            )
            clean_state = (
                prior
                + np.einsum(
                    "bij,bj->bi", model.target["K"][mode, gain_index], local_residual
                )
                + np.einsum(
                    "bij,bj->bi", model.target["G"][mode, gain_index],
                    clean_neighbor_innovation,
                )
            )
            labels[:, tick, qi] = (
                (attacked_state[:, 2:] - paths.x[:, tick, 2:]) ** 2
            ).sum(axis=1) - (
                (clean_state[:, 2:] - paths.x[:, tick, 2:]) ** 2
            ).sum(axis=1)

        q_reference = np.where(alarm, 0.8, 1.0)
        reference_index = np.where(alarm, reference_q_index, 6)
        reference_neighbor_sum = received[:, 26] + perturbation + (
            q_reference - 1.0
        )[:, None] * (y[:, sender] + perturbation)
        reference_h_sum = other_h_sum[None] + q_reference[:, None, None] * model.H[sender][None]
        reference_innovation = reference_neighbor_sum - np.einsum(
            "bi,bji->bj", prior, reference_h_sum
        )
        reference_post = (
            prior
            + np.einsum(
                "bij,bj->bi", model.target["K"][mode, reference_index],
                local_residual,
            )
            + np.einsum(
                "bij,bj->bi", model.target["G"][mode, reference_index],
                reference_innovation,
            )
        )
    return labels


def literal_single_step_damage(model, paths, delta, masks):
    """Independent per-neighbor reference implementation for deterministic tests."""
    count = len(paths.x)
    output = np.zeros((count, T, 7), dtype=float)
    reference_post = np.zeros((count, 4), dtype=float)
    reference_q_index = int(np.flatnonzero(np.isclose(Q, 0.8))[0])
    sender = int(model.links[6, 1])
    other_neighbors = [j for j in model.neighbors[26] if j != sender]
    for tick in range(1, T):
        y = paths.y[:, tick]
        alarm = masks[:, tick, 6]
        mode = alarm.astype(int)
        prior = reference_post @ model.A.T
        local = y[:, 26] - prior @ model.H[26].T
        for qi, q in enumerate(Q):
            eff = np.where(alarm, q, 1.0)
            gain_index = np.where(alarm, qi, 6)
            clean_innovation = np.zeros((count, 2))
            attack_innovation = np.zeros((count, 2))
            for neighbor in other_neighbors:
                residual = y[:, neighbor] - prior @ model.H[neighbor].T
                clean_innovation += residual
                attack_innovation += residual
            clean_sender = y[:, sender] - prior @ model.H[sender].T
            attack_sender = (
                y[:, sender] + delta[:, tick, 6] - prior @ model.H[sender].T
            )
            clean_innovation += eff[:, None] * clean_sender
            attack_innovation += eff[:, None] * attack_sender
            clean = (
                prior
                + np.einsum("bij,bj->bi", model.target["K"][mode, gain_index], local)
                + np.einsum("bij,bj->bi", model.target["G"][mode, gain_index], clean_innovation)
            )
            attacked = (
                prior
                + np.einsum("bij,bj->bi", model.target["K"][mode, gain_index], local)
                + np.einsum("bij,bj->bi", model.target["G"][mode, gain_index], attack_innovation)
            )
            output[:, tick, qi] = (
                (attacked[:, 2:] - paths.x[:, tick, 2:]) ** 2
            ).sum(1) - (
                (clean[:, 2:] - paths.x[:, tick, 2:]) ** 2
            ).sum(1)

        qref = np.where(alarm, 0.8, 1.0)
        qref_index = np.where(alarm, reference_q_index, 6)
        ref_clean_innovation = np.zeros((count, 2))
        ref_attack_innovation = np.zeros((count, 2))
        for neighbor in other_neighbors:
            residual = y[:, neighbor] - prior @ model.H[neighbor].T
            ref_clean_innovation += residual
            ref_attack_innovation += residual
        clean_sender = y[:, sender] - prior @ model.H[sender].T
        attack_sender = y[:, sender] + delta[:, tick, 6] - prior @ model.H[sender].T
        ref_clean_innovation += qref[:, None] * clean_sender
        ref_attack_innovation += qref[:, None] * attack_sender
        reference_post = (
            prior
            + np.einsum("bij,bj->bi", model.target["K"][mode, qref_index], local)
            + np.einsum("bij,bj->bi", model.target["G"][mode, qref_index], ref_attack_innovation)
        )
    return output


def path_cell_means(scores, runs, labels, threshold, boundaries):
    normalized = scores[:, 50:91, 6] / threshold
    run = runs[:, 50:91]
    fine = make_cell(normalized, run, boundaries)
    score_bin = np.searchsorted(boundaries, normalized, side="right")
    parent = 9 + score_bin
    alarm = run > 0
    n = len(scores)
    values = np.full((n, 13, 7), np.nan)
    events = np.zeros((n, 13), dtype=np.int16)
    y = labels[:, 50:91]
    for path in range(n):
        for cell in range(13):
            if cell < 9:
                selected = alarm[path] & (fine[path] == cell)
            elif cell < 12:
                selected = alarm[path] & (parent[path] == cell)
            else:
                selected = alarm[path]
            if selected.any():
                values[path, cell] = y[path, selected].mean(0)
                events[path, cell] = int(selected.sum())
    return values, events


def fit_risk_table(records, threshold, scene, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    alarm_score = np.concatenate([
        r["scores"][:FIT, 50:91, 6][r["runs"][:FIT, 50:91] > 0] / threshold
        for r in records
    ])
    boundaries = np.quantile(alarm_score, [1 / 3, 2 / 3])
    values_by_fixture, events_by_fixture = [], []
    for record in records:
        v, e = path_cell_means(
            record["scores"][:FIT], record["runs"][:FIT],
            record["labels"][:FIT], threshold, boundaries
        )
        values_by_fixture.append(v)
        events_by_fixture.append(e)
    values = np.concatenate(values_by_fixture)
    events = np.concatenate(events_by_fixture)
    fixture_path_count = len(values) // len(records)
    if any(len(record["labels"]) != fixture_path_count for record in records):
        raise ValueError("Continuous and burst risk-fit samples must have equal path counts.")
    path_count = len(values)
    support = np.isfinite(values[:, :, 1]).sum(0)
    fixture_support = {
        record["fixture"]: np.isfinite(fixture_values[:, :, 1]).sum(0).astype(int).tolist()
        for record, fixture_values in zip(records, values_by_fixture)
    }
    fixture_events = {
        record["fixture"]: fixture_event_counts.sum(0).astype(int).tolist()
        for record, fixture_event_counts in zip(records, events_by_fixture)
    }
    means = np.zeros((13, 7))
    se = np.zeros_like(means)
    for cell in range(13):
        valid = np.isfinite(values[:, cell, 1])
        if valid.any():
            means[cell] = np.nanmean(values[valid, cell], axis=0)
        if valid.sum() >= 2:
            se[cell] = np.nanstd(values[valid, cell], axis=0, ddof=1) / np.sqrt(valid.sum())

    rng = np.random.default_rng(
        np.random.SeedSequence([*NAMESPACE, scene, 0, 0, BOOTSTRAP_SEED_SUFFIX, 0])
    )
    # Stratify path resampling by fixture to preserve the frozen equal mixture.
    multipliers = np.zeros((bootstrap_replicates, path_count), dtype=np.int16)
    for fixture_index in range(len(records)):
        first = fixture_index * fixture_path_count
        last = first + fixture_path_count
        multipliers[:, first:last] = rng.multinomial(
            fixture_path_count,
            np.full(fixture_path_count, 1 / fixture_path_count),
            size=bootstrap_replicates,
        )
    maximum = np.zeros(bootstrap_replicates)
    for cell in range(13):
        valid = np.isfinite(values[:, cell, 1])
        positive = se[cell, 1:] > 0
        if support[cell] < 30 or not positive.any():
            continue
        sample = values[valid, cell]
        weights = multipliers[:, valid]
        denominator = weights.sum(1)
        boot_mean = weights @ sample / np.maximum(denominator[:, None], 1)
        statistic = np.zeros((bootstrap_replicates, 6))
        statistic[:, positive] = (
            means[cell, 1:][positive] - boot_mean[:, 1:][:, positive]
        ) / se[cell, 1:][positive]
        maximum = np.maximum(maximum, statistic.max(1))
    critical = max(0.0, float(np.quantile(maximum, .95)))
    upper = means + critical * se
    means[:, 0] = 0.0
    upper[:, 0] = 0.0

    fallback = []
    fallback_level = []
    for leaf in range(9):
        parent = 9 + leaf // 3
        if support[leaf] >= 30:
            fallback.append(leaf)
            fallback_level.append("score_by_run")
        elif support[parent] >= 30:
            fallback.append(parent)
            fallback_level.append("score_only")
        elif support[12] >= 30:
            fallback.append(12)
            fallback_level.append("pooled_alarm")
        else:
            raise RuntimeError("Pooled target-alarm cell has fewer than 30 paths.")
    return dict(
        boundaries=boundaries.tolist(),
        support_paths=support.astype(int).tolist(),
        support_paths_by_fixture=fixture_support,
        event_counts=events.sum(0).astype(int).tolist(),
        event_counts_by_fixture=fixture_events,
        means=means.tolist(),
        standard_errors=se.tolist(),
        simultaneous_upper=upper.tolist(),
        critical_value=critical,
        fallback_cell=fallback,
        fallback_level=fallback_level,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_unit="whole path, joint max over supported cells and nonzero q",
        risk_identity="A-reference-prior U1 conditional single-step damage mean",
        is_closed_loop_mse_bound=False,
    )


def prepare_health_cost_affine(model):
    f_full = model.target["F"][0, 6]
    o_full = model.target["Omega"][0, 6]
    series = np.eye(4)
    coefficient = np.zeros((4, 4))
    for _ in range(60):
        coefficient += series.T @ W_MISSING @ series
        series = f_full @ series

    noise_cov = np.zeros((4, 4))
    noise_cost = 0.0
    for _ in range(59):
        noise_cov = f_full @ noise_cov @ f_full.T + o_full
        noise_cost += np.trace(W_MISSING @ noise_cov)

    f_alarm = model.target["F"][1]
    o_alarm = model.target["Omega"][1]
    linear = np.einsum("qji,jk,qkl->qil", f_alarm, coefficient, f_alarm) / 60.
    constant = (
        np.einsum("ij,qji->q", coefficient, o_alarm) + noise_cost
    ) / 60.
    return linear, constant


def health_cost_vector(covariance, affine):
    linear, constant = affine
    return np.einsum("qij,...ji->...q", linear, covariance) + constant


def fixed_health_cost(model, affine):
    initial = model.target["Pplus"][0]
    vector = health_cost_vector(initial, affine)
    return vector - vector[-1]


def select_q(objective, legal):
    candidate_cost = np.where(legal, objective, np.inf)
    best = candidate_cost.min()
    return int(np.flatnonzero(legal & (objective <= best + 1e-10))[-1])


def covariance_scalar(dynamic_h, fixed_h, legal):
    denominator = float(fixed_h[legal] @ fixed_h[legal])
    if denominator <= 1e-24:
        return 0.0, True
    scalar = float(dynamic_h[legal] @ fixed_h[legal]) / denominator
    return max(scalar, 0.0), False


def simulate_policy(model, graph, paths, scores, masks, delta, table, threshold):
    n = len(paths.x)
    state = np.zeros((n, 4, 50, 4))
    state_cov = np.broadcast_to(
        model.target["Pplus"][0], (n, 4, 4, 4)
    ).copy()
    history = np.full((n, 4), 7 ** 6 - 1, dtype=np.int32)
    alarm_run = np.zeros(n, dtype=np.int32)
    actions = np.zeros((n, 4, T), dtype=np.int8)
    losses = np.zeros((n, 4, 5), dtype=float)
    # Loss columns are target missing block window/tail/post20, network window/post20.
    zero_scalar = np.zeros((n, 4), dtype=np.int32)
    risk_fallback = np.zeros((n, 4), dtype=np.int32)
    risk_fallback_mask = np.zeros((n, 4, T), dtype=np.uint8)
    risk_fallback_levels = np.zeros((n, 4, 3), dtype=np.int32)
    dynamic_cost_history = np.zeros((n, 4, T, len(Q)), dtype=float)
    dynamic_absolute_history = np.zeros_like(dynamic_cost_history)
    target_sender = int(model.links[6, 1])
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    base_f, base_k, base_g = (model.base[name] for name in ("F", "K", "G"))
    target_f = model.target["F"]
    target_k = model.target["K"]
    target_g = model.target["G"]
    target_o = model.target["Omega"]
    affine = prepare_health_cost_affine(model)
    hstar = fixed_health_cost(model, affine)
    table_bounds = np.asarray(table["boundaries"])
    table_lookup = np.asarray(table["fallback_cell"])
    table_upper = np.asarray(table["simultaneous_upper"])
    shared_masks = masks.copy()
    if shared_masks.shape != (n, T, 7):
        raise ValueError("Four policies require one shared H1 mask.")

    for tick in range(1, T):
        bits = shared_masks[:, tick]
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]]
            )
        target_alarm = bits[:, 6]
        alarm_run = np.where(target_alarm, alarm_run + 1, 0)
        normalized = scores[:, tick, 6] / threshold
        fine_cell = cell_index(normalized, alarm_run, table_bounds)
        cell = table_lookup[fine_cell]
        risk = table_upper[cell]
        valid = graph[history]

        dynamic_by_policy = health_cost_vector(state_cov, affine)
        dynamic_delta_by_policy = dynamic_by_policy - dynamic_by_policy[:, :, -1:]
        dynamic_absolute_history[:, :, tick] = dynamic_by_policy
        dynamic_cost_history[:, :, tick] = dynamic_delta_by_policy
        dynamic_scalar = np.zeros((n, 4))
        for policy in range(4):
            projected_h = np.where(valid[:, policy], hstar[None, :], 0.0)
            projected_dynamic = np.where(
                valid[:, policy], dynamic_delta_by_policy[:, policy], 0.0
            )
            numerator = np.sum(projected_dynamic * projected_h, axis=1)
            denominator = np.sum(projected_h ** 2, axis=1)
            zero_scalar[:, policy] += (denominator <= 1e-24) & target_alarm
            dynamic_scalar[:, policy] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator),
                where=denominator > 1e-24
            ).clip(0.0)

        selected = np.full((n, 4), 6, dtype=np.int8)
        for path in np.flatnonzero(target_alarm):
            allowed = valid[path]
            selected[path, 0] = select_q(risk[path], allowed[0])
            selected[path, 1] = select_q(risk[path] + hstar, allowed[1])
            selected[path, 2] = select_q(risk[path] + dynamic_scalar[path, 2] * hstar, allowed[2])
            selected[path, 3] = select_q(
                risk[path] + dynamic_delta_by_policy[path, 3], allowed[3]
            )
            fallback = np.isin(cell[path], [9, 10, 11, 12])
            risk_fallback[path] += fallback
            risk_fallback_mask[path, :, tick] = fallback
            risk_fallback_levels[path, :, 0] += cell[path] == fine_cell[path]
            risk_fallback_levels[path, :, 1] += np.isin(cell[path], [9, 10, 11])
            risk_fallback_levels[path, :, 2] += cell[path] == 12
        if not np.all(valid[np.arange(n)[:, None], np.arange(4)[None], selected]):
            raise AssertionError("B policy selected an action outside the M7 extension graph.")
        actions[:, :, tick] = selected
        history = (history * 7 + selected) % (7 ** 6)
        q_actual = np.where(target_alarm[:, None], Q[selected], 1.0)
        mode = target_alarm.astype(int)

        next_state = np.empty_like(state)
        for policy in range(4):
            f = target_f[mode, selected[:, policy]]
            k_gain = target_k[mode, selected[:, policy]]
            g_gain = target_g[mode, selected[:, policy]]
            packet = received[:, 26] + delta[:, tick, 6] + (
                q_actual[:, policy] - 1.0
            )[:, None] * (
                y[:, target_sender] + delta[:, tick, 6]
            )
            next_state[:, policy, 26] = (
                np.einsum("bij,bj->bi", f, state[:, policy, 26])
                + np.einsum("bij,bj->bi", k_gain, y[:, 26])
                + np.einsum("bij,bj->bi", g_gain, packet)
            )
            state_cov[:, policy] = (
                f @ state_cov[:, policy] @ f.swapaxes(-1, -2)
                + target_o[mode, selected[:, policy]]
            )

        # Other receivers use the same masks and raw messages. The target link terminates at 26.
        all_mode = shared_masks[:, tick].astype(np.int64) @ BITS
        base_next = (
            np.einsum("bnij,bnj->bni", base_f[all_mode], state[:, 0])
            + np.einsum("bnij,bnj->bni", base_k[all_mode], y)
            + np.einsum("bnij,bnj->bni", base_g[all_mode], received)
        )
        for policy in range(4):
            next_state[:, policy, :26] = base_next[:, :26]
            next_state[:, policy, 27:] = base_next[:, 27:]
        state = next_state

        error = (state - paths.x[:, tick, None, None, :]) ** 2
        missing = error[:, :, 26, 2:].sum(axis=-1)
        network = error.sum(axis=(2, 3)) / 50.0
        if START <= tick < STOP:
            losses[:, :, 0] += missing / 60.0
            losses[:, :, 3] += network / 60.0
        if tick >= STOP:
            losses[:, :, 1] += missing / 40.0
        if tick >= 20:
            losses[:, :, 2] += missing / 130.0
            losses[:, :, 4] += network / 130.0

    return losses, actions, dict(
        zero_scalar_denominator_counts=zero_scalar.sum(axis=0).tolist(),
        risk_fallback_counts=risk_fallback.sum(axis=0).tolist(),
        risk_fallback_level_counts=risk_fallback_levels.sum(axis=0).tolist(),
        target_alarm_events=int(shared_masks[:, :, 6].sum()),
        h1_masks_identical_by_construction=True,
    ), dynamic_cost_history, dynamic_absolute_history, risk_fallback_mask


def action_summary(actions, target_alarm, window):
    alarm = np.asarray(target_alarm, dtype=bool)
    window_actions = actions[:, :, window]
    window_alarm = alarm[:, window]
    counts = np.zeros((len(STRATEGIES), len(Q)), dtype=np.int64)
    alarm_weight_means = []
    window_weight_means = []
    for policy in range(len(STRATEGIES)):
        policy_actions = window_actions[:, policy]
        counts[policy] = [
            int(((policy_actions == qi) & window_alarm).sum())
            for qi in range(len(Q))
        ]
        alarm_count = int(window_alarm.sum())
        alarm_weight_means.append(
            float(Q[policy_actions[window_alarm]].mean()) if alarm_count else None
        )
        window_weight_means.append(float(Q[policy_actions].mean()))

    pairs = {
        "RH_dynamic_minus_R1": (0, 3),
        "RH_dynamic_minus_RH_static": (1, 3),
        "RH_dynamic_minus_RH_scalar": (2, 3),
    }
    disagreements = {}
    for name, (left, right) in pairs.items():
        denominator = int(window_alarm.sum())
        different = (window_actions[:, left] != window_actions[:, right]) & window_alarm
        disagreements[name] = {
            "different_alarm_actions": int(different.sum()),
            "target_alarm_events": denominator,
            "rate": float(different.sum() / denominator) if denominator else None,
        }
    return {
        "alarm_action_counts_by_policy_and_q": counts.tolist(),
        "mean_effective_weight_on_target_alarm_by_policy": alarm_weight_means,
        "mean_effective_weight_over_window_by_policy": window_weight_means,
        "action_disagreements": disagreements,
    }


def clopper_pearson(successes, trials, confidence=0.95):
    tail = (1.0 - confidence) / 2.0
    lower = (
        float(beta_distribution.ppf(tail, successes, trials - successes + 1))
        if successes else 0.0
    )
    upper = (
        float(beta_distribution.ppf(1.0 - tail, successes + 1, trials - successes))
        if successes < trials else 1.0
    )
    return [lower, upper]


def simulate_full_accept_reference(model, paths, scores, masks, delta):
    """Diagnostic q=1 reference; never used to fit or select the four policies."""
    n = len(paths.x)
    state = np.zeros((n, 50, 4), dtype=float)
    target_sender = int(model.links[6, 1])
    base_f, base_k, base_g = (model.base[name] for name in ("F", "K", "G"))
    target_f = model.target["F"]
    target_k = model.target["K"]
    target_g = model.target["G"]
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    losses = np.zeros((n, 5), dtype=float)
    for tick in range(1, T):
        bits = masks[:, tick]
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]]
            )
        alarm = bits[:, 6]
        mode = alarm.astype(int)
        next_state = np.empty_like(state)
        target_packet = received[:, 26] + delta[:, tick, 6]
        next_state[:, 26] = (
            np.einsum("bij,bj->bi", target_f[mode, 6], state[:, 26])
            + np.einsum("bij,bj->bi", target_k[mode, 6], y[:, 26])
            + np.einsum("bij,bj->bi", target_g[mode, 6], target_packet)
        )
        global_mode = bits.astype(np.int64) @ BITS
        other = (
            np.einsum("bnij,bnj->bni", base_f[global_mode], state)
            + np.einsum("bnij,bnj->bni", base_k[global_mode], y)
            + np.einsum("bnij,bnj->bni", base_g[global_mode], received)
        )
        next_state[:, :26] = other[:, :26]
        next_state[:, 27:] = other[:, 27:]
        state = next_state
        error = (state - paths.x[:, tick, None, :]) ** 2
        missing = error[:, 26, 2:].sum(axis=1)
        network = error.sum(axis=(1, 2)) / 50.0
        if START <= tick < STOP:
            losses[:, 0] += missing / 60.0
            losses[:, 3] += network / 60.0
        if tick >= STOP:
            losses[:, 1] += missing / 40.0
        if tick >= 20:
            losses[:, 2] += missing / 130.0
            losses[:, 4] += network / 130.0
    return losses


def literal_policy_replay(model, paths, delta, masks, actions):
    """Independent per-neighbor update for the deterministic smoke audit."""
    count = len(paths.x)
    state = np.zeros((count, 4, 50, 4), dtype=float)
    edge_lookup = {tuple(map(int, edge)): i for i, edge in enumerate(model.links)}
    losses = np.zeros((count, 4, 5), dtype=float)
    for tick in range(1, T):
        prediction = state @ model.A.T
        y = paths.y[:, tick]
        next_state = np.empty_like(state)
        global_mode = masks[:, tick].astype(np.int64) @ BITS
        alarm = masks[:, tick, 6]
        for receiver, neighbors in enumerate(model.neighbors):
            local = y[:, None, receiver] - prediction[:, :, receiver] @ model.H[receiver].T
            innovation = np.zeros((count, 4, 2), dtype=float)
            for sender in neighbors:
                residual = y[:, None, sender] - prediction[:, :, receiver] @ model.H[sender].T
                edge = edge_lookup.get((receiver, sender))
                if edge is not None and edge < 6:
                    residual *= (~masks[:, tick, edge])[:, None, None]
                elif edge == 6:
                    packet = y[:, None, sender] + delta[:, tick, 6, None, :]
                    residual = packet - prediction[:, :, receiver] @ model.H[sender].T
                    effective = np.where(alarm[:, None], Q[actions[:, :, tick]], 1.0)
                    residual *= effective[..., None]
                innovation += residual
            if receiver == 26:
                mode = alarm.astype(int)
                index = actions[:, :, tick]
                k_gain = model.target["K"][mode[:, None], index]
                g_gain = model.target["G"][mode[:, None], index]
                next_state[:, :, receiver] = (
                    prediction[:, :, receiver]
                    + np.einsum("bpij,bpj->bpi", k_gain, local)
                    + np.einsum("bpij,bpj->bpi", g_gain, innovation)
                )
            else:
                k_gain = model.base["K"][global_mode, receiver]
                g_gain = model.base["G"][global_mode, receiver]
                next_state[:, :, receiver] = (
                    prediction[:, :, receiver]
                    + np.einsum("bij,bpj->bpi", k_gain, local)
                    + np.einsum("bij,bpj->bpi", g_gain, innovation)
                )
        state = next_state
        error = (state - paths.x[:, tick, None, None, :]) ** 2
        missing = error[:, :, 26, 2:].sum(axis=-1)
        network = error.sum(axis=(2, 3)) / 50.0
        if START <= tick < STOP:
            losses[:, :, 0] += missing / 60.0
            losses[:, :, 3] += network / 60.0
        if tick >= STOP:
            losses[:, :, 1] += missing / 40.0
        if tick >= 20:
            losses[:, :, 2] += missing / 130.0
            losses[:, :, 4] += network / 130.0
    return losses


def full_accept_reference(model, paths, masks, delta):
    """Target q=1 diagnostic reference with the same other-edge binary gates."""
    count = len(paths.x)
    state = np.zeros((count, 50, 4), dtype=float)
    loss = np.zeros((count, 5), dtype=float)
    raw = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    for tick in range(1, T):
        bits = masks[:, tick]
        y = paths.y[:, tick]
        received = raw[:, tick].copy()
        for edge in range(6):
            received[:, model.links[edge, 0]] -= (
                bits[:, edge, None] * y[:, model.links[edge, 1]]
            )
        received[:, 26] += delta[:, tick, 6]
        mode = bits.astype(np.int64) @ BITS
        state = (
            np.einsum("bnij,bnj->bni", model.base["F"][mode], state)
            + np.einsum("bnij,bnj->bni", model.base["K"][mode], y)
            + np.einsum("bnij,bnj->bni", model.base["G"][mode], received)
        )
        error = (state - paths.x[:, tick, None, :]) ** 2
        missing = error[:, 26, 2:].sum(axis=-1)
        network = error.sum(axis=(1, 2)) / 50.0
        if START <= tick < STOP:
            loss[:, 0] += missing / 60.0
            loss[:, 3] += network / 60.0
        if tick >= STOP:
            loss[:, 1] += missing / 40.0
        if tick >= 20:
            loss[:, 2] += missing / 130.0
            loss[:, 4] += network / 130.0
    return loss


def verify_health_intervention(model):
    affine = prepare_health_cost_affine(model)
    initial = model.target["Pplus"][0]
    direct = np.zeros(7)
    for qi in range(7):
        covariance = (
            model.target["F"][1, qi] @ initial @ model.target["F"][1, qi].T
            + model.target["Omega"][1, qi]
        )
        for step in range(60):
            direct[qi] += np.trace(W_MISSING @ covariance) / 60.0
            if step < 59:
                f = model.target["F"][0, 6]
                covariance = f @ covariance @ f.T + model.target["Omega"][0, 6]
    analytic = health_cost_vector(initial, affine)
    np.testing.assert_allclose(analytic, direct, rtol=2e-12, atol=2e-12)
    return float(np.max(np.abs(analytic - direct)))


def threshold_calibration(model, scene, count=FIT, namespace=NAMESPACE):
    maxima = []
    for batch in range(count // BATCH):
        p = health.physical(model, BATCH, make_seed(scene, 0, 0, 0, batch, namespace))
        maxima.extend(health.scores(model, p)[:, 4:].max(axis=(1, 2)))
    maxima = np.asarray(maxima)
    rank = int(np.ceil((count + 1) * 0.95))
    return float(np.partition(maxima, rank - 1)[rank - 1]), maxima


def run_scene(scene, spec, gains, path_count=N, fit_count=FIT,
              calibration_count=FIT, evaluation_count=EVALUATION,
              namespace=NAMESPACE, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    model, graph, graph_path = configure(spec, gains)
    threshold, calibration_maxima = threshold_calibration(
        model, scene, calibration_count, namespace
    )
    fit_records = []
    full_records = {}
    for fixture in ("continuous", "burst"):
        paths, delta, scores, masks = generate_paths(
            model, spec, scene, fixture, path_count, 2,
            threshold=threshold, namespace=namespace
        )
        runs = alarm_runs(masks[:, :, 6])
        labels = single_step_damage(model, paths, delta, masks)
        fit_records.append(dict(fixture=fixture, scores=scores[:fit_count], runs=runs[:fit_count],
                                labels=labels[:fit_count]))
        full_records[fixture] = (paths, delta, scores, masks, runs)
    risk_table = fit_risk_table(
        fit_records, threshold, scene, bootstrap_replicates
    )
    if risk_table["simultaneous_upper"][12][1] < -1e-12:
        raise AssertionError("Pooled risk UCB has impossible negative excess-risk upper.")

    health_paths, health_scores, health_masks = generate_health(
        model, scene, threshold, 1, evaluation_count, namespace
    )
    target_health_path_alarm = health_masks[:, :, 6].any(axis=1)
    all_health_path_alarm = health_masks[:, :, 4:].any(axis=(1, 2))
    (health_loss, health_actions, health_status, health_dynamic,
     health_dynamic_absolute, health_fallback) = simulate_policy(
        model, graph, health_paths, health_scores, health_masks,
        np.zeros((evaluation_count, T, 7, 2)), risk_table, threshold
    )
    health_full_reference = full_accept_reference(
        model, health_paths, health_masks,
        np.zeros((evaluation_count, T, 7, 2))
    )
    health_full_accept = simulate_full_accept_reference(
        model, health_paths, health_scores, health_masks,
        np.zeros((evaluation_count, T, 7, 2)),
    )
    hit_all = int(all_health_path_alarm.sum())
    hit_target = int(target_health_path_alarm.sum())
    health_actions_summary = action_summary(
        health_actions, health_masks[:, :, 6], slice(20, T)
    )

    result = dict(
        scene=scene, spec=list(spec), threshold=threshold,
        calibration_path_count=calibration_count,
        calibration_rank=int(np.ceil((calibration_count + 1) * .95)),
        calibration_maxima_sha256=hashlib.sha256(calibration_maxima.tobytes()).hexdigest(),
        health_evaluation_paths=evaluation_count,
        target_alarm_paths=hit_target,
        target_fwer=hit_target / evaluation_count,
        target_fwer_CP95=clopper_pearson(hit_target, evaluation_count),
        all_edge_fwer_paths=hit_all,
        all_edge_fwer=hit_all / evaluation_count,
        all_edge_fwer_CP95=clopper_pearson(hit_all, evaluation_count),
        health_loss_mean=health_loss.mean(axis=0).tolist(),
        health_full_accept_reference_mean=health_full_reference.mean(axis=0).tolist(),
        health_full_accept_loss_mean=health_full_accept.mean(axis=0).tolist(),
        health_actions=health_actions_summary,
        health_status=health_status,
        risk_table=risk_table,
        attacks={},
    )
    arrays = dict(
        health_masks=health_masks.astype(np.uint8),
        health_actions=health_actions,
        health_dynamic_health_cost=health_dynamic,
        health_dynamic_health_cost_absolute=health_dynamic_absolute,
        health_risk_fallback=health_fallback,
        health_effective_weights=Q[health_actions],
        health_loss=health_loss,
        health_full_accept_reference=health_full_reference,
        health_full_accept_loss=health_full_accept,
        health_scores=health_scores,
        health_calibration_maxima=calibration_maxima,
    )
    for fixture, (paths, delta, scores, masks, runs) in full_records.items():
        eval_slice = slice(fit_count, path_count)
        eval_paths = SimpleNamespace(x=paths.x[eval_slice], y=paths.y[eval_slice])
        eval_delta = delta[eval_slice]
        eval_scores = scores[eval_slice]
        eval_masks = masks[eval_slice]
        (loss, actions, status, dynamic_cost,
         dynamic_cost_absolute, fallback_mask) = simulate_policy(
            model, graph, eval_paths, eval_scores, eval_masks,
            eval_delta, risk_table, threshold
        )
        full_accept_loss = simulate_full_accept_reference(
            model, eval_paths, eval_scores, eval_masks, eval_delta
        )
        active = active_attack(fixture)
        attack_alarm = eval_masks[:, START:STOP, 6]
        active_alarm = attack_alarm[:, active]
        hit = active_alarm.any(axis=1)
        attack_actions_summary = action_summary(
            actions, eval_masks[:, :, 6], slice(START, STOP)
        )
        fallback_rate = [
            float(value / max(status["target_alarm_events"], 1))
            for value in status["risk_fallback_counts"]
        ]
        result["attacks"][fixture] = dict(
            evaluation_paths=path_count - fit_count,
            path_miss_count=int((~hit).sum()),
            path_miss_fraction=float((~hit).mean()),
            attack_tick_unalarmed_fraction=float((~active_alarm).mean()),
            longest_target_alarm_run_mean=float(runs[eval_slice, START:STOP].max(axis=1).mean()),
            loss_mean=loss.mean(axis=0).tolist(),
            full_accept_reference_loss_mean=full_accept_loss.mean(axis=0).tolist(),
            risk_fallback_rate_by_policy=fallback_rate,
            risk_fallback_level_counts=status["risk_fallback_level_counts"],
            action_summary=attack_actions_summary,
            policy_status=status,
        )
        arrays[f"{fixture}_masks"] = eval_masks.astype(np.uint8)
        arrays[f"{fixture}_scores"] = eval_scores
        arrays[f"{fixture}_loss"] = loss
        arrays[f"{fixture}_full_accept_loss"] = full_accept_loss
        arrays[f"{fixture}_actions"] = actions
        arrays[f"{fixture}_dynamic_health_cost"] = dynamic_cost
        arrays[f"{fixture}_dynamic_health_cost_absolute"] = dynamic_cost_absolute
        arrays[f"{fixture}_risk_fallback"] = fallback_mask
        arrays[f"{fixture}_effective_weights"] = Q[actions]

    return result, arrays, graph_path


def run_smoke():
    spec = health.scenarios()[0]
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        model, graph, _ = configure(spec, gains)
        health_affine_error = verify_health_intervention(model)
        physical = health.physical(model, 64, make_seed(0, 0, 0, 5, SMOKE_NAMESPACE))
        clean_score = h1_score(model, physical)
        assert clean_score.shape == (64, T, 7)
        health_threshold, _ = calibrate_threshold(
            model, 0, FIT, SMOKE_NAMESPACE
        )
        health_test, health_scores, health_masks = generate_health(
            model, 0, health_threshold, 9, 64, SMOKE_NAMESPACE
        )
        health_full = simulate_full_accept_reference(
            model, health_test, health_scores, health_masks,
            np.zeros((64, T, 7, 2)),
        )
        health_full_independent = full_accept_reference(
            model, health_test, health_masks, np.zeros((64, T, 7, 2))
        )
        np.testing.assert_allclose(
            health_full_independent, health_full, rtol=2e-9, atol=2e-8
        )
        packed_health_masks = np.sum(
            health_masks * BITS[None, None, :], axis=2
        ).astype(np.uint8)
        historical_eval = health.evaluate(model, health_test, packed_health_masks)
        np.testing.assert_allclose(
            health_full[:, 2], historical_eval["missing"][:, 6],
            rtol=2e-9, atol=2e-8
        )
        np.testing.assert_allclose(
            health_full[:, 4], historical_eval["network"][:, 6],
            rtol=2e-9, atol=2e-8
        )
        sys.path.insert(0, str(D1_ROOT / "paper1_history_ablation_20261123"))
        from ablation import residuals as frozen_residuals, scores_all as frozen_scores
        small_paths = SimpleNamespace(x=physical.x[:4], y=physical.y[:4])
        frozen_h1 = frozen_scores(
            frozen_residuals(model, small_paths), model.Theta
        )["H1"]
        np.testing.assert_allclose(clean_score[:4], frozen_h1, rtol=1e-12, atol=1e-12)
        result, arrays, _ = run_scene(
            0, spec, gains, path_count=64, fit_count=32,
            calibration_count=256, evaluation_count=64,
            namespace=(20261124, 4401, 99), bootstrap_replicates=128
        )
    assert set(result["risk_table"]["support_paths_by_fixture"]) == {"continuous", "burst"}
    assert set(result["risk_table"]["event_counts_by_fixture"]) == {"continuous", "burst"}
    with tempfile.TemporaryDirectory(dir=HERE) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        json_path = temporary_directory / "atomic.json"
        npz_path = temporary_directory / "atomic.npz"
        atomic_json(json_path, {"status": "ok"})
        atomic_npz(npz_path, {"value": np.arange(3)})
        assert json.loads(json_path.read_text(encoding="utf-8")) == {"status": "ok"}
        with np.load(npz_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["value"], np.arange(3))
        assert json_path.read_bytes().endswith(b"\n")
    assert result["health_evaluation_paths"] == 64
    assert set(result["attacks"]) == {"continuous", "burst"}
    assert arrays["health_actions"].shape == (64, 4, T)
    for fixture in ("continuous", "burst"):
        assert arrays[f"{fixture}_actions"].shape == (32, 4, T)
        assert np.isfinite(arrays[f"{fixture}_loss"]).all()
        assert arrays[f"{fixture}_masks"].shape == (32, T, 7)
        assert np.isfinite(arrays[f"{fixture}_scores"]).all()
        np.testing.assert_allclose(
            arrays[f"{fixture}_dynamic_health_cost"][..., 6], 0.0,
            rtol=0.0, atol=2e-12,
        )
    for fixture in ("continuous", "burst"):
        physical = health.physical(
            model, 64,
            make_seed(0, int(fixture == "burst"), int(spec[3] == "iid"), 7, 0, SMOKE_NAMESPACE)
        )
        delta = make_attack_delta(
            64, spec, fixture, 0, 99, SMOKE_NAMESPACE
        )
        score = h1_score(model, physical, delta)
        masks = score > result["threshold"]
        table = result["risk_table"]
        # The single shared mask is the only detector output passed to policy evaluation.
        policies = []
        for _ in STRATEGIES:
            policies.append(masks.copy())
        assert all(np.array_equal(policies[0], candidate) for candidate in policies[1:])
        raw_target = np.einsum("ij,bjm->bim", model.adjacency, physical.y[:, 1])[:, 26]
        sender = int(model.links[6, 1])
        for alarm in (False, True):
            for q in Q:
                effective = q if alarm else 1.0
                aggregate = raw_target + delta[:, 1, 6] + (effective - 1.0) * (
                    physical.y[:, 1, sender] + delta[:, 1, 6]
                )
                individual = sum(
                    (
                        (physical.y[:, 1, neighbor]
                         + (delta[:, 1, 6] if neighbor == sender else 0.0))
                        * (effective if neighbor == sender else 1.0)
                        for neighbor in model.neighbors[26]
                    ),
                    start=np.zeros((64, 2)),
                )
                np.testing.assert_allclose(aggregate, individual, rtol=1e-13, atol=1e-13)
            label_paths = health.physical(
                model, 4,
                make_seed(0, int(fixture == "burst"), int(spec[3] == "iid"),
                          2, 7, SMOKE_NAMESPACE)
            )
            label_delta = make_attack_delta(
                4, spec, fixture, 0, 7, SMOKE_NAMESPACE
            )
            label_scores = h1_score(model, label_paths, label_delta)
            label_masks = label_scores > result["threshold"]
            aggregate_labels = single_step_damage(
                model, label_paths, label_delta, label_masks
            )
            literal_labels = literal_single_step_damage(
                model, label_paths, label_delta, label_masks
            )
            np.testing.assert_allclose(
                aggregate_labels, literal_labels, rtol=1e-9, atol=1e-9
            )
            if label_masks[:, :, 6].any():
                np.testing.assert_array_equal(
                    aggregate_labels[:, :, 0][label_masks[:, :, 6]], 0.0
                )
        direct_paths = SimpleNamespace(x=physical.x[:2], y=physical.y[:2])
        direct_delta = delta[:2]
        direct_scores = h1_score(model, direct_paths, direct_delta)
        direct_masks = direct_scores > result["threshold"]
        # Exercise the full policy interface under an exogenous mask and verify
        # the independent per-neighbor recursion exactly reproduces its four losses.
        smoke_table = table
        direct_loss, direct_actions, _, _, _, _ = simulate_policy(
            model, graph, direct_paths, direct_scores, direct_masks,
            direct_delta, smoke_table, result["threshold"]
        )
        literal_loss = literal_policy_replay(
            model, direct_paths, direct_delta, direct_masks, direct_actions
        )
        np.testing.assert_allclose(literal_loss, direct_loss, rtol=2e-9, atol=2e-8)
        q1_actions = np.full_like(direct_actions, 6)
        q1_literal = literal_policy_replay(
            model, direct_paths, direct_delta, direct_masks, q1_actions
        )
        q1_fast = simulate_full_accept_reference(
            model, direct_paths, direct_scores, direct_masks, direct_delta
        )
        np.testing.assert_allclose(q1_literal[:, 0], q1_fast, rtol=2e-9, atol=2e-8)
        q1_independent = full_accept_reference(
            model, direct_paths, direct_masks, direct_delta
        )
        np.testing.assert_allclose(q1_independent, q1_fast, rtol=2e-9, atol=2e-8)
    return dict(status="PASS_SMOKE", scene=0,
                attack_batches=2, health_paths=64,
                policy_count=4,
                direct_health_intervention_max_difference=health_affine_error,
                full_accept_matches_frozen_B1_update=True,
                dynamic_cost_is_relative_to_q1=True,
                source_fixture_support_saved=True,
                atomic_output_roundtrip=True)


def preflight():
    branch = subprocess.check_output(
        ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
    ).strip()
    if branch != "d2-history-value-fusion-20261124":
        raise ValueError(f"D2-B runner must execute on its frozen D2 branch, found {branch}.")
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("D2-B formal runs require a clean committed worktree.")
    protocol_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "af279ba^{commit}"], text=True
    ).strip()
    if protocol_commit != "af279ba71b0edbdc10131f9318338ebd2599a119":
        raise ValueError("The frozen D2-B protocol commit could not be verified.")
    current_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("The checked-out D2-B protocol is not an ancestor of HEAD.")
    d1_head = subprocess.check_output(
        ["git", "-C", str(D1), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_d1 = "7e32a41b9802438f6645d9048ee46386caa53418"
    if d1_head != expected_d1:
        raise ValueError(f"Unexpected frozen D1 checkout: {d1_head}")
    expected_configuration = "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d"
    if digest(modes.CONFIG) != expected_configuration:
        raise ValueError("Frozen common-G physical configuration hash mismatch.")
    b1_plan = json.loads(
        (D1 / "B1_HEALTH_RUN.json").read_text(encoding="utf-8")
    )
    b1_policy = json.loads(
        (D1 / "b1_health_results" / "policy_lock.json").read_text(encoding="utf-8")
    )
    b2_run = json.loads(
        (D1 / "b2_attack_results" / "run_manifest.json").read_text(encoding="utf-8")
    )
    b2_response = json.loads(
        (D1 / "b2_weight_response_results" / "response_manifest.json").read_text(encoding="utf-8")
    )
    b1_status = b1_plan.get("authorization_status")
    if b1_status != "posthoc_execution_under_B0_definitions_pending_manual_confirmation":
        raise ValueError("B1 evidence identity changed or is missing.")
    if b1_plan.get("confirmatory_claims_allowed") is not False:
        raise ValueError("B1 posthoc data must not be labeled confirmatory.")
    if tuple(b1_plan["seed_namespace"]) == NAMESPACE:
        raise ValueError("D2-B seed namespace collides with B1.")
    if tuple(b2_run["seed_namespace"]) == NAMESPACE:
        raise ValueError("D2-B seed namespace collides with B2.")
    historical_namespaces = {
        "B1": tuple(b1_plan["seed_namespace"]),
        "B2": tuple(b2_run["seed_namespace"]),
        "D2-A-prime": tuple(b2_response["seed_namespace"]),
    }
    for label, old_namespace in historical_namespaces.items():
        if NAMESPACE[:len(old_namespace)] == old_namespace or old_namespace[:len(NAMESPACE)] == NAMESPACE:
            raise ValueError(f"D2-B seed namespace overlaps {label}.")
    if len(b1_policy["rows"]) != 12 or len(health.scenarios()) != 12:
        raise ValueError("The frozen D2 design requires twelve physical scenes.")
    if any(float(row["B"]) != float(row["D"]) for row in b1_policy["rows"]):
        raise ValueError("The frozen D1 B/D policy-equivalence check changed.")
    b1_run_record = json.loads(
        (D1 / "B1_HEALTH_RUN.json").read_text(encoding="utf-8")
    )
    if b1_run_record.get("confirmatory_claims_allowed") is not False:
        raise ValueError("B1 posthoc health paths must remain exploratory.")
    for directory, manifest_name, inputs_name, outputs_name in (
        (D1 / "b2_attack_results", "run_manifest.json", "inputs_sha256", "outputs_sha256"),
        (D1 / "b2_weight_response_results", "response_manifest.json", "input_sha256", "output_sha256"),
    ):
        record = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
        for relative, expected in record[inputs_name].items():
            source = D1_ROOT / relative.replace("\\", "/")
            if digest(source) != expected:
                raise ValueError(f"Frozen D1 dependency hash mismatch: {relative}")
        for filename, expected in record[outputs_name].items():
            if digest(directory / filename) != expected:
                raise ValueError(f"Frozen D1 output hash mismatch: {filename}")
    mode_verification = json.loads(
        (D1 / "b1_mode_results" / "verification.json").read_text(encoding="utf-8")
    )
    if mode_verification["status"] != "PASS":
        raise ValueError("Frozen D1 mode-table verification is not PASS.")
    if mode_verification["healthy_paths_generated"] or mode_verification["attack_paths_generated"]:
        raise ValueError("D1 mode verification scope is not deterministic-only.")
    graph_hashes = {}
    for scene, spec in enumerate(health.scenarios()):
        graph_file = GRAPH_DIR / f"{spec[1]}_{spec[0]}_target{spec[2]}_qwords.npz"
        if not graph_file.exists():
            raise FileNotFoundError(graph_file)
        graph_hashes[str(scene)] = digest(graph_file)
    input_hashes = {
        "D2_B_PROTOCOL.md": digest(D2_PROTOCOL),
        "D2_MODEL_CONTRACT.md": digest(D2_MODEL),
        "b_run.py": digest(Path(__file__)),
        "physical_configuration.json": digest(modes.CONFIG),
        "mode_tables.npz": digest(modes.OUTPUT / "mode_tables.npz"),
        "D1A_link_value.json": digest(D1 / "results" / "D1A_link_value.json"),
        "D1_mode_verification.json": digest(D1 / "b1_mode_results" / "verification.json"),
        "B1_HEALTH_RUN.json": digest(D1 / "B1_HEALTH_RUN.json"),
        "B1_POSTHOC_STATUS.md": digest(D1 / "B1_POSTHOC_STATUS.md"),
        "B1_policy_lock.json": digest(D1 / "b1_health_results" / "policy_lock.json"),
        "b1_health.py": digest(D1 / "b1_health.py"),
        "b1_mode_models.py": digest(D1 / "b1_mode_models.py"),
        "attack_waveform_generator": digest(
            D1_ROOT / "paper1_sscd_commonrandom_20261122" / "watson_crn.py"
        ),
        "H1_residual_and_score_source": digest(
            D1_ROOT / "paper1_history_ablation_20261123" / "ablation.py"
        ),
        "physical_path_generator": digest(
            D1_ROOT / "paper1_partial_observability_20261123" / "run_partial50.py"
        ),
        "M7_graphs": graph_hashes,
    }
    return {
        "D2_branch": branch,
        "D2_commit_at_preflight": current_commit,
        "D2_protocol_commit": protocol_commit,
        "D1_commit": d1_head,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "B1_authorization_status": b1_plan.get("authorization_status"),
        "B1_seed_namespace": b1_plan["seed_namespace"],
        "B2_seed_namespace": b2_run["seed_namespace"],
        "D2_A_prime_seed_namespace": b2_response["seed_namespace"],
        "D2B_seed_namespace": list(NAMESPACE),
        "seed_prefixes_disjoint": True,
        "physical_configuration_sha256": digest(modes.CONFIG),
        "mode_table_verification": mode_verification["status"],
        "B1_confirmatory_claims_allowed": b1_run_record.get("confirmatory_claims_allowed"),
        "M7_graphs_present": True,
        "input_sha256": input_hashes,
    }


def run_all(scene_indices=None):
    run_preflight = preflight()
    requested_all_scenes = scene_indices is None
    if scene_indices is None:
        scene_indices = list(range(12))
    else:
        scene_indices = sorted(set(scene_indices))
    if not scene_indices or any(scene not in range(12) for scene in scene_indices):
        raise ValueError("Select one or more registered scene indices.")
    run_identity = {
        "protocol_commit": run_preflight["D2_protocol_commit"],
        "execution_commit": run_preflight["D2_commit_at_preflight"],
        "protocol_sha256": run_preflight["input_sha256"]["D2_B_PROTOCOL.md"],
        "model_contract_sha256": run_preflight["input_sha256"]["D2_MODEL_CONTRACT.md"],
        "runner_sha256": run_preflight["input_sha256"]["b_run.py"],
        "input_sha256": run_preflight["input_sha256"],
        "seed_namespace": list(NAMESPACE),
        "scenes": scene_indices,
        "paths_per_attack_fixture": N,
        "fit_paths_per_attack_fixture": FIT,
        "health_calibration_paths": FIT,
        "health_evaluation_paths": EVALUATION,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    if OUT.exists():
        if not PROGRESS.exists():
            raise FileExistsError(f"Refusing to overwrite completed or unregistered output: {OUT}")
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if progress.get("run_identity") != run_identity:
            raise ValueError("Partial D2-B output belongs to a different frozen run identity.")
        results = progress["results"]
        completed = set(progress["completed_scenes"])
        for scene in completed:
            filename = f"scene_{int(scene):02}.npz"
            output = OUT / filename
            if not output.exists() or digest(output) != progress["output_sha256"].get(filename):
                raise ValueError(f"Completed scene output is missing or corrupt: {filename}")
    else:
        OUT.mkdir()
        results, completed = {}, set()
        progress = {
            "status": "in_progress_D2_B_development_screening",
            "run_identity": run_identity,
            "completed_scenes": [],
            "results": results,
            "output_sha256": {},
        }
        atomic_json(PROGRESS, progress)
    with np.load(modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        for scene in scene_indices:
            if scene in completed:
                continue
            result, arrays, _ = run_scene(scene, health.scenarios()[scene], gains)
            results[str(scene)] = result
            filename = f"scene_{scene:02}.npz"
            atomic_npz(OUT / filename, arrays)
            progress["results"] = results
            progress["completed_scenes"] = sorted([*completed, scene])
            progress["output_sha256"][filename] = digest(OUT / filename)
            atomic_json(PROGRESS, progress)
            completed.add(scene)
            print("completed B scene", scene, flush=True)
    if completed != set(scene_indices):
        raise RuntimeError("D2-B scene completion manifest is inconsistent.")
    manifest = dict(
        status=("complete_D2_B_development_screening" if requested_all_scenes
                else "partial_D2_B_development_screening"),
        protocol_commit=run_preflight["D2_protocol_commit"],
        execution_commit=run_preflight["D2_commit_at_preflight"],
        preflight=run_preflight,
        seed_namespace=list(NAMESPACE),
        attack_paths_per_fixture=N,
        attack_fit_paths_per_fixture=FIT,
        attack_evaluation_paths_per_fixture=EVALUATION,
        health_calibration_paths=FIT,
        health_evaluation_paths=EVALUATION,
        scenes=scene_indices,
        strategies=list(STRATEGIES),
        new_paths=True,
        claim_scope="development_screening_conditional_risk_proxy_not_closed_loop_bound",
        random_stream_rules={
            "physical": "SeedSequence([20261124,4401,1,scene,fixture,watermark,stage,batch])",
            "attack_stage": 2, "health_calibration_stage": 0, "health_evaluation_stage": 1,
            "watermark_streams_distinct_by_watermark_index": True,
        },
        input_sha256=run_preflight["input_sha256"],
        results=results,
        output_sha256={
            path.name: digest(path)
            for path in sorted(OUT.glob("scene_*.npz"))
        },
    )
    atomic_json(OUT / "B_RUN.json", manifest)
    PROGRESS.unlink()
    print(json.dumps({"status": manifest["status"], "scenes": scene_indices,
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(run_smoke(), indent=2))
    else:
        run_all()

