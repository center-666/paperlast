"""B1-1 deterministic mode-dependent common-G model construction.

This module builds steady-state gains for each candidate target weight and
binary H1 gate mask. It does not generate health paths, H1 scores, or attacks.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json"
B0 = HERE / "B0_REGISTRATION.json"
OUTPUT = HERE / "b1_mode_results"
WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
TARGETS = {"U": (35,), "R": (35, 1)}
ORIGINAL_LINKS = ((0, 1), (2, 3), (15, 16), (17, 31), (14, 49), (17, 34))
RECEIVER = 26
RANK_TOL = 1e-10
PROJECTOR_TOL = 1e-8
DARE_TOL = 1e-8
PSD_TOL = 1e-10
W_MISSING = np.diag([0.0, 0.0, 1.0, 1.0])


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def load_inputs() -> dict:
    expected = "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d"
    if digest(CONFIG) != expected:
        raise ValueError("Frozen physical configuration hash mismatch.")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def components(adjacency: np.ndarray) -> list[list[int]]:
    pending = set(range(len(adjacency)))
    result = []
    while pending:
        start = min(pending)
        seen = {start}
        todo = [start]
        while todo:
            node = todo.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
        result.append(sorted(seen))
        pending -= seen
    return result


def build_topology(config: dict, name: str) -> tuple[np.ndarray, dict]:
    base = np.asarray(config["graph"]["adjacency"], dtype=np.uint8)
    if name == "U":
        if np.flatnonzero(base[RECEIVER]).tolist() != [16, 28, 35]:
            raise ValueError("U topology receiver neighborhood changed.")
        return base, {"candidate_search": [], "replacement": None}

    if name != "R":
        raise ValueError(name)
    same_mode = sorted(int(node) for node in np.flatnonzero(base[RECEIVER])
                       if int(node) % 2 == RECEIVER % 2)
    attack_arcs = {(int(item["receiver"]), int(item["sender"]))
                   for item in config["attack_links"]}
    if same_mode != [16, 28]:
        raise ValueError("R same-mode candidate set changed.")
    search = []
    selected = None
    selected_graph = None
    for candidate in range(len(base)):
        reasons = []
        if candidate % 2 == RECEIVER % 2:
            reasons.append("same_mode")
        if base[RECEIVER, candidate]:
            reasons.append("already_adjacent")
        if ((RECEIVER, candidate) in attack_arcs or
                (candidate, RECEIVER) in attack_arcs):
            reasons.append("directed_attack_arc_conflict")
        proposed = base.copy()
        proposed[RECEIVER, 16] = proposed[16, RECEIVER] = 0
        proposed[RECEIVER, candidate] = proposed[candidate, RECEIVER] = 1
        if len(components(proposed)) != 1:
            reasons.append("replacement_disconnects_graph")
        search.append({"candidate": candidate, "rejection_reasons": reasons,
                       "connected_components": len(components(proposed))})
        if not reasons:
            selected = candidate
            selected_graph = proposed
            break
    if selected != 1:
        raise ValueError(f"R candidate search selected {selected}, expected 1.")
    return selected_graph, {
        "candidate_search": search,
        "replacement": {"removed_undirected_edge": [16, 26],
                         "added_undirected_edge": [1, 26]},
    }


def make_model(config: dict, adjacency: np.ndarray, dynamics: str) -> dict:
    a2 = np.asarray(config[f"A_{dynamics}"], dtype=float)
    q2 = np.asarray(config["Q"], dtype=float)
    h2 = np.asarray(config["H"], dtype=float)
    r = np.asarray(config["R"], dtype=float)
    a = block_diag(a2, a2)
    q = block_diag(q2, q2)
    h = np.zeros((50, 2, 4), dtype=float)
    for node in range(50):
        block = node % 2
        h[node, :, 2 * block:2 * block + 2] = h2[node]
    neighbors = [np.flatnonzero(row).astype(int).tolist() for row in adjacency]
    return {"A": a, "A2": a2, "Q": q, "Q2": q2, "H": h, "R": r,
            "neighbors": neighbors, "adjacency": adjacency}


def observability(c: np.ndarray, a: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    matrix = np.vstack([c @ np.linalg.matrix_power(a, power)
                        for power in range(a.shape[0])])
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular > RANK_TOL))
    basis = right[rank:].T
    return rank, singular, basis @ basis.T


def weighted_measurement(model: dict, receiver: int,
                         weights: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    hsum = np.zeros((2, 4), dtype=float)
    rsum = np.zeros((2, 2), dtype=float)
    for sender in model["neighbors"][receiver]:
        weight = weights.get(sender, 1.0)
        hsum += weight * model["H"][sender]
        rsum += weight * weight * model["R"][sender]
    return np.vstack([model["H"][receiver], hsum]), block_diag(
        model["R"][receiver], rsum)


def design_local(model: dict, receiver: int, weights: dict[int, float],
                 allow_u_fallback: bool) -> dict:
    c, v = weighted_measurement(model, receiver, weights)
    rank, singular, projector = observability(c, model["A"])
    fallback = False
    if rank == 4:
        pminus = solve_discrete_are(model["A"].T, c.T, model["Q"], v)
        innovation = c @ pminus @ c.T + v
        gain = np.linalg.solve(innovation, c @ pminus).T
        pplus = pminus - pminus @ c.T @ np.linalg.solve(innovation, c @ pminus)
        pminus = (pminus + pminus.T) / 2
        pplus = (pplus + pplus.T) / 2
    elif allow_u_fallback:
        if receiver != RECEIVER or rank != 2 or \
                np.linalg.norm(projector - W_MISSING, ord=2) > PROJECTOR_TOL:
            raise ValueError("U q=0 mode has an unregistered missing subspace.")
        pminus_visible = solve_discrete_are(model["A2"].T, c[:, :2].T,
                                             model["Q2"], v)
        innovation = c[:, :2] @ pminus_visible @ c[:, :2].T + v
        gain_visible = np.linalg.solve(
            innovation, c[:, :2] @ pminus_visible).T
        gain = np.vstack([gain_visible, np.zeros((2, 4))])
        pminus = np.full((4, 4), np.nan)
        pplus = np.full((4, 4), np.nan)
        fallback = True
    else:
        raise ValueError(f"Unexpected rank-deficient mode at receiver {receiver}.")
    identity = np.eye(4) - gain @ c
    f_matrix = identity @ model["A"]
    omega = identity @ model["Q"] @ identity.T + gain @ v @ gain.T
    omega = (omega + omega.T) / 2
    if np.linalg.eigvalsh(omega).min() < -PSD_TOL:
        raise ArithmeticError("Omega is not positive semidefinite.")
    residual = None if fallback else float(np.linalg.norm(
        pplus - f_matrix @ pplus @ f_matrix.T - omega, ord="fro"))
    if residual is not None and residual > DARE_TOL:
        raise ArithmeticError(f"DARE fixed-point residual {residual} exceeds tolerance.")
    return {"C": c, "V": v, "K": gain[:, :2], "G": gain[:, 2:],
            "L": gain, "F": f_matrix, "Omega": omega,
            "Pminus": pminus, "Pplus": pplus, "rank": rank,
            "singular_values": singular, "projector": projector,
            "fallback": fallback, "dare_residual": residual}


def mode_links(topology: str, target: int) -> tuple[tuple[int, int], ...]:
    if target not in TARGETS[topology]:
        raise ValueError("Target is not registered for this topology.")
    return ORIGINAL_LINKS + ((RECEIVER, target),)


def build_mode_table(config: dict, topology: str, dynamics: str,
                     target: int, q_target: float) -> tuple[dict, dict]:
    adjacency, topology_record = build_topology(config, topology)
    model = make_model(config, adjacency, dynamics)
    links = mode_links(topology, target)
    mode_count = 1 << len(links)
    arrays = {name: np.zeros((mode_count, 50, *shape), dtype=float)
              for name, shape in (("C", (4, 4)), ("V", (4, 4)),
                                  ("K", (4, 2)), ("G", (4, 2)),
                                  ("L", (4, 4)), ("F", (4, 4)),
                                  ("Omega", (4, 4)), ("Pminus", (4, 4)),
                                  ("Pplus", (4, 4)))}
    arrays["Pminus"][:] = np.nan
    arrays["Pplus"][:] = np.nan
    ranks = np.zeros((mode_count, 50), dtype=int)
    fallback = np.zeros((mode_count, 50), dtype=bool)
    local_cache = {}
    for mask in range(mode_count):
        for receiver in range(50):
            weights = {sender: 1.0 for sender in model["neighbors"][receiver]}
            for bit, (link_receiver, sender) in enumerate(links):
                if link_receiver != receiver or not (mask & (1 << bit)):
                    continue
                weights[sender] = q_target if bit == len(links) - 1 else 0.0
            weight_key = tuple((sender, float(weights.get(sender, 1.0)))
                               for sender in model["neighbors"][receiver])
            cache_key = (receiver, weight_key)
            if cache_key not in local_cache:
                local_cache[cache_key] = design_local(
                    model, receiver, weights,
                    allow_u_fallback=(topology == "U" and target == 35 and
                                      receiver == RECEIVER and
                                      q_target == 0.0))
            local = local_cache[cache_key]
            ranks[mask, receiver] = local["rank"]
            fallback[mask, receiver] = local["fallback"]
            for name in arrays:
                if np.isfinite(local[name]).all():
                    arrays[name][mask, receiver] = local[name]
    if np.any(ranks < 2):
        raise ArithmeticError("Mode table contains a rank-deficient local model below rank 2.")
    summary = {
        "topology": topology, "dynamics": dynamics, "target_sender": target,
        "target_receiver": RECEIVER, "q_target": q_target,
        "monitored_links_receiver_sender_order": [list(link) for link in links],
        "mode_count": mode_count, "topology_record": topology_record,
        "rank_min": int(ranks.min()), "rank_max": int(ranks.max()),
        "fallback_mode_count": int(fallback.sum()),
        "full_dare_mode_count": int((~fallback).sum()),
    }
    return arrays, summary


def propagate_mode_sequence(f_matrices: np.ndarray, omegas: np.ndarray,
                            mode_sequence: list[int], initial: np.ndarray) -> dict:
    covariance = np.asarray(initial, dtype=float).copy()
    trace = np.empty((len(mode_sequence) + 1, 4, 4), dtype=float)
    trace[0] = covariance
    max_asymmetry = 0.0
    for step, mode in enumerate(mode_sequence, start=1):
        covariance = f_matrices[mode] @ covariance @ f_matrices[mode].T + omegas[mode]
        max_asymmetry = max(max_asymmetry, float(np.linalg.norm(covariance - covariance.T)))
        covariance = (covariance + covariance.T) / 2
        if np.linalg.eigvalsh(covariance).min() < -PSD_TOL:
            raise ArithmeticError("Mode-sequence covariance lost PSD.")
        trace[step] = covariance
    return {"trace": trace,
            "maximum_symmetry_residual_fro": max_asymmetry,
            "minimum_eigenvalue": float(min(
                np.linalg.eigvalsh(item).min() for item in trace))}


def run() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite B1-1 output directory.")
    config = load_inputs()
    registration = json.loads(B0.read_text(encoding="utf-8"))
    if registration["weighted_mode_table"]["required"] is not True:
        raise ValueError("B0 does not require weighted mode tables.")
    OUTPUT.mkdir()
    arrays_for_archive = {}
    report = {
        "status": "complete_deterministic_B1_1_mode_models",
        "scope": "No healthy paths, H1 scores, or attacks",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "input_sha256": {"physical_configuration.json": digest(CONFIG),
                          "B0_REGISTRATION.json": digest(B0)},
        "mode_definition": {
            "weights": list(WEIGHTS), "rank_tolerance": RANK_TOL,
            "projector_tolerance": PROJECTOR_TOL, "dare_tolerance": DARE_TOL,
            "binary_original_links_receiver_sender": [list(link) for link in ORIGINAL_LINKS],
            "mode_key": "candidate_target_weight_and_binary_gate_mask",
            "steady_state_and_switching_separated": True,
        },
        "tables": {}, "sequence_checks": [],
    }
    for topology in ("U", "R"):
        for dynamics in ("unstable", "stable"):
            for target in TARGETS[topology]:
                for q_target in WEIGHTS:
                    key = f"{topology}_{dynamics}_target{target}_q{q_target:g}"
                    arrays, summary = build_mode_table(
                        config, topology, dynamics, target, q_target)
                    report["tables"][key] = {
                        "topology": topology, "dynamics": dynamics,
                        "target_sender": target, "q_target": q_target,
                        "mode_count": summary["mode_count"],
                        "rank_min": summary["rank_min"],
                        "rank_max": summary["rank_max"],
                        "fallback_mode_count": summary["fallback_mode_count"],
                        "full_dare_mode_count": summary["full_dare_mode_count"],
                        "monitored_links_receiver_sender_order": summary[
                            "monitored_links_receiver_sender_order"],
                        "topology_record": summary["topology_record"],
                    }
                    for name, value in arrays.items():
                        arrays_for_archive[f"{key}_{name}"] = value
                    f = arrays["F"][:, RECEIVER]
                    omega = arrays["Omega"][:, RECEIVER]
                    for sequence_name, sequence in (
                        ("all_mode_zero", [0] * 20),
                        ("alternating_target", [0, 64] * 10),
                        ("consecutive_and_simultaneous", [64, 65, 127, 63, 0, 0, 64, 0] * 3),
                    ):
                        sequence = [mode for mode in sequence
                                    if mode < summary["mode_count"]]
                        check = propagate_mode_sequence(
                            f, omega, sequence, np.zeros((4, 4)))
                        report["sequence_checks"].append({
                            "table": key, "sequence": sequence_name,
                            "length": len(sequence), "masks": sequence,
                            "trace": check["trace"].tolist(),
                            "minimum_eigenvalue": check["minimum_eigenvalue"],
                            "maximum_symmetry_residual_fro": check[
                                "maximum_symmetry_residual_fro"],
                        })
    np.savez_compressed(OUTPUT / "mode_tables.npz", **arrays_for_archive)
    report["archive_sha256"] = {"mode_tables.npz": digest(OUTPUT / "mode_tables.npz")}
    (OUTPUT / "B1_1_mode_models.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"],
                      "tables": len(report["tables"]),
                      "sequence_checks": len(report["sequence_checks"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    run()
