"""Independent verifier for the deterministic B1-1 mode tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json"
B0 = HERE / "B0_REGISTRATION.json"
RESULTS = HERE / "b1_mode_results"
Q_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
ORIGINAL_LINKS = ((0, 1), (2, 3), (15, 16), (17, 31), (14, 49), (17, 34))
RECEIVER = 26
RANK_TOL = 1e-10
PROJECTOR_TOL = 1e-8
DARE_TOL = 1e-8
PSD_TOL = 1e-10
W_MISSING = np.diag([0.0, 0.0, 1.0, 1.0])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(adjacency: np.ndarray) -> list[list[int]]:
    pending = set(range(50))
    result = []
    while pending:
        root = min(pending)
        seen = {root}
        todo = [root]
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


def graph(config: dict, name: str) -> np.ndarray:
    base = np.asarray(config["graph"]["adjacency"], dtype=np.uint8)
    if name == "U":
        assert np.flatnonzero(base[RECEIVER]).tolist() == [16, 28, 35]
        return base
    same_mode = sorted(int(node) for node in np.flatnonzero(base[RECEIVER])
                       if int(node) % 2 == RECEIVER % 2)
    assert same_mode == [16, 28]
    attacks = {(int(item["receiver"]), int(item["sender"]))
               for item in config["attack_links"]}
    for candidate in range(50):
        reasons = []
        if candidate % 2 == RECEIVER % 2:
            reasons.append("same_mode")
        if base[RECEIVER, candidate]:
            reasons.append("already_adjacent")
        if ((RECEIVER, candidate) in attacks or
                (candidate, RECEIVER) in attacks):
            reasons.append("directed_attack_arc_conflict")
        proposed = base.copy()
        proposed[RECEIVER, 16] = proposed[16, RECEIVER] = 0
        proposed[RECEIVER, candidate] = proposed[candidate, RECEIVER] = 1
        if len(components(proposed)) != 1:
            reasons.append("replacement_disconnects_graph")
        if not reasons:
            assert candidate == 1
            return proposed
    raise AssertionError("R topology search found no candidate.")


def model(config: dict, adjacency: np.ndarray, dynamics: str) -> dict:
    a2 = np.asarray(config[f"A_{dynamics}"], dtype=float)
    q2 = np.asarray(config["Q"], dtype=float)
    h2 = np.asarray(config["H"], dtype=float)
    r = np.asarray(config["R"], dtype=float)
    h = np.zeros((50, 2, 4), dtype=float)
    for node in range(50):
        block = node % 2
        h[node, :, 2 * block:2 * block + 2] = h2[node]
    return {"A": block_diag(a2, a2), "A2": a2,
            "Q": block_diag(q2, q2), "Q2": q2, "H": h, "R": r,
            "neighbors": [np.flatnonzero(row).astype(int).tolist()
                          for row in adjacency]}


def obs(C: np.ndarray, A: np.ndarray) -> tuple[int, np.ndarray]:
    matrix = np.vstack([C @ np.linalg.matrix_power(A, power)
                        for power in range(4)])
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular > RANK_TOL))
    basis = right[rank:].T
    return rank, basis @ basis.T


def local(model_data: dict, receiver: int, weights: dict[int, float],
          fallback_allowed: bool) -> dict:
    hsum = np.zeros((2, 4))
    rsum = np.zeros((2, 2))
    for sender in model_data["neighbors"][receiver]:
        q = weights.get(sender, 1.0)
        hsum += q * model_data["H"][sender]
        rsum += q * q * model_data["R"][sender]
    C = np.vstack([model_data["H"][receiver], hsum])
    V = block_diag(model_data["R"][receiver], rsum)
    rank, projector = obs(C, model_data["A"])
    fallback = False
    if rank == 4:
        Pminus = solve_discrete_are(model_data["A"].T, C.T,
                                    model_data["Q"], V)
        innovation = C @ Pminus @ C.T + V
        L = np.linalg.solve(innovation, C @ Pminus).T
        Pplus = Pminus - Pminus @ C.T @ np.linalg.solve(innovation, C @ Pminus)
        Pminus = (Pminus + Pminus.T) / 2
        Pplus = (Pplus + Pplus.T) / 2
    elif fallback_allowed:
        assert receiver == RECEIVER and rank == 2
        assert np.linalg.norm(projector - W_MISSING, ord=2) <= PROJECTOR_TOL
        Cvisible = C[:, :2]
        Pvisible = solve_discrete_are(model_data["A2"].T, Cvisible.T,
                                      model_data["Q2"], V)
        innovation = Cvisible @ Pvisible @ Cvisible.T + V
        Lvisible = np.linalg.solve(innovation, Cvisible @ Pvisible).T
        L = np.vstack([Lvisible, np.zeros((2, 4))])
        Pminus = np.full((4, 4), np.nan)
        Pplus = np.full((4, 4), np.nan)
        fallback = True
    else:
        raise AssertionError(f"Unexpected rank {rank} at receiver {receiver}.")
    identity = np.eye(4) - L @ C
    F = identity @ model_data["A"]
    Omega = identity @ model_data["Q"] @ identity.T + L @ V @ L.T
    Omega = (Omega + Omega.T) / 2
    residual = None if fallback else float(np.linalg.norm(
        Pplus - F @ Pplus @ F.T - Omega, ord="fro"))
    if residual is not None:
        assert residual <= DARE_TOL
        assert np.linalg.eigvalsh(Pminus).min() >= -PSD_TOL
        assert np.linalg.eigvalsh(Pplus).min() >= -PSD_TOL
    assert np.linalg.eigvalsh(Omega).min() >= -PSD_TOL
    return {"C": C, "V": V, "L": L, "K": L[:, :2], "G": L[:, 2:],
            "F": F, "Omega": Omega, "Pminus": Pminus, "Pplus": Pplus,
            "rank": rank, "fallback": fallback, "residual": residual}


def assert_array(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if np.isnan(expected).all():
        assert np.isnan(actual).all(), label
        return
    assert np.allclose(actual, expected, atol=3e-9, rtol=0), label


def main() -> None:
    report = json.loads((RESULTS / "B1_1_mode_models.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert digest(CONFIG) == report["input_sha256"]["physical_configuration.json"]
    assert digest(B0) == report["input_sha256"]["B0_REGISTRATION.json"]
    assert report["mode_definition"]["steady_state_and_switching_separated"]
    with np.load(RESULTS / "mode_tables.npz") as stored:
        checked_tables = 0
        checked_modes = 0
        max_difference = 0.0
        for topology in ("U", "R"):
            targets = (35,) if topology == "U" else (35, 1)
            adjacency = graph(config, topology)
            assert len(components(adjacency)) == 1
            for dynamics in ("unstable", "stable"):
                model_data = model(config, adjacency, dynamics)
                for target in targets:
                    links = ORIGINAL_LINKS + ((RECEIVER, target),)
                    for q_target in Q_GRID:
                        key = f"{topology}_{dynamics}_target{target}_q{q_target:g}"
                        table = report["tables"][key]
                        assert table["mode_count"] == 128
                        assert table["monitored_links_receiver_sender_order"] == [
                            list(link) for link in links]
                        matrices = {name: stored[f"{key}_{name}"] for name in
                                    ("C", "V", "L", "K", "G", "F", "Omega", "Pminus", "Pplus")}
                        masks = np.arange(128)
                        fallback_count = 0
                        for receiver in range(50):
                            bits = [(bit, j) for bit, (i, j) in enumerate(links) if i == receiver]
                            relevant = sum(1 << bit for bit, _ in bits)
                            for pattern in np.unique(masks & relevant):
                                matches = (masks & relevant) == pattern
                                weights = {j: (q_target if bit == 6 else 0.0)
                                           for bit, j in bits if pattern & (1 << bit)}
                                expected = local(model_data, receiver, weights,
                                                 topology == "U" and receiver == 26 and q_target == 0)
                                for name, data in matrices.items():
                                    actual = data[matches, receiver]
                                    assert_array(actual, expected[name], f"{key}/{receiver}/{pattern}/{name}")
                                    if np.isfinite(expected[name]).all():
                                        max_difference = max(max_difference, float(np.max(abs(actual - expected[name]))))
                                fallback_count += int(matches.sum()) * expected["fallback"]
                                checked_modes += int(matches.sum())
                        assert fallback_count == (64 if topology == "U" and q_target == 0 else 0)
                        assert table["fallback_mode_count"] == fallback_count
                        assert table["rank_min"] == (2 if fallback_count else 4)
                        # Unroll each exogenous schedule into a sum of propagated noise covariances.
                        for case in (c for c in report["sequence_checks"] if c["table"] == key):
                            trace = np.asarray(case["trace"])
                            contributions = []
                            for step, mask in enumerate(case["masks"], start=1):
                                f, o = matrices["F"][mask, 26], matrices["Omega"][mask, 26]
                                contributions = [(f @ phi, noise) for phi, noise in contributions]
                                contributions.append((np.eye(4), o))
                                direct = sum(phi @ noise @ phi.T for phi, noise in contributions)
                                np.testing.assert_allclose(trace[step], direct, rtol=2e-12, atol=1e-11)
                        checked_tables += 1
        assert checked_tables == len(report["tables"])
    verification = {
        "status": "PASS",
        "tables_checked": checked_tables,
        "mode_receiver_combinations_checked": checked_modes,
        "maximum_finite_matrix_difference": max_difference,
        "healthy_paths_generated": False,
        "attack_paths_generated": False,
        "mode_steady_state_and_switching_separated": True,
    }
    (RESULTS / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
