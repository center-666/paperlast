"""D1-A deterministic finite-horizon link-value calculations."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.linalg import block_diag, null_space, solve_discrete_are


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = "5919036776ace8cf5263d4d6a2078184fd58039b"
CONFIG = ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json"
REGISTRATION = HERE / "D1A_registration.json"
AMENDMENT = HERE / "D1A_AUDIT_AMENDMENT.md"
OUTPUT = HERE / "results"
WEIGHTS = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
HORIZONS = (10, 20, 40, 60)
RANK_TOL = 1e-10
PROJECTOR_TOL = 1e-8
DARE_RESIDUAL_TOL = 1e-8
PSD_TOL = 1e-10
W26 = np.diag([0.0, 0.0, 1.0, 1.0])
RECEIVER = 26


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_config() -> dict:
    if sha256_file(CONFIG) != "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d":
        raise ValueError("Frozen physical configuration hash mismatch.")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    remaining = set(range(len(adjacency)))
    components = []
    while remaining:
        root = min(remaining)
        stack = [root]
        seen = {root}
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(seen))
        remaining -= seen
    return components


def make_models(config: dict) -> tuple[dict, dict]:
    base = np.asarray(config["graph"]["adjacency"], dtype=np.uint8)
    same_mode = sorted(j for j in np.flatnonzero(base[RECEIVER])
                       if int(j) % 2 == RECEIVER % 2)
    if same_mode != [16, 28]:
        raise ValueError(f"Frozen same-mode receiver neighbors changed: {same_mode}.")
    remove_sender = same_mode[0]
    attack_arcs = {(int(row["sender"]), int(row["receiver"]))
                   for row in config["attack_links"]}
    search = []
    chosen = None
    chosen_adjacency = None
    for candidate in range(len(base)):
        reasons = []
        if candidate % 2 == RECEIVER % 2:
            reasons.append("same_mode")
        if base[RECEIVER, candidate]:
            reasons.append("already_adjacent")
        if ((candidate, RECEIVER) in attack_arcs or
                (RECEIVER, candidate) in attack_arcs):
            reasons.append("directed_attack_arc_conflict")
        proposed = base.copy()
        proposed[remove_sender, RECEIVER] = 0
        proposed[RECEIVER, remove_sender] = 0
        proposed[candidate, RECEIVER] = 1
        proposed[RECEIVER, candidate] = 1
        components = connected_components(proposed)
        if len(components) != 1:
            reasons.append("replacement_disconnects_graph")
        search.append({"candidate": candidate, "rejection_reasons": reasons,
                       "connected_components": len(components)})
        if not reasons:
            chosen = candidate
            chosen_adjacency = proposed
            break
    if chosen != 1:
        raise ValueError(f"Deterministic R topology selected {chosen}, expected 1.")
    r_neighbors = np.flatnonzero(chosen_adjacency[RECEIVER]).astype(int).tolist()
    if r_neighbors != [1, 28, 35]:
        raise ValueError(f"Unexpected R receiver neighborhood {r_neighbors}.")
    return ({"name": "U", "adjacency": base,
             "search": [], "replacement": None},
            {"name": "R", "adjacency": chosen_adjacency,
             "search": search, "replacement": {"removed_undirected_edge": [16, 26],
                                                "added_undirected_edge": [1, 26]}})


def model_arrays(config: dict, topology: dict, dynamics: str) -> dict:
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
    return {"A": a, "A2": a2, "Q": q, "Q2": q2,
            "H": h, "R": r, "adjacency": topology["adjacency"],
            "neighbors": [np.flatnonzero(row).astype(int).tolist()
                          for row in topology["adjacency"]]}


def observability(C: np.ndarray, A: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.vstack([C @ np.linalg.matrix_power(A, power)
                        for power in range(A.shape[0])])
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > RANK_TOL))
    null_basis = right[rank:].T
    projector = null_basis @ null_basis.T
    return rank, singular_values, null_basis, projector


def weighted_observation(model: dict, receiver: int,
                         weights: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    active = model["neighbors"][receiver]
    hsum = sum((weights.get(sender, 1.0) * model["H"][sender]
                for sender in active), start=np.zeros((2, 4)))
    rsum = sum((weights.get(sender, 1.0) ** 2 * model["R"][sender]
                for sender in active), start=np.zeros((2, 2)))
    c = np.vstack([model["H"][receiver], hsum])
    v = block_diag(model["R"][receiver], rsum)
    return c, v


def fallback_allowed(model: dict, receiver: int, c: np.ndarray) -> dict:
    rank, singular_values, null_basis, projector = observability(c, model["A"])
    difference = float(np.linalg.norm(projector - W26, ord=2))
    if receiver != RECEIVER or rank != 2 or difference > PROJECTOR_TOL:
        raise ValueError("Rank-deficient mode does not have the registered W26 missing subspace.")
    return {"rank": rank, "singular_values": singular_values,
            "null_basis": null_basis, "projector": projector,
            "projector_difference_norm": difference}


def design_gain(model: dict, receiver: int, weights: dict[int, float],
                *, allow_visible_block_fallback: bool) -> dict:
    c, v = weighted_observation(model, receiver, weights)
    rank, singular_values, _, _ = observability(c, model["A"])
    if rank == 4:
        pminus = solve_discrete_are(model["A"].T, c.T, model["Q"], v)
        innovation_covariance = c @ pminus @ c.T + v
        l = np.linalg.solve(innovation_covariance, c @ pminus).T
        pplus = pminus - pminus @ c.T @ np.linalg.solve(
            innovation_covariance, c @ pminus)
        pminus = (pminus + pminus.T) / 2
        pplus = (pplus + pplus.T) / 2
        fallback = None
        dare_residual = float(np.linalg.norm(
            model["A"] @ pplus @ model["A"].T + model["Q"] - pminus, ord="fro"))
    elif allow_visible_block_fallback:
        fallback = fallback_allowed(model, receiver, c)
        if receiver % 2 != 0:
            raise ValueError("D0 fallback is registered only for even receiver 26.")
        c_visible = c[:, :2]
        pminus_visible = solve_discrete_are(model["A2"].T, c_visible.T,
                                            model["Q2"], v)
        innovation_covariance = c_visible @ pminus_visible @ c_visible.T + v
        l_visible = np.linalg.solve(innovation_covariance,
                                    c_visible @ pminus_visible).T
        l = np.vstack([l_visible, np.zeros((2, 4))])
        pminus = np.full((4, 4), np.nan)
        pplus = np.full((4, 4), np.nan)
        dare_residual = None
        pminus_minimum_eigenvalue = None
        pplus_minimum_eigenvalue = None
    else:
        raise ValueError(f"Rank-deficient observation mode at receiver {receiver}.")
    f = (np.eye(4) - l @ c) @ model["A"]
    identity_minus_lc = np.eye(4) - l @ c
    omega = (identity_minus_lc @ model["Q"] @ identity_minus_lc.T
             + l @ v @ l.T)
    omega = (omega + omega.T) / 2
    if not np.all(np.isfinite(l)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(omega)):
        raise ArithmeticError("Gain design produced non-finite matrices.")
    omega_minimum = float(np.linalg.eigvalsh(omega).min())
    if omega_minimum < -PSD_TOL:
        raise ArithmeticError(f"Omega is not positive semidefinite: {omega_minimum}.")
    if pplus is not None and np.isfinite(pplus).all():
        pminus_minimum_eigenvalue = float(np.linalg.eigvalsh(pminus).min())
        pplus_minimum_eigenvalue = float(np.linalg.eigvalsh(pplus).min())
        if min(pminus_minimum_eigenvalue, pplus_minimum_eigenvalue) < -PSD_TOL:
            raise ArithmeticError("DARE covariance is not positive semidefinite.")
        fixed_point_residual = float(np.linalg.norm(
            pplus - f @ pplus @ f.T - omega, ord="fro"))
        if fixed_point_residual > DARE_RESIDUAL_TOL:
            raise ArithmeticError(f"Posterior covariance fixed point residual {fixed_point_residual} exceeds {DARE_RESIDUAL_TOL}.")
        if np.linalg.eigvalsh((pplus + pplus.T) / 2).min() < -PSD_TOL:
            raise ArithmeticError("DARE posterior covariance is not positive semidefinite.")
    else:
        fixed_point_residual = None
    return {"C": c, "V": v, "L": l, "K": l[:, :2], "G": l[:, 2:],
            "Pminus": pminus, "Pplus": pplus, "F": f, "Omega": omega,
            "rank": rank, "singular_values": singular_values,
            "dare_residual": dare_residual,
            "fixed_point_residual": fixed_point_residual,
            "Pminus_minimum_eigenvalue": pminus_minimum_eigenvalue,
            "Pplus_minimum_eigenvalue": pplus_minimum_eigenvalue,
            "fallback": fallback}


def covariance_trace(design: dict, p0: np.ndarray, horizon: int) -> tuple[np.ndarray, dict]:
    trajectory = np.empty((horizon + 1, 4, 4), dtype=float)
    trajectory[0] = p0
    maximum_symmetry_residual = 0.0
    minimum_eigenvalue = float(np.linalg.eigvalsh((p0 + p0.T) / 2).min())
    for step in range(horizon):
        raw_next = (design["F"] @ trajectory[step] @ design["F"].T
                    + design["Omega"])
        maximum_symmetry_residual = max(
            maximum_symmetry_residual,
            float(np.linalg.norm(raw_next - raw_next.T, ord="fro")))
        trajectory[step + 1] = (raw_next + raw_next.T) / 2
        step_minimum = float(np.linalg.eigvalsh(trajectory[step + 1]).min())
        minimum_eigenvalue = min(minimum_eigenvalue, step_minimum)
        if step_minimum < -PSD_TOL:
            raise ArithmeticError("Covariance recursion lost positive semidefiniteness.")
    audit = {"maximum_symmetry_residual_fro": maximum_symmetry_residual,
             "minimum_eigenvalue_over_trajectory": minimum_eigenvalue}
    return trajectory, audit


def risk(trajectory: np.ndarray, horizon: int) -> float:
    return float(sum(np.trace(W26 @ covariance) for covariance in trajectory[1:horizon + 1]))


def transfer_gain(f: np.ndarray, b: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    selector = np.eye(4)[2:, :]
    transfer = np.zeros((2 * horizon, 2 * horizon), dtype=float)
    for output_step in range(horizon):
        for input_step in range(output_step + 1):
            block = selector @ np.linalg.matrix_power(f, output_step - input_step) @ b
            transfer[2 * output_step:2 * output_step + 2,
                     2 * input_step:2 * input_step + 2] = block
    singular = np.linalg.svd(transfer, compute_uv=False)
    return transfer, float(singular[0] ** 2)


def matrix_hash(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def sender_parity_audit(model: dict, sender: int) -> dict:
    stack = np.vstack([model["H"][sender] @ np.linalg.matrix_power(model["A"], step)
                       for step in range(5)])
    parity = null_space(stack.T).T
    return {"sender": sender,
            "observable_stack_shape": list(stack.shape),
            "observable_stack_rank": int(np.linalg.matrix_rank(stack, tol=RANK_TOL)),
            "parity_basis_shape": list(parity.shape),
            "annihilation_residual_fro": float(np.linalg.norm(parity @ stack, ord="fro"))}


def topology_record(topology: dict, config: dict) -> dict:
    adjacency = topology["adjacency"]
    degrees = adjacency.sum(axis=1).astype(int)
    old_degrees = np.asarray(config["graph"]["adjacency"], dtype=np.uint8).sum(axis=1)
    h1 = np.asarray(config["H"][1], dtype=float)
    h35 = np.asarray(config["H"][35], dtype=float)
    r1 = np.asarray(config["R"][1], dtype=float)
    r35 = np.asarray(config["R"][35], dtype=float)
    old_attack_arcs = [[int(row["sender"]), int(row["receiver"])]
                       for row in config["attack_links"]]
    target_links = ([{"risk_scan_index_zero_based": 6, "sender": 35, "receiver": 26}]
                    if topology["name"] == "U" else
                    [{"risk_scan_index_zero_based": 6, "sender": 35, "receiver": 26},
                     {"risk_scan_index_zero_based": 7, "sender": 1, "receiver": 26}])
    return {"name": topology["name"], "adjacency": adjacency.astype(int).tolist(),
            "adjacency_sha256": matrix_hash(adjacency),
            "undirected_edge_count": int(adjacency.sum() // 2),
            "connected_components": connected_components(adjacency),
            "degree_changes": {str(node): {"old": int(old_degrees[node]),
                                            "new": int(degrees[node]),
                                            "difference": int(degrees[node] - old_degrees[node])}
                               for node in range(len(adjacency))},
            "H1": h1.tolist(), "H35": h35.tolist(),
            "R1": r1.tolist(), "R35": r35.tolist(),
            "H1_minus_H35": (h1 - h35).tolist(),
            "R1_minus_R35": (r1 - r35).tolist(),
            "H_difference_fro": float(np.linalg.norm(h1 - h35, ord="fro")),
            "R_difference_fro": float(np.linalg.norm(r1 - r35, ord="fro")),
            "replacement": topology["replacement"],
            "changed_directed_arcs": ({"removed": [], "added": []}
                                      if topology["name"] == "U" else
                                      {"removed": [[16, 26], [26, 16]],
                                       "added": [[1, 26], [26, 1]]}),
            "original_six_attack_arcs": old_attack_arcs,
            "D1A_risk_scan_link_indices_zero_based": target_links,
            "link_index_note": "Indices identify the D1-A risk-scan table only. They do not redefine the frozen detector's six-edge mask.",
            "added_direction_parity_roles": ([] if topology["name"] == "U" else
                [{"sender": 1, "receiver": 26, "risk_scan_index_zero_based": 7,
                  "role": "new target direction; sender-1 parity audit"},
                 {"sender": 26, "receiver": 1, "risk_scan_index_zero_based": None,
                  "role": "reciprocal healthy direction; sender-26 parity audit"}]),
            "redesign_nodes": [1, 16, 26] if topology["name"] == "R" else [26],
            "candidate_search": topology["search"]}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite D1-A output directory: {OUTPUT}")
    config = load_config()
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    registered_hashes = registration["pre_execution_hashes"]
    for name in (".gitattributes", "link_value_design.py", "verify_link_value.py",
                 "D1A_AUDIT_AMENDMENT.md"):
        actual = sha256_file(HERE / name)
        if registered_hashes.get(name) != actual:
            raise ValueError(f"D1-A pre-execution hash mismatch for {name}.")
    source_paths = {
        "physical_configuration.json": CONFIG,
        "D1A_registration.json": REGISTRATION,
        "D1A_AUDIT_AMENDMENT.md": AMENDMENT,
        "run_partial50.py": ROOT / "paper1_partial_observability_20261123" / "run_partial50.py",
        "run_critical.py": ROOT / "paper1_critical_complement_20261123" / "run_critical.py",
        "sweep_link_weight.py": ROOT / "paper1_history_ablation_20261123" / "sweep_link_weight.py",
        "ablation.py": ROOT / "paper1_history_ablation_20261123" / "ablation.py",
        "watson_crn.py": ROOT / "paper1_sscd_commonrandom_20261122" / "watson_crn.py",
        "design_common_g.py": ROOT / "paper1_common_g50_gain_acceptance_20261121" / "design_common_g.py",
        "run_common_g_50.py": ROOT / "paper1_common_g50_gain_acceptance_20261121" / "run_common_g_50.py",
        "source_model.py": ROOT / "paper1_50node_tv_gate_theory_20261121" / "source_model.py",
        "replay.py": ROOT / "paper1_50node_tv_gate_theory_20261121" / "replay.py",
    }
    expected_source_hashes = {
        "physical_configuration.json": "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d",
        "run_partial50.py": "94b53739598ea46b2dc40595b338a6d368d3d5e398d1e2797e785582e17b026e",
        "run_critical.py": "f98d97f3fa92d684da90d6e272979a9cfb3ce5cd5b9df6f0086330d6f0af8c80",
        "sweep_link_weight.py": "64749de7ff62f3994b545bc11a58260d69cdc43ef4d9c2862fd32368d3e60219",
        "ablation.py": "e440f2ca5c775161ff12e03b7b0cc0ca1de66b5753c0e30ad2f54d0e32697895",
        "watson_crn.py": "e3c3af17fed3a6afb2c71aa8a33ede942ba80796b7b34862ec6768bc59068f7a",
        "design_common_g.py": "68f9ef06e5f7c1696e68ac850b045716a225f90d1dd60625f659238bb8f10669",
        "run_common_g_50.py": "22ea30c33321da57aac0bf56b987221b529fc5020ada74b6cc789615f42d096c",
        "source_model.py": "21c2b1e58a4b88db1f44e9ba41660168c577e001873a522e3dc17d70dd83b336",
        "replay.py": "05e68f702ccea35d4bd5d4c1396c4acd9dbc79f4c6659150b1e283f279ddbabb",
    }
    frozen_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    for name, expected_hash in expected_source_hashes.items():
        if frozen_sha[name] != expected_hash:
            raise ValueError(f"Frozen upstream source hash mismatch for {name}.")
    u_topology, r_topology = make_models(config)
    topologies = {"U": u_topology, "R": r_topology}
    topology_documents = {name: topology_record(topology, config)
                          for name, topology in topologies.items()}
    arrays = {}
    report = {
        "status": "complete_deterministic_D1A",
        "scope": "No Monte Carlo paths; no D1-B or D1-C",
        "frozen_upstream_commit": FROZEN,
        "registration_commit": "0738004d9cbf7ea53e20023c04e03b5c691981aa",
        "input_sha256": frozen_sha,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform()},
        "noise_assumptions": [
            "process noise is temporally white with covariance Q",
            "measurement noises are independent across sensors",
            "measurement noises are independent across time",
            "measurement noise is independent of process noise",
        ],
        "matrix_dimensions": {"x": [4], "local_measurement": [2],
                              "C": [4, 4], "V": [4, 4],
                              "K": [4, 2], "G": [4, 2], "L": [4, 4],
                              "F": [4, 4], "Omega": [4, 4],
                              "Pminus": [4, 4], "Pplus": [4, 4],
                              "delta_packet": [2], "S1": [2, 2], "S35": [2, 2]},
        "attack_input_definition": {
            "delta_packet": "M_k a_k, the already-watermarked additive packet perturbation",
            "M_k_shape": [2, 2], "M_k_is_time_varying": True,
            "injection_matrices": {"S1": np.eye(2).tolist(), "S35": np.eye(2).tolist()},
            "B_delta": "q_j G(q) S_j; do not apply M_k again",
        },
        "numerical_rules": {"q": WEIGHTS.tolist(), "horizons": list(HORIZONS),
                            "absolute_observability_singular_value_tolerance": RANK_TOL,
                            "unobservable_projector_tolerance": PROJECTOR_TOL,
                            "DARE_fixed_point_residual_tolerance": DARE_RESIDUAL_TOL,
                            "attack_injection_matrices": {"S1": np.eye(2).tolist(),
                                                          "S35": np.eye(2).tolist()}},
        "topologies": topology_documents,
        "models": {},
    }
    report["D0_regression"] = check_d0_regression(config, u_topology)
    for topology_name, topology in topologies.items():
        for dynamics in ("unstable", "stable"):
            model = model_arrays(config, topology, dynamics)
            if topology_name == "U":
                if model["neighbors"][26] != [16, 28, 35]:
                    raise ValueError("U topology no longer matches frozen D0 neighbors.")
                endpoint_nodes = [1, 16, 26]
                targets = [(35, {35})]
            else:
                if model["neighbors"][26] != [1, 28, 35]:
                    raise ValueError("R topology does not have registered receiver neighbors.")
                endpoint_nodes = [1, 16, 26]
                targets = [(35, {35}), (1, {1})]
            model_key = f"{topology_name}_{dynamics}"
            network_h = model["H"].reshape(-1, 4)
            network_rank, network_singular, _, _ = observability(network_h, model["A"])
            record = {"network_observability_rank": network_rank,
                      "network_observability_singular_values": network_singular.tolist(),
                      "local_observability_ranks": {
                          str(node): observability(model["H"][node], model["A"])[0]
                          for node in range(50)},
                      "redesign_nodes": endpoint_nodes,
                      "endpoint_mode_table_note": "Nodes 1 and 16 have one all-retained row because no D1-A attacked directed arc enters either receiver; node 26 has the registered single-link q scan rows.",
                      "endpoint_designs": {}, "link_scans": {}}
            parity_senders = [35] if topology_name == "U" else [1, 16, 26, 35]
            record["sender_parity_audit"] = {
                str(sender): sender_parity_audit(model, sender)
                for sender in parity_senders}
            for receiver in endpoint_nodes:
                endpoint_weights = {sender: 1.0 for sender in model["neighbors"][receiver]}
                endpoint = design_gain(model, receiver, endpoint_weights,
                                       allow_visible_block_fallback=False)
                record["endpoint_designs"][str(receiver)] = {
                    "neighbors": model["neighbors"][receiver],
                    "rank": endpoint["rank"],
                    "singular_values": endpoint["singular_values"].tolist(),
                    "dare_residual": endpoint["dare_residual"],
                    "fixed_point_residual": endpoint["fixed_point_residual"],
                    "Pminus_minimum_eigenvalue": endpoint["Pminus_minimum_eigenvalue"],
                    "Pplus_minimum_eigenvalue": endpoint["Pplus_minimum_eigenvalue"],
                    "norm_F_2": float(np.linalg.norm(endpoint["F"], 2)),
                }
                for matrix_name in ("C", "V", "L", "K", "G", "Pminus", "Pplus", "F", "Omega"):
                    arrays[f"{model_key}_endpoint{receiver}_{matrix_name}"] = endpoint[matrix_name]
            p_keep = design_gain(model, RECEIVER,
                                 {sender: 1.0 for sender in model["neighbors"][RECEIVER]},
                                 allow_visible_block_fallback=False)["Pplus"]
            if p_keep is None or not np.isfinite(p_keep).all():
                raise ArithmeticError("Full-accept DARE did not provide finite Pplus.")
            arrays[f"{model_key}_P0_keep"] = p_keep
            arrays[f"{model_key}_P0_zero"] = np.zeros((4, 4))
            model_radius = float(max(abs(np.linalg.eigvals(model["A"][2:, 2:]))))
            for target_sender, _ in targets:
                target_key = f"{model_key}_edge{target_sender}_to26"
                scan = {"target_sender": target_sender,
                        "risk_scan_index_zero_based": 6 if target_sender == 35 else 7,
                        "fixed_other_complementary_weight": 1.0,
                        "q_values": WEIGHTS.tolist(), "weights": [], "kappa": {}}
                trajectories = {"keep": [], "zero": []}
                for qvalue in WEIGHTS:
                    weights = {sender: 1.0 for sender in model["neighbors"][RECEIVER]}
                    weights[target_sender] = float(qvalue)
                    design = design_gain(model, RECEIVER, weights,
                                         allow_visible_block_fallback=(RECEIVER == 26))
                    item = {"q": float(qvalue), "weights": {str(k): v for k, v in weights.items()},
                            "rank": design["rank"],
                            "singular_values": design["singular_values"].tolist(),
                            "dare_residual": design["dare_residual"],
                            "fixed_point_residual": design["fixed_point_residual"],
                            "Pminus_minimum_eigenvalue": design["Pminus_minimum_eigenvalue"],
                            "Pplus_minimum_eigenvalue": design["Pplus_minimum_eigenvalue"],
                            "norm_F_2": float(np.linalg.norm(design["F"], 2)),
                            "norm_Omega_2": float(np.linalg.norm(design["Omega"], 2)),
                            "minimum_eigenvalue_Omega": float(np.linalg.eigvalsh(
                                (design["Omega"] + design["Omega"].T) / 2).min()),
                            "fallback": design["fallback"] is not None}
                    scan["weights"].append(item)
                    for name in ("C", "V", "L", "K", "G", "Pminus", "Pplus", "F", "Omega"):
                        arrays[f"{target_key}_q{qvalue:g}_{name}"] = design[name]
                    covariance_audits = {}
                    for initial_name, initial in (("keep", p_keep), ("zero", np.zeros((4, 4)))):
                        trajectory, covariance_audits[initial_name] = covariance_trace(
                            design, initial, max(HORIZONS))
                        trajectories[initial_name].append(trajectory)
                        arrays[f"{target_key}_q{qvalue:g}_P_{initial_name}"] = trajectory
                    item["covariance_audit_by_initial"] = covariance_audits
                    delta = float(qvalue) * design["G"] @ np.eye(2)
                    arrays[f"{target_key}_q{qvalue:g}_B_delta"] = delta
                    item["attack_transfer_gain_by_horizon"] = {
                        str(horizon): transfer_gain(design["F"], delta, horizon)[1]
                        for horizon in HORIZONS}
                scan["risk"] = {initial_name: [] for initial_name in ("keep", "zero")}
                for initial_name in ("keep", "zero"):
                    scan["risk"][initial_name] = [
                        {"q": float(qvalue), "J_by_horizon": {
                            str(horizon): risk(trajectories[initial_name][qi], horizon)
                            for horizon in HORIZONS}}
                        for qi, qvalue in enumerate(WEIGHTS)]
                    j0 = {h: risk(trajectories[initial_name][0], h) for h in HORIZONS}
                    j1 = {h: risk(trajectories[initial_name][-1], h) for h in HORIZONS}
                    scan["kappa"][initial_name] = {
                        str(h): float((j0[h] - j1[h]) / (j1[h] + 1e-12))
                        for h in HORIZONS}
                # Store observability geometry for full, singly weakened, and deleted links.
                weight_one = {sender: 1.0 for sender in model["neighbors"][RECEIVER]}
                geometries = {}
                for label, qvalue in (("keep", 1.0), ("delete", 0.0)):
                    local_weights = dict(weight_one)
                    local_weights[target_sender] = qvalue
                    c, _ = weighted_observation(model, RECEIVER, local_weights)
                    rank_value, singular, null_basis, projector = observability(c, model["A"])
                    geometries[label] = {"rank": rank_value,
                                         "singular_values": singular.tolist(),
                                         "null_basis": null_basis.tolist(),
                                         "unobservable_projector": projector.tolist(),
                                         "projector_difference_norm_W26": float(np.linalg.norm(projector-W26, 2))}
                scan["observability"] = geometries
                scan["open_loop_missing_block_spectral_radius"] = model_radius
                record["link_scans"][str(target_sender)] = scan
            report["models"][model_key] = record
    report["topology_endpoint_differences"] = {}
    for dynamics in ("unstable", "stable"):
        model_key_u = f"U_{dynamics}"
        model_key_r = f"R_{dynamics}"
        report["topology_endpoint_differences"][dynamics] = {}
        for receiver in (1, 16, 26):
            matrices = {}
            for matrix_name in ("C", "V", "L", "K", "G", "F", "Omega"):
                before = arrays[f"{model_key_u}_endpoint{receiver}_{matrix_name}"]
                after = arrays[f"{model_key_r}_endpoint{receiver}_{matrix_name}"]
                matrices[matrix_name] = {
                    "maximum_absolute_difference": float(np.max(np.abs(after - before))),
                    "frobenius_difference": float(np.linalg.norm(after - before, ord="fro")),
                }
            report["topology_endpoint_differences"][dynamics][str(receiver)] = {
                "U_neighbors": model_arrays(config, u_topology, dynamics)["neighbors"][receiver],
                "R_neighbors": model_arrays(config, r_topology, dynamics)["neighbors"][receiver],
                "matrix_differences_R_minus_U": matrices,
            }
    save_outputs(report, topology_documents, arrays, frozen_sha)


def check_d0_regression(config: dict, u_topology: dict) -> dict:
    """Compare U receiver-26 gains against the frozen D0 design implementation."""
    import sys
    sys.path.insert(0, str(ROOT / "paper1_critical_complement_20261123"))
    sys.path.insert(0, str(ROOT / "paper1_history_ablation_20261123"))
    from run_critical import critical_model
    from sweep_link_weight import design_weighted_gains as frozen_weighted_design

    checks = {}
    for dynamics in ("unstable", "stable"):
        model_arrays_u = model_arrays(config, u_topology, dynamics)
        reference_model, _, _ = critical_model(dynamics, config)
        for qvalue in (0.0, 0.8, 1.0):
            actual_weights = {sender: 1.0 for sender in model_arrays_u["neighbors"][RECEIVER]}
            actual_weights[35] = qvalue
            actual = design_gain(model_arrays_u, RECEIVER, actual_weights,
                                 allow_visible_block_fallback=(qvalue == 0.0))
            if qvalue == 0.0:
                expected_mode = 1 << 6
                expected = {name: getattr(reference_model, name)[expected_mode, RECEIVER]
                            for name in ("K", "G", "F")}
            else:
                tables = frozen_weighted_design(reference_model, qvalue)
                expected = {"K": tables[0][0, RECEIVER],
                            "G": tables[1][0, RECEIVER],
                            "F": tables[2][0, RECEIVER]}
            differences = {name: float(np.max(np.abs(actual[name] - expected[name])))
                           for name in ("K", "G", "F")}
            maximum = max(differences.values())
            if maximum > 2e-10:
                raise ArithmeticError(f"D0 regression failed for {dynamics}, q={qvalue}: {differences}")
            checks[f"{dynamics}_q{qvalue:g}"] = {
                "max_abs_difference": maximum,
                "matrix_differences": differences,
                "reference": "frozen D0 critical-mode gains for q=0; frozen D0 weighted gain designer for q=0.8,1",
            }
    return checks


def save_outputs(report: dict, topologies: dict, arrays: dict, source_hashes: dict) -> None:
    OUTPUT.mkdir(parents=False)
    results = OUTPUT
    for topology_name, record in topologies.items():
        filename = "topology_unique.json" if topology_name == "U" else "topology_redundant.json"
        (results / filename).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8", newline="\n")
    np.savez_compressed(results / "gain_tables.npz", **{
        key: value for key, value in arrays.items()
        if any(key.endswith("_" + name) for name in
               ("C", "V", "L", "K", "G", "Pminus", "Pplus", "F", "Omega", "B_delta"))})
    np.savez_compressed(results / "covariance_traces.npz", **{
        key: value for key, value in arrays.items() if "_P_keep" in key or "_P_zero" in key})
    artifact_hashes = {path.name: sha256_file(path) for path in results.iterdir()
                       if path.is_file() and path.suffix in {".json", ".npz"}}
    report["result_sha256"] = artifact_hashes
    (results / "D1A_link_value.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    (results / "REPRODUCTION.md").write_text(
        "# D1-A deterministic reproduction\n\n"
        "Run `python link_value_design.py` followed by `python verify_link_value.py`. "
        "The programs create deterministic matrix, topology, gain and covariance artifacts only. "
        "They do not generate Monte Carlo attack paths. The output directory is never overwritten.\n\n"
        "`sha256.txt` records all pinned inputs and generated artifacts except itself, whose digest "
        "cannot be included in its own manifest. The independent verifier checks the listed inputs "
        "and result artifacts.\n",
        encoding="utf-8", newline="\n")
    hashes = {f"inputs/{name}": value for name, value in source_hashes.items()}
    hashes.update({f"results/{path.name}": sha256_file(path)
                   for path in results.iterdir() if path.is_file()})
    hashes.update({f"source/{name}": sha256_file(HERE / name)
                   for name in (".gitattributes", "link_value_design.py",
                                "verify_link_value.py", "D1A_DESIGN.md")})
    (results / "sha256.txt").write_text("".join(f"{value}  {name}\n" for name, value in sorted(hashes.items())),
                                       encoding="utf-8", newline="\n")


if __name__ == "__main__":
    run()
