"""Independent deterministic verification of D1-A matrices and recursions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, null_space, solve_discrete_are


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = ROOT / "paper1_common_g50_gain_acceptance_20261121" / "physical_configuration.json"
RESULTS = HERE / "results"
Q_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
HORIZONS = (10, 20, 40, 60)
RANK_CUTOFF = 1e-10
PROJECTOR_CUTOFF = 1e-8
DARE_CUTOFF = 1e-8
PSD_CUTOFF = 1e-10
W_MISSING = np.diag([0.0, 0.0, 1.0, 1.0])


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_config() -> dict:
    raw = CONFIG.read_bytes()
    if digest(raw) != "c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d":
        raise AssertionError("Verifier did not read the pinned physical configuration.")
    return json.loads(raw)


def pinned_input_paths() -> dict[str, Path]:
    return {
        "physical_configuration.json": CONFIG,
        "D1A_registration.json": HERE / "D1A_registration.json",
        "D1A_AUDIT_AMENDMENT.md": HERE / "D1A_AUDIT_AMENDMENT.md",
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


def verify_hash_manifest(summary: dict) -> dict:
    manifest_path = RESULTS / "sha256.txt"
    manifest = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest_value, relative_name = line.split("  ", maxsplit=1)
        if relative_name in manifest:
            raise AssertionError(f"Duplicate SHA-256 manifest entry: {relative_name}")
        manifest[relative_name] = digest_value

    input_paths = pinned_input_paths()
    if set(summary["input_sha256"]) != set(input_paths):
        raise AssertionError("Input hash inventory does not match the registered source inventory.")
    expected_entries = {f"inputs/{name}" for name in input_paths}
    expected_entries.update(
        f"results/{path.name}" for path in RESULTS.iterdir()
        if path.is_file() and path.name not in {"sha256.txt", "verification.json"})
    expected_entries.update(
        f"source/{name}" for name in
        (".gitattributes", "link_value_design.py", "verify_link_value.py", "D1A_DESIGN.md"))
    if set(manifest) != expected_entries:
        raise AssertionError("SHA-256 manifest does not cover the complete registered input and output set.")
    checked = {}
    for name, path in input_paths.items():
        observed = digest(path.read_bytes())
        if observed != summary["input_sha256"][name]:
            raise AssertionError(f"Input digest differs from result record: {name}")
        if manifest.get(f"inputs/{name}") != observed:
            raise AssertionError(f"Input digest differs from sha256.txt: {name}")
        checked[f"inputs/{name}"] = observed

    for relative_name, expected in manifest.items():
        if relative_name.startswith("results/"):
            path = HERE / relative_name
        elif relative_name.startswith("source/"):
            path = HERE / relative_name.removeprefix("source/")
        elif relative_name.startswith("inputs/"):
            continue
        else:
            raise AssertionError(f"Unexpected hash-manifest namespace: {relative_name}")
        observed = digest(path.read_bytes())
        if observed != expected:
            raise AssertionError(f"Hash-manifest mismatch: {relative_name}")
        checked[relative_name] = observed
    return checked


def reconstruct_graph(config: dict, label: str) -> tuple[np.ndarray, dict]:
    adjacency = np.asarray(config["graph"]["adjacency"], dtype=np.uint8)
    if label == "U":
        if np.flatnonzero(adjacency[26]).tolist() != [16, 28, 35]:
            raise AssertionError("Frozen U neighbors differ.")
        return adjacency, {"removed": None, "added": None, "candidate_history": []}
    if label != "R":
        raise ValueError(label)
    directed_attacks = {(int(item["sender"]), int(item["receiver"]))
                        for item in config["attack_links"]}
    same_mode = sorted(int(node) for node in np.flatnonzero(adjacency[26])
                       if int(node) % 2 == 26 % 2)
    if same_mode != [16, 28]:
        raise AssertionError(f"Unexpected same-mode neighbors {same_mode}.")
    removed_node = same_mode[0]
    tested = []
    selected = None
    selected_graph = None
    for node in range(len(adjacency)):
        why = []
        if node % 2 == 26 % 2:
            why.append("same_mode")
        if adjacency[26, node]:
            why.append("already_adjacent")
        if ((node, 26) in directed_attacks or (26, node) in directed_attacks):
            why.append("directed_attack_arc_conflict")
        proposed = adjacency.copy()
        proposed[removed_node, 26] = proposed[26, removed_node] = 0
        proposed[node, 26] = proposed[26, node] = 1
        components = components_of(proposed)
        if len(components) != 1:
            why.append("replacement_disconnects_graph")
        tested.append({"candidate": node, "rejection_reasons": why,
                       "connected_components": len(components)})
        if not why:
            selected = node
            selected_graph = proposed
            break
    if selected != 1:
        raise AssertionError(f"Ascending independent search selected {selected}.")
    return selected_graph, {"removed": [removed_node, 26], "added": [selected, 26],
                            "candidate_history": tested}


def components_of(adjacency: np.ndarray) -> list[list[int]]:
    pending = set(range(len(adjacency)))
    result = []
    while pending:
        start = min(pending)
        seen = {start}
        todo = [start]
        while todo:
            current = todo.pop()
            for neighbor in np.flatnonzero(adjacency[current]):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
        result.append(sorted(seen))
        pending -= seen
    return result


def construct_physical(config: dict, adjacency: np.ndarray, dynamics: str) -> dict:
    a2 = np.asarray(config["A_" + dynamics], dtype=float)
    q2 = np.asarray(config["Q"], dtype=float)
    old_h = np.asarray(config["H"], dtype=float)
    h4 = np.zeros((50, 2, 4), dtype=float)
    for node in range(50):
        block = 0 if node % 2 == 0 else 2
        h4[node, :, block:block + 2] = old_h[node]
    return {"A": block_diag(a2, a2), "A2": a2,
            "Q": block_diag(q2, q2), "Q2": q2, "H": h4,
            "R": np.asarray(config["R"], dtype=float),
            "neighbors": [np.flatnonzero(row).astype(int).tolist() for row in adjacency]}


def rank_and_null(C: np.ndarray, A: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.vstack([C @ np.linalg.matrix_power(A, t) for t in range(4)])
    _, values, right = np.linalg.svd(obs, full_matrices=True)
    rank = int(np.count_nonzero(values > RANK_CUTOFF))
    basis = right[rank:].T
    projection = basis @ basis.T
    return rank, values, basis, projection


def make_measurements(model: dict, receiver: int,
                      q_by_sender: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    hsum = np.zeros((2, 4))
    rsum = np.zeros((2, 2))
    for sender in model["neighbors"][receiver]:
        qval = q_by_sender.get(sender, 1.0)
        hsum += qval * model["H"][sender]
        rsum += qval * qval * model["R"][sender]
    return (np.vstack([model["H"][receiver], hsum]),
            block_diag(model["R"][receiver], rsum))


def rebuild_gain(model: dict, receiver: int, q_by_sender: dict[int, float],
                 use_registered_fallback: bool) -> dict:
    c, v = make_measurements(model, receiver, q_by_sender)
    rank, singular, null_basis, projector = rank_and_null(c, model["A"])
    if rank == 4:
        pminus = solve_discrete_are(model["A"].T, c.T, model["Q"], v)
        innovation = c @ pminus @ c.T + v
        l = np.linalg.solve(innovation, c @ pminus).T
        pplus = pminus - pminus @ c.T @ np.linalg.solve(innovation, c @ pminus)
        pminus = (pminus + pminus.T) / 2
        pplus = (pplus + pplus.T) / 2
        dare_residual = float(np.linalg.norm(model["A"] @ pplus @ model["A"].T
                                             + model["Q"] - pminus, ord="fro"))
        pminus_min = float(np.linalg.eigvalsh(pminus).min())
        pplus_min = float(np.linalg.eigvalsh(pplus).min())
        if min(pminus_min,pplus_min) < -PSD_CUTOFF:
            raise AssertionError("Independent DARE covariance PSD check failed.")
        fallback = False
    elif use_registered_fallback:
        projector_error = float(np.linalg.norm(projector - W_MISSING, ord=2))
        if receiver != 26 or rank != 2 or projector_error > PROJECTOR_CUTOFF:
            raise AssertionError("U fallback is not aligned to W26.")
        c_visible = c[:, :2]
        pminus_visible = solve_discrete_are(model["A2"].T, c_visible.T,
                                            model["Q2"], v)
        innovation = c_visible @ pminus_visible @ c_visible.T + v
        l_visible = np.linalg.solve(innovation, c_visible @ pminus_visible).T
        l = np.vstack([l_visible, np.zeros((2, 4))])
        pminus = np.full((4, 4), np.nan)
        pplus = np.full((4, 4), np.nan)
        dare_residual = None
        pminus_min = None
        pplus_min = None
        fallback = True
    else:
        raise AssertionError(f"Unexpected rank-deficient mode for receiver {receiver}: {rank}.")
    f = (np.eye(4) - l @ c) @ model["A"]
    identity = np.eye(4) - l @ c
    omega = identity @ model["Q"] @ identity.T + l @ v @ l.T
    omega = (omega + omega.T) / 2
    fixed = None if fallback else float(np.linalg.norm(pplus - f @ pplus @ f.T - omega, ord="fro"))
    return {"C": c, "V": v, "L": l, "K": l[:, :2], "G": l[:, 2:],
            "Pminus": pminus, "Pplus": pplus, "F": f, "Omega": omega,
            "rank": rank, "singular": singular, "null_basis": null_basis,
            "projector": projector, "dare_residual": dare_residual,
            "fixed_point_residual": fixed, "Pminus_minimum_eigenvalue": pminus_min,
            "Pplus_minimum_eigenvalue": pplus_min, "fallback": fallback}


def propagate_gain(design: dict, initial: np.ndarray, steps: int) -> tuple[np.ndarray, dict]:
    path = np.empty((steps + 1, 4, 4))
    path[0] = initial
    maximum_symmetry_residual = 0.0
    minimum_eigenvalue = float(np.linalg.eigvalsh((initial + initial.T) / 2).min())
    for step in range(steps):
        raw = design["F"] @ path[step] @ design["F"].T + design["Omega"]
        maximum_symmetry_residual = max(maximum_symmetry_residual,
                                        float(np.linalg.norm(raw-raw.T,ord="fro")))
        next_p = raw
        path[step + 1] = (next_p + next_p.T) / 2
        step_minimum = float(np.linalg.eigvalsh(path[step + 1]).min())
        minimum_eigenvalue = min(minimum_eigenvalue,step_minimum)
        if step_minimum < -PSD_CUTOFF:
            raise AssertionError("Independent covariance PSD check failed.")
    audit = {"maximum_symmetry_residual_fro": maximum_symmetry_residual,
             "minimum_eigenvalue_over_trajectory": minimum_eigenvalue}
    return path, audit


def projected_risk(path: np.ndarray, length: int) -> float:
    return float(sum(np.trace(W_MISSING @ path[t]) for t in range(1, length + 1)))


def transfer_norm(f: np.ndarray, b: np.ndarray, length: int) -> float:
    projection = np.eye(4)[2:, :]
    matrix = np.zeros((2 * length, 2 * length))
    for t in range(length):
        for s in range(t + 1):
            matrix[2*t:2*t+2, 2*s:2*s+2] = projection @ np.linalg.matrix_power(f, t-s) @ b
    return float(np.linalg.svd(matrix, compute_uv=False)[0] ** 2)


def assert_close(actual: np.ndarray, expected: np.ndarray, name: str,
                 maximum: dict, tolerance: float = 2e-9) -> None:
    difference = float(np.max(np.abs(actual - expected)))
    maximum[name] = max(maximum.get(name, 0.0), difference)
    if difference > tolerance:
        raise AssertionError(f"{name} differs by {difference}, tolerance {tolerance}.")


def main() -> None:
    config = read_config()
    report_path = RESULTS / "D1A_link_value.json"
    with np.load(RESULTS / "gain_tables.npz") as stored, np.load(RESULTS / "covariance_traces.npz") as covariances:
        summary = json.loads(report_path.read_text(encoding="utf-8"))
        verification = {"status": "PASS", "models_checked": 0,
                        "maximum_matrix_difference": {}, "fixed_point_residuals": {},
                        "topology_checks": {}, "archive_hashes_checked": {}}
        expected_dimensions = {"x": [4], "local_measurement": [2],
                               "C": [4,4], "V": [4,4], "K": [4,2],
                               "G": [4,2], "L": [4,4], "F": [4,4],
                               "Omega": [4,4], "Pminus": [4,4], "Pplus": [4,4],
                               "delta_packet": [2], "S1": [2,2], "S35": [2,2]}
        if summary["matrix_dimensions"] != expected_dimensions:
            raise AssertionError("Registered matrix dimension table mismatch.")
        if summary["attack_input_definition"]["injection_matrices"] != {
                "S1": np.eye(2).tolist(), "S35": np.eye(2).tolist()}:
            raise AssertionError("Attack input matrix definition differs from S1=S35=I2.")
        verification["manifest_hashes_checked"] = verify_hash_manifest(summary)
        registration = json.loads((HERE / "D1A_registration.json").read_text(encoding="utf-8"))
        for filename, expected in registration["pre_execution_hashes"].items():
            observed = digest((HERE / filename).read_bytes())
            if observed != expected:
                raise AssertionError(f"Registered pre-execution digest mismatch: {filename}")
        for topology_name in ("U", "R"):
            adjacency, replacement = reconstruct_graph(config, topology_name)
            graph_record = json.loads((RESULTS / ("topology_unique.json" if topology_name == "U"
                                                   else "topology_redundant.json")).read_text(encoding="utf-8"))
            assert np.array_equal(adjacency, np.asarray(graph_record["adjacency"], dtype=np.uint8))
            assert graph_record["candidate_search"] == replacement["candidate_history"]
            assert len(components_of(adjacency)) == 1
            old_degrees = np.asarray(config["graph"]["adjacency"],dtype=np.uint8).sum(axis=1)
            new_degrees = adjacency.sum(axis=1)
            declared_degrees = graph_record["degree_changes"]
            if set(declared_degrees) != {str(node) for node in range(50)}:
                raise AssertionError("Topology report must include degree changes for every node.")
            for node in range(50):
                declared = declared_degrees[str(node)]
                old_degree = int(old_degrees[node])
                new_degree = int(new_degrees[node])
                if declared != {"old":old_degree,"new":new_degree,
                                "difference":new_degree-old_degree}:
                    raise AssertionError(f"Node-degree change mismatch at node {node}.")
            if graph_record["connected_components"] != components_of(adjacency) or \
                    graph_record["undirected_edge_count"] != int(adjacency.sum()//2):
                raise AssertionError("Topology connectivity summary mismatch.")
            if digest(np.ascontiguousarray(adjacency).tobytes()) != graph_record["adjacency_sha256"]:
                raise AssertionError(f"Topology adjacency SHA-256 mismatch for {topology_name}.")
            h1 = np.asarray(config["H"][1],dtype=float)
            h35 = np.asarray(config["H"][35],dtype=float)
            r1 = np.asarray(config["R"][1],dtype=float)
            r35 = np.asarray(config["R"][35],dtype=float)
            assert np.array_equal(np.asarray(graph_record["H1"]),h1)
            assert np.array_equal(np.asarray(graph_record["H35"]),h35)
            assert np.array_equal(np.asarray(graph_record["R1"]),r1)
            assert np.array_equal(np.asarray(graph_record["R35"]),r35)
            assert np.array_equal(np.asarray(graph_record["H1_minus_H35"]),h1-h35)
            assert np.array_equal(np.asarray(graph_record["R1_minus_R35"]),r1-r35)
            verification["topology_checks"][topology_name] = {
                "receiver_neighbors": np.flatnonzero(adjacency[26]).astype(int).tolist(),
                "connected": True,
                "edge_count": int(adjacency.sum() // 2),
                "degree_1_16_26": [int(adjacency[node].sum()) for node in [1,16,26]],
            }
            if topology_name == "R":
                assert replacement["removed"] == [16,26] and replacement["added"] == [1,26]
                assert graph_record["redesign_nodes"] == [1,16,26]
                assert graph_record["changed_directed_arcs"] == {
                    "removed": [[16,26],[26,16]], "added": [[1,26],[26,1]]}
                assert graph_record["D1A_risk_scan_link_indices_zero_based"] == [
                    {"risk_scan_index_zero_based":6,"sender":35,"receiver":26},
                    {"risk_scan_index_zero_based":7,"sender":1,"receiver":26}]
                assert graph_record["added_direction_parity_roles"] == [
                    {"sender":1,"receiver":26,"risk_scan_index_zero_based":7,
                     "role":"new target direction; sender-1 parity audit"},
                    {"sender":26,"receiver":1,"risk_scan_index_zero_based":None,
                     "role":"reciprocal healthy direction; sender-26 parity audit"}]
            for sender, parity in summary["models"][f"{topology_name}_unstable"]["sender_parity_audit"].items():
                sender_id = int(sender)
                # Rebuild the sender's physical measurement block independently.
                local_h = np.asarray(config["H"][sender_id],dtype=float)
                h4 = np.zeros((2,4),dtype=float)
                block = 0 if sender_id % 2 == 0 else 2
                h4[:,block:block+2] = local_h
                a2 = np.asarray(config["A_unstable"],dtype=float)
                a4 = block_diag(a2,a2)
                stack = np.vstack([h4 @ np.linalg.matrix_power(a4,power)
                                   for power in range(5)])
                parity_basis = null_space(stack.T)
                observed_parity = {
                    "sender": sender_id,
                    "observable_stack_shape": list(stack.shape),
                    "observable_stack_rank": int(np.linalg.matrix_rank(stack,tol=RANK_CUTOFF)),
                    "parity_basis_shape": list(parity_basis.T.shape),
                    "annihilation_residual_fro": float(np.linalg.norm(parity_basis.T @ stack,ord="fro")),
                }
                if observed_parity["observable_stack_shape"] != parity["observable_stack_shape"] or \
                        observed_parity["observable_stack_rank"] != parity["observable_stack_rank"] or \
                        observed_parity["parity_basis_shape"] != parity["parity_basis_shape"] or \
                        abs(observed_parity["annihilation_residual_fro"]-parity["annihilation_residual_fro"])>1e-10:
                    raise AssertionError(f"Sender parity audit mismatch for node {sender_id}.")
            for dynamics in ("unstable", "stable"):
                model = construct_physical(config, adjacency, dynamics)
                model_key = f"{topology_name}_{dynamics}"
                prefix = f"{topology_name}_{dynamics}"
                for receiver in [1,16,26]:
                    if receiver not in [int(x) for x in summary["models"][model_key]["endpoint_designs"]]:
                        raise AssertionError("Endpoint gain matrix missing from result summary.")
                    weights = {sender:1.0 for sender in model["neighbors"][receiver]}
                    gain = rebuild_gain(model, receiver, weights, False)
                    for matrix_name in ("C","V","L","K","G","Pminus","Pplus","F","Omega"):
                        assert_close(gain[matrix_name], stored[f"{prefix}_endpoint{receiver}_{matrix_name}"],
                                     f"{prefix}_endpoint{receiver}_{matrix_name}",
                                     verification["maximum_matrix_difference"])
                    endpoint_record = summary["models"][model_key]["endpoint_designs"][str(receiver)]
                    if endpoint_record["rank"] != gain["rank"] or \
                            abs(endpoint_record["fixed_point_residual"]-gain["fixed_point_residual"])>2e-10:
                        raise AssertionError("Endpoint DARE summary mismatch.")
                targets = [35] if topology_name == "U" else [35,1]
                expected_indices = {35:6, 1:7}
                for target in targets:
                    scan_key = f"{prefix}_edge{target}_to26"
                    scan_summary = summary["models"][model_key]["link_scans"][str(target)]
                    if scan_summary["risk_scan_index_zero_based"] != expected_indices[target]:
                        raise AssertionError("D1-A link-scan index mismatch.")
                    keep_weights = {sender:1.0 for sender in model["neighbors"][26]}
                    keep = rebuild_gain(model,26,keep_weights,False)
                    p0_keep = keep["Pplus"]
                    recomputed_risks = {"keep": {}, "zero": {}}
                    for qi,qval in enumerate(Q_GRID):
                        weights = dict(keep_weights)
                        weights[target] = qval
                        rebuilt = rebuild_gain(model,26,weights,
                                               receiver == 26)
                        for matrix_name in ("C","V","L","K","G","F","Omega"):
                            assert_close(rebuilt[matrix_name], stored[f"{scan_key}_q{qval:g}_{matrix_name}"],
                                         f"{scan_key}_q{qval:g}_{matrix_name}",
                                         verification["maximum_matrix_difference"])
                        if rebuilt["fallback"]:
                            assert np.isnan(stored[f"{scan_key}_q{qval:g}_Pminus"]).all()
                            assert np.isnan(stored[f"{scan_key}_q{qval:g}_Pplus"]).all()
                        else:
                            for matrix_name in ("Pminus","Pplus"):
                                assert_close(rebuilt[matrix_name], stored[f"{scan_key}_q{qval:g}_{matrix_name}"],
                                             f"{scan_key}_q{qval:g}_{matrix_name}",
                                             verification["maximum_matrix_difference"])
                            if rebuilt["fixed_point_residual"] > DARE_CUTOFF:
                                raise AssertionError("Independent fixed point residual exceeds registered limit.")
                        weight_record = scan_summary["weights"][qi]
                        if weight_record["rank"] != rebuilt["rank"] or \
                                weight_record["fallback"] != rebuilt["fallback"]:
                            raise AssertionError("Registered weight rank or fallback label mismatch.")
                        for field in ("dare_residual", "fixed_point_residual",
                                      "Pminus_minimum_eigenvalue", "Pplus_minimum_eigenvalue"):
                            actual_value = rebuilt["fixed_point_residual"] if field == "fixed_point_residual" else (
                                rebuilt["dare_residual"] if field == "dare_residual" else rebuilt[field])
                            declared_value = weight_record[field]
                            if actual_value is None or declared_value is None:
                                if actual_value is not None or declared_value is not None:
                                    raise AssertionError(f"Inapplicable DARE field mismatch: {field}")
                            elif abs(actual_value-declared_value)>2e-10:
                                raise AssertionError(f"DARE numerical summary mismatch: {field}")
                        if np.linalg.eigvalsh(rebuilt["Omega"]).min() < -PSD_CUTOFF:
                            raise AssertionError("Independent Omega PSD check failed.")
                        verification["fixed_point_residuals"][f"{scan_key}_q{qval:g}"] = rebuilt["fixed_point_residual"]
                        trajectory, keep_audit = propagate_gain(rebuilt,p0_keep,60)
                        assert_close(trajectory, covariances[f"{scan_key}_q{qval:g}_P_keep"],
                                     f"{scan_key}_q{qval:g}_P_keep", verification["maximum_matrix_difference"])
                        zero_trajectory, zero_audit = propagate_gain(rebuilt,np.zeros((4,4)),60)
                        assert_close(zero_trajectory, covariances[f"{scan_key}_q{qval:g}_P_zero"],
                                     f"{scan_key}_q{qval:g}_P_zero", verification["maximum_matrix_difference"])
                        for initial_name, actual_audit in (("keep",keep_audit),("zero",zero_audit)):
                            declared_audit = weight_record["covariance_audit_by_initial"][initial_name]
                            if abs(actual_audit["maximum_symmetry_residual_fro"]-
                                   declared_audit["maximum_symmetry_residual_fro"])>2e-10 or \
                                    abs(actual_audit["minimum_eigenvalue_over_trajectory"]-
                                        declared_audit["minimum_eigenvalue_over_trajectory"])>2e-10:
                                raise AssertionError("Covariance trajectory audit mismatch.")
                        for length in HORIZONS:
                            if not np.isfinite(projected_risk(trajectory,length)):
                                raise AssertionError("Non-finite finite-horizon risk.")
                        recomputed_risks["keep"][qval] = {str(length): projected_risk(trajectory,length)
                                                           for length in HORIZONS}
                        recomputed_risks["zero"][qval] = {str(length): projected_risk(zero_trajectory,length)
                                                           for length in HORIZONS}
                        bdelta = qval * rebuilt["G"] @ np.eye(2)
                        assert_close(bdelta, stored[f"{scan_key}_q{qval:g}_B_delta"],
                                     f"{scan_key}_q{qval:g}_B_delta",
                                     verification["maximum_matrix_difference"])
                        for length in HORIZONS:
                            independent_transfer = transfer_norm(rebuilt["F"],bdelta,length)
                            declared = scan_summary["weights"][qi]["attack_transfer_gain_by_horizon"][str(length)]
                            if abs(independent_transfer - declared) > 2e-9:
                                raise AssertionError("Finite-horizon attack transfer mismatch.")
                    for geometry_name, qvalue in (("keep",1.0),("delete",0.0)):
                        geometry_weights = dict(keep_weights)
                        geometry_weights[target] = qvalue
                        geometry_c, _ = make_measurements(model,26,geometry_weights)
                        geometry_rank, geometry_singular, _, geometry_projector = rank_and_null(
                            geometry_c,model["A"])
                        declared_geometry = scan_summary["observability"][geometry_name]
                        if geometry_rank != declared_geometry["rank"] or \
                                not np.allclose(geometry_singular,declared_geometry["singular_values"],
                                                rtol=1e-12,atol=1e-12) or \
                                not np.allclose(geometry_projector,declared_geometry["unobservable_projector"],
                                                rtol=0,atol=2e-10):
                            raise AssertionError("Finite-weight observability geometry mismatch.")
                        projector_distance = float(np.linalg.norm(geometry_projector-W_MISSING,ord=2))
                        if abs(projector_distance-declared_geometry["projector_difference_norm_W26"])>2e-10:
                            raise AssertionError("Missing-subspace projector distance mismatch.")
                    risk_summary = summary["models"][model_key]["link_scans"][str(target)]["risk"]
                    for initial_name in ("keep", "zero"):
                        for qi,qval in enumerate(Q_GRID):
                            declared = risk_summary[initial_name][qi]["J_by_horizon"]
                            for length in HORIZONS:
                                actual = recomputed_risks[initial_name][qval][str(length)]
                                if abs(actual - declared[str(length)]) > 2e-8 * max(1.0, abs(actual)):
                                    raise AssertionError("Finite-horizon J mismatch.")
                        j0 = recomputed_risks[initial_name][Q_GRID[0]]
                        j1 = recomputed_risks[initial_name][Q_GRID[-1]]
                        for length in HORIZONS:
                            value = (j0[str(length)] - j1[str(length)]) / (j1[str(length)] + 1e-12)
                            expected = summary["models"][model_key]["link_scans"][str(target)]["kappa"][initial_name][str(length)]
                            if abs(value - expected) > 2e-10:
                                raise AssertionError("Link marginal risk indicator mismatch.")
                    verification["models_checked"] += 1
        verification["endpoint_topology_differences_checked"] = {}
        for dynamics in ("unstable", "stable"):
            verification["endpoint_topology_differences_checked"][dynamics] = {}
            for receiver in (1,16,26):
                declared = summary["topology_endpoint_differences"][dynamics][str(receiver)]
                u_neighbors = np.flatnonzero(reconstruct_graph(config,"U")[0][receiver]).astype(int).tolist()
                r_neighbors = np.flatnonzero(reconstruct_graph(config,"R")[0][receiver]).astype(int).tolist()
                if declared["U_neighbors"] != u_neighbors or declared["R_neighbors"] != r_neighbors:
                    raise AssertionError("Endpoint neighbor set differs from the constructed topology.")
                for matrix_name, measures in declared["matrix_differences_R_minus_U"].items():
                    left = stored[f"R_{dynamics}_endpoint{receiver}_{matrix_name}"]
                    right = stored[f"U_{dynamics}_endpoint{receiver}_{matrix_name}"]
                    difference = left - right
                    maximum = float(np.max(np.abs(difference)))
                    frobenius = float(np.linalg.norm(difference,ord="fro"))
                    if abs(maximum-measures["maximum_absolute_difference"])>2e-10 or \
                            abs(frobenius-measures["frobenius_difference"])>2e-10:
                        raise AssertionError("Endpoint topology matrix difference mismatch.")
                verification["endpoint_topology_differences_checked"][dynamics][str(receiver)] = True
        # Verify every archive listed in the report and stored SHA manifest.
        for filename, expected in summary["result_sha256"].items():
            observed = digest((RESULTS / filename).read_bytes())
            if observed != expected:
                raise AssertionError(f"Archive hash mismatch for {filename}.")
            verification["archive_hashes_checked"][filename] = observed
    (RESULTS / "verification.json").write_text(json.dumps(verification, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
