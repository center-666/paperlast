"""Healthy-model quality perturbations and frozen-contract mode construction."""
import ast
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.linalg import block_diag, null_space, pinv

HERE = Path(__file__).resolve().parent
D2 = HERE.parent / "paper1_D2_history_value_fusion_20261124"
sys.path.insert(0, str(D2))
import d2b2_run as frozen
import audit_graph

Q = frozen.common.Q
W = np.diag([0., 0., 1., 1.])
OUT = HERE / "quality_models"
SPECS = tuple((d, t, 35, f) for d in ("unstable", "stable")
              for t in ("U", "R") for f in (.5, 1., 2.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def construct(dynamics, topology, factor):
    config = frozen.modes.load_inputs()
    config = copy.deepcopy(config)
    config["R"][35] = (np.array(config["R"][35])*factor).tolist()
    adjacency, _ = frozen.modes.build_topology(config, topology)
    arrays = frozen.modes.make_model(config, adjacency, dynamics)
    model = SimpleNamespace(**arrays)
    model.links = np.array(frozen.modes.mode_links(topology, 35))
    cache = {}

    def local(receiver, weights):
        key = (receiver, tuple(sorted(weights.items())))
        if key not in cache:
            cache[key] = frozen.modes.design_local(
                arrays, receiver, weights,
                allow_u_fallback=topology == "U" and receiver == 26 and weights.get(35) == 0.)
        return cache[key]

    base = {name: [] for name in ("F", "K", "G")}
    for mode in range(128):
        rows = {name: [] for name in base}
        for receiver in range(50):
            weights = {int(sender): 1. for sender in model.neighbors[receiver]}
            for edge, (i, j) in enumerate(model.links[:6]):
                if i == receiver and mode & (1 << edge):
                    weights[int(j)] = 0.
            gain = local(receiver, weights)
            for name in rows:
                rows[name].append(gain[name])
        for name in base:
            base[name].append(rows[name])
    model.base = {name: np.array(value) for name, value in base.items()}
    quantities = ("F", "K", "G", "Omega", "C", "V", "Pplus", "Pminus")
    target = {name: [] for name in quantities}
    target_designs = []
    for q in Q:
        weights = {int(sender): 1. for sender in model.neighbors[26]}
        weights[35] = float(q)
        target_designs.append(local(26, weights))
    full = target_designs[-1]
    for name in quantities:
        target[name] = np.stack((
            np.stack([full[name] for _ in Q]),
            np.stack([gain[name] for gain in target_designs]),
        ))
    model.target = target
    model.target["Pplus"] = np.stack((full["Pplus"], full["Pplus"]))
    raw_source = frozen.health.SOURCE.read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(raw_source).body
                if isinstance(n, ast.FunctionDef) and n.name == "build_parity_space")
    scope = {
        "np": np, "block_diag": block_diag, "null_space": null_space,
        "Params": SimpleNamespace(**config["defaults"]), "Q": model.Q2,
        "H": np.moveaxis(np.array(config["H"]), 0, 2), "R": np.moveaxis(model.R, 0, 2),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(frozen.health.SOURCE), "exec"), scope)
    parity, covariance = scope[node.name](model.A2)
    model.Z = np.stack([parity[j] for _, j in model.links])
    model.Theta = np.stack([covariance[j] for _, j in model.links])
    model.precision = np.stack([pinv(matrix) for matrix in model.Theta])
    return model, config, target_designs, cache


def physical(model, count, root):
    # Separate physical process and measurement components support paired quality worlds.
    rng = np.random.default_rng(np.random.SeedSequence([*root, 0]))
    x = np.empty((count, 150, 4))
    x[:, 0] = rng.normal(size=(count, 4))
    chol = np.linalg.cholesky(model.Q)
    for tick in range(1, 150):
        x[:, tick] = x[:, tick-1] @ model.A.T+rng.normal(size=(count, 4)) @ chol.T
    normal = np.random.default_rng(np.random.SeedSequence([*root, 1])).normal(
        size=(count, 150, 50, 2))
    standard = np.sqrt(np.diagonal(model.R, axis1=1, axis2=2))
    y = np.einsum("jmn,btn->btjm", model.H, x)+normal*standard[None, None]
    return SimpleNamespace(x=x, y=y)


def health_costs(model, designs):
    p0 = model.target["Pplus"][0]
    full_f = designs[-1]["F"]
    full_o = designs[-1]["Omega"]
    one_step_curves = []
    constant_costs = []
    trajectories = []
    for gain in designs:
        current = gain["F"] @ p0 @ gain["F"].T+gain["Omega"]
        one_step = float(np.trace(W @ current))
        for _ in range(59):
            current = full_f @ current @ full_f.T+full_o
            one_step += float(np.trace(W @ current))
        one_step_curves.append(one_step/60)
        current = p0.copy()
        total = 0.
        trace = [current]
        for _ in range(60):
            current = gain["F"] @ current @ gain["F"].T+gain["Omega"]
            trace.append(current)
            total += float(np.trace(W @ current))
        constant_costs.append(total)
        trajectories.append(trace)
    kappa = (constant_costs[0]-constant_costs[-1])/(constant_costs[-1]+1e-12)
    epsilon = .05-.04*max(kappa, 0)/(1+max(kappa, 0))
    return np.array(one_step_curves), kappa, epsilon, np.array(trajectories)


def policy_reachability(graph, health_mask, requested):
    masks = np.stack((
        graph & (np.arange(7)[None] >= requested),
        graph & health_mask[0], graph & health_mask[1],
    ))
    choices = masks.argmax(axis=2).astype(np.int8)
    reachable = np.zeros((3, 7**6), dtype=bool)
    for policy in range(3):
        reachable[policy, -1] = True
        todo = deque([7**6-1])
        while todo:
            state = todo.popleft()
            assert masks[policy, state].any()
            for action in (6, int(choices[policy, state])):
                successor = (state*7+action) % 7**6
                if not reachable[policy, successor]:
                    reachable[policy, successor] = True
                    todo.append(successor)
    witnesses = {}
    for left, right, label in ((0, 1, "T_H"), (0, 2, "T_V"), (1, 2, "H_V")):
        start = (7**6-1, 7**6-1)
        todo = deque([(start, [])])
        seen = {start}
        witness = None
        while todo:
            state, word = todo.popleft()
            a, b = int(choices[left, state[0]]), int(choices[right, state[1]])
            if a != b:
                witness = {"alarm_word": word+[1], "history_pair": list(state),
                           "action_indices": [a, b], "q": [float(Q[a]), float(Q[b])]}
                break
            for alarm in (0, 1):
                action = a if alarm else 6
                successor = tuple((s*7+action) % 7**6 for s in state)
                if successor not in seen:
                    seen.add(successor)
                    todo.append((successor, word+[alarm]))
        witnesses[label] = {"first_divergence": witness,
                            "paired_states_checked": len(seen), "strictly_equal": witness is None}
    return masks, choices, reachable, witnesses


def main():
    OUT.mkdir(exist_ok=True)
    old = json.loads((D2 / "b2_results/B2_RUN.json").read_text(encoding="utf-8"))
    rows = []
    inputs = {
        "source_sha256": sha(Path(__file__)), "configuration_sha256": sha(frozen.modes.CONFIG),
        "D1_mode_source_sha256": sha(Path(frozen.modes.__file__)),
        "M7_enumerator_sha256": sha(Path(audit_graph.__file__)),
        "D2_B2_run_sha256": sha(D2 / "b2_results/B2_RUN.json"),
    }
    with np.load(frozen.modes.OUTPUT / "mode_tables.npz") as original:
        for index, (dynamics, topology, sender, factor) in enumerate(SPECS):
            name = f"{topology}_{dynamics}_R35x{factor:g}"
            target_file = OUT / f"{name}.json"
            if target_file.exists():
                rows.append(json.loads(target_file.read_text(encoding="utf-8")))
                continue
            model, config, designs, cache = construct(dynamics, topology, factor)
            curves, kappa, epsilon, traces = health_costs(model, designs)
            previous = next(r for r in old["results"].values()
                            if r["spec"][:3] == [dynamics, topology, sender])
            original_epsilon = previous["epsilon"]
            relative = curves-curves[-1]
            margin_h = original_epsilon*curves[-1]-relative
            margin_v = epsilon*curves[-1]-relative
            health_ok = np.stack((margin_h >= -1e-10, margin_v >= -1e-10))
            key = f"{topology}_{dynamics}_target35_q"
            fake_c = {f"{key}{q:g}_C": np.broadcast_to(designs[a]["C"], (128, 50, 4, 4))
                      for a, q in enumerate(Q)}
            graph_folder = OUT / name
            graph = audit_graph.enumerate_graph(
                model, fake_c, (dynamics, topology, sender, "negative"), graph_folder)
            graph_file = graph_folder / f"{key}words.npz"
            graph_report_file = graph_folder / f"{key}graph.json"
            graph_report = json.loads(graph_report_file.read_text(encoding="utf-8"))
            frozen.atomic_json(graph_report_file, graph_report)
            requested = 4 if dynamics == "unstable" else 3
            if topology == "R":
                requested = 0
            masks, choices, reach, witness = policy_reachability(graph, health_ok, requested)
            regression = {}
            if factor == 1.:
                differences = {}
                for quantity in ("F", "K", "G", "Omega"):
                    target = np.stack([original[f"{key}{q:g}_{quantity}"][[0, 64], 26] for q in Q], axis=1)
                    differences[quantity] = float(np.max(np.abs(model.target[quantity]-target)))
                    np.testing.assert_allclose(model.target[quantity], target, rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(curves, previous["static_nominal_health_cost_absolute"],
                                           rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(kappa, previous["kappa_L60"], rtol=2e-10, atol=2e-10)
                original_graph_file = D2 / "experiment_A/graphs" / f"{key}words.npz"
                with np.load(original_graph_file) as archived_graph:
                    np.testing.assert_array_equal(graph, archived_graph["legal"])
                regression = {"gain_max_abs_error": differences,
                              "cost_max_abs_error": float(np.max(abs(curves-previous["static_nominal_health_cost_absolute"]))),
                              "kappa_abs_error": abs(kappa-previous["kappa_L60"]),
                              "M7_legal_bits_identical": True}
            affected = sorted(set(model.neighbors[35]+[35]))
            saved = {f"base_{name}": value for name, value in model.base.items()}
            saved.update({f"target_{name}": value for name, value in model.target.items()})
            saved.update(policy_masks=masks, policy_choices=choices,
                         policy_reachable=reach, covariance_traces=traces,
                         R=model.R, H=model.H, A=model.A, Q=model.Q,
                         H1_parity=model.Z, H1_covariance=model.Theta)
            archive = OUT / f"{name}.npz"
            frozen.atomic_npz(archive, saved)
            row = {
                "configuration": index, "name": name, "dynamics": dynamics, "topology": topology,
                "R35_factor": factor, "target_sender": sender,
                "affected_gain_receivers": affected, "unique_local_designs": len(cache),
                "health_cost": curves.tolist(), "relative_health_cost": relative.tolist(),
                "kappa_L60": kappa, "epsilon_original": original_epsilon, "epsilon_updated": epsilon,
                "budget_margin_H": margin_h.tolist(), "budget_margin_V": margin_v.tolist(),
                "health_feasible_H": Q[health_ok[0]].tolist(), "health_feasible_V": Q[health_ok[1]].tolist(),
                "fixed_T_q": float(Q[requested]),
                "policy_reachable_states_T_H_V": reach.sum(axis=1).tolist(),
                "reachable_selected_q_T_H_V": [Q[np.unique(choices[p, reach[p]])].tolist() for p in range(3)],
                "first_divergence": witness, "regression_factor1": regression,
                "archive_sha256": sha(archive), "M7_archive_sha256": sha(graph_file),
                "M7_report": graph_report,
            }
            frozen.atomic_json(target_file, row)
            rows.append(row)
            print("quality", name, "q", row["reachable_selected_q_T_H_V"], flush=True)
    frozen.atomic_json(OUT / "MODELS.json", {"status": "PASS", "inputs": inputs, "configurations": rows})


if __name__ == "__main__":
    main()
