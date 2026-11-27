"""Independent sample, calibration, score and literal-filter checks for task4."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import frontend
import models_quality as quality
import task3
import task4_confirm as confirm

HERE = Path(__file__).resolve().parent
OUT = HERE/"task4_confirm"
Q = quality.frozen.Q
POLICIES = confirm.POLICIES
WATERMARKS = confirm.WATERMARKS


def independent_replay(model, graph, archive, watermark, policy, start=0, count=1,
                       requested=0.):
    stop = start+count
    x, y = archive["x"][start:stop], archive["y"][start:stop]
    delta = archive[watermark+"_delta"][start:stop]
    gates = archive[watermark+"_gates"][start:stop].astype(bool)
    actions = archive[watermark+"_"+policy+"_actions"][start:stop].astype(int)
    n = len(x)
    state = np.zeros((n, 50, 4))
    total_sq = np.zeros((n, 150, 50))
    missing_sq = np.zeros((n, 150))
    max_missing_sq = np.zeros((n, 150))
    violations = 0
    history = np.full(n, 7**6-1, dtype=np.int64)
    raw = (y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    edge_lookup = {tuple(map(int, edge)): e for e, edge in enumerate(model.links)}
    target_sender = int(model.links[6, 1])
    for tick in range(1, 150):
        bit = gates[:, tick]
        alarm = bit[:, 6]
        prediction = state @ model.A.T
        legal = graph[history]
        chosen = actions[:, tick]
        expected = np.full(n, 6, dtype=int)
        alarm_indices = np.flatnonzero(alarm)
        if policy == "A_q0_raw":
            expected[alarm_indices] = 0
        elif policy == "q1_full":
            expected[:] = 6
        else:
            lower = 0 if policy == "A_M7_min" else int(np.argmin(abs(Q-requested)))
            for path in alarm_indices:
                expected[path] = int(np.flatnonzero(
                    legal[path] & (np.arange(7) >= lower))[0])
        np.testing.assert_array_equal(chosen, expected)
        violations += int(np.count_nonzero(alarm & ~legal[np.arange(n), chosen]))
        neighbor_innovation = np.zeros((n, 50, 2))
        received = raw[:, tick].copy()
        for edge in range(6):
            receiver, sender = model.links[edge]
            keep = ~bit[:, edge]
            received[:, receiver] -= bit[:, edge, None]*y[:, tick, sender]
        for receiver, senders in enumerate(model.neighbors):
            for sender in senders:
                edge = edge_lookup.get((receiver, sender))
                packet = y[:, tick, sender]
                if edge is None:
                    neighbor_innovation[:, receiver] += (
                        packet-prediction[:, receiver] @ model.H[sender].T)
                elif edge < 6:
                    keep = ~bit[:, edge]
                    neighbor_innovation[:, receiver] += keep[:, None]*(
                        packet-prediction[:, receiver] @ model.H[sender].T)
                else:
                    decoded = packet+delta[:, tick, edge]
                    raw_innovation = decoded-prediction[:, receiver] @ model.H[sender].T
                    effective = np.where(alarm, Q[chosen], 1.)
                    neighbor_innovation[:, receiver] += effective[:, None]*raw_innovation
        mode = alarm.astype(int)
        f = model.target["F"][mode, chosen]
        k = model.target["K"][mode, chosen]
        g = model.target["G"][mode, chosen]
        effective = np.where(alarm, Q[chosen], 1.)
        target_packet = (received[:, 26]+delta[:, tick, 6]
                         +(effective-1)[:, None]*(y[:, tick, target_sender]+delta[:, tick, 6]))
        all_mode = bit.astype(np.int64) @ quality.frozen.BITS
        local_innovation = y[:, tick]-np.einsum("jmn,bjn->bjm", model.H, prediction)
        next_state = (
            prediction
            + np.einsum("bnij,bnj->bni", model.base["K"][all_mode], local_innovation)
            + np.einsum("bnij,bnj->bni", model.base["G"][all_mode], neighbor_innovation)
        )
        next_state[:, 26] = (
            np.einsum("bij,bj->bi", f, state[:, 26])
            + np.einsum("bij,bj->bi", k, y[:, tick, 26])
            + np.einsum("bij,bj->bi", g, target_packet)
        )
        state = next_state
        square = (state-x[:, tick, None])**2
        total_sq[:, tick] = square.sum(axis=-1)
        missing_sq[:, tick] = square[:, 26, 2:].sum(axis=-1)
        max_missing_sq[:, tick] = square[:, 26, 2:].max(axis=-1)
        history = (history*7+chosen) % 7**6
    return total_sq, missing_sq, max_missing_sq, violations


def independent_ls_samples(model, operators, archive, watermark):
    packets = archive["y"][:4, :, model.links[:, 1]]+archive[watermark+"_delta"][:4]
    keys = archive[watermark+"_keys"][:4]
    scores = frontend.scores(packets, keys, operators)
    np.testing.assert_allclose(scores, archive[watermark+"_scores"][:4],
                               rtol=2e-10, atol=2e-7)
    maximum_error = 0.
    for path in range(4):
        for tick in (5, 8, 52, 60, 109, 124):
            width = min(8, tick)
            start = tick-width+1
            for edge in (0, 6):
                value = frontend.independent_score(
                    packets[path, start:tick+1, edge],
                    keys[path, start:tick+1, edge],
                    operators[width][1][edge]["whitening"])
                maximum_error = max(maximum_error, abs(value-scores[path, tick, edge]))
                np.testing.assert_allclose(value, scores[path, tick, edge],
                                           rtol=2e-10, atol=2e-7)
    return float(maximum_error)


def verify():
    run = quality.json.loads((OUT/"RUN.json").read_text(encoding="utf-8"))
    lock = quality.json.loads((HERE/"TASK4_CONFIRM_LOCK.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete_task4_confirmatory_data"
    assert run["identity"]["selection"] == "S_fixed_table_T"
    assert run["identity"]["namespace"] == [20261126, 94017, 85]
    lineage_path = OUT / "ARCHIVE_MANIFEST.json"
    if lineage_path.exists():
        lineage = quality.json.loads(lineage_path.read_text(encoding="utf-8"))
        assert lineage["run_sha256"] == confirm.sha(OUT / "RUN.json")
        assert lineage["confirm_lock_sha256"] == confirm.sha(HERE / "TASK4_CONFIRM_LOCK.json")
        assert lineage["protocol_sha256"] == confirm.sha(HERE / "TASK4_CONFIRM_PROTOCOL.md")
        assert lineage["execution_commit"] == run["execution_commit"]
        published_base = lineage["source_history"]["light_branch_base"]
        base_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", published_base, "HEAD"],
            cwd=HERE, check=False)
        if base_ancestor.returncode != 0:
            shallow = subprocess.check_output(
                ["git", "rev-parse", "--is-shallow-repository"],
                cwd=HERE, text=True).strip()
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=HERE, text=True).strip()
            assert shallow == "true"
            assert branch == lineage["source_history"]["published_branch"]
        original_history = subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["lock_creation_parent_commit"],
             run["execution_commit"]],
            cwd=HERE, check=False)
        if original_history.returncode != 0:
            assert lineage["source_history"]["protocol_and_lock_commit"]
            assert lineage["source_history"]["original_results_commit"]
    else:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["lock_creation_parent_commit"],
             run["execution_commit"]],
            cwd=HERE, check=True)
    for filename, expected in run["archives"].items():
        assert confirm.sha(OUT/filename) == expected
    for filename, expected in lock["source_hashes"].items():
        assert confirm.sha(HERE/filename) == expected
    repeat_count = 0
    replay_count = 0
    max_score_error = 0.
    max_loss_error = 0.
    m7_violations = {name: 0 for name in POLICIES}
    for config_index, name in enumerate(task3.QUALITY_ORDER):
        model = task3.audit_frontend.load_model(config_index)
        operators = frontend.operators(model)
        graph = task3.configure(config_index)[1]
        model_record = task3.read_quality_model(config_index)
        for fixture in confirm.FIXTURES:
            for purpose in ("attack", "health"):
                path = OUT/f"{name}_{fixture}_{purpose}.npz"
                with np.load(path) as archive:
                    expected_n = confirm.N if purpose == "attack" else confirm.HEALTH_N
                    assert len(archive["path_id"]) == expected_n
                    np.testing.assert_array_equal(archive["path_id"], np.arange(expected_n))
                    assert np.all(archive["configuration_index"] == config_index)
                    # Both code rows intentionally share the path and uniform primitive.
                    # CRN means equal initial states. Later states follow their
                    # separately frozen iid and no-repeat transition kernels.
                    np.testing.assert_array_equal(archive["iid_keys"][:, 0],
                                                  archive["no_repeat_keys"][:, 0])
                    for watermark in WATERMARKS:
                        score_error = independent_ls_samples(model, operators, archive, watermark)
                        max_score_error = max(max_score_error, score_error)
                        gate = archive[watermark+"_gates"].astype(bool)
                        expected_rank = int(np.ceil((2048+1)*.95))
                        family_index = ("H1", "iid_matched", "no_repeat_matched").index(
                            watermark+"_matched")
                        with np.load(HERE/"frontend_health"/
                                     f"config{config_index:02}_purpose2.npz") as calibration:
                            expected_threshold = float(np.sort(
                                calibration["maxima"][:, family_index])[expected_rank-1])
                        np.testing.assert_allclose(
                            run["configs"][name]["thresholds"][watermark],
                            expected_threshold, rtol=0, atol=0)
                        for policy in POLICIES:
                            actions = archive[watermark+"_"+policy+"_actions"].astype(int)
                            if policy in ("A_q0_raw", "S_fixed_table_T"):
                                # q choice cannot feed back into the frozen matched score stream.
                                np.testing.assert_array_equal(gate, archive[watermark+"_gates"])
                            if policy == "S_fixed_table_T":
                                request = confirm.target_request(model_record)
                                request_index = int(np.argmin(abs(Q-request)))
                                assert np.all(actions[~gate[:, :, 6]] == 6)
                                alarms = gate[:, :, 6]
                                assert np.all(actions[alarms] >= request_index)
                            for row in range(2):
                                total, missing, max_missing, violations = independent_replay(
                                    model, graph, archive, watermark, policy,
                                    start=row, count=1,
                                    requested=confirm.target_request(model_record))
                                # Replay is path- and input-specific, so compare each saved first sample.
                                expected = archive[watermark+"_"+policy+"_losses"][row]
                                reconstructed_missing = missing[0]
                                attack = reconstructed_missing[50:110].mean()
                                recovery = reconstructed_missing[110:150].mean()
                                health = reconstructed_missing[20:150].mean()
                                if purpose == "attack":
                                    active = (np.ones(60, dtype=bool) if fixture == "continuous"
                                              else (np.arange(60) % 10 < 6))
                                    attack_values = missing[0, 50:110]
                                    reduced = np.array([
                                        attack_values.mean(),
                                        attack_values[active].mean(),
                                        attack_values[~active].mean() if (~active).any() else 0.,
                                        total[0, 50:110].sum(axis=1).mean()/50,
                                        missing[0, 110:150].mean(),
                                        missing[0, 20:150].mean(),
                                        total[0, 20:150].sum(axis=1).mean()/50,
                                        max_missing[0, 50:110].max(),
                                    ])
                                    expected_indices = np.arange(8)
                                else:
                                    reduced = np.array([
                                        0., 0., 0., 0., 0.,
                                        missing[0, 20:150].mean(),
                                        total[0, 20:150].sum(axis=1).mean()/50,
                                        0.,
                                    ])
                                    expected_indices = np.array([5, 6])
                                np.testing.assert_allclose(reduced[expected_indices],
                                                           expected[expected_indices],
                                                           rtol=3e-8, atol=2e-6)
                                max_loss_error = max(max_loss_error, float(np.max(abs(
                                    reduced[expected_indices]-expected[expected_indices]))))
                                m7_violations[policy] += violations
                                replay_count += 1
                        keys = archive["no_repeat_keys"]
                        codes = np.zeros(keys.shape[:3], dtype=np.int8)
                        for code, rotation in enumerate(frontend.ROTATIONS):
                            codes[np.max(abs(keys-rotation), axis=(-2, -1)) < 1e-12] = code
                        repeats = int(np.count_nonzero(codes[:, 1:] == codes[:, :-1]))
                        assert repeats == 0
                        repeat_count += repeats
    report = {
        "status": "PASS", "run_sha256": confirm.sha(OUT/"RUN.json"),
        "lock_sha256": confirm.sha(HERE/"TASK4_CONFIRM_LOCK.json"),
        "archives_verified": len(run["archives"]),
        "source_hashes_verified": len(lock["source_hashes"]),
        "no_repeat_transitions_verified": repeat_count,
        "independent_LS_score_cases": 2*12*2*4*6*2,
        "maximum_score_error": max_score_error,
        "independent_literal_replay_cases": replay_count,
        "maximum_literal_endpoint_error": max_loss_error,
        "M7_violations": m7_violations,
        "same_physical_path_and_uniform_probe_paired": True,
    }
    quality.frozen.atomic_json(OUT/"VERIFY.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
