"""Fit development weights and evaluate them on disjoint archived paths."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "class_weight_development"
sys.path.insert(0, str(HERE))
DEV = Path(os.environ.get("D2_TASK3_DEV_DIR", str(HERE/"task3_dev"))).resolve()

import frontend
import models_quality as quality
import task3

Q = np.asarray(quality.Q, dtype=float)
C_GRID = np.asarray([0., .125, .25, .5, .75, 1., 1.5, 2., 2.5, 3., 4., 6., 8., 12., np.inf])
METHODS = ("unified", "topology_two", "three_class", "frozen_T", "saturation")
SEED = [20261127, 94017, 271, 1]
BOOT_REPS = 20000


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def config_group(record):
    if record["topology"] == "R":
        return "R"
    return "U_unstable" if record["dynamics"] == "unstable" else "U_stable"


def t_request(record):
    if record["topology"] == "R":
        return 0.
    return .8 if record["dynamics"] == "unstable" else .6


def quantize_weight(weight):
    lower = np.searchsorted(Q, weight, side="left").clip(0, len(Q)-1)
    upper = np.maximum(lower-1, 0)
    # lower is the grid point at or above the raw weight. Midpoint ties select it.
    lower_distance = abs(Q[lower]-weight)
    upper_distance = abs(Q[upper]-weight)
    index = np.where(lower_distance <= upper_distance+1e-12, lower, upper)
    return index.astype(int)


def make_model(index):
    return task3.audit_frontend.load_model(index), task3.configure(index)[1]


def prepare_data(archive, fixture, purpose, threshold):
    prefix = "health_" if purpose == "health" else fixture + "_"
    return {
        "paths": SimpleNamespace(x=archive[prefix+"x"], y=archive[prefix+"y"]),
        "delta": archive[prefix+"delta"],
        "matched": archive[prefix+"matched"],
        "gates": archive[prefix+"gates"].astype(bool),
        "active": np.ones(60, dtype=bool) if fixture == "continuous" else np.arange(60)%10 < 6,
        "thresholds": {"no_repeat_matched": float(threshold)},
    }


def saturated_request(model, data, state, covariance, tick, c):
    prediction = state @ model.A.T
    pminus = model.A @ covariance @ model.A.T + model.Q
    sender = int(model.links[6, 1])
    measurement = model.H[sender]
    residual = (data["paths"].y[:, tick, sender] + data["delta"][:, tick, 6]
                - prediction @ measurement.T)
    innovation_cov = measurement @ pminus @ measurement.T + model.R[sender]
    whitened = np.linalg.solve(innovation_cov, residual[..., None])[..., 0]
    distance = np.sqrt(np.maximum(np.einsum("bi,bi->b", residual, whitened), 0.))
    if np.isinf(c):
        weight = np.ones(len(state))
    elif c == 0:
        weight = np.zeros(len(state))
    else:
        weight = np.minimum(1., np.divide(c, distance, out=np.ones_like(distance), where=distance > 0))
    return quantize_weight(weight), weight


def replay(model, graph, data, request, mode="constant", c=None):
    paths, delta, gates = data["paths"], data["delta"], data["gates"]
    n = len(paths.x)
    state = np.zeros((n, 4))
    covariance = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    history = np.full(n, 7**6-1, dtype=np.int64)
    actions = np.full((n, 150), len(Q)-1, dtype=np.int8)
    requests = np.full((n, 150), len(Q)-1, dtype=np.int8)
    raw_weights = np.ones((n, 150), dtype=float)
    losses = np.zeros((n, 8))
    active = data["active"]
    raw_network = (paths.y.swapaxes(-1, -2) @ model.adjacency.T).swapaxes(-1, -2)
    lookup = {tuple(map(int, edge)): i for i, edge in enumerate(model.links[:6])}
    senders = model.neighbors[26]
    sender = int(model.links[6, 1])
    violations = projections = 0
    request_index = int(np.argmin(abs(Q-float(request)))) if mode == "constant" else None

    for tick in range(1, 150):
        bits = gates[:, tick]
        alarm = bits[:, 6]
        received = raw_network[:, tick, 26].copy()
        for neighbor in senders:
            edge = lookup.get((26, int(neighbor)))
            if edge is not None:
                received -= bits[:, edge, None]*paths.y[:, tick, int(neighbor)]

        if mode == "saturation":
            desired, raw_request = saturated_request(model, data, state, covariance, tick, c)
        else:
            desired = np.full(n, request_index, dtype=int)
            raw_request = Q[desired]
        desired = np.where(alarm, desired, len(Q)-1)
        requests[:, tick] = desired
        raw_weights[:, tick] = np.where(alarm, raw_request, 1.)
        legal = graph[history]
        candidates = legal & (np.arange(len(Q))[None, :] >= desired[:, None])
        if not candidates.any(axis=1).all():
            raise AssertionError("The M7 graph must keep q=1 available.")
        chosen = candidates.argmax(axis=1)
        violations += int(np.count_nonzero(alarm & ~legal[np.arange(n), chosen]))
        projections += int(np.count_nonzero(alarm & (chosen != desired)))
        actions[:, tick] = chosen
        history = (history*7+chosen) % 7**6
        effective_q = np.where(alarm, Q[chosen], 1.)
        prediction = state @ model.A.T
        packet = (received + delta[:, tick, 6]
                  +(effective_q-1.)[:, None]*(paths.y[:, tick, sender]+delta[:, tick, 6]))
        mode_index = alarm.astype(int)
        f = model.target["F"][mode_index, chosen]
        k = model.target["K"][mode_index, chosen]
        g = model.target["G"][mode_index, chosen]
        state = (np.einsum("bij,bj->bi", f, state)
                 + np.einsum("bij,bj->bi", k, paths.y[:, tick, 26])
                 + np.einsum("bij,bj->bi", g, packet))
        if mode == "saturation":
            covariance = (np.einsum("bij,bjk,bmk->bim", f, covariance, f)
                          + model.target["Omega"][mode_index, chosen])
        squared = (state-paths.x[:, tick])**2
        missing = squared[:, 2:].sum(axis=1)
        if 50 <= tick < 110:
            losses[:, 0] += missing/60.
            if active[tick-50]:
                losses[:, 1] += missing/max(int(active.sum()), 1)
            else:
                losses[:, 2] += missing/max(60-int(active.sum()), 1)
        elif tick >= 110:
            losses[:, 4] += missing/40.
        if tick >= 20:
            losses[:, 5] += missing/130.
    return {"actions": actions, "requests": requests, "raw_weights": raw_weights,
            "losses": losses, "M7_violations": violations, "M7_projections": projections}


def load_training(index):
    model, graph = make_model(index)
    threshold = task3.frontend_thresholds(index)["no_repeat_matched"]
    ops = frontend.operators(model)
    fixtures = []
    for fixture_index, fixture in enumerate(("continuous", "burst")):
        data = task3.batch_data(
            model, ops, index, fixture_index, 4, 256, task3.frontend_thresholds(index))
        fixtures.append((fixture, data))
    return model, graph, fixtures


def fit():
    dev = DEV
    run = json.loads((dev/"RUN.json").read_text(encoding="utf-8"))
    records, training = [], []
    for index, name in enumerate(task3.QUALITY_ORDER):
        record = task3.read_quality_model(index)
        records.append(record)
        model, graph, fixtures = load_training(index)
        loss_grid = {float(q): [] for q in Q}
        sat_grid = {float(c): [] for c in C_GRID}
        for fixture, data in fixtures:
            for q in Q:
                result = replay(model, graph, data, q)
                loss_grid[float(q)].append(float(result["losses"][:, 0].mean()))
            for c in C_GRID:
                result = replay(model, graph, data, 1., mode="saturation", c=float(c))
                sat_grid[float(c)].append(float(result["losses"][:, 0].mean()))
        training.append({"record": record, "losses": loss_grid, "saturation": sat_grid})
        print("FIT", name, flush=True)

    def choose(values):
        best = min(values.values())
        return max(float(q) for q, value in values.items() if abs(value-best) <= 1e-12)
    pooled = {str(float(q)): float(np.mean([x["losses"][float(q)][f]
                                           for x in training for f in range(2)])) for q in Q}
    classes = {}
    for group in ("U", "R", "U_unstable", "U_stable"):
        subset = [x for x in training if
                  (x["record"]["topology"] == group if group in ("U", "R")
                   else config_group(x["record"]) == group)]
        classes[group] = {str(float(q)): float(np.mean(
            [x["losses"][float(q)][f] for x in subset for f in range(2)])) for q in Q}
    sat = {str(float(c)): float(np.mean([x["saturation"][float(c)][f]
                                         for x in training for f in range(2)])) for c in C_GRID}
    cbest = max((float(c) for c in C_GRID),
                key=lambda c: (-sat[str(c)], c if np.isfinite(c) else 1e300))
    parameters = {
        "frozen_commit": "c6d05204602a78ad603a2ee39676e11c1c3663ef",
        "fit_namespace": [20261126, 94017, 83, 4],
        "paths_per_configuration_fixture": 256,
        "evaluation_paths_per_configuration_fixture": 256,
        "healthy_paths_per_configuration": 256,
        "grid": Q.tolist(), "saturation_c_grid": [
            float(c) if np.isfinite(c) else "inf" for c in C_GRID],
        "unified_q": choose(pooled),
        "topology_two_q": {g: choose(classes[g]) for g in ("U", "R")},
        "three_class_q": {g: choose(classes[g]) for g in ("U_unstable", "U_stable", "R")},
        "frozen_T": {"U_unstable": .8, "U_stable": .6, "R": 0.},
        "saturation_c": cbest,
        "training_losses": {"pooled": pooled, "classes": classes, "saturation": sat},
        "evaluation_archive_sha256": {
            f"config{i:02}.npz": sha(DEV/f"config{i:02}.npz") for i in range(12)},
        "evaluation_archive_namespace": "task3_dev purpose-5 evaluation and purpose-6 healthy paths",
        "task3_RUN_sha256": sha(dev/"RUN.json"),
        "task3_protocol_sha256": sha(HERE/"TASK3_DEVELOPMENT_PROTOCOL.md"),
        "source_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE/"class_weight_development/PROTOCOL.md"),
    }
    atomic = OUT/"FIT_PARAMETERS.json"
    atomic.write_text(json.dumps(parameters, indent=2)+"\n", encoding="utf-8", newline="\n")
    print("FROZEN FIT PARAMETERS",
          json.dumps({k: parameters[k] for k in ("unified_q", "topology_two_q",
                                                  "three_class_q", "frozen_T", "saturation_c")}, indent=2))


def smoke():
    midpoint = quantize_weight(np.asarray([.5, .85]))
    np.testing.assert_array_equal(Q[midpoint], [.6, .9])
    model, graph = make_model(2)
    threshold = task3.frontend_thresholds(2)["no_repeat_matched"]
    with np.load(DEV/"config02.npz") as archive:
        data = {
            "paths": SimpleNamespace(x=archive["continuous_x"], y=archive["continuous_y"]),
            "delta": archive["continuous_delta"],
            "matched": archive["continuous_matched"],
            "gates": archive["continuous_gates"].astype(bool),
            "active": np.ones(60, dtype=bool),
            "thresholds": {"no_repeat_matched": threshold},
        }
        for request, column in ((.8, 2), (0., 1)):
            result = replay(model, graph, data, request)
            np.testing.assert_array_equal(result["actions"], archive["continuous_actions"][:, column])
            endpoint = [0, 1, 2, 4, 5]
            np.testing.assert_allclose(result["losses"][:, endpoint],
                                       archive["continuous_losses"][:, column, endpoint],
                                       rtol=0, atol=2e-6)
        saturation = replay(model, graph, data, 1., mode="saturation", c=1.)
        assert saturation["M7_violations"] == 0
        assert np.isfinite(saturation["raw_weights"]).all()
        assert np.all((saturation["raw_weights"] >= 0.) & (saturation["raw_weights"] <= 1.))
    print("PASS target-local M7 and endpoint smoke replay", flush=True)


def evaluate():
    lock = OUT/"FIT_PARAMETERS.json"
    params = json.loads(lock.read_text(encoding="utf-8"))
    assert params["source_sha256"] == sha(Path(__file__))
    assert params["protocol_sha256"] == sha(HERE/"class_weight_development/PROTOCOL.md")
    all_rows, stored = [], []
    indices = []
    for index, config in enumerate(task3.QUALITY_ORDER):
        model, graph = make_model(index)
        record = task3.read_quality_model(index)
        threshold = task3.frontend_thresholds(index)["no_repeat_matched"]
        with np.load(DEV/f"config{index:02}.npz") as archive:
            methods = {
                "unified": params["unified_q"],
                "topology_two": params["topology_two_q"]["U" if record["topology"]=="U" else "R"],
                "three_class": params["three_class_q"][config_group(record)],
                "frozen_T": params["frozen_T"][config_group(record)],
            }
            for fixture in ("continuous", "burst"):
                data = {
                    "paths": SimpleNamespace(x=archive[fixture+"_x"], y=archive[fixture+"_y"]),
                    "delta": archive[fixture+"_delta"],
                    "matched": archive[fixture+"_matched"],
                    "gates": archive[fixture+"_gates"].astype(bool),
                    "active": np.ones(60, dtype=bool) if fixture=="continuous" else np.arange(60)%10<6,
                    "thresholds": {"no_repeat_matched": threshold},
                }
                for method,q in methods.items():
                    result = replay(model,graph,data,q)
                    if method=="frozen_T":
                        np.testing.assert_array_equal(result["actions"],archive[fixture+"_actions"][:,2])
                        endpoint=[0,1,2,4,5]
                        np.testing.assert_allclose(result["losses"][:,endpoint],
                                                   archive[fixture+"_losses"][:,2,endpoint],
                                                   atol=2e-6,rtol=0)
                    item={"config":config,"topology":record["topology"],"dynamics":record["dynamics"],
                          "factor":record["R35_factor"],"fixture":fixture,"method":method,**result}
                    stored.append(item)
                    all_rows.append({k:item[k] for k in ("config","topology","dynamics","factor","fixture","method","M7_violations","M7_projections")}|{
                        "full":float(result["losses"][:,0].mean()),
                        "burst_on":float(result["losses"][:,1].mean()),
                        "burst_off":float(result["losses"][:,2].mean()),
                        "recovery":float(result["losses"][:,4].mean())})
            health={
                "paths":SimpleNamespace(x=archive["health_x"],y=archive["health_y"]),
                "delta":archive["health_delta"],"matched":archive["health_matched"],
                "gates":archive["health_gates"].astype(bool),"active":np.zeros(60,dtype=bool),
                "thresholds":{"no_repeat_matched":threshold}}
            for method,q in methods.items():
                result=replay(model,graph,health,q)
                item={"config":config,"topology":record["topology"],"dynamics":record["dynamics"],
                      "factor":record["R35_factor"],"fixture":"health","method":method,**result}
                stored.append(item)
                all_rows.append({k:item[k] for k in ("config","topology","dynamics","factor","fixture","method","M7_violations","M7_projections")}|{
                    "full":float(result["losses"][:,0].mean()),"burst_on":0.,"burst_off":0.,
                    "recovery":float(result["losses"][:,4].mean()),"health":float(result["losses"][:,5].mean())})
        indices.append(index)
        print("EVALUATE",config,flush=True)
    write_csv(OUT/"evaluation.csv",all_rows)
    matrices={}
    methods_order=("unified","topology_two","three_class","frozen_T","saturation")
    for item in stored:
        matrices[item["config"],item["fixture"],item["method"]]=item["losses"]
    groups={"all":set(task3.QUALITY_ORDER),
            "U":{x for x in task3.QUALITY_ORDER if x.startswith("U_")},
            "R":{x for x in task3.QUALITY_ORDER if x.startswith("R_")},
            "U_unstable":{x for x in task3.QUALITY_ORDER if x.startswith("U_unstable")},
            "U_stable":{x for x in task3.QUALITY_ORDER if x.startswith("U_stable")},
            "primary":{ "U_unstable_R35x2"}}
    comparisons=(("topology_two","unified","two_minus_unified"),
                 ("three_class","topology_two","three_minus_two"),
                 ("three_class","unified","three_minus_unified"),
                 ("frozen_T","unified","T_minus_unified"),
                 ("frozen_T","three_class","T_minus_three"),
                 ("saturation","frozen_T","saturation_minus_T"),
                 ("saturation","topology_two","saturation_minus_two"))
    metrics={"full":0,"burst_on":1,"burst_off":2,"recovery":4,"health":5}
    rng=np.random.default_rng(np.random.SeedSequence(SEED))
    paired=[]
    for group,configs in groups.items():
        for left,right,label in comparisons:
            for metric,col in metrics.items():
                strata=[]
                fixture_names=("continuous","burst") if metric!="health" else ("health",)
                for config in sorted(configs):
                    for fixture in fixture_names:
                        a=matrices[config,fixture,left][:,col]
                        b=matrices[config,fixture,right][:,col]
                        strata.append(a-b)
                point=float(np.mean([v.mean() for v in strata]))
                draws=np.zeros(BOOT_REPS)
                for values in strata:
                    n=len(values)
                    for start in range(0,BOOT_REPS,500):
                        count=min(500,BOOT_REPS-start)
                        weights=rng.multinomial(n,np.full(n,1/n),size=count)/n
                        draws[start:start+count]+=weights@values/len(strata)
                paired.append({"group":group,"comparison":label,"metric":metric,
                               "difference":point,"lower95":float(np.quantile(draws,.025)),
                               "upper95":float(np.quantile(draws,.975)),
                               "independent_paths":sum(len(v) for v in strata)})
    write_csv(OUT/"paired.csv",paired)
    receipt={"status":"complete_development_evaluation","fit_parameters_sha256":sha(lock),
             "evaluation_archive_sha256":{f"config{i:02}.npz":sha(DEV/f"config{i:02}.npz") for i in range(12)},
             "evaluation_seed_namespace":[20261126,94017,83,5],
             "methods":list(methods_order),"groups":list(groups),
             "bootstrap_seed":SEED,"bootstrap_replicates":BOOT_REPS,
             "intervals":"Path-level paired percentile bootstrap within configuration and fixture, equally weighted strata.",
             "source_sha256":sha(Path(__file__)),"protocol_sha256":sha(HERE/"class_weight_development/PROTOCOL.md")}
    atomic_json(OUT/"EVALUATION.json",receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("smoke","fit","evaluate"))
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.mode=="smoke": smoke()
    elif args.mode=="fit": fit()
    else: evaluate()


if __name__=="__main__":
    main()
