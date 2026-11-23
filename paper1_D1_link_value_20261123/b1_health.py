"""B1 healthy paired common-G replay. Requires only frozen physical noise and healthy observations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy.linalg import block_diag, null_space, pinv

import b1_mode_models as modes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "b1_health_results"
PLAN = json.loads((HERE / "B1_HEALTH_RUN.json").read_text(encoding="utf-8"))
B0 = json.loads((HERE / "B0_REGISTRATION.json").read_text(encoding="utf-8"))
Q = np.asarray(B0["candidate_weights"])
N = PLAN["path_count_each_split_per_scenario"]
BATCH = PLAN["batch_size"]
BITS = 1 << np.arange(7)
INDEX = np.arange(50)
sys.path.insert(0, str(ROOT / "paper1_history_ablation_20261123"))
from ablation import residuals, scores_all
from run_partial50 import physical

SOURCE = ROOT / "paper1_original50_source_20261121/source/simulation_revised_detection_metrics_complete_v2.py"


def scenarios():
    return [(model, topology, target, label) for model in PLAN["dynamics_order"]
            for topology, target in PLAN["link_order"] for label in PLAN["label_order"]]


def seed(index, stage, batch=0):
    return np.random.SeedSequence([*PLAN["seed_namespace"], index, stage, batch])


def setup(spec, gains):
    dynamics, topology, target, _ = spec
    config = modes.load_inputs()
    adjacency, _ = modes.build_topology(config, topology)
    model = SimpleNamespace(**modes.make_model(config, adjacency, dynamics))
    model.links = np.asarray(modes.mode_links(topology, target))
    # Execute only the frozen parity constructor. No source simulation or waveform generation.
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config["source_sha256"]
    node = next(n for n in ast.parse(raw.decode("utf-8-sig")).body
                if isinstance(n, ast.FunctionDef) and n.name == "build_parity_space")
    scope = dict(np=np, block_diag=block_diag, null_space=null_space,
                 Params=SimpleNamespace(**config["defaults"]), Q=model.Q2,
                 H=np.moveaxis(np.asarray(config["H"]), 0, 2), R=np.moveaxis(model.R, 0, 2))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    parity, theta = scope[node.name](model.A2)
    model.Z = np.stack([parity[j] for _, j in model.links])
    model.Theta = np.stack([theta[j] for _, j in model.links])
    model.precision = np.stack([pinv(v) for v in model.Theta])
    key = f"{topology}_{dynamics}_target{target}_q"
    model.base = {n: gains[f"{key}1_{n}"] for n in ("K","G","F")}
    model.target = {}
    for name in ("F","K","G","Omega"):
        # Original six send to other receivers; column 6 alone selects node 26's row.
        rows = []
        for q in Q:
            table = gains[f"{key}{q:g}_{name}"][:, 26]
            assert np.all(table[:64] == table[0]) and np.all(table[64:] == table[64])
            rows.append(table[[0,64]])
        model.target[name] = np.stack(rows, axis=1)  # target bit, candidate, matrix
    model.safety = []
    for q in Q:
        f = gains[f"{key}{q:g}_F"]
        model.safety.append(float(np.linalg.norm(f, ord=2, axis=(-2,-1)).max()))
    model.safety = np.maximum(model.safety, model.safety[-1])
    return model


def scores(model, paths):
    r = residuals(model, paths)
    out = np.zeros((len(paths.x), 150, 7))
    cur = r[:, 1:]
    out[:,5:] = 0.5 * np.einsum("btei,eij,btej->bte", cur, model.precision, cur, optimize=True)
    return out


def mv(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


def evaluate(model, paths, masks):
    count = len(paths.x)
    net = np.zeros((count, 50, 4))
    target = np.zeros((count, len(Q), 4))
    sums = np.zeros((count, len(Q), 2))
    all_other = np.zeros(count)
    r, j = model.links.T
    # Frozen common-G uses raw measurements and each node's own previous estimate.
    raw = (paths.y.swapaxes(-1,-2) @ model.adjacency.T).swapaxes(-1,-2)
    for k in range(1,150):
        mask = masks[:,k]
        bits = (mask[:,None] & BITS) != 0
        y = paths.y[:,k]
        received = raw[:,k].copy()
        for e in range(6):
            received[:,r[e]] -= bits[:,e,None] * y[:,j[e]]
        net = (mv(model.base["F"][mask], net) + mv(model.base["K"][mask], y)
               + mv(model.base["G"][mask], received))
        alarm = bits[:,6].astype(int)
        packet = (received[:,26,None,:]
                  - (alarm[:,None] * (1-Q)[None,:])[...,None] * y[:,None,j[6],:])
        target = (mv(model.target["F"][alarm], target)
                  + mv(model.target["K"][alarm], y[:,None,26,:])
                  + mv(model.target["G"][alarm], packet))
        if k >= 20:
            err = (target - paths.x[:,k,None,:])**2
            sums[:,:,0] += err[:,:,2:].sum(axis=-1)
            sums[:,:,1] += err.sum(axis=-1)
            other_err = (net - paths.x[:,k,None,:])**2
            other_err[:,26] = 0
            all_other += other_err.sum(axis=(1,2))
    primary, full = (sums / 130).transpose(2,0,1)
    network = (all_other[:,None] / 130 + full) / 50
    return dict(missing=primary, full=full, network=network)


def literal_check(model, paths, masks, optimized):
    """Receiver-by-receiver raw innovations, including shared-receiver simultaneous alarms."""
    states = np.zeros((len(paths.x), len(Q), 50, 4))
    loss = np.zeros((len(paths.x), len(Q), 3))
    for k in range(1,150):
        mask = masks[:,k]
        bits = (mask[:,None] & BITS) != 0
        pred = states @ model.A.T
        y = paths.y[:,k]
        for i, neighbors in enumerate(model.neighbors):
            innovation = np.zeros_like(pred[:,:,i,:2])
            for j in neighbors:
                v = y[:,None,j,:] - pred[:,:,i] @ model.H[j].T
                for e,(ri,sj) in enumerate(model.links):
                    if ri == i and sj == j:
                        v = v * np.where(bits[:,e,None], Q if e == 6 else 0., 1.)[...,None]
                innovation += v
            if i == 26:
                K, G = (model.target[name][bits[:,6].astype(int)] for name in ("K","G"))
            else:
                K, G = (model.base[name][mask,i,None] for name in ("K","G"))
            states[:,:,i] = pred[:,:,i] + mv(K, y[:,None,i,:] - pred[:,:,i] @ model.H[i].T) + mv(G, innovation)
        if k >= 20:
            err = (states - paths.x[:,k,None,None,:])**2
            loss[:,:,0] += err[:,:,26,2:].sum(axis=-1)
            loss[:,:,1] += err[:,:,26].sum(axis=-1)
            loss[:,:,2] += err.sum(axis=-1).mean(axis=-1)
    diff = max(float(np.max(abs(optimized[name]-loss[:,:,col]/130))) for col,name
               in enumerate(("missing","full","network")))
    np.testing.assert_allclose(loss/130, np.stack(list(optimized.values()),axis=-1), rtol=2e-9, atol=1e-8)
    return diff


def smoke():
    audit = []
    with np.load(modes.OUTPUT / "mode_tables.npz") as gains:
        for idx,spec in enumerate(scenarios()[::2]):
            model = setup(spec, gains)
            paths = physical(model, 2, seed(idx,2))
            actual = scores(model, paths)
            original = scores_all(residuals(model, paths),model.Theta)["H1"]
            np.testing.assert_allclose(actual,original,rtol=1e-12,atol=1e-12)
            masks = np.stack([np.arange(150)%128, (np.arange(150)+64)%128]).astype(np.uint8)
            loss_err = literal_check(model, paths, masks, evaluate(model,paths,masks))
            audit.append(dict(scene=spec[:3], H1_max_abs=float(np.max(abs(actual-original))),
                              path_mse_max_abs=loss_err))
    return audit


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = smoke()
    with TemporaryDirectory(dir=HERE) as tmp, np.load(modes.OUTPUT / "mode_tables.npz") as gains:
        dest = Path(tmp)/"health"
        dest.mkdir()
        for idx,spec in enumerate(scenarios()):
            model = setup(spec,gains)
            maxima = []
            for batch in range(N//BATCH):
                paths = physical(model,BATCH,seed(idx,0,batch))
                maxima.extend(scores(model,paths)[:,4:].max(axis=(1,2)))
            maxima = np.asarray(maxima)
            rank = int(np.ceil((N+1)*0.95))
            threshold = np.partition(maxima,rank-1)[rank-1]
            records = {name:[] for name in ("masks","missing","network","full","holdout_max")}
            for batch in range(N//BATCH):
                paths = physical(model,BATCH,seed(idx,1,batch))
                score = scores(model,paths)
                masks = ((score>threshold) @ BITS).astype(np.uint8)
                records["holdout_max"].append(score[:,4:].max(axis=(1,2)))
                records["masks"].append(masks)
                for name,values in evaluate(model,paths,masks).items():
                    records[name].append(values)
            arrays = {name:np.concatenate(values) for name,values in records.items()}
            np.savez_compressed(dest/f"scene_{idx:02}.npz", **arrays, calibration_max=maxima,
                                threshold=threshold, safety=model.safety)
            print(idx, spec, "finished", flush=True)
        (dest/"smoke.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8",newline="\n")
        dest.rename(OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
    else:
        run()
