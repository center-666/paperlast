"""Registered triad-watermark mechanism screen, using exact conditional models.

Current-step comparisons share their entire observed history. Actual unknown-
amplitude V1a is not replaced by the favorable fixed-waveform candidate bank.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import block_diag, solve_discrete_are, solve_triangular
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "common_g_reference_revision_20261115"))
from common_g_reference import load_model, common_step


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def model_set(reg):
    models = {name: load_model(name) for name in reg["frozen_models"]}
    r = reg["regional_model"]
    model = SimpleNamespace(name=reg["additional_model"], A=np.array(r["A"]), Q=np.array(r["Q"]),
                            H=[np.array(h) for h in r["H"]], R=[np.array(r["R_each"])] * 3,
                            neighbors=[[1], [0, 2], [1]], K=[np.zeros((2, 2)) for _ in range(3)],
                            G=[np.zeros((2, 2)) for _ in range(3)])
    obs = np.vstack([model.H[1], model.H[0] + model.H[2]])
    noise = block_diag(model.R[1], model.R[0] + model.R[2])
    prior = solve_discrete_are(model.A.T, obs.T, model.Q, noise)
    gain = np.linalg.solve(obs @ prior @ obs.T + noise, obs @ prior).T
    model.K[1], model.G[1] = gain[:, :2], gain[:, 2:]
    models[model.name] = model
    return models


def attack_templates(model, reg):
    rows = np.vstack([model.H[0] @ np.linalg.matrix_power(model.A, lag) for lag in range(reg["duration"])])
    direction = np.array([-1.89, .42])
    xi = direction * np.sqrt(reg["coded_energy"] / np.sum((rows @ direction) ** 2))
    attacks = np.zeros((3, reg["horizon"], 2))
    for i, onset in enumerate(reg["onsets"], 1):
        attacks[i, onset:onset + reg["duration"]] = (rows @ xi).reshape(-1, 2)
        np.testing.assert_allclose(np.sum(attacks[i] ** 2), reg["coded_energy"], atol=2e-12)
    return attacks, xi


def radial_angle_grid(radial, angles, normal):
    x, wx = laggauss(radial)
    theta = np.arange(angles) * np.pi / angles
    u = np.stack([np.cos(theta), np.sin(theta)], -1)
    matrices = .5 * np.eye(2) + x[:, None, None, None] * (u[:, :, None] * u[:, None, :])[None]
    matrices = matrices.reshape(-1, 2, 2)
    weights = np.repeat(wx, angles) / angles
    features = np.tile(np.stack([np.cos(4 * theta), np.sin(4 * theta)], -1), (radial, 1))
    hn, hw = hermgauss(normal)
    nodes = np.stack(np.meshgrid(hn * np.sqrt(2), hn * np.sqrt(2), indexing="ij"), -1).reshape(-1, 2)
    node_weights = np.outer(hw, hw).reshape(-1) / np.pi
    return matrices, weights, features, nodes, node_weights


class Bank:
    def __init__(self, model, reg, attacks):
        self.model, self.reg, self.attacks = model, reg, attacks
        self.H = np.vstack([model.H[1], model.H[2], model.H[0]])
        self.R = block_diag(model.R[1], model.R[2], model.R[0])
        self.Kcg = model.K[1]
        self.Gcg = model.G[1]
        self.Lcg = np.hstack([self.Kcg, self.Gcg, self.Gcg])
        self.Jcg = np.eye(2) - self.Lcg @ self.H
        self.Fcg = self.Jcg @ model.A
        self.means = np.zeros((3, 2))
        self.P = np.eye(2)
        self.weights = np.array(reg["candidate_prior"], dtype=float)
        self.common = np.zeros(2)
        self.raw = np.zeros(2)
        self.discard = np.zeros(2)
        self.k = 0

    def state(self):
        if self.k == 0:
            means, P = self.means, self.P
        else:
            means = self.means @ self.model.A.T
            P = self.model.A @ self.P @ self.model.A.T + self.model.Q
        S = self.H @ P @ self.H.T + self.R
        chol = np.linalg.cholesky(S)
        K = np.linalg.solve(S, self.H @ P).T
        Pplus = P - K @ self.H @ P
        Pplus = (Pplus + Pplus.T) / 2
        return dict(means=means, P=P, S=S, chol=chol, K=K, Pplus=Pplus,
                    weights=self.weights.copy(), attacks=self.attacks[:, self.k].copy(),
                    common=self.common.copy(), k=self.k)

    def update(self, state, z, matrix):
        delta = state["attacks"] @ matrix.T
        mu = state["means"] @ self.H.T
        mu[:, -2:] += delta
        residual = z - mu
        white = solve_triangular(state["chol"], residual.T, lower=True).T
        logs = np.log(np.maximum(state["weights"], 1e-300)) - .5 * np.sum(white ** 2, axis=1)
        self.weights = np.exp(logs - logsumexp(logs))
        self.means = state["means"] + residual @ state["K"].T
        self.P = state["Pplus"]
        corrected = z[-2:] - self.weights @ delta
        _, self.common = common_step(self.common, z[:2], [z[2:4], corrected], self.model.A,
                                     self.model.H[1], [self.model.H[2], self.model.H[0]], self.Kcg, self.Gcg)
        _, self.raw = common_step(self.raw, z[:2], [z[2:4], z[-2:]], self.model.A,
                                  self.model.H[1], [self.model.H[2], self.model.H[0]], self.Kcg, self.Gcg)
        _, self.discard = common_step(self.discard, z[:2], [z[2:4]], self.model.A,
                                      self.model.H[1], [self.model.H[2]], self.Kcg, self.Gcg)
        self.k += 1
        return dict(bayes=self.weights @ self.means, common=self.common.copy(), raw=self.raw.copy(),
                    discard=self.discard.copy())


def exact_gA(Ktarget, attacks, weights):
    centered = attacks - weights @ attacks
    Va = (centered.T * weights) @ centered
    Q = Ktarget.T @ Ktarget
    u, w = (Q[0, 0] - Q[1, 1]) / 2, Q[0, 1]
    g = .25 * np.array([u * (Va[0, 0] - Va[1, 1]) - 2 * w * Va[0, 1],
                        w * (Va[0, 0] - Va[1, 1]) + 2 * u * Va[0, 1]])
    return g, Q, Va


def integrate(bank, state, grid):
    """Integrate only the <=2 informative Gaussian coordinates; all other
    Gaussian coordinates in actual common-G risk are integrated analytically.
    """
    matrices, mw, feature, nodes, nw = grid
    weights, means, K = state["weights"], state["means"], state["K"]
    delta = np.einsum("bij,hj->bhi", matrices, state["attacks"])
    mu = np.broadcast_to(means @ bank.H.T, (len(matrices), 3, 6)).copy()
    mu[:, :, -2:] += delta
    t = means[None] - mu @ K.T
    tbar = weights @ t
    tc = t - tbar[:, None]
    A = np.einsum("h,bhi,bhi->b", weights, tc, tc)
    diff = mu[:, 1:] - mu[:, :1]
    white = np.linalg.solve(state["chol"], diff.swapaxes(1, 2))
    U, _, _ = np.linalg.svd(white, full_matrices=False)
    projection = np.einsum("bdq,bdh->bhq", U, np.linalg.solve(state["chol"], (mu - mu[:, :1]).swapaxes(1, 2)))
    y = projection[:, :, None, :] + nodes[None, None]
    dist = y[:, :, :, None, :] - projection[:, None, None, :, :]
    logits = -.5 * np.sum(dist ** 2, axis=-1) + np.log(np.maximum(weights, 1e-300))
    posterior = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    tp = np.einsum("bhnj,bjd->bhnd", posterior, t)
    I_by_h = np.sum(np.sum((tp - tbar[:, None, None]) ** 2, axis=-1) * nw, axis=-1)
    I = I_by_h @ weights
    bayes_by_h = np.trace(state["Pplus"]) + np.sum(
        np.sum((tp - t[:, :, None]) ** 2, axis=-1) * nw, axis=-1)
    variance_identity_error = bayes_by_h @ weights - (np.trace(state["Pplus"]) + A - I)

    # Actual finite-bank posterior-mean compensation in the original common-G.
    decoded_mean = np.einsum("bhnj,bjd->bhnd", posterior, delta)
    DU = np.einsum("ad,bdq->baq", (bank.Lcg - K) @ state["chol"], U)
    informative_noise = np.einsum("baq,nq->bna", DU, nodes)
    constant = bank.Fcg @ state["common"] + mu @ bank.Lcg.T - means[None]
    cg_residual = constant[:, :, None] + informative_noise[:, None] - decoded_mean @ bank.Gcg.T
    full_noise = (bank.Lcg - K) @ state["S"] @ (bank.Lcg - K).T
    missing_variance = np.trace(full_noise) - np.sum(DU ** 2, axis=(1, 2))
    assert np.min(missing_variance) > -1e-9
    common_by_h = np.trace(state["Pplus"]) + missing_variance[:, None] + np.sum(
        np.sum(cg_residual ** 2, axis=-1) * nw, axis=-1)

    pA = posterior[..., 1] + posterior[..., 2]
    active_t = np.einsum("bhnj,bjd->bhnd", posterior[..., 1:], t[:, 1:]) / np.maximum(pA[..., None], 1e-300)
    activity = pA * (1 - pA) * np.sum((active_t - t[:, None, None, 0]) ** 2, axis=-1)
    onset = posterior[..., 1] * posterior[..., 2] / np.maximum(pA, 1e-300) * np.sum((t[:, 1] - t[:, 2]) ** 2, axis=-1)[:, None, None]
    activity = np.sum(activity * nw, axis=-1) @ weights
    onset = np.sum(onset * nw, axis=-1) @ weights

    gA, Q, Va = exact_gA(K[:, -2:], state["attacks"], weights)
    numerical_gA = (mw * A) @ feature
    np.testing.assert_allclose(gA, numerical_gA, atol=2e-11, rtol=2e-10)
    base = lambda values: np.einsum("b,b...->...", mw, values)
    sensitivity = lambda values: np.einsum("b,b...,bq->...q", mw, values, feature)
    return jsonable(dict(gA=gA, gI=sensitivity(I), A=base(A), I=base(I),
                         bayes=base(bayes_by_h), g_bayes=sensitivity(bayes_by_h),
                         common=base(common_by_h), g_common=sensitivity(common_by_h),
                         activity=base(activity), g_activity=sensitivity(activity),
                         onset=base(onset), g_onset=sensitivity(onset),
                         oracle=np.trace(state["Pplus"]), Q=Q, Va=Va,
                         dispersion_coefficient_error=float(np.max(np.abs(gA - numerical_gA))),
                         variance_identity_weighted_abs_error=float(mw @ np.abs(variance_identity_error)),
                         variance_identity_sensitivity_error=(mw * variance_identity_error) @ feature))


def choose(gA, gI, ghealthy, allowance, rho, minimum):
    gA, gI, ghealthy = map(np.asarray, (gA, gI, ghealthy))
    # Equality is imposed even when its absolute effect is small.
    if np.linalg.norm(gA) > 1e-18:
        n = np.array([-gA[1], gA[0]]) / np.linalg.norm(gA)
        low, high = -rho, rho
        h = ghealthy @ n
        if h > 1e-18:
            high = min(high, allowance / h)
        elif h < -1e-18:
            low = max(low, allowance / h)
        if low > high:
            return np.zeros(2)
        t = high if gI @ n > 0 else low
        v = t * n
    else:
        if np.linalg.norm(gI) < 1e-20:
            return np.zeros(2)
        v = rho * gI / np.linalg.norm(gI)
        if ghealthy @ v > allowance + 1e-15:
            hnorm = np.linalg.norm(ghealthy)
            if hnorm < 1e-20:
                return np.zeros(2)
            unit = ghealthy / hnorm
            coordinate = np.clip(allowance / hnorm, -rho, rho)
            perp = np.array([-unit[1], unit[0]])
            v = coordinate * unit + np.sign(gI @ perp) * np.sqrt(max(0, rho ** 2 - coordinate ** 2)) * perp
    if gI @ v < minimum:
        return np.zeros(2)
    assert abs(gA @ v) < 1e-10
    assert ghealthy @ v <= allowance + 1e-9
    return v


def draw_region(model, reg, path_id):
    rng = np.random.default_rng(reg["regional_seeds"]["physical_base"] + path_id)
    T = reg["horizon"]
    x = np.empty((T, 2))
    x[0] = rng.standard_normal(2)
    for k in range(1, T):
        x[k] = model.A @ x[k - 1] + np.linalg.cholesky(model.Q) @ rng.standard_normal(2)
    y = np.stack([x @ h.T + rng.standard_normal((T, 2)) @ np.linalg.cholesky(r).T
                  for h, r in zip(model.H, model.R)], axis=1)
    rng = np.random.default_rng(reg["regional_seeds"]["watermark_base"] + path_id)
    beta = rng.standard_normal((T, 2)) * np.sqrt(.5)
    matrices = .5 * np.eye(2) + beta[:, :, None] * beta[:, None, :]
    return x, y, matrices


def observations(model, reg, path_id, archive):
    if model.name in reg["frozen_models"]:
        return (archive[f"{model.name}__physical__x"][path_id],
                archive[f"{model.name}__physical__y"][path_id],
                archive[f"{model.name}__M_target"][path_id])
    return draw_region(model, reg, path_id)


def frozen_digest():
    names = ["common_g_reference_revision_20261115/common_g_reference.py",
             "common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json",
             "JFI_reconstruction_attack_aware_soft_accommodation_V1/accommodation.py",
             "attack_gain_switch_audit_20261115/screen_arrays.npz"]
    return {name: sha256((BASE / name).read_bytes()).hexdigest() for name in names}


def screen(reg, models, grid, refine_grid):
    rows, replays, refinements = [], [], []
    ids = reg["development_path_ids"] + reg["evaluation_path_ids"]
    with np.load(BASE / "attack_gain_switch_audit_20261115/screen_arrays.npz", allow_pickle=False) as archive:
        for name, model in models.items():
            attacks, xi = attack_templates(model, reg)
            for path_id in ids:
                truth, y, matrices = observations(model, reg, path_id, archive)
                for h in range(3):
                    bank = Bank(model, reg, attacks)
                    outputs = {method: [] for method in ("bayes", "common", "raw", "discard")}
                    for k in range(reg["horizon"]):
                        state = bank.state()
                        if k in reg["evaluation_ticks"]:
                            stats = integrate(bank, state, grid)
                            vectors = {"uniform": np.zeros(2)}
                            for budget in reg["healthy_constraints"]:
                                label = f"information_health_{budget:g}"
                                vectors[label] = choose(stats["gA"], stats["gI"], stats["g_common"][0],
                                                        budget * stats["common"][0], reg["watermark"]["rho"],
                                                        reg["minimum_predicted_information_gain"])
                            row = dict(model=name, path_id=path_id, true_h=h, tick=k,
                                       partition="development" if path_id in reg["development_path_ids"] else "evaluation",
                                       weights=state["weights"], stats=stats, vectors=vectors)
                            rows.append(jsonable(row))
                            if path_id == 16 and k == 8:
                                refined = integrate(bank, state, refine_grid)
                                v = vectors["information_health_0.02"]
                                refinements.append(jsonable(dict(model=name, true_h=h, tick=k,
                                    coarse_gI=stats["gI"], refined_gI=refined["gI"], chosen_v=v,
                                    coarse_gain=np.asarray(stats["gI"]) @ v,
                                    refined_gain=np.asarray(refined["gI"]) @ v,
                                    refined_healthy_change=np.asarray(refined["g_common"])[0] @ v,
                                    gI_error=np.linalg.norm(np.asarray(stats["gI"]) - refined["gI"]))))
                        z = np.r_[y[k, 1], y[k, 2], y[k, 0] + matrices[k] @ attacks[h, k]]
                        out = bank.update(state, z, matrices[k])
                        for method in outputs:
                            outputs[method].append(out[method])
                    if name == reg["additional_model"] and path_id in reg["evaluation_path_ids"]:
                        for method, values in outputs.items():
                            squared = (np.array(values) - truth) ** 2
                            replays.append(dict(path_id=path_id, true_h=h, method=method,
                                                total_mse=float(np.mean(np.sum(squared, axis=1))),
                                                last_eight_mse=float(np.mean(np.sum(squared[-8:], axis=1))),
                                                last_eight_missing_mode_mse=float(np.mean(squared[-8:, 1]))))
                if path_id in (0, 7, 16, 23, 31):
                    print(f"screen {name} path {path_id} completed", flush=True)
    return rows, replays, refinements


def summarize(reg, rows, replays, refinements):
    results = {}
    for name in [*reg["frozen_models"], reg["additional_model"]]:
        dev = [r for r in rows if r["model"] == name and r["partition"] == "development"]
        val = [r for r in rows if r["model"] == name and r["partition"] == "evaluation"]
        # Development histories are drawn separately under each true candidate;
        # weight those strata by the registered candidate prior to reconstruct
        # the Bayesian predictive-history law.
        prior = np.asarray(reg["candidate_prior"], dtype=float)
        hist_weights = np.asarray([prior[r["true_h"]] for r in dev], dtype=float)
        hist_weights /= hist_weights.sum()
        average = lambda key: np.average(np.asarray([r["stats"][key] for r in dev]), axis=0,
                                         weights=hist_weights)
        fixed = choose(average("gA"), average("gI"), average("g_common")[0],
                       .02 * average("common")[0], reg["watermark"]["rho"], reg["minimum_predicted_information_gain"])
        tables = {}
        for label in ("uniform", "information_health_0", "information_health_0.02", "development_fixed"):
            details = []
            for row in val:
                stat, h = row["stats"], row["true_h"]
                v = fixed if label == "development_fixed" else np.array(row["vectors"][label])
                weights = np.array(row["weights"])
                details.append(dict(path_id=row["path_id"], h=h, tick=row["tick"], active=bool(np.linalg.norm(v) > 0),
                    information_gain=float(np.array(stat["gI"]) @ v),
                    dispersion_change=float(np.array(stat["gA"]) @ v),
                    predicted_mixture_bayes_change=float(weights @ (np.array(stat["g_bayes"]) @ v)),
                    true_conditional_bayes_change=float(np.array(stat["g_bayes"])[h] @ v),
                    true_conditional_common_change=float(np.array(stat["g_common"])[h] @ v),
                    healthy_common_change=float(np.array(stat["g_common"])[0] @ v),
                    uniform_true_bayes=stat["bayes"][h], uniform_true_common=stat["common"][h]))
            table = {key: float(np.mean([d[key] for d in details])) for key in details[0] if key not in ("path_id", "h", "tick", "active")}
            table["histories"] = len(details)
            table["active_histories"] = sum(d["active"] for d in details)
            table["max_abs_dispersion_change"] = max(abs(d["dispersion_change"]) for d in details)
            table["by_true_candidate"] = {str(h): {key: float(np.mean([d[key] for d in details if d["h"] == h])) for key in (
                "information_gain", "true_conditional_bayes_change", "true_conditional_common_change", "healthy_common_change")}
                for h in range(3)}
            # Paired path means: three candidate worlds and three ticks share a path.
            for key in ("true_conditional_bayes_change", "true_conditional_common_change"):
                path_means = np.array([np.mean([d[key] for d in details if d["path_id"] == i]) for i in reg["evaluation_path_ids"]])
                table[key + "_paired_path_se"] = (float(np.std(path_means, ddof=1) / np.sqrt(len(path_means)))
                                                    if len(path_means) > 1 else None)
            tables[label] = table
        results[name] = dict(development_fixed_v=fixed.tolist(), comparisons=tables)
    engineering = {}
    for h in range(3):
        engineering[str(h)] = {}
        for method in ("raw", "discard", "common", "bayes"):
            selected = [r for r in replays if r["true_h"] == h and r["method"] == method]
            engineering[str(h)][method] = {key: float(np.mean([r[key] for r in selected])) for key in (
                "total_mse", "last_eight_mse", "last_eight_missing_mode_mse")}
    return dict(one_step=results, regional_gaussian_watermark_replay=engineering, quadrature_refinement=refinements,
                disclaimer="Finite known-waveform candidates; one-step comparisons share historical policy. No full adaptive-loop, unknown-amplitude or V1a superiority claim.")


def geometry(models, reg):
    out = {}
    for name, model in models.items():
        local = np.vstack([model.H[1] @ np.linalg.matrix_power(model.A, t) for t in range(2)])
        trusted = np.vstack([np.vstack([model.H[1], model.H[2]]) @ np.linalg.matrix_power(model.A, t) for t in range(2)])
        aggregate = np.vstack([np.vstack([model.H[1], model.H[0] + model.H[2]]) @ np.linalg.matrix_power(model.A, t) for t in range(2)])
        rng = np.random.default_rng(2026112099)
        beta = rng.standard_normal((4, 2)) * np.sqrt(.5)
        watermarks = .5 * np.eye(2) + beta[:, :, None] * beta[:, None, :]
        augmented = {}
        for kind in ("identity", "Gaussian"):
            blocks = []
            for t in range(4):
                at = np.linalg.matrix_power(model.A, t)
                trusted_t = np.vstack([model.H[1], model.H[2]]) @ at
                matrix = np.eye(2) if kind == "identity" else watermarks[t]
                blocks.append(np.block([[trusted_t, np.zeros((4, 2))],
                                        [model.H[0] @ at, matrix @ model.H[0] @ at]]))
            design = np.vstack(blocks)
            _, singular, vt = np.linalg.svd(design)
            null = vt[np.linalg.matrix_rank(design):]
            augmented[kind] = dict(rank=int(np.linalg.matrix_rank(design)), singular_values=singular.tolist(),
                                   state_null_projection=float(np.linalg.norm(null[:, :2])))
        gain = np.hstack([model.K[1], model.G[1], model.G[1]])
        full_F = (np.eye(2) - gain @ np.vstack([model.H[1], model.H[2], model.H[0]])) @ model.A
        drop_F = (np.eye(2) - model.K[1] @ model.H[1] - model.G[1] @ model.H[2]) @ model.A
        out[name] = dict(local_observability_rank=int(np.linalg.matrix_rank(local)),
                         trusted_remaining_observability_rank=int(np.linalg.matrix_rank(trusted)),
                         actual_aggregate_observability_rank=int(np.linalg.matrix_rank(aggregate)),
                         full_F_spectral_radius=float(max(abs(np.linalg.eigvals(full_F)))),
                         discard_F_spectral_radius=float(max(abs(np.linalg.eigvals(drop_F)))),
                         known_onset_noiseless_augmented_state_attack=augmented)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    reg = read(HERE / "registration.json")
    before = frozen_digest()
    models = model_set(reg)
    write(HERE / "results/geometry.json", geometry(models, reg))
    if args.smoke:
        reg = deepcopy(reg)
        reg["development_path_ids"] = [0]
        reg["evaluation_path_ids"] = [16]
        reg["evaluation_ticks"] = [8]
        reg["quadrature"] = dict(radial=4, angles=16, normal_1d=7)
        reg["refinement"] = dict(radial=8, angles=32, normal_1d=11)
    q, r = reg["quadrature"], reg["refinement"]
    grid = radial_angle_grid(q["radial"], q["angles"], q["normal_1d"])
    refine = radial_angle_grid(r["radial"], r["angles"], r["normal_1d"])
    rows, replays, refinements = screen(reg, models, grid, refine)
    summary = jsonable(summarize(reg, rows, replays, refinements))
    after = frozen_digest()
    assert before == after
    suffix = "_smoke" if args.smoke else ""
    write(HERE / f"results/histories{suffix}.json", rows)
    write(HERE / f"results/regional_paths{suffix}.json", replays)
    write(HERE / f"results/summary{suffix}.json", summary)
    write(HERE / f"results/provenance{suffix}.json", dict(frozen_hashes_before=before, frozen_hashes_after=after,
        registration_sha256=sha256((HERE / "registration.json").read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__, history_comparisons=len(rows),
        newly_generated_regional_physical_paths=len(reg["development_path_ids"] + reg["evaluation_path_ids"]),
        inferential_status="development and evaluation diagnostic; path SE descriptive, not simultaneous certification"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
