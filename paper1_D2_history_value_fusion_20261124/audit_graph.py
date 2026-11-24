"""Numerical enumeration for D2-A, using the frozen D1 matrices."""
import json
from pathlib import Path
import numpy as np

Q = np.array([0., .2, .4, .6, .8, .9, 1.])
HISTORIES = 7 ** 6
WORD_COUNT = 7 * HISTORIES


def enumerate_graph(model, gain_archive, spec, output):
    dynamics, topology, sender, _ = spec
    key = f"{topology}_{dynamics}_target{sender}_q"
    f = model.target["F"][1]
    c = np.array([gain_archive[f"{key}{q:g}_C"][64, 26] for q in Q])
    others = np.delete(model.base["F"], 26, axis=1)
    other_upper = float(np.linalg.norm(others, ord=2, axis=(-2, -1)).max())
    if other_upper >= 1:
        raise ValueError("Other nodes need a separate window audit.")
    eig = np.linalg.eigvals(model.A)
    if not (np.all(abs(eig) >= 1) or np.all(abs(eig) < 1)):
        raise ValueError("Mixed spectral subspaces need the restricted observation SVD.")
    unstable = bool(np.all(abs(eig) >= 1))
    g = np.empty(WORD_COUNT)
    sigma = np.empty_like(g)
    condition = np.empty_like(g)
    rank = np.empty(WORD_COUNT, dtype=np.uint8)
    powers = np.array([np.linalg.matrix_power(model.A, k) for k in range(7)])
    for start in range(0, WORD_COUNT, 4096):
        ids = np.arange(start, min(start + 4096, WORD_COUNT))
        digits = ids[:, None] // (7 ** np.arange(6, -1, -1)) % 7
        product = np.broadcast_to(np.eye(4), (len(ids), 4, 4))
        for qidx in digits.T:
            product = f[qidx] @ product
        observation = (c[digits] @ powers).reshape(len(ids), 28, 4)
        sv = np.linalg.svd(observation, compute_uv=False)
        rank[ids] = (sv > 1e-10).sum(1)
        sigma[ids] = sv[:, -1]
        condition[ids] = np.divide(sv[:, 0], sv[:, -1], out=np.full(len(ids), np.inf),
                                   where=rank[ids] == 4)
        g[ids] = np.linalg.svd(product, compute_uv=False)[:, 0]
    edges = (g <= 1. - 1e-8) & ((rank == 4) if unstable else True)
    dest = np.arange(WORD_COUNT).reshape(-1, 7) % HISTORIES
    edges = edges.reshape(-1, 7)
    keep = np.ones(HISTORIES, bool)
    while True:
        new = keep & edges[:, -1] & keep[dest[:, -1]]
        if np.array_equal(keep, new):
            break
        keep = new
    assert keep[-1]
    viable = edges & keep[:, None] & keep[dest]
    reach = np.zeros_like(keep)
    reach[-1] = True
    while True:
        new = reach.copy()
        new[dest[viable & reach[:, None]]] = True
        if np.array_equal(reach, new):
            break
        reach = new
    legal = viable & reach[:, None]
    assert np.all(legal[reach, -1])
    ok = legal.ravel()
    witness = int(np.argmax(np.where(ok, g, -1.)))
    word = (witness // 7 ** np.arange(6, -1, -1) % 7).tolist()
    exact_order = np.eye(4)
    for idx in reversed(word):
        exact_order = exact_order @ f[idx]
    np.testing.assert_allclose(np.linalg.norm(exact_order, 2), g[witness], atol=2e-14)
    if unstable and topology == "U":
        assert rank[0] == 2 and not ok[0] and g[0] > 1.
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{key}words.npz", norm=g, sigma_min=sigma,
                        obs_rank=rank, condition_number=condition, legal=legal)
    report = dict(numerical_only=True, words=WORD_COUNT, full_unstable_subspace=unstable,
                  viable_states=int(keep.sum()), reachable_states=int(reach.sum()),
                  admitted=int(ok.sum()), single_choice_states=int((legal.sum(1) == 1).sum()),
                  rho_hat=float(g[witness]), witness=Q[word].tolist(), other_single_upper=other_upper,
                  network_window_upper_hat=max(float(g[witness]), other_upper ** 7),
                  all_zero_norm=float(g[0]), all_zero_rank=int(rank[0]),
                  rank_deficient_admitted=int(((rank < 4) & ok).sum()),
                  min_admitted_sigma=float(sigma[ok].min()),
                  max_finite_admitted_condition=float(condition[ok & np.isfinite(condition)].max()))
    (output / f"{key}graph.json").write_text(json.dumps(report, indent=2) + "\n")
    return legal
