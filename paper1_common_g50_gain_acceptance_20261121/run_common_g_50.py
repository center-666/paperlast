"""50-node common-G MODE-LOOKUP runner and small reproducible CLI.

Every neighbor of receiver i uses the same G_i for the CURRENT retained set.
This is a new steady gain protocol; no covariance is propagated to set gains.
The legacy source's parity/plugin-KL score is reused without changing it.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
LEGACY_VERIFIED = HERE.parent / 'paper1_50node_tv_gate_theory_20261121'
sys.path.insert(0, str(LEGACY_VERIFIED))
from source_model import load_model as load_source_model, matvec
from replay import scores as legacy_scores
from design_common_g import EXPECTED_CONFIGURATION_SHA256


def load_common_g(label):
    table = json.loads((HERE / 'gain_table.json').read_text(encoding='utf-8'))
    raw = (HERE / 'physical_configuration.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_CONFIGURATION_SHA256 or table['physical_configuration_sha256'] != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError('Frozen physical configuration hash mismatch.')
    model = load_source_model(label)  # Hash-checked source parity/attack only.
    if json.loads(raw) != model.cfg:
        raise ValueError('Source and gain physical configurations differ.')
    data = table['models'][label]
    row_index = np.asarray(data['global_mask_to_local_row'], dtype=int)
    rows = data['rows']
    model.K_lookup = np.asarray([r['K'] for r in rows])[row_index]
    model.G_lookup = np.asarray([r['G'] for r in rows])[row_index]
    # Reconstruct F from the actual coefficients rather than rounded diagnostic F.
    F = []
    for row in rows:
        i, active = row['receiver'], row['active_neighbors']
        Hsum = model.H[active].sum(axis=0) if active else np.zeros((2, 2))
        F.append((np.eye(2) - np.asarray(row['K']) @ model.H[i]
                  - np.asarray(row['G']) @ Hsum) @ model.A)
    model.F_lookup = np.asarray(F)[row_index]
    model.gain_data = data
    model.adjacency = np.asarray(model.cfg['graph']['adjacency'], float)
    model.neighbor_Hsum = np.einsum('ij,jab->iab', model.adjacency, model.H)
    model.edge_index = {tuple(map(int, pair)): e for e, pair in enumerate(model.links)}
    return model


def physical_paths(model, count, seed):
    """Independent physical RNG, unaffected by watermark sampling decisions."""
    rng = np.random.default_rng(seed)
    x = np.empty((count, model.T, 2))
    x[:, 0] = rng.normal(size=(count, 2))
    for k in range(1, model.T):
        x[:, k] = x[:, k - 1] @ model.A.T + rng.normal(size=(count, 2))
    y = np.einsum('jmn,btn->btjm', model.H, x)
    y += rng.normal(size=y.shape) * np.sqrt(.5)
    return SimpleNamespace(x=x, y=y)


def encode_decode(model, physical, seed, wm_on):
    """M=.5I+beta beta' with beta~N(0,.5I); sender/time key shared by links.

    Local measurements are not round-tripped. Returned packets are the remote
    sender packets. When wm_on=False M=I, independently of the gate policy.
    """
    B = len(physical.x)
    if not wm_on:
        return physical.y.copy(), np.broadcast_to(np.eye(2), (B, 150, 50, 2, 2))
    beta = np.random.default_rng(seed).normal(size=(B, 150, 50, 2)) * np.sqrt(.5)
    M = .5 * np.eye(2) + beta[..., :, None] * beta[..., None, :]
    encoded = np.linalg.solve(M, physical.y[..., None])[..., 0]
    decoded = matvec(M, encoded)
    return decoded, M


def decoded_attack(model, matrices, edges=None):
    out = np.zeros((matrices.shape[0], 150, 6, 2))
    for e in (range(6) if edges is None else edges):
        sender = int(model.links[e, 1])
        out[:, :, e] = matvec(matrices[:, :, sender], model.attack[None, :, e])
    return out


def run_common_g(model, physical, delta=None, *, sender_packets=None,
                 gate_on=False, threshold=42.0, gate_override=None,
                 implementation='aggregate'):
    """Run all50 nodes, all150 times; scores always see UNPRUNED raw packets.

    delta is decoded channel injection on the six registered directed edges.
    No gate ever drops a local sensor, and no edge outside the six is gated.
    Aggregate update uses the UNWEIGHTED neighbor measurement sum; it never
    uses the old stacked-KF weighted sufficient statistics or Riccati arrays.
    """
    B = len(physical.x)
    if delta is None:
        delta = np.zeros((B, 150, 6, 2))
    if sender_packets is None:
        sender_packets = physical.y
    if np.shape(delta) != (B, 150, 6, 2) or np.shape(sender_packets) != (B, 150, 50, 2):
        raise ValueError('Unexpected packet array shape.')
    score = legacy_scores(model, SimpleNamespace(x=physical.x, y=sender_packets), delta)
    gates = score > threshold if gate_on else np.zeros((B, 150, 6), bool)
    if gate_override is not None:
        gates = np.asarray(gate_override, dtype=bool).copy()
        if gates.shape != (B, 150, 6):
            raise ValueError('gate_override must have shape (B,150,6).')
    masks = np.sum(gates * (1 << np.arange(6)), axis=-1).astype(np.int64)
    post = np.zeros((B, 150, 50, 2))
    for k in range(1, 150):
        pred = post[:, k - 1] @ model.A.T
        K, G = model.K_lookup[masks[:, k]], model.G_lookup[masks[:, k]]
        local = physical.y[:, k] - matvec(model.H[None], pred)
        if implementation == 'aggregate':
            neighbor_y = np.einsum('ij,bjm->bim', model.adjacency, sender_packets[:, k])
            Hsum = np.broadcast_to(model.neighbor_Hsum, (B, 50, 2, 2)).copy()
            for e, (i, j) in enumerate(model.links):
                drop = gates[:, k, e]
                neighbor_y[:, i] += (~drop)[:, None] * delta[:, k, e] - drop[:, None] * sender_packets[:, k, j]
                Hsum[:, i] -= drop[:, None, None] * model.H[j]
            innovation = neighbor_y - matvec(Hsum, pred)
        elif implementation == 'literal':
            innovation = np.zeros_like(pred)
            for i, neighbors in enumerate(model.cfg['graph']['neighbors']):
                for j in neighbors:
                    e = model.edge_index.get((i, j))
                    packet = sender_packets[:, k, j]
                    keep = np.ones(B, bool)
                    if e is not None:
                        packet = packet + delta[:, k, e]
                        keep = ~gates[:, k, e]
                    innovation[:, i] += keep[:, None] * (packet - pred[:, i] @ model.H[j].T)
        else:
            raise ValueError('implementation must be aggregate or literal.')
        post[:, k] = pred + matvec(K, local) + matvec(G, innovation)
    return SimpleNamespace(post=post, scores=score, gates=gates, masks=masks)


def paired_identity(model, physical, healthy, attacked, delta):
    """Exact endogenous-mode identity, including current gain/mask mismatch.

    Precondition: both runs use ideal sender_packets=physical.y. A numerical
    roundtrip packet error adds its own forcing; test it separately instead of
    silently interpreting it as the attack-only identity. Acceptance D uses
    ideal packets; acceptance B separately checks actual floating roundtrips.
    """
    b = attacked.post - healthy.post
    mismatch = np.zeros_like(b)
    forcing = np.zeros_like(b)
    residual = np.zeros_like(b)
    for k in range(1, 150):
        pred0 = healthy.post[:, k - 1] @ model.A.T
        Ka, Ga = model.K_lookup[attacked.masks[:, k]], model.G_lookup[attacked.masks[:, k]]
        K0, G0 = model.K_lookup[healthy.masks[:, k]], model.G_lookup[healthy.masks[:, k]]
        local = physical.y[:, k] - matvec(model.H[None], pred0)
        raw = np.einsum('ij,bjm->bim', model.adjacency, physical.y[:, k]) - matvec(model.neighbor_Hsum[None], pred0)
        ia, i0 = raw.copy(), raw.copy()
        inj = np.zeros_like(raw)
        for e, (i, j) in enumerate(model.links):
            nu = physical.y[:, k, j] - pred0[:, i] @ model.H[j].T
            ia[:, i] -= attacked.gates[:, k, e, None] * nu
            i0[:, i] -= healthy.gates[:, k, e, None] * nu
            inj[:, i] += (~attacked.gates[:, k, e, None]) * delta[:, k, e]
        mismatch[:, k] = matvec(Ka - K0, local) + matvec(Ga, ia) - matvec(G0, i0)
        forcing[:, k] = matvec(Ga, inj)
        residual[:, k] = b[:, k] - matvec(model.F_lookup[attacked.masks[:, k]], b[:, k - 1]) - mismatch[:, k] - forcing[:, k]
    return SimpleNamespace(b=b, mismatch=mismatch, forcing=forcing, residual=residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['unstable', 'stable'], default='unstable')
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026112101)
    parser.add_argument('--wm-on', action='store_true')
    parser.add_argument('--gate-on', action='store_true')
    parser.add_argument('--attack', choices=['none', 'all', '0', '1', '2', '3', '4', '5'], default='none')
    parser.add_argument('--implementation', choices=['aggregate', 'literal'], default='aggregate')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError('trials must be positive.')
    model = load_common_g(args.model)
    physical = physical_paths(model, args.trials, args.seed)
    packets, M = encode_decode(model, physical, args.seed + 1, args.wm_on)
    edges = [] if args.attack == 'none' else (None if args.attack == 'all' else [int(args.attack)])
    delta = decoded_attack(model, M, edges)
    result = run_common_g(model, physical, delta, sender_packets=packets,
                          gate_on=args.gate_on, implementation=args.implementation)
    loss = np.sum((result.post - physical.x[:, :, None]) ** 2, axis=-1)
    summary = {'architecture': 'NEW common-G steady current-mode lookup',
               'model': args.model, 'trials': args.trials, 'seed': args.seed,
               'wm_on': args.wm_on, 'gate_on': args.gate_on, 'attack': args.attack,
               'score': 'source raw parity s4/buffer4 Gaussian plugin-KL; cutoff42; NOT chi-square',
               'network_mse_post20': float(loss[:, 20:].mean()),
               'affected_mse_post20': float(loss[:, 20:, model.receiver_ids].mean()),
               'startup_k5_all_six_gated_fraction': float(result.gates[:, 5].all(axis=-1).mean()),
               'warning': 'Engineering smoke test, not SSCD advantage or calibrated-power validation.'}
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
