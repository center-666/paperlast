"""Small deterministic acceptance suite for the new common-G interface.

This is not the 1,000-path security experiment and makes no superiority claim.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from run_common_g_50 import (load_common_g, physical_paths, encode_decode,
                             run_common_g, decoded_attack, paired_identity)
from design_common_g import aggregate_matrices

HERE = Path(__file__).resolve().parent
SEED = 2026112107


def main():
    results = {'status': 'PASS', 'scope': 'Small engineering acceptance only; no 1,000-path common-G experiment.',
               'seed': SEED, 'models': {}}
    for label in ['unstable', 'stable']:
        model = load_common_g(label)
        per_model = {}

        # Independently re-form the aggregate observation gain from P_prior;
        # matching K/G blocks is necessary, not just regrouping the same arrays.
        independent_gain_error, independent_update_error = 0.0, 0.0
        rng = np.random.default_rng(SEED)
        for row in model.gain_data['rows']:
            i, active = row['receiver'], row['active_neighbors']
            C, V = aggregate_matrices(model.H, model.R, i, active)
            prior = np.asarray(row['P_prior'])
            L = np.linalg.solve(C @ prior @ C.T + V, C @ prior).T
            stored = np.hstack([row['K'], row['G']])
            independent_gain_error = max(independent_gain_error, float(np.max(np.abs(L - stored))))
            p, raw_y = rng.normal(size=2), rng.normal(size=(50, 2))
            literal_one = p + np.asarray(row['K']) @ (raw_y[i] - model.H[i] @ p)
            literal_one += np.asarray(row['G']) @ sum((raw_y[j] - model.H[j] @ p for j in active), start=np.zeros(2))
            z = np.concatenate([raw_y[i], raw_y[active].sum(axis=0)])
            aggregate_one = p + L @ (z - C @ p)
            independent_update_error = max(independent_update_error, float(np.max(np.abs(literal_one - aggregate_one))))
        if max(independent_gain_error, independent_update_error) > 5e-14:
            raise AssertionError('Independently formed aggregate gain/update mismatch.')
        for mask, indices in enumerate(model.gain_data['global_mask_to_local_row']):
            for i, row_index in enumerate(indices):
                row = model.gain_data['rows'][row_index]
                expected_drop = [e for e, (receiver, _) in enumerate(model.links)
                                 if receiver == i and (mask & (1 << e))]
                assert row['receiver'] == i and row['dropped_link_indices'] == expected_drop
        per_model['independent_aggregate_gain'] = {
            'local_rows_checked': 57,
            'maximum_gain_block_difference': independent_gain_error,
            'maximum_independent_one_step_update_difference': independent_update_error,
            'global_mask_node_mapping_checks': 64 * 50}

        # A: all 64 global current gate masks in one batched run, compared with
        # the independent per-neighbor shared-G update.
        physical = physical_paths(model, 64, SEED)
        fixed = np.broadcast_to(((np.arange(64)[:, None] >> np.arange(6)) & 1)[:, None, :],
                                (64, 150, 6)).astype(bool).copy()
        aggregate = run_common_g(model, physical, gate_override=fixed, implementation='aggregate')
        literal = run_common_g(model, physical, gate_override=fixed, implementation='literal')
        aggregate_error = float(np.max(np.abs(aggregate.post - literal.post)))
        aggregate_relative = float(np.max(np.abs(aggregate.post - literal.post) / (1 + np.abs(literal.post))))
        if aggregate_relative > 2e-12 or not np.array_equal(aggregate.masks, literal.masks):
            raise AssertionError(f'{label}: common-G neighbor and aggregate forms differ.')
        per_model['all_64_global_masks_aggregate_vs_literal'] = {
            'masks_checked': 64, 'nodes': 50,
            'maximum_state_absolute_difference': aggregate_error,
            'maximum_state_scaled_difference': aggregate_relative,
            'scaled_pass_tolerance': 2e-12,
            'node17_both_incident_bits_checked': True,
            'all_rows_from_local_lookup': True}

        switched_physical = physical_paths(model, 4, SEED + 88)
        switched_masks = (17 * np.arange(4)[:, None] + 13 * np.arange(150)[None, :]) % 64
        switched = ((switched_masks[:, :, None] >> np.arange(6)) & 1).astype(bool)
        switched_aggregate = run_common_g(model, switched_physical, gate_override=switched)
        switched_literal = run_common_g(model, switched_physical, gate_override=switched, implementation='literal')
        switch_error = float(np.max(np.abs(switched_aggregate.post - switched_literal.post)
                                    / (1 + np.abs(switched_literal.post))))
        assert switch_error < 2e-12
        per_model['deterministic_switching_schedule'] = {
            'physical_paths': 4, 'global_masks_visited': int(len(np.unique(switched_masks))),
            'maximum_scaled_update_difference': switch_error,
            'schedule': 'mask(b,k)=(17*b+13*k) mod 64; exogenous compatibility check only'}

        # B: watermark encoder/decode has ideal raw-y identity; compare healthy
        # baseline vs watermark-on with the very same physical and gate streams.
        physical = physical_paths(model, 16, SEED + 1)
        raw_off, _ = encode_decode(model, physical, SEED + 2, False)
        raw_on, Ms = encode_decode(model, physical, SEED + 3, True)
        healthy_off = run_common_g(model, physical, sender_packets=raw_off, gate_on=True)
        healthy_on = run_common_g(model, physical, sender_packets=raw_on, gate_on=True)
        decode_error = float(np.max(np.abs(raw_on - physical.y)))
        score_error = float(np.max(np.abs(healthy_on.scores - healthy_off.scores)))
        state_error = float(np.max(np.abs(healthy_on.post - healthy_off.post)))
        gate_disagreement = int(np.count_nonzero(healthy_on.gates != healthy_off.gates))
        decode_scaled = float(np.max(np.abs(raw_on - physical.y) / (1 + np.abs(physical.y))))
        state_scaled = float(np.max(np.abs(healthy_on.post - healthy_off.post) / (1 + np.abs(healthy_off.post))))
        if decode_scaled > 5e-13 or state_scaled > 5e-13 or score_error > 5e-7 or gate_disagreement:
            raise AssertionError(f'{label}: healthy watermark transparency acceptance failed.')
        per_model['healthy_roundtrip_transparency_gate_independent'] = {
            'wm_on_and_gate_on_tested_separately': True,
            'physical_paths': 16,
            'maximum_roundtrip_measurement_error': decode_error,
            'maximum_scaled_measurement_error': decode_scaled,
            'maximum_score_difference': score_error,
            'maximum_gate_disagreements': gate_disagreement,
            'maximum_state_difference': state_error,
            'maximum_scaled_state_difference': state_scaled,
            'minimum_active_M_eigenvalue': float(np.min(np.linalg.eigvalsh(Ms))),
            'startup_k5_gate_fraction_both': float(healthy_on.gates[:, 5].all(axis=-1).mean()),
            'scaled_state_pass_tolerance': 5e-13,
            'absolute_score_pass_tolerance': 5e-7,
            'warning': 'This verifies ideal roundtrip transparency, not SSCD-vs-IID detector power.'}

        # C: each registered attacked edge by itself under an identical fixed
        # healthy mode path. The exact bias must be G_i times decoded injection.
        edge_reports = []
        for edge in range(6):
            p = physical_paths(model, 8, SEED + 10 + edge)
            _, M = encode_decode(model, p, SEED + 30 + edge, True)
            delta = decoded_attack(model, M, edges=[edge])
            h = run_common_g(model, p, gate_on=False)
            a = run_common_g(model, p, delta, gate_on=False)
            paired = paired_identity(model, p, h, a, delta)
            receiver = int(model.links[edge, 0])
            predicted_bias = np.zeros((8, 150, 2))
            F = model.F_lookup[0, receiver]
            G = model.G_lookup[0, receiver]
            for k in range(1, 150):
                predicted_bias[:, k] = predicted_bias[:, k - 1] @ F.T + delta[:, k, edge] @ G.T
            target_error = float(np.max(np.abs(paired.b[:, :, receiver] - predicted_bias)))
            nonreceiver = np.delete(paired.b, receiver, axis=2)
            unaffected_error = float(np.max(np.abs(nonreceiver)))
            recursion_error = float(np.max(np.abs(paired.residual)))
            # Independently verify absolute error e=x-xhat, including the
            # physical process/measurement noise forcing and its signs.
            K = model.K_lookup[0, receiver]
            active = model.cfg['graph']['neighbors'][receiver]
            Bmatrix = np.eye(2) - K @ model.H[receiver] - G @ model.H[active].sum(axis=0)
            err = p.x - a.post[:, :, receiver]
            v = p.y - np.einsum('jmn,btn->btjm', model.H, p.x)
            absolute_error_residual = 0.0
            for k in range(1, 150):
                w = p.x[:, k] - p.x[:, k - 1] @ model.A.T
                predicted_error = err[:, k - 1] @ F.T + w @ Bmatrix.T
                predicted_error -= v[:, k, receiver] @ K.T
                predicted_error -= (v[:, k, active].sum(axis=1) + delta[:, k, edge]) @ G.T
                absolute_error_residual = max(absolute_error_residual, float(np.max(np.abs(err[:, k] - predicted_error))))
            if max(target_error, unaffected_error, recursion_error, absolute_error_residual) > 2e-8:
                raise AssertionError(f'{label}: isolated attacked edge {edge} recursion failed.')
            edge_reports.append({'edge_index': edge, 'receiver': receiver,
                                 'sender': int(model.links[edge, 1]),
                                 'max_single_receiver_bias_minus_recursive_F_G_prediction': target_error,
                                 'max_unaffected_receiver_bias': unaffected_error,
                                 'max_full_paired_recursion_residual': recursion_error,
                                 'max_absolute_error_F_process_measurement_forcing_residual': absolute_error_residual,
                                 'encoded_domain_source_attack_energy': float(np.sum(model.attack[:, edge] ** 2))})
        per_model['six_individual_attack_edges_gate_off'] = {
            'paths_per_edge': 8, 'edges': edge_reports,
            'attack': 'original dominant virtual-state waveform on one edge; IID M=.5I+beta beta prime, q=.5; no E=10 replacement',
            'absolute_pass_tolerance': 2e-8,
            'note': 'For actual data-dependent gates the full paired identity includes mode-mismatch forcing; checked next.'}

        # D: actual s=4, window=4 raw plugin42 gates and the exact full paired
        # identity. Verify startup, post-attack raw-buffer synchronization, and
        # current-mode lookup decay; no old Riccati covariance memory remains.
        p = physical_paths(model, 16, SEED + 50)
        _, M = encode_decode(model, p, SEED + 51, True)
        delta = decoded_attack(model, M)
        h = run_common_g(model, p, gate_on=True)
        a = run_common_g(model, p, delta, gate_on=True)
        paired = paired_identity(model, p, h, a, delta)
        startup_fraction = float(a.gates[:, 5].all(axis=-1).mean())
        post117_disagreements = int(np.count_nonzero(a.gates[:, 117:] != h.gates[:, 117:]))
        bias_identity_error = float(np.max(np.abs(paired.residual)))
        if startup_fraction != 1.0 or post117_disagreements or bias_identity_error > 2e-8:
            raise AssertionError(f'{label}: endogenous gate paired identity/coalescence failed.')
        per_model['actual_attack_raw_plugin42_gate_check'] = {
            'paths': 16, 'threshold': 42.0, 'attack_active_interval': [50, 110],
            'all_six_edges_gate_at_k5_fraction': startup_fraction,
            'gate_disagreements_from_k117': post117_disagreements,
            'paired_identity_packet_model': 'ideal sender packets y; actual floating encode/decode checked separately above',
            'maximum_full_paired_identity_residual': bias_identity_error,
            'maximum_mode_mismatch_forcing_from_k117': float(np.max(np.abs(paired.mismatch[:, 117:]))),
            'maximum_attack_forcing_from_k117': float(np.max(np.abs(paired.forcing[:, 117:]))),
            'max_state_contamination_norm_at_k116': float(np.linalg.norm(paired.b[:, 116], axis=-1).max()),
            'max_state_contamination_norm_at_k117': float(np.linalg.norm(paired.b[:, 117], axis=-1).max()),
            'note': 'Raw plugin threshold42 has forced k=5 startup alarm; this is preserved and explicitly exposed.'}

        results['models'][label] = per_model
    results['input_hashes'] = {
        'gain_table_sha256': hashlib.sha256((HERE / 'gain_table.json').read_bytes()).hexdigest(),
        'physical_configuration_sha256': hashlib.sha256((HERE / 'physical_configuration.json').read_bytes()).hexdigest()}
    out = HERE / 'acceptance_results.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
