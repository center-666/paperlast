# B2 execution formula correction

## Finding

The first B2 replay removed the target attack perturbation completely whenever H1 alarmed, then multiplied only the clean sender measurement by the alarm weight q. That assumes the detector can perfectly separate and remove the attack component. It does not match the registered link-weighting model.

D1-A defines the actual watermarked packet perturbation as `delta_pkt = M_k a_k` and its weighted attack input as `B_delta(q) = q G(q) S`. B0 applies the alarm weight to the complete target-link packet. For alarm bit g_k, the effective target-link weight is q on alarm and 1 otherwise. The weighted aggregate contribution is therefore `q_k (y_sender + delta_pkt)`, and the target attack-state input is `G(q_k) q_k delta_pkt`.

## Corrected replay

The attack perturbation is added to the registered directed-link packet before target-link weighting. On alarm, the candidate q scales both the clean measurement and the perturbation. Without alarm, the link weight returns to 1. The remaining links, physical paths, thresholds, policy table, random seeds, and B1 health data were not changed.

The first-pass and denominator-corrected archives are retained in `b2_attack_results_pre_metric_correction` and `b2_attack_results_pre_input_correction`. Their attack propagation formula does not match the D1-B objective, so they must not be cited as D1 attack-performance evidence. The final `b2_attack_results` archive replays the same B2 random streams under the corrected formula and has an independent path-prefix verification.

This correction changes the alarm-time attack input and therefore replaces the earlier attack MSE and forcing outputs. It does not change B1 health results. B1 remains a posthoc execution under B0 definitions, and B2 remains descriptive conditional on the frozen B1 thresholds and policy table.
