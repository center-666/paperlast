# 历史 50 节点堆叠 KF：理论核查与公平配对结果

**中文执行摘要：本批是旧堆叠 KF，不是后来新附件中的共 G。** 每个模型512条健康校准、1024条独立评估。阈值42在k5有秩亏导致的启动必报警；完整健康路径假警为100%，不能拿post20约0.33%/0.36%的点假警率冒充全程误报。独立校准后负耦合在原攻击方向上改善MSE与eligible TPR，但不门控的原始MSE差未被MC清楚分辨；已知六候选边的静态丢弃基线又在全网2%健康额度内通过，必须正面比较。新共G附件在本批计算后到达，以下结果不得改名沿用。

下列英文表采用代码原字段名，完整说明和公式见 [THEORY.md](THEORY.md)。文档只读取已有JSON，没有新增试验。登记运行环境为Python3.14.9、NumPy2.5.4、SciPy1.18.1、NetworkX3.7、Matplotlib3.11.2；文档整理用的解释器不是本批长验证环境的替代说明。

**All runs in this directory use the recovered historical unconstrained stacked KF, NOT common G. The new common-G attachment arrived AFTER these runs and was not part of their input, calibration or result. A later common-G phase requires its own provenance and must not relabel these existing JSON/NPZ files.**

Documentation only: no new simulation was run to assemble this report. For EACH physical model there are 512 independent healthy calibration paths and 1,024 independent evaluation paths, evaluated in 16 batches of64. All 50 physical sensor streams were generated; raw-measurement-sharing noninterference permits recomputing only the five affected receivers when gates change. The original main and arbitrary-attack optimizer were not called.

## 1. Inputs, coupling and metric definitions

- Keep source A/H/Q/R, 50 nodes and298 edges, horizon150, six candidate attacked links, support[50,110), and magnitudes6/10. This is not supplementary coded energy10.
- Compare IID, negative theta=-1.315236976827179 and positive theta=+1.315236976827179. Share physical noise, radial spectra and the first attacked M across laws; use separate physical/radial/initial-direction/transition RNG streams. Equal spectra do not imply equal realized encoded-packet energy.
- Receiver modes: wm_no_gate (watermark but no rejection), legacy42 (original plugin score and threshold42), calibrated (same score, independently calibrated common threshold). Thresholds are never separately tuned for favorable watermark results.
- Primary MSE period20..149; attack50..109; after_support110..149; after_buffer_recovery117..149. The source mean-path RMSE is preserved separately and is not sqrt(expected MSE).
- TPR is the fraction of eligible time/link alarm decisions at57..109, averaged over six preconfigured candidate links. It is not probability of ever detecting an attack and is not unknown-location whole-graph detection.
- Every SE uses an independent physical path as its statistical unit; time/link dependence stays within each path.

## 2. Original function-body replay validation

source_model.py loads hash-pinned original functions, extending only diagnostic returns. Four original single-path calls cover both models with/without attack. Information-form replay uses the same packets and watermark realization; all four gate paths match.

| Model | Attack | Max estimate error | Max P error | Max score error | Max score error within5 of threshold42 |
|---|---|---:|---:|---:|---:|
| unstable | False | 1.8626451e-09 | 1.1391582e-13 | 5.059789e-09 | 4.1754618e-09 |
| unstable | True | 4.6566129e-10 | 1.1391582e-13 | 0.000208179 | 1.3627854e-09 |
| stable | False | 2.717826e-13 | 1.6775123e-13 | 1.5133281e-09 | 1.5133281e-09 |
| stable | True | 2.3270275e-13 | 1.6775123e-13 | 5.1867005e-08 | 2.4990783e-09 |

The larger absolute unstable attacked score error occurs at large scores, while threshold-near errors are small and gate decisions match on these four paths. This is not an all-input bitwise equivalence theorem; exact equivalence follows from the information-form identity.

## 3. Real energy, moment design and lower bounds

The design uses the original six-link coded energy and the14-dimensional effective support of four overlapping8D parity residuals from eight2D packets. Information budget=.1 nats per step, hence .7 for this eight-packet window; not a total150-time/50-node secrecy budget.

| Model | Original coded energy | IID robust Jensen DG LB | Candidate LB | IID worst DG UB witness | Strict proxy witness |
|---|---:|---:|---:|---:|---|
| unstable | 59006840.793 | 98.621881 | 107.02485 | 100.49644 | True |
| stable | 25067.5698831 | 184.74349 | 200.16345 | 186.4237 | True |

Both321-point screens choose the negative kernel. Candidate LB exceeds an IID UB from an admissible direction, providing a numerical strict worst-covariance-proxy witness. This is neither a global parameter optimum nor actual plugin-score power certification. Robust energy-shell moment design and actual original-dominant-direction MC have different quantifiers.

| Model | IID original-direction displacement risk | Mean floor | Mean+radial floor | Budgeted signed-family relative change bound |
|---|---:|---:|---:|---:|
| unstable | 100.33522 | 68.102626 | 90.00533 | 0.19466% |
| stable | 0.034651453 | 0.022781092 | 0.030717116 | 0.19882% |

These are no-gate contamination risks with150-time weighting and no healthy error added. The MC primary table instead weights130 times20..149; comparing100.335 directly with116.256 is invalid. theory_checks.json independently compares dense input-output and cost-to-go contractions at12 points.

## 4. Healthy false alarms: threshold42 has a startup artifact

At k5 only two8D residual samples exist, so sample covariance rank<=1. THEORY.md proves a data-independent plugin-score floor42.4282788>42. Every healthy decision path therefore has a startup alarm; reporting only post20 point-FPR conceals a gate that really occurred.

| Model | Calibrated threshold | Calibration path FWER | Legacy42 eval full FWER | Legacy42 eval post20 any alarm | Legacy42 eval post20 point-FPR | Calibrated eval full FWER [95% CP] |
|---|---:|---:|---:|---:|---:|---|
| unstable | 60.67994737 | 4.8828% | 100.0000% | 66.0156% | 0.32890% | 4.4922% [3.3073%, 5.9468%] |
| stable | 59.95191033 | 4.8828% | 100.0000% | 69.0430% | 0.36433% | 5.6641% [4.3287%, 7.2609%] |

The threshold is the empirical95th percentile of512 healthy path maxima over six links and actual decision times4..149. Evaluation paths are independent. Stable heldout FWER is above5% but its interval includes5%; there is no guaranteed5% certificate. Zero post20 alarms under calibrated thresholds is an observation, not proof of zero probability.

### Healthy estimation costs, shared across watermark kernels

| Model | Healthy receiver mode | Network MSE all150 | Network MSE post20 | Affected5 MSE post20 | Source mean-path RMSE post20 |
|---|---|---:|---:|---:|---:|
| unstable | wm_no_gate | 0.07301203651 | 0.05917443561 | 0.04864285802 | 0.2391573021 |
| unstable | legacy42 | 0.07302159276 | 0.05917661137 | 0.04866461568 | 0.2391616009 |
| unstable | calibrated | 0.07301206237 | 0.05917443561 | 0.04864285802 | 0.2391573021 |
| unstable | static_known_candidates | 0.07365145755 | 0.05981724096 | 0.05507091157 | 0.2404565478 |
| stable | wm_no_gate | 0.07173947697 | 0.05900034568 | 0.04837115688 | 0.238787795 |
| stable | legacy42 | 0.07174970237 | 0.05900324553 | 0.04840015538 | 0.2387936714 |
| stable | calibrated | 0.07173955378 | 0.05900034568 | 0.04837115688 | 0.238787795 |
| stable | static_known_candidates | 0.07237799886 | 0.05964387624 | 0.05480646249 | 0.240088897 |

The comparison across receiver modes includes admission/gating costs. Within one receiver mode, ideal healthy encode/decode makes the kernel choices pathwise identical; do not count identical healthy outputs as separately tuned budgets.

## 5. Full fair paired outcome tables

### unstable

| Receiver/kernel | Network MSE post20 | Affected5 MSE post20 | Attack network MSE | After-support network MSE | Eligible TPR | Source mean-path RMSE post20 |
|---|---:|---:|---:|---:|---:|---:|
| wm_no_gate_iid | 116.2556456 | 1162.013354 | 251.7676549 | 0.1351823064 | 99.853209% | 4.296836202 |
| wm_no_gate_negative | 114.935546 | 1148.812358 | 248.9098872 | 0.1315100968 | 99.896509% | 4.291945204 |
| wm_no_gate_positive | 117.6687241 | 1176.144139 | 254.8263874 | 0.1395887009 | 99.763844% | 4.315627479 |
| legacy42_iid | 0.05998366784 | 0.05673518032 | 0.06091690502 | 0.05938707743 | 99.853209% | 0.2407400038 |
| legacy42_negative | 0.05995243433 | 0.05642284529 | 0.06084923243 | 0.05938707743 | 99.896509% | 0.2406816221 |
| legacy42_positive | 0.06004738501 | 0.05737235207 | 0.0610549589 | 0.05938707743 | 99.763844% | 0.2408578773 |
| calibrated_iid | 0.06175273208 | 0.07442582272 | 0.06475120841 | 0.0593860596 | 98.801715% | 0.2437231907 |
| calibrated_negative | 0.06150521207 | 0.0719506226 | 0.06421491506 | 0.0593860596 | 99.075337% | 0.2433015153 |
| calibrated_positive | 0.06221880151 | 0.07908651704 | 0.06576102552 | 0.0593860596 | 98.354277% | 0.2444739613 |

### stable

| Receiver/kernel | Network MSE post20 | Affected5 MSE post20 | Attack network MSE | After-support network MSE | Eligible TPR | Source mean-path RMSE post20 |
|---|---:|---:|---:|---:|---:|---:|
| wm_no_gate_iid | 0.09901502085 | 0.4485179085 | 0.1457271755 | 0.05906233193 | 92.495209% | 0.2971383512 |
| wm_no_gate_negative | 0.09889492234 | 0.4473169235 | 0.1454669736 | 0.05906231463 | 93.902933% | 0.2970342196 |
| wm_no_gate_positive | 0.09917736036 | 0.4501413037 | 0.1460790425 | 0.05906213481 | 89.897492% | 0.2973022613 |
| legacy42_iid | 0.0601327793 | 0.05969549311 | 0.06144214891 | 0.05912080697 | 92.495209% | 0.2410523408 |
| legacy42_negative | 0.05999237879 | 0.05829148795 | 0.06113766979 | 0.05912122397 | 93.902933% | 0.2407704264 |
| legacy42_positive | 0.06036404548 | 0.06200815483 | 0.061945755 | 0.0591170129 | 89.897492% | 0.2415110044 |
| calibrated_iid | 0.06466358825 | 0.1050035825 | 0.07128551315 | 0.05908266942 | 69.123428% | 0.2493721602 |
| calibrated_negative | 0.06400056398 | 0.09837333984 | 0.06984758767 | 0.05908472877 | 72.487041% | 0.2481709506 |
| calibrated_positive | 0.06563241676 | 0.1146918677 | 0.07338555928 | 0.05908129291 | 63.853675% | 0.2511381279 |

wm_no_gate still records passive threshold42 scores, so its TPR matches legacy42 for the same watermark; those are not independent power experiments.

### Paired changes relative to IID

| Model | Mode/kernel | Network MSE delta post20 | Path SE | Eligible TPR delta (percentage points) | Path SE(pp) |
|---|---|---:|---:|---:|---:|
| unstable | wm_no_gate/negative | -1.320099636 | 0.99641271 | 0.043300 | 0.011144 |
| unstable | wm_no_gate/positive | 1.413078478 | 1.0747896 | -0.089365 | 0.014594 |
| unstable | legacy42/negative | -3.123350326e-05 | 7.7795e-06 | 0.043300 | 0.011144 |
| unstable | legacy42/positive | 6.371717504e-05 | 9.4583435e-06 | -0.089365 | 0.014594 |
| unstable | calibrated/negative | -0.0002475200117 | 2.8489031e-05 | 0.273622 | 0.032341 |
| unstable | calibrated/positive | 0.000466069432 | 3.7982259e-05 | -0.447438 | 0.040163 |
| stable | wm_no_gate/negative | -0.000120098504 | 0.00014707251 | 1.407724 | 0.089586 |
| stable | wm_no_gate/positive | 0.0001623395149 | 0.00015732114 | -2.597718 | 0.103607 |
| stable | legacy42/negative | -0.0001404005167 | 1.0970198e-05 | 1.407724 | 0.089586 |
| stable | legacy42/positive | 0.0002312661712 | 1.3524724e-05 | -2.597718 | 0.103607 |
| stable | calibrated/negative | -0.0006630242694 | 3.4781624e-05 | 3.363613 | 0.153650 |
| stable | calibrated/positive | 0.0009688285158 | 4.0153935e-05 | -5.269752 | 0.165744 |

No-gate negative-kernel MSE differences are below2 path SE in both models, so MC does not resolve a large raw-MSE benefit. Gated negative-kernel gains exist for the registered original direction, while positive hurts; this does not follow from fixed-gain Q alone. Legacy42 has forced healthy startup alarms, so fair full-decision FWER comparisons require the calibrated version.

## 6. Alarm delays and eligible windows are different metrics

These are first alarm delays from attack time50 within50..109. Every model/policy/link has observed detection fraction1. This includes partially attacked boundary windows; first_full_window_alarm_delay separately preserves eligible windows starting57.

| Model/mode/kernel | 1->0 | 3->2 | 16->15 | 31->17 | 49->14 | 34->17 |
|---|---:|---:|---:|---:|---:|---:|
| unstable/legacy42/iid | 1.52637 | 1.99414 | 2.08887 | 1.56152 | 2.08984 | 1.99316 |
| unstable/legacy42/negative | 1.47754 | 1.88477 | 1.99219 | 1.54688 | 1.96582 | 1.87793 |
| unstable/legacy42/positive | 1.59668 | 2.06543 | 2.18066 | 1.65723 | 2.19141 | 2.12598 |
| unstable/calibrated/iid | 3.44531 | 4.35156 | 4.44922 | 3.45898 | 4.29980 | 4.26562 |
| unstable/calibrated/negative | 3.28613 | 4.09082 | 4.20801 | 3.27148 | 4.04980 | 4.09570 |
| unstable/calibrated/positive | 3.70312 | 4.53125 | 4.70117 | 3.69141 | 4.56641 | 4.59961 |
| stable/legacy42/iid | 0.64062 | 0.66699 | 0.68750 | 0.72656 | 0.90820 | 0.69727 |
| stable/legacy42/negative | 0.64551 | 0.65820 | 0.65723 | 0.70898 | 0.88086 | 0.67188 |
| stable/legacy42/positive | 0.67188 | 0.65625 | 0.68262 | 0.76758 | 0.95508 | 0.70215 |
| stable/calibrated/iid | 2.15430 | 2.01172 | 2.05176 | 2.31836 | 2.57520 | 2.11914 |
| stable/calibrated/negative | 2.05078 | 1.91504 | 1.90820 | 2.18945 | 2.40820 | 1.89258 |
| stable/calibrated/positive | 2.42383 | 2.13770 | 2.12598 | 2.53320 | 2.90527 | 2.13867 |

The gate is recomputed at every time, not permanent isolation at first alarm. Startup k5 alarms do not make these attack-window delay fields zero. Fraction-of-eligible-decisions TPR and observed ever-detection fraction1 need not coincide.

## 7. Path identities, first packet and post-stop recovery

| Model | First attacked tick across-kernel max difference | Max paired-bias recursion error | Max MSE-decomposition error | Max Riccati-bound violation | Gate disagreement after117 |
|---|---:|---:|---:|---:|---:|
| unstable | 0 | 1.6819669e-10 | 7.2759576e-12 | 0 | 0 |
| stable | 0 | 8.9789287e-15 | 8.8817842e-16 | 0 | 0 |

Last injection109; five-packet residual recovers at114; four-residual buffer at117. Gates agree from117, but estimates and P need not immediately be exactly equal. Tail decay does not undo previously emitted attack-period error. curves retains displacement, state cross-term, gain-mismatch and admitted-attack forcings separately; these are not independent orthogonal terms.

## 8. Strong simple baseline: always drop the six known candidate edges

The source already monitors only these six candidate links. Always ignoring them requires no attack-activity/magnitude knowledge and gives exactly zero effect for this communication-only attack. This is a known-candidate-set baseline, not unknown-location defense. Equality of attacked and healthy outputs is a pathwise identity, not a claim that another attack MC was run.

| Model | Healthy no-gate network MSE post20 | Static-drop network MSE post20 | Healthy no-gate affected MSE post20 | Static-drop affected MSE post20 | Analytic network healthy cost post20 | Analytic affected healthy cost post20 |
|---|---:|---:|---:|---:|---:|---:|
| unstable | 0.0591744356 | 0.0598172410 | 0.0486428580 | 0.0550709116 | 1.08484% | 13.21777% |
| stable | 0.0590003457 | 0.0596438762 | 0.0483711569 | 0.0548064625 | 1.08446% | 13.21013% |

At a predeclared2% NETWORK-average healthy allowance this static baseline passes in both models, while affected-group cost is about13.21% and fails a2% affected-group allowance. Do not switch denominator after seeing this result. The static candidate-set baseline beats the attacked network MSE of the dynamic gated schemes here; this challenges a necessity/advantage claim for the complicated method in this registered problem. It does not settle unknown attack locations. Every source H is full rank, so the separate regional indispensable-neighbor motivation is not an existing fact of this network.

## 9. Recommendations and fields of record

- Keep deterministic TV moments, endogenous paired identity, rank14 support, first-packet invariance and mean/radial lower floors as independently checkable theory results.
- Retain the original-direction calibrated negative-kernel evidence, but do not claim universal power, global parameter optimality or complex-method necessity against the known-six baseline.
- Archive the newly arrived common-G input in a separate phase with its own gains, code, calibration and runtime provenance; never relabel these stacked-KF paths.

| Content | Source of values |
|---|---|
| Registration, sizes, seeds, environment | `registration.json:validation_sizes,seeds,environment,source` |
| Original-function replay | `source_validation.json:rows[*]` |
| Actual energy, Jensen witness, floors and family bounds | `design_results.json:models[*].design` |
| Rank14 support | `design_results.json:models[*].buffers[*]` |
| Thresholds and full512 calibration maxima | `<model>_validation.json:calibration` |
| MSE, source RMSE, TPR, SE | `<model>_validation.json:summaries.<mode>_<policy>__<metric>` |
| Paired differences | `<model>_validation.json:paired_comparisons` |
| Six-link delays | `<model>_validation.json:delay_summaries` |
| Identity errors and gate recovery | `<model>_validation.json:identity_checks,gate_path_diagnostics,curves` |
| Static known-set baseline | `observer_results.json:models[*].fixed_known_candidates_drop` |
| Path-level data | `<model>_paired_paths.npz` |
| Dense Q, startup floor and transparency checks | `theory_checks.json` |

Reproduction order: prepare_design.py, validate_source.py, theory_checks.py, observer_checks.py, long_validation.py. Copy this directory before rerunning because scripts overwrite corresponding JSON/NPZ outputs. This report was assembled read-only from existing results; no new experiment was executed.
