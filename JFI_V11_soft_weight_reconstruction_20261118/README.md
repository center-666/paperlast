# JFI V1.1 软补偿权重重构与闭环消融

> 在 GitHub 检出中，从本目录运行实验脚本时，将 `--baseline ..` 指向仓库根目录。当前研究决策见 [V1.2](../JFI_V12_budget_and_risk_aware_compensation_20261118/README.md)。

日期 2026-11-18。第一阶段复现了旧 V1 的 12 条联配轨迹并比较六种权重。旧 2% 当拍名义上限是 V1b 受攻 MSE 较差的主要原因，叠加的一标准差也有影响。V1.1 以同一辨识器和同一共 G 更新实现可证明的名义整段健康 MSE 限制。独立配对样本中它仍落后于 V1a，普通 A0 的报警前危险也有增加，因而保留 V1a 为主要受攻性能基线。

|材料|用途|
|---|---|
|[V1b 源码审计与权重理论](V1b源码审计与权重理论.md)|回答七项源码问题，推导条件 MSE 与名义轨迹保证|
|[闭环消融与独立验证](闭环消融与独立验证.md)|冻结协议、六方案消融、健康精度、受攻 MSE、风险、辨识误差及历史反馈|
|[第四、第五章修订段落](第四第五章修订段落.md)|独立技术文字，含未通过的受攻性能结果|
|[V1.2 修改建议](V1.2修改建议.md)|将未用健康额度留给后续时刻的唯一提案，尚未实现|
|[analytical_weight.py](analytical_weight.py)|V1.1 的因果标量权重，给定现有观察器输出和同拍共 G 预览可计算|
|[test_v11.py](test_v11.py)|5 个确定性检查，含完整联合 Gaussian 回归交叉验证|
|[第一阶段注册](phase1_registration.json)、[独立验证冻结](phase2_registration.json)|锁定顺序、种子、档案分组、脚本及输入哈希|
|[最终复核](reproduction_checks.json)|保护范围、运行检查、结果文件哈希|

## 复算

在本目录运行，Python 环境采用 `requirements.txt` 中的 NumPy 与 SciPy 版本。使用 Windows PowerShell 可先设 `$env:OPENBLAS_NUM_THREADS='1'`。`weight_experiment.py` 与 `validate_v11.py` 从原 V1 目录导入辨识器，默认使用项目内 `交付/paperlast_audit_20261113/` 的冻结输入。若镜像位于其他位置，可给两个实验脚本提供 `--baseline`。复算 V1.1 独立结果前，程序会检查冻结代码和第一阶段存档的哈希。用 `--out` 可把复算另存到新目录。

```powershell
python -B -X utf8 weight_experiment.py
python -B -X utf8 test_v11.py
python -B -X utf8 validate_v11.py
```

`TailBudgetPolicy(prepared)` 为可复用的接收端权重接口。初始化仍采用 V1 的 `PreparedModel`、`OnlineEvidence` 和 `CommonGPath` 实例。在第 `k` 拍，依次从原始报文取得 `ev = observer.observe(local, healthy, target, decoder)`，从本策略得 `u = path.preview(local, healthy, target)`，通过 `choice = policy(k, u, ev)` 计算权重，再调用 `path.update(local, healthy, target - choice.alpha * ev.delta)`。监测仍使用 `ev.mixture_alarm`。此接口对当前输入和初始声明时域有效，当前权重不读取模拟真值或原始干净目标报文。

## 结果数据字段

- [消融总表](phase1_results/ablation.json)和[独立总表](phase2_results/heldout.json)给出方法顺序、每场景评价数据及配对差。独立总表的 `models` 对应 48 条新物理路径，`matched_historical_A1_models` 另存 32 条旧物理联配留出回放。`A0_P3_proxy` 沿用原有限候选档案，不作全局最坏声称。新 A1 样本把留出失真记录作为外给输入。
- `phase1_results/*_step_logs.npz`、`phase2_results/*_step_logs.npz` 中键为 `情形__字段`。观察器共享字段的轴顺序为路径、拍、分量，含无攻击 `p_null`、已开始 `p_started`、当拍活动 `p_active`、全后验 `onset_weights`、失真估计 `delta_hat`、后验协方差 `delta_cov`、一标准差 `sigma_margin` 及模型间占比。`decoded_difference` 与 `state_truth` 仅由评价器记录。
- 方法字段依次为路径、方法、拍；向量字段再加最后一维。`posterior_raw_weight` 是解析比例，`posterior_clipped_weight` 和 `margin_weight` 是不带、带余量的候选，`original_cap` 和 `applied_cap` 记录旧上限及本法上限，`alpha` 是执行值。两个绑定标记记录实际起限幅作用的项。用 `preview`、`post`、`history_feedback`、`correction` 可复算历史和补偿效应。未使用相应上限的策略在名义尾部预算字段中记零。`common_alarm` 是全部方法共享的实际统计报警，`native_V0_alarm` 是原 V0 的五包奇偶报警。
- `matched_archive` 文件保留历史记录中噪声、水印、失真的配对，可独立于新噪声外给回放阅读。任何一组档案的配对结果均不替代针对所有攻击序列的统一保证。
