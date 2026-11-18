# JFI V1.2 健康预算结转与风险感知补偿

**技术选择是以 V1a 作为当前原 A0 主算法。** 两组名义健康模型在联合 95% 置信下的整段 MSE 相对增量上界是 +1.4141% 和 +1.3799%。V1.2a 结转减少了名义截断，提供了需要显式健康上界时的替代方案。普通 A0 的受攻 MSE 和报警前危险仍以 V1a 较低。保留软权重作失配研究与健康保证对照。V1.2b 可针对报警前危险建模，作为下一步机制研究。

|材料|内容|
|---|---|
|[01 V1a 健康精度核验](01_V1a健康精度核验.md)|旧 48 条配对、冻结模型期望及联合置信界、逐拍对照|
|[02 结转理论与全补偿条件](02_结转理论与全补偿条件.md)|因果账本、整段 2% 证明、完全使用后验校正的单拍最优条件|
|[03 闭环配对与报警前危险](03_闭环配对与报警前危险.md)|原路径回归、新路径受攻 MSE、首包和危险时刻、诊断及基线|
|[04 V1.2b 风险模型与近邻](04_V1.2b风险模型与近邻.md)|低维接收端目标、健康反事实、最坏风险的适用边界|
|[第四第五章修改稿](第四第五章修改稿.md)|可供独立审阅的算法、理论和实验段落|
|[复算及保护范围](results/delivery_checks.json)|日志与汇总交叉检查、算法集的结果哈希和受保护输入|

## 运行

在 GitHub 检出后进入本目录。使用冻结 V1、V1.1 的 Python / NumPy / SciPy 环境，将仓库根目录用 `--baseline ..` 传给实验脚本。源码、原始记录及新种子见 [注册](registration.json)。完整本机原稿的哈希核验结果见 [交付复核](results/delivery_checks.json)。

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
python -B -X utf8 -c "import pathlib, unittest, test_v12 as t; t.DEFAULT_BASELINE=pathlib.Path('..').resolve(); unittest.main(module=t)"
python -B -X utf8 receiver_experiment.py --phase development --baseline ..
python -B -X utf8 nominal_audit.py --baseline ..
python -B -X utf8 receiver_experiment.py --phase validation --baseline ..
python -B -X utf8 audit_pairs.py --baseline ..
python -B -X utf8 archived_joint_check.py --baseline ..
~~~

每条新接收路径需要一个 `CarryBudgetPolicy(prepared)`。每拍用原 `OnlineEvidence.observe` 更新到本拍，基于本方法自身历史按冻结共 G 计算 `preview`，再调用 `step(k, preview, evidence)`。将其 `alpha` 乘辨识的失真均值，从目标报文扣除后调用原共 G 更新。返回动作、同拍上限是否起限、名义预算增量和余额。已有 BF 报警从原始接收信息产生。单独接收器试验范围限定为声明时域和原协议。

## 结果索引

- [名义积分](results/nominal_audit.json)含协方差解析序列及健康点估计与上界。`results/*_bounded_health_scores.npy` 的列依次为 V0、V1a、V1b、V1.1，每行是当段条件均值误差范数平方的时间平均。
- [旧开发样本](results/development/summary.json)、[新验证](results/validation/summary.json)及[历史 A1 联配](results/archived_joint.json)分别保留。新验证的 A1 是新物理数据上固定档案失真的回放。P3 对照为历史代理记录，风险证据限当前输入。每模型 96 条新路径在所有方法、情形中成对复用。
- `results/development/*_paths.npz`、`results/validation/*_paths.npz` 的键是 `情形__字段`。方法字段轴为路径、方法、时刻，状态或补偿向量再加分量。`alpha/proposal/cap/cap_binding` 记录权重及限幅，`opening/allocation/available/nominal_charge/balance` 记录结转账。`post`、`state` 供复算误差。`displacement_squared`、`peak_paired_displacement_squared`、`first_danger`、`first_alarm_common`、`first_alarm_native`、`support_mse`、`tail_mse` 和 `before_alarm` 汇集实际配对表现。无首次事件记为 32。部分方法自带其他统计量，主比较用相同 BF，另存各法初始报警。
- 随机起点下 `paired_healthy_oracle_post` 是为已知起点 GLS 保存的同起点健康对照。首次计算后发现这一对照所用时标有误，代码按同一时标修正并用原种子重跑。注册文件留有修正前其余九种方法的后验哈希，验证逐项相同。种子预检识别了第一份编号与旧队列的重复，已在任何新样本生成之前换成不重叠的编号。
- [逐路径危险审计](results/paired_risk_audit.json)记录旧 V1.1 与 V1a 的不一致路径，以及首包相对 V0 的实测变化。
