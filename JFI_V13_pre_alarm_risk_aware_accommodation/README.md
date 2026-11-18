# JFI V1.3 报警前风险感知的在线攻击补偿

**结论是保留 V1a 为 A0 主算法，V1.2a 为解析名义精度约束配置。** V1.3 实现了两世界偏移的工作模型预测及因果权重。两组独立健康样本均支持整段 2% 约束。每模型 2,048 条配对路径上，尚无足够证据支持 V1.3 降低严格报警前危险。A1 首包的压力测试预测严重低估实际越界。保留该证据作为方法边界。

|阅读入口|内容|
|---|---|
|[条件危险模型与理论](01_条件危险模型与理论.md)|合法信息、健康误补偿、候选积分、状态误差与风险转移条件|
|[闭环校准与研发决定](02_闭环校准与研发决定.md)|旧 Pareto 与离线单步范围、新配对风险、健康上界、校准及恢复|
|[近邻与论文定位](03_近邻与论文定位.md)|2025–2026 年重构、Bayesian 模式和序贯报警|
|[第四第五章新增技术内容](第四第五章新增技术内容.md)|独立审阅稿中的算法及结果|
|[Pareto 图](results/legacy_pareto.png) · [校准图](results/independent_calibration.png)|旧样本的误差—危险关系、新样本的单拍预测|
|[设计](design.json) · [冻结规则](frozen_policy.json) · [交付核验](results/delivery_checks.json)|独立种子、样本数量、代码哈希、候选选择、数据复核|

## 调用和复算

Python 环境见 `requirements.txt`。在本机工作区可省略 `--baseline`，由原 V1 定位冻结交付副本。GitHub 检出时从本目录以 `--baseline ..` 指向仓库根，V1、V1.1、V1.2 是相邻目录。

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
python -B -X utf8 experiments.py --phase validation --baseline ..
python -B -X utf8 experiments.py --phase healthy_check --baseline ..
python -B -X utf8 experiments.py --phase historical_joint --baseline ..
python -B -X utf8 offline_diagnostic.py --baseline ..
python -B -X utf8 audit.py --baseline ..
python -B -X utf8 figures.py
python -B -X utf8 verify_delivery.py --baseline ..
~~~

开发阶段的 `legacy_screen.py` 重算旧 Pareto。在同一 GitHub 检出中新建 V1.3 的隔离副本时，从已有 `design.json` 和匹配的旧轨迹汇总开始，还可用 `experiments.py --phase development --baseline ..` 重算开发选择。现有冻结文件存在时入口会保留已存决策。会话采用同一次核验的 `frozen_policy.json` 验证，而未从验证数据重拟合。

在线 `OnlineRiskReceiver(prepared, common_step, 16.0)` 每条独立路径创建一次。逐拍把可信本地、健康邻居、实际目标解码值、解码矩阵交给 `step`。输出估计、标量权重、报警及报警前条件危险分数。候选来自 `risk_policy.py`，原混合似然用等价正态方程批算，经原 SVD 验证。线上函数没有真起点、干净目标、状态真值和真实失真参数。

## 数据口径

- 验证路径包 `results/validation/*_paths.npz` 的键为 `情形__字段`，方法次序见汇总。`alpha`、`offset_sq`、`risk_score` 依次为路径／方法／时刻。概率和偏移三维字段用 float32 储存，事件时刻、计数与摘要从原双精度产生。阈值复算已与路径计数比对。
- `risk` 表示整条配对路径上首次危险严格早于首次报警。`first_packet` 是失真首包是否达阈，`first_alarm` 包括可能更早的正常误报，未发生记为 32。`risk_score` 是当拍条件代理，首次报警后留空。`mse`、`tail_mse`、`healthy_mse` 保留路径级损失。
- A1-H/C 证据由固定失真的外给回放和原历史联配组成。配对差和校准结论限这组输入。受攻攻击参数的逐防御重优化、实时反馈攻击及整个 A1 集合的最坏风险未获验证。
