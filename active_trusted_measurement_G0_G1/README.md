# 主动可信量测核验 G0/G1

**最终决定C：不进入G2。** 可信本地H₁已经覆盖原自主FDI全部有效方向。独立辅助标量传感只显示小幅、可由标准联合检验和固定采样解释的功效收益；没有找到独立中心性质。

基线28c363343b2545a65b4abee915cff04a17fbad80；参考实现0ea9c7d。日期2026-11-16。

1. [量测来源与权限](01_量测来源与权限.md)
2. [联合隐蔽性充要条件](02_联合隐蔽性充要条件.md)
3. [噪声、功效与未知起点](03_可信量测噪声与检测功效.md)
4. [标准联合检测强基线](04_标准联合检测强基线.md)
5. [最坏攻击与损害](05_最坏攻击方向与资源约束.md)
6. [直接近邻排重](06_直接近邻理论排重.md)
7. [研究决定](07_G0G1研究决定.md)
8. [证据等级](evidence_table.md)

## 最小复算

Python3、numpy2.2.5、scipy1.15.3、sympy1.14.0。目录须与既有common_g_reference_revision_20261115、attack_gain_switch_audit_20261115相邻。依次运行：

~~~text
python structural_certificates.py
python batch_power_and_attacks.py
python prior_baseline_check.py
python online_bounds.py
python independent_checks.py
python build_reports.py
~~~

不必联网或运行prepare_protocol/update_literature_register；这些是本机一次性登记工具，不是科学计算依赖。online较慢，但只计算Gaussian边缘概率和矩阵，复用已有秘密路径，不抽新物理噪声。

registration.json为事前协议。其中“rigorous bounds”指解析概率不等式，当前报告/JSON另外明确浮点端点没有区间认证。全历史已知先验强基线是审查附录，不追溯改注册参数。NPZ保存攻击信息、损害及重叠协方差；JSON保留方向、能量和首报界。

原稿/阶段稿/冻结底座/第二篇不改；唯一文献库按项目约定维护。当前非新算法成功报告、全局风险证书或原50节点复现。Huang C&S全文仍缺。

核查回执：[本机保护范围](delivery_validation.json)、[公开目录复算](public_reproduction.json)、[输入与换行映射](scientific_inputs.json)、[文献入库](literature_registration.json)。公开复算的6份JSON内容一致、3份生成报告字节一致、160个数组逐项完全一致；JSON对象键顺序不作为科学差异。
