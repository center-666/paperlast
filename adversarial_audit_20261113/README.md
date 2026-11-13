# 对抗式审查入口（2026-11-13）

先读[独立审证总报告](独立审证总报告.md)、[反例报告](反例与失效条件报告.md)及[两轴评估](投稿独立性评估.md)。数学轴附加条件下通过，投稿轴应停止独立方法主张。

[定理证据表](定理证据表.md)区分精确证明、有限枚举、浮点诊断、成熟理论与未完成验证。

复算脚本均为本轮独立实现，没有调用原安全词/图/尾部脚本：

- `independent_cert.py`：SymPy有理矩阵幂＋闭式卷积，重建安全词、尾部及图/预测。
- `find_counterexamples.py`：有限确定性搜索见证；`certify_counterexamples.py`用Fraction精确复核发布的舍入参数。
- `public_period_audit.py`：公开周期有限枚举＋交替全时域minimax精确证明。
- `physical_origin_audit.py`：原始参数DARE重建、有理上下解与K/F/g误差包围。
- `bridge_margin_check.py`：将独立DARE误差界传播到有限安全裕度和无限尾部，未调用旧鲁棒性脚本。
- `detection_privacy_boundary.py`：从fresh分类重建图和过去符号预测，检查预定加性FDI的支持几何。
- `verify_audit_delivery.py`：仅在本机完整项目核保护/输入哈希，不是独立证据本身。

脚本依赖Python、NumPy、SciPy、SymPy、mpmath；按上述顺序运行。物理来源核查涉及本机旧来源文件，公开审阅仓库不包含全部原TeX和PDF；缺这些文件时不能宣称完成本机全项目审计。固定参数及证明用数值列在JSON，支持单独核查。没有物理噪声Monte Carlo。

公开包的文本按UTF-8/LF传输，`input_transport.json`核对原始输入与仓库输入仅行尾表示不同、数值文本一致；本机证据保留原字节哈希。`verify_audit_delivery.py`只适用于完整本机项目，不能在简化审阅仓库中假装完成原稿保护核查。`SHA256.json`仅列本次新增包，不覆盖仓库原有证书。

“独立”表示第二套实现与对抗交叉检查，不是外部独立同行评审。原始研究记录和证书全部保持。
