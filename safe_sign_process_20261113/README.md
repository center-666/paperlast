# 2026-11-13 风险约束秘密符号时序

先读[研究决策](Codex研究决策报告.md)与[主报告](主研究报告_风险约束秘密符号过程.md)，再看[定理证据表](定理证据表.md)。决定为有条件继续，未认定投稿创新。

主要复算顺序：

1. `python recheck_safe_words.py`：Decimal→Fraction独立安全词认证。
2. `python tail_robustness.py`：无限尾部、矩阵扰动域及13项预设敏感性。
3. `python graph_process.py`：完整图/SCC/PF链、概率及预测性。
4. `python unknown_onset_checks.py`：合法确定路径上的时钟平移交叉检查。
5. `python compare_detection.py`：旧检测器、有限符号词加权，已知起点诊断。
6. `python permission_counterexamples.py`：A3预补偿精确反例。
7. `python verify_delivery.py`：仅在本机完整论文项目中执行保护范围、结构与数据一致性检查；公开仓库不含被保护原稿，不能直接跑此全项目检查。

本机依赖Python、NumPy、SciPy、SymPy。输入为同一第一篇目录的`checks/水印收束_理论与映射核验_20261105.json`；检测脚本复用`glrt_screening_20261112/minimal_screen.py`，不覆盖它。所有脚本仅在本新目录写结果。

完整图数据约0.9MB；`graph_summary.json`为较小摘要，不能替代完整P/π/状态与边。安全词完整列表见`safe_words.json`。原稿、阶段TeX/PDF、旧报告及统一文献库保持。
