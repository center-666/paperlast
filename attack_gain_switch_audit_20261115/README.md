# 原攻击与检测触发增益：第一轮审查

基线`0ea9c7db47f3e1642bbe5cba9ec6f914e4c6981a`。原参考目录不改。本轮由用户新任务明确允许检测后重配K/G，保留原共G原量测求和架构与自主编码域FDI；没有恢复旧ρ、MAC、符号码本或新检测器路线。

结论：B2在本次有限样本中改善B1的MSE，但没有改善八组首警前危险频率；B3与B1完全等价。Ding2024式38/Algorithm2、Han2025式13直接覆盖本轮核心增益动作，当前不认定新方法增量。另有合法重接纳严格反例，故不宣布安全融合成功。

- [攻击—检测—损害对应](01_原攻击检测损害对应表.md)
- [四基线与数值](02_四组基线与最小数值比较.md)
- [完整切换误差与证明](03_检测依赖切换误差递推与证明.md)
- [直接近邻公式/定理/算法](04_最近邻公式定理算法比较.md)
- [最终研究判断](05_第一轮研究判断.md)

`registration.json`在采样前冻结，128样本/模型，共256条32拍物理输入；两个编码和两个方向复用同输入。不做50节点正式实验，不解闭环P3最坏优化。整体或每检测器误报、目标/网络MSE、首警/首危险、残余、重接纳和编码成本分别报告。

注册文件内`sampling_has_started:false`记录注册瞬间；执行完成状态见`screen_results.json`。不为更新进度而修改事前配置。

复算（Python3.10，numpy2.2.5/scipy1.15.3/sympy1.14.0）：

```text
python run_screen.py
python audit_math_and_results.py
python certify_reentry_counterexample.py
```

需相邻冻结`common_g_reference_revision_20261115/`；只导入其函数，禁止写入该目录。`prepare_registration.py`与`certify_attack_inputs.py`为已执行的事前准备，带防重写保护，不应在已冻结目录重新运行。`record_literature_scope.py`仅本机唯一库维护，不属于公开复算。

`screen_results.json`/`screen_arrays.npz`保存全部小筛选数据；`analytic_preflight.json`保存模式与P1，`independent_math_audit.json`给精确模式证书和第二条仿射重算；`reentry_counterexample_exact.json`给严格有限反例。显示小数与浮点统计不冒充物理区间认证或全领域原创性验收。

所谓独立数学核查是同一工作过程另写代数路径，不是外部同行审证。重接纳附加诊断落实事前要求的瞬态/累计污染指标；反例从已冻结样本按既定情形、样本号、时刻顺序取首个，不生成新种子或调整D/E。

公开包不含第三方PDF、页面截图或全文提取。唯一库仅追加5个既有ID的本轮阅读记录，保留190条/135本地PDF/46完整独立学位论文及全部旧键/记录。原稿、阶段TeX/PDF、旧证书和第二篇保持。
