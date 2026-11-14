# 乘性水印感知融合：JFI修订最小筛选

日期：2026-11-14。基线提交eebb740。结论 **C：停止当前ρ候选，创新性不通过**。

阅读入口：

1. [研究决定与停止理由](研究决定_C_当前候选停止.md)。
2. [审稿意见—防御设计矩阵](审稿意见_防御设计对应表.md)。
3. [公式、证明与反例边界](候选公式_误差递推与安全边界.md)。
4. [正常精度—安全风险—成本结果](最小对照结果与停止决定.md)。
5. [近邻定理级差异与独立性限制](近邻独立性_水印条件增益封顶.md)。

`frozen_scope.json`先冻结E=10、D=.1、η=1/5/10/20%、原Gaussian分布和五节点聚合模型。没有50节点实验或原稿修改。该模型由上轮显式补入Niu聚合MMSE，未冒充原50节点图/最终增益复现。

复算依赖Python/NumPy/SciPy/SymPy，按顺序：

- `rho_theory_check.py`：有理恒等式、端点收缩和γ=.09安全界。
- `rho_numeric.py`：正常协方差求解及同预算参数，生成rho_normal_screen.json。
- `rho_first_packet.py`、`rho_integral_crosscheck.py`：有限首包风险搜索与固定方向积分复核，非全局证书。
- `rho_expected_damage.py`、`rho_numeric_checks.py`：P1/P3及二阶矩关联复核。
- `raw_watermark_resources.py`：m=2原/归一化Gaussian与符号成本及期望损害。
- `clip_baseline.py`：普通裁剪基线的训练/独立验证；有实际Gaussian噪声抽样，不能称无MC。
- `check_clip_risk_net.py`、`check_candidate_first_mc.py`：只在冻结η20首包子问题上，使用与初筛不同的新鲜种子、Lipschitz覆盖/浓缩界核对；正常MSE仍是估计，非完整变持续风险定理。
- `assemble_screen_tables.py`：只汇总已保存结果；不重新选参数。
- `verify_delivery.py`：本机保护文件/交付一致性核查；简化公开仓库缺原稿和全部来源，不可假称完成本机来源审计。

瞬时界、全时域偏移界、平均损害、单次检出率、首包风险、完整事件风险及正常MSE的证据等级均分开。`evidence_table.md`列出严格证明、浮点、有限样本与未完成项。

原稿与阶段TeX/PDF、旧研究记录、唯一文献库不改。本轮没有MAC/调度/恢复/RL新机制，也没有将普通裁剪写成新贡献。第三方全文和提取文本不上传审阅仓库。

公开文本按UTF-8/LF传输；来源/保护哈希保留本机原字节表示，行尾不同可能导致这些来源哈希与公开文本哈希不同。包内SHA256清单核对Git实际字节。数学参数不因行尾改变；本机保护核查仍须在完整工作区运行。
