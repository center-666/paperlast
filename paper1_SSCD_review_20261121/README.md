# 第一篇论文：SSCD 建议审阅、可运行代码及独立验证

2026-11-21，Asia/Shanghai。基线 `fbc003d`；原型来源为用户建议指定分支的 `548a031874b2867c863a66771d929ed8d4c2b719`。本轮继续担任审核者和合作者，不另开第二篇论文，不重写原稿或合并来源研究分支。

**结论：采纳“有符号耦合 + 系统感知优化 + iid 退路”作为候选主线。原型的有符号互信息证明有符号错误，默认搜索与反解区间不匹配会报错；本轮复现后给出修正版。新增有效攻击维数为 2 时的连续方向目标误差括界与外层剩余 gap，不再把有限模板最小值称作全体攻击最坏值。控制例中算法选择负耦合/正耦合，并在冲突集合保留 iid；区域模型的实际稳健功效改善仍未建立。**

主要文件：

- [审阅意见和理论深化](review.md)：修正 Gibbs 符号，推导有符号局部改进条件，区分最坏风险预算与逐方向非恶化，给出二维连续方向/外层搜索界。
- [新代码 `sscd.py`](sscd.py)：紧致域内稳定统计与反解、带符号采样器、协方差、原共 G 风险、通用有限模板筛选。计算域明确为 $\theta\in[-20,20]$；不伪称实现任意端点的特殊函数渐近。
- [连续有效二维攻击求解 `continuous_design.py`](continuous_design.py)：分析能量球面上全部参数方向，报告离散与区间剩余 gap；不是任意高维攻击的全局优化器或形式化区间证书。
- [验证报告](experiments.md)、[主实验脚本](run_checks.py)、[完整结果](results.json)、[留出方向与起点敏感性](heldout_scope.py)及[结果](heldout_results.json)。
- [六项回归测试](test_sscd.py)、[依赖](requirements.txt)、[登记](registration.json)。
- [原文与来源哈希](input_manifest.json)、[用户本轮原始建议](inputs/01_SSCD_advice.txt)、[未改动的来源 README](source_snapshot/README.md)与[原代码](source_snapshot/sscd.py)、[本轮对话](dialogue.md)。来源错误保留在快照中，审阅及修正版另存，避免混淆。
- [前轮 ACMW 矩与原共 G 桥接](../paper1_ACMW_theory_audit_20261121/README.md)。

## 证据范围

有符号统计与原共 G 风险的接口适用于规定的外生 Markov 水印、预定自主攻击及固定无门控共 G。每拍 Gaussian 边缘和理想健康输出保持不变。独立校准和单窗非 Gaussian 功效没有被当作原网络整段 BF、检测延迟或实际攻击者秘密性。

在五个登记模型中，数值连续代理与状态预算检查成功，外层目标 gap 均低于 $5\times10^{-4}$。该界使用浮点范数、矩和求积，没有向外舍入；仍须按数值验证结果表述。一般测量维数的代码不等于一般攻击参数维数的连续最坏问题已经解决。原 50 节点数据和评价器尚未完成完整来源绑定。

本轮的具体研究进展是可执行的**预算约束下有符号设计原型**和明确的验收界，不是新水印已普遍优于 iid、整个学报论文已成立或原创性已清查。停止后及逐时风险的独立矩核查也已保存于 `heldout_results.json`。

`best` 和 `feasible` 描述的是已登记模型中的代理与矩约束，不代表实际检测/原网络部署验收已经通过。区域模型就是需要保留这一界限的例子。

## 接入接口

对一个来源绑定的受检链接/窗口，先提供完整攻击映射 `T_full`、检测窗内映射 `T_detect`、原共 G 输出映射 `K_full`、输出权重 `W`、奇偶矩阵 `Z` 和名义协方差 `Theta`。它们必须在同一有效攻击参数坐标中。以 $p=2$ 为例：

```python
from continuous_design import (
    parity_energy_coefficients, state_risk_polynomials, screen_continuum_2d,
)

E = sum(T.T @ T for T in T_full)  # 真正完整攻击的编码能量 Gram
detector = parity_energy_coefficients(T_detect, Z, Theta, E, 10.0, q=0.5)
state = state_risk_polynomials(T_full, K_full, W, q=0.5, lam=0.5)

result = screen_continuum_2d(
    detector, state, E, 10.0,
    dependence_budget=B_I, state_risk_budget=B_J,
    window_length=len(T_detect), measurement_dimension=m,
    directionwise_tolerance=0.0,
)
```

`B_J` 应事先定义，例如 iid 的最坏污染预算；若 iid 都不可行，程序会拒绝把它冒充安全退路。`B_I` 在该接口中是 `window_length` 的总相关预算。若相同核运行 32 拍，总相关是 $31I$，不能拿五拍的 $4I$ 预算冒充整个时域预算。本轮按每步 .1 nats 登记，五拍对应 .4 nats，32 拍对应 3.1 nats。

所有回归与实验入口见 [experiments.md](experiments.md)。任意高维攻击球、全网多链同时设计和在线历史知情攻击不由这段接口自动解决。
