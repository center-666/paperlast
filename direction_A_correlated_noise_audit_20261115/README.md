# 第一篇方向A：相关测量噪声必要性审查（G0/G1）

2026-11-15，基线`8941dff`。**决定C：停止方向A立项，不进入G2。** 当前没有建立独立防御中心性质；这不是全领域未知相关FDI问题均已解决的声明。

建议按顺序阅读：

1. [当前决定及五个验收回答](05_当前A_B_C决策.md)
2. [研究边界与公式映射](00_研究边界与原稿公式映射.md)
3. [G0时间检测不变性证明](01_G0_检测统计不变性证明.md)
4. [G0空间/联合/聚合影响](02_G0_空间与聚合相关性分析.md)
5. [G1核心文献逐式对照](03_G1_核心文献逐式对照.md)
6. [A—H强基线覆盖](04_G1_强基线覆盖与独立性.md)
7. [证据分级表](evidence_table.md)

详细分工核读在`g0_independent_review.md`、`literature_one.md`和`literature_two.md`。三项重要纠正已合入主报告：单源边缘不变不等于跨源总体报警law不变；当前异构五节点L≥2自主模板不能用标量归因歧义替代；奇异条件空间协方差必须区分共同支持及支持分离。

## 最小复算

依赖Python3、NumPy、SciPy、SymPy。使用本机环境时Python为`D:/conda/envs/zx/python.exe -X utf8`。从本目录执行：

```text
python analytic_checks.py
python independent_g0_checks.py
python check_ma_trace_step.py
python literature_formula_checks.py --math-only
```

前两项读取相邻目录中已公开的`original_estimator_gate_20261114/five_node_aggregate_model.json`，SHA见`analysis_parameters.json`；不调用旧脚本。第一项用精确有理Lyapunov解，第二项独立使用符号公式/浮点Lyapunov。重叠的四个模型-相关性案例trace最大差约`2.78e-17`，不称第三方同行审证。其余两个是局部文献适用性反例，不否定文献所有动态定理。

`freeze_inputs.py`、`verify_delivery.py`与`append_library.py`是本机保护/归档工具，依赖完整本机工作区，非公开包的必要数学步骤。`append_library.py`已经执行且拒绝重复写入。公开数学复算不需要第三方PDF，也不需要全库副本。

## 原文件与唯一库

570个旧输入核对无未授权改变；原稿/正式源17、归档14成员、阶段7TeX+1PDF、旧50节点来源/输出20项分别通过历史hash核查。唯一库只新增CNA01—04，183旧记录逐条保持；export及完整hash check通过，现187条/128本地PDF记录/46完整独立学位论文。没有修改原稿、阶段稿或第二篇，没有新增物理样本。

本机`literature_work/`保存实际核读的PDF/提取/图片及来源查询响应；**不向公共仓库再分发第三方全文、图片或整份私人文献库**。两篇Zhang 2026全文仍缺，摘要级状态不改变。
