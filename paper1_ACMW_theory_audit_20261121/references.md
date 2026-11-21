# 本轮外部原始参考及读取范围

本轮核心定理主要依据用户两份附件、分支源文件和本机已审阅的原共 G/奇偶模型重新推导。下列资料仅用于核实分布命名与积分工具，不据此宣称完成全领域排重。

1. S. Sra, D. Karp, *The multivariate Watson distribution: Maximum-likelihood estimation and other aspects*，作者/机构保存的正式论文 PDF：<https://optml.mit.edu/papers/2013_sra_karp_jmva.pdf>。本轮阅读式(2.1)附近，确认密度 $\exp[\kappa(\mu^\top x)^2]/{}_1F_1(1/2;p/2;\kappa)$ 属已有 Watson 轴向分布。ACMW 负指数倾斜不能只以分布形式本身声称原创；本轮补充推导的是它与给定残差/状态矩和约束的接口。
2. SciPy 官方 `scipy.special.roots_jacobi` 文档：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_jacobi.html>。读取 Gauss-Jacobi 权重 $(1-x)^\alpha(1+x)^\beta$ 的定义，用于匹配 Beta$(1/2,(m-1)/2)$ 的端点权重。工具文档不认证本轮积分误差；本轮仅报告多阶稳定性检查。

用户附件中关于 Zhou、Ding、切换水印和其他 2025–2026 论文的叙述在原文中保留。本轮未重新取得所有正文逐条对照，不把这些叙述直接升级为已确认的新颖性结论。
