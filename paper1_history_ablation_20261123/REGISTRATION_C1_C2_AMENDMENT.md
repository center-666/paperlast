# 实验 C1/C2 运行级设计锁定，2026-11-23

本文件在实验 C 的实现代码和抽样之前补足具体随机种子、样本量、EWMA 递推及可信恢复规则。它扩展主注册，不改变预登记终点。

## C1 新种子重复

C1 root=2026122301。每模型校准2048条，独立健康留出2048条，每个动力学×故障时序攻击1024条。按模型 mi∈{0,1}，校准 seed=root+100000+10000mi，留出 seed=root+200000+10000mi。攻击时序索引 fi，连续0/突发1，物理 seed=root+300000+10000mi+1000fi，水印径向/初始/转移 seed=physical+11/+12/+13。iid、negative、positive共享CRN latent variables。每个检测器用七条边、全时间路径最大分数独立校准阈值，one-based rank 1947，严格大于阈值报警。健康留出集不用于调阈值。检测器按原B定义 H4、H2、H1、EWMA-A。

## C2 先检测后更新和可信样本更新

C2 root=2026122401。每模型 EWMA gate calibration 2048条、可信参考校准2048条、独立健康留出2048条、连续和突发攻击各1024条。EWMA gate calibration seed=root+100000+10000mi，trusted-H1 reference calibration seed=root+200000+10000mi，留出 seed=root+300000+10000mi。故障 fixture 索引 fi，物理 seed=root+400000+10000mi+1000fi，水印径向/初始/转移 seed=physical+11/+12/+13。C1和C2全部 seed namespaces 两两不重叠。

- H4为原始检测器和基线。
- EWMA-A逐字复现B中的当前样本先更新统计、再计算同拍插件KL分数。
- EWMA-B当前分数用更新前的均值及协方差评价，再更新。均值和协方差初值为0及各链路Θ。prequential score 固定为 (q_B=0.5(r-mu)^	op(Sigma+10^{-6}I)^{-1}(r-mu))，其后用更新式 (mu'=\lambdamu+(1-\lambda)r)，(Sigma'=\lambdaSigma+\lambda(1-\lambda)(r-mu)(r-mu)^	op)。
- EWMA-C和B使用相同的prequential score。另以固定参考协方差定义可信分数 q_H=0.5 r' Θ^{-1}r。只有连续五个q_H均低于独立校准的可信阈值才开始参数更新。运行中任意一个q_H超阈值都会立刻停止更新、清零连续正常计数。阈值重新连续达标五拍即可恢复，避免一次误报后永久冻结。门控阈值和可信阈值来自不同健康校准批。

B/C共用B的均值与中心协方差更新式，遗忘因子λ=0.5。A保留B既有当前样本包含式插件KL统计定义。因此A/B对比包含当前样本自包含和score definition的变化，不将其结果纯解释成单一self-inclusion效应；C再检验trusted-sample更新及恢复规则。所有检测器的最终门控阈值均由各自的七链路全时域路径最大分数独立校准，one-based rank 1947，严格大于阈值报警。固定阈值的另一个独立健康留出集评估FWER。

C2只在negative watermarked attack作为主机制试验，连续/突发和稳定/不稳定均完整运行。C1三种水印全跑。C1/C2都不根据数据调λ、恢复连续拍数、残差窗口或阈值。
