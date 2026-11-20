# 仓库证据索引与研究边界

日期：2026-11-20。只读依据：`db3bbd77bc2bd34291cedce70fdaf01bb6113313`。本轮未重新运行旧实验；下面的数字与历史判断均来自已经阅读的仓库记录。

## 1. 原共 G 及编码攻击模型

[common_g_reference.py](../../common_g_reference_revision_20261115/common_g_reference.py)：

- 约 35 行：`J()` 与 `F()`；$F=(I-KH_1-G\sum_jH_j)A$。
- 约 55 行：`common_step()`，各邻居原始量测的创新先求和，再乘共同 $G$。不是每个邻居自由增益的融合器。
- 约 73 行：`finite_autonomous_attack()`，支持区间为 `[Ta, Ta+L)`，能量在编码域统计。
- 约 91 行：`original_watermarks()`，$M=0.5I+\beta\beta^\top$，$\beta\sim N(0,0.5I)$。
- 约 97 行：`transmit_decode()`，发送 $M^{-1}y$，编码域加 $a$，接收端解码为 $y+Ma$。
- 约 127 行：`bias_recursion()`，保存原共 G 的污染传播。

[五节点模型登记](../../common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json)注明该模型是与原结构相容的新增实例，补充增益设计采用聚合 MMSE。不能称为原 50 节点实验的复现。

## 2. V1a 已有能力

[accommodation.py](../../JFI_reconstruction_attack_aware_soft_accommodation_V1/accommodation.py) 的 `observe()`，约 231—290 行：

- 使用完整可信前缀条件化目标量测，并按完整协方差白化；
- 已计算名义完整信息状态均值 `nominal_state_mean`；
- 已混合未知起点，计算 `mixture_state_mean`、`state_mean_uncertainty` 与 `posterior_state_cov`；
- `state_spread` 包括组内和组间方差，不能把实际固定 A0 下的校准失配直接解释成漏算了混合项。

最终原共 G 输出在同文件约 367 行起的 `step()` 仍采用当拍污染修正再更新。**已计算全前缀状态后验参照，不等于最终输出就是全信息状态 MMSE。**

[登记](../../JFI_reconstruction_attack_aware_soft_accommodation_V1/registration.json)给出接收节点 1、目标节点 0、健康邻居 2、32 拍时域、10 拍攻击模板、编码能量参考 10 等冻结配置。它们不是本轮重新选定的参数。

## 3. 真实状态 MSE 的接口早已存在

[污染后验与融合性能](../../JFI_watermark_assisted_fusion_feasibility_20261119/02_污染后验与融合性能.md) §1、§3 已给出

$$L_k=C_kS_k^{-1},\qquad P_{F,k}=P_{T,k}-C_kS_k^{-1}C_k^\top,$$

$$\mu_k=m_{T,k}+L_k\{Z-\mathbb E[d_h\mid\mathcal F_k]\},\qquad \operatorname{Cov}(x_k\mid\mathcal F_k)=P_{F,k}+\operatorname{Cov}(L_kd_h\mid\mathcal F_k).$$

因此本轮将 $L_k(d_i-d_j)$ 接入 B1 的双假设任务损失是衔接与纠正，不是首次提出状态回归公式。它与原共 G 的 $\mathsf C_k(d_i-d_j)$ 一般不同。

## 4. 早期校准审计的证据口径

[早期污染误差与水印作用](../../JFI_A0_watermark_calibration_audit_20261120/01_早期污染误差与水印作用.md)：固定保存的水印与 A0，对健康噪声积分，首包修正均方约为工作协方差迹的 7.36 和 4.82 倍。

该量是当前 $G(\delta_k-\widehat\delta_k)$ 的平方风险，**不是状态 MSE 的倍数，也不是全体 A0 的统一结论**。主要差距来自活动质量收缩、起点混合及预测方差改变，不能靠简单放大协方差迹解决。

[结构信息价值诊断](../../JFI_A0_structure_value_diagnosis/README.md)的有限档案标签反事实表明，活动标签贡献了主要净风险改善，精确起点有较小追加收益。这说明 B1 不宜只围绕两个活动起点优化，而忽略无攻击/活动的混淆；诊断仍不能当作在线方法或样本外保证。

## 5. 方向与时序设计的既有限定

[逐拍等能量与方向四阶边界](../../JFI_A0_watermark_direction_moment_audit_20261120/逐拍等能量与水印方向四阶边界.md)：在径向与角独立、单拍二阶角矩不变、相关公式另要求各拍独立的子类中，冻结 A0 前三拍平方证据的相对变化界小于 0.73%，当拍平方污染影响变化界小于 1.97%。

这两个量不能用来约束实际补偿误差或最终状态 MSE；亦不是对径向角耦合或时间相关的一般不可能性结论。

[同边缘时间结构的最小判据](../../JFI_original_review_return_20261120/03_同边缘时间结构的最小判据.md)明确：同单拍边缘不决定联合起点试验；应比较距离矩阵的分布，不能只比较平均距离或信息矩阵的平均值。本轮没有完成新的时序政策或性能验证。

## 6. 额外可信信息与文献排重

[G0/G1 研究决定](../../active_trusted_measurement_G0_G1/07_G0G1研究决定.md)记录：既有可信本地 $H_1$ 已满列秩；该轮额外可信采样展示的收益可由标准联合 GLRT 和固定采样解释，未建立新的独立机制。不能把再次增加可信采样当成尚未研究的空白。

[既有文献笔记](../../JFI_watermark_assisted_fusion_feasibility_20261119/01_文献直读与排重.md)和[后续两篇正文核对](../../JFI_A0_watermark_calibration_audit_20261120/02_两篇直接近邻正文核对.md)已有以下竞争方向：

- 双水印与结构化攻击重构；
- 按闭环性能损害设计乘性水印；
- 目标导向实验设计与后验目标协方差；
- 保护信息、联合误差相关与安全融合。

以上是既有记录的内容索引，本轮没有重新读取这些论文的全部正文或完成外部查新。不能以更换“信息获取”“状态损害”名称认定原创，也不能从记录中的摘要范围反推原论文未覆盖什么。

## 7. 对下一步的约束

既有停止决定限定的是当时的具体候选，不阻止用户本轮授权的理论研究。但要重开新方法，应出现明确的真实风险命题：合法因果水印与同资源下的可达改善、固定 A0 或集合失配的可靠保证、以及相对成熟方法的独立机制。

当前档案中的推导不替代这些结果，也不修改旧基线或旧结论。
