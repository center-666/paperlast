# 转移约束旋转水印：外部对照文献审计（第一批）

日期：2026-11-28。工作分支由联合分析提交 `0e13edb039fa3ba9454b46d29e54e8d58cfafe44` 新建。本文只整理文献接口与确认前可执行性，不生成独立确认样本。可读取的出版原文、旧路径回放和新样本确认分开登记。

## 当前判定

| 文献或编码 | 接口类别 | 本基准可比较的对象 | 当前判定 |
|---|---|---|---|
| Miao 等，TCNS 2017，Algorithm 1 | 集中式原方法不可直接接入；已有本地二维 Givens 适配 | 本地静态 Givens＋共同合法匹配 GLRT；每 sender 固定角，不是每拍时变编码 | 已有 W5 已查看路径结果只用于开发参照。Algorithm 1 的可行旋转和集中式全输出实现不同，已在前一交付明确。 |
| Huang 等，arXiv:2511.00963v1，§4.2 Theorem 5 | 仅理论可比 | 通道发送节点局部估计向量，按有向通道编码；与本平台发送二维原始测量的报文语义不同 | 该定理针对攻击者使用估计编码矩阵的模型失配，不是有限阈值、特定输入族下的前三报文概率保证。没有等价接口，当前不设性能臂。 |
| Baldoni 等，IEEE Access 2021 | 可能的离散排列编码本地适配，原法直接复现尚未确定 | 需从原文还原可行矩阵集合、Fibonacci (p)-序列调度和攻击者假设后，再映射到二维报文 | 当前仅核到出版元数据和摘要，未核递推、取模、初始相位及移动起点规则。C 组暂不实现、不比较。取得原文前不能冻结 A−C 或声称覆盖了该文方法。 |
| Gallo 等，Automatica 2025，DOI 10.1016/j.automatica.2025.112301 | 闭环设计目标的理论/任务范围对照 | 其题名为 “Switching multiplicative watermark design against covert attacks”，对照水印切换优化目标与本文接收器端有限窗边检出＋估计损失 | 可帮助定位优化目标差异，不能替代同一报文接口和共同接收器的外部性能比较；本轮未声称复现其算法。 |
| no-repeat / iid / 禁相反四态 | 同平台直接运行或既有验证参照 | 同一二维报文接口、密钥匹配器、噪声与 common-G | A 与 B 必须先通过旧 W5 逐路径回归。禁相反结果引用已有独立机制确认，不因新对照重跑。 |

## Huang：§4.2–4.3 的可比边界

审读来源为 Jiahao Huang, Marios M. Polycarpou, Wen Yang, Fangfei Li, Yang Tang, “Secure Distributed Consensus Estimation under False Data Injection Attacks: A Defense Strategy Based on Partial Channel Coding,” arXiv:2511.00963v1，2025-11-02。核对文件的 PDF SHA-256 为 `EEDAB27D8DF9952C7F6FCE84C1EE959756597FB0BE834B4CDA320EA3E2CA9F01`。Section 4.2 从论文页码 9 开始，Theorem 5 位于页码 9，§4.3 位于页码 10。可复核版本为 arXiv v1。

| 项 | Huang 原文 | 本平台 |
|---|---|---|
| 通道载荷 | 节点 (j) 到节点 (i) 的局部估计向量，经 (M_{ij}(k)) 编码，合法端用逆矩阵解码，式 (16)–(17) | 每个发送节点发送本地 (y_j=H_jx+
u_j)，维数2；接收端把解码创新用于 common-G |
| 共享/同步 | 每个有向通道两端持相同 PRNG seed，使 (M_{ij}(k)) 同步；seed 是需对攻击者隐藏的密钥，可为每信道不同 | no-repeat 四态矩阵按发送端时刻生成；一个 sender 的同拍矩阵供所有出边共用。转为每边独立种子会改变本平台密钥共享范围 |
| 编码对象 | 选择被编码的有向信道。§4.3 的式 (19)–(20)按接收节点邻接测量矩阵选择需要编码的通道，最小化编码数 | 六条特定监测链路，另有一条目标融合边。已有水印仅作用于发送端二维报文 |
| 攻击者假设 | 编码方案与矩阵对攻击者保密，攻击者可监听编码通道和未保护通道。Theorem 5 称若攻击者按 (widetilde M_{ij}(k)
e M_{ij}(k)) 重设计攻击，则不能保证残差继续严格隐蔽 | 本实验的合法匹配 GLRT读取真实密钥。未建模攻击者学习密钥或重设计攻击的过程 |
| 理论结论 | “不能保证严格隐蔽”是对所定义重设计攻击与矩阵失配的断言 | 不给固定报警阈值下前三报文检出率，不给统一正态/非中心参数下界，不给全路径误报控制后的逐边功效，也不自动给出估计误差优势 |
| 论文终点 | 严格隐蔽性/估计发散条件与仿真报警率；示例检测阈值按统计残差检验设置 | 5% 左右的全路径多边 FWER 校准后，三份真实受扰报文内逐边至少报警一次，另算共识闭环 MSE |

两者在载荷维度、编码位置、通道密钥作用域、攻击者知识和检测终点上都不同。若要把 Huang 编码转为 2D 原始测量编码，必须重推其系统与 detector 对象，不能以更换 (M_{ij}) 后跑原 GLRT称作复现。本轮故不设置 Huang 数值对照。

## Baldoni：需要原文才能冻结的规则

Baldoni、Battisti、Carli、Pascucci 的出版条目已核对为 Sara Baldoni et al., “On the Use of Fibonacci Sequences for Detecting Injection Attacks in Cyber Physical Systems,” IEEE Access 9 (2021), 41787–41798, DOI 10.1109/ACCESS.2021.3065228。Crossref、OpenAlex 和 Semantic Scholar 的记录将论文列为开放获取，摘要说明其以测量系统输出编码，使用排列矩阵和秘密 Fibonacci (p)-序列产生的模式。IEEE PDF 与机构仓储的获取端点对当前运行环境返回 418/403，尚未取得 PDF，也没有引用原文公式或实验细节。

摘要与题名不足以实现其矩阵集合及调度。确认以下条目之前不构造 C 或 D：

- 可行排列矩阵的完整集合、筛选条件、排序和退化/空集情形。
- (p) 的定义、合法参数域、p-Fibonacci 递推、每项取模和初始状态。
- 原文序列索引如何从实际报文时刻初始化、是否允许偏移、编码开始时如何移动起点。
- 原法是否每条信道、发送端或接收组独立使用种子，矩阵是否周期或按密钥更新。
- 原文如何把所选排列作用到测量输出，以及逆排列如何与物理模型和检测残差对齐。
- 文中连续攻击检测指标、采样窗口、阈值/误报定义和维数，与本研究的公平比较边界。

拟比较设计是 A no-repeat 与 C 文献本地适配的唯一主比较。若原始可行集合在已登记二维 sender 上为空或退化，应先按任务书记录实际适用格，并仅在受扰结果封存前决定 A−D 是否替代。D 固定四态字母表，只采用忠实还原的文献索引调度，属于控制实验，不可称为 Baldoni 编码原方法。若无法从原文把调度适配到本地时序，C/D 均不运行，转为报告不可比原因。

## Miao 与 Gallo 的结论用语

Miao et al., “Coding Schemes for Securing Cyber-Physical Systems Against Stealthy Data Injection Attacks,” IEEE Transactions on Control of Network Systems 4(1), 106–117 (2017), DOI 10.1109/TCNS.2016.2573039。其 Algorithm 1 的集中式支撑覆盖 Givens 构造与本地二维适配的差别，及本地适配的 W5 开发结果，见本目录 `MIAO_PAPER1_INTERFACE.md`、`MIAO_RESULTS.json`。本文只能说已比较一个明确实现的本地二维适配，不能将结果概括为优于 Miao 原论文完整方法。

Gallo, Anand, Teixeira, Ferrari, “Switching multiplicative watermark design against covert attacks,” Automatica (2025), DOI 10.1016/j.automatica.2025.112301。此工作用于说明切换水印以隐蔽攻击下闭环控制增益为设计目标。当前研究的主端点是合法匹配接收器在有限窗的逐边早检及融合估计损失。需要逐项引用原文算法、约束或定理时，需独立审读该文全文，不能仅由题名推断其具体实现细节。

## 执行关卡

1. A/B/no-repeat/禁相反历史对照核验使用 W5 split16 旧路径，仅判定回归。核验早检定义、六边逐路径平均、五节点 MSE、首次报警后的大幅误接纳掩码及每一分母。
2. 对文献调度先写由原文推导出的独立短序列 oracle 与对照实现，验证短序列、编码逆变换和路径时间索引。逐 sender 报告单拍访问、相邻转移、最长重复段和独立相位数量；固定矩阵/PRNG seed 是否每路径重抽要在冻结前决定。
3. 每编码单独训练全路径 max-FWER 阈值，并用独立健康留出报告 FWER 和精确二项区间。主比较共享物理路径，但水印独立密钥批次作为分层/外层聚类单位。若阈值以有限次健康校准路径估计，主效应区间需注明条件于该阈值，或者在独立批次确认中把阈值校准不确定性纳入最外层重抽。
4. 从 W5 旧差值方差预先估算样本量，列出天花板下的配对分歧、独立密钥种子数、稀有事件率及计算预算敏感性。不得用全零非参数 bootstrap 区间证明零事件风险。
5. 文献逐项审计、C/D 适用范围、主比较、样本数、阈值协议、seed namespace 和分析脚本锁定后，才生成新的独立确认路径。

## 当前停止项

Baldoni C/D 的算法公式逐项核验尚缺出版原文 PDF。新确认样本的随机流、样本量与阈值均尚未锁定，当前不能把 A−C/A−D称为已预登记比较。A、B 在旧 W5 上的端点已由只读脚本复算核对，逐路径回归结果见 [W5 回归表](watermark_research/W5_AB_REGRESSION_RESULTS.md)、[逐场景机器读数](watermark_research/W5_AB_REGRESSION_SCENES.csv) 和 [总表](watermark_research/W5_AB_REGRESSION_OVERALL.csv)。原归档没有每拍门控时序，所以本次不能重建逐拍报警事件流。Gallo 的算法全文未在本批审读，暂只作设计目标文献。

## 执行前来源与一致性核查

- W5 A/B 只读端点脚本已按 RUN.json 哈希校验12个攻击归档，各格512条，输出主端点和首次报警后大幅报文路径事件。重算得到 no-repeat 对 iid 的前三报文差 +1.1203 pp [1.0200, 1.2261]、攻击 MSE差 −0.006589 [−0.013659, −0.001836]、恢复 MSE差 +0.0000411 [+0.0000330, +0.0000493]。误接纳事件为 A独有0、B独有11、共同0、均无6133；登记尾事件为0、8、0、6136。事件区间来自非参数 bootstrap，不能用作稀有风险上界。
- 归档没有逐拍分数/报警/门控时间序列，故任务书要求的逐拍动作掩码核对无法用当前已发布 W5 endpoint NPZ 完成。必须有原始 replay archive 或重跑旧 seed 的模拟器，后者只作回归。
- sender级精确相对裕量门槛范围经 JSON 重读为不稳定 **1.08966–1.30535**、稳定 **1.07380–1.31699**。此前四舍五入到三位小数为1.090–1.305和1.074–1.317。该界对象是完整两拍白化信号，不是固定匹配 GLRT 投影裕量。
- 若 A−C 作为主比较，确认单位至少应跨独立物理路径和独立文献密钥实现抽取。若每场景只固定一个 Miao 角向量或 Fibonacci 相位，即便大量物理路径重用这组密钥，区间也条件于该密钥实现，不能代表随机实现平均效果。
### 出版元数据来源

- Huang 等 arXiv v1，2025-11-02：[全文 PDF](https://arxiv.org/pdf/2511.00963v1)，[版本记录](https://arxiv.org/abs/2511.00963v1)。
- Baldoni 等 IEEE Access 2021, 9:41787–41798, DOI [10.1109/ACCESS.2021.3065228](https://doi.org/10.1109/ACCESS.2021.3065228)，公开元数据：[OpenAlex](https://api.openalex.org/works/W3135366127)，[Semantic Scholar](https://www.semanticscholar.org/paper/fb58ed31b4d23c5bb7a72ae85bd35a0e34f2a17f)。
- Miao 等，TCNS 2017, DOI [10.1109/TCNS.2016.2573039](https://doi.org/10.1109/TCNS.2016.2573039)。
- Gallo 等，Automatica 2025, DOI [10.1016/j.automatica.2025.112301](https://doi.org/10.1016/j.automatica.2025.112301)。
