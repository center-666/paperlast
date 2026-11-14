# 第一关近邻核读分工2：Chen、Bhattacharya、Zhang/Yang、Gupta

2026-11-14。只读唯一库并审查第一关机制；没有算法、实验或原文件修改。下列“未核”是证据缺口，不是文献没有该性质。临时全文及提取件位于`literature_work`，不作为平行文献库；由主过程统一决定登记/归档。

## 1. 先查唯一库的结果

|指定文献|本轮开始时唯一库状态|本轮实际取得与阅读|
|---|---|---|
|Chen等，JAS2026，doi10.1109/JAS.2025.125900|未检出|取得官方正式13页PDF；定向文本核PDF2、5—7及12—13参考文献，视觉核PDF6—7。不是全文证明审证|
|Bhattacharya等，TECS2026，doi10.1145/3777903|未检出|官方检索可见正文片段；直接HTML/PDF返回403。**未得可连续核读全文及PDF页码**|
|Zhang/Yang，JFI2026，doi10.1016/j.jfranklin.2026.108914|既有FPN10，仅题录/公开片段，无本地PDF|仍仅出版社预览、引言和章节片段；未得核心完整正文|
|Gupta等，Signal Processing2026，doi10.1016/j.sigpro.2026.110516|既有SRA01无期刊PDF；SRA02为3作者2022 arXiv v3|期刊仍仅预览；额外只读既有SRA02 PDF3—4、6—8与参考文献，不能冒充4作者2026终版|

Chen PDF通过官网公开`downloadpdf`正常流程获取：先`article/checkArticlePdf`返回`result:true`，再导出。没有绕过登录；取回PDF后验证题名、DOI、作者及13页印刷1301—1313。抓取与哈希在`literature_2_fetch.json`、`literature_2_followup.json`、`literature_2_sources.json`。

## 2. Chen：已存在历史输出、检测迟滞补偿与状态重构，不能混成原共G确认

来源：[官方文章页](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125900)、[官方PDF导出](https://www.ieee-jas.net/article/exportPdf?id=c9a9d9b1-cd64-4aa4-9ad4-90ae38070701&language=en)。

|必核项目|原文定位与结论|
|---|---|
|问题/权限|PDF2/1302式(1)、PDF6/1306式(36)：连续时间可观非线性系统、加性输出异常；不是固定自主编码域FDI。状态/输入/扰动和噪声有界，Assumption1要求Lipschitz|
|估计/检测|PDF5—7/1305—1307：固定时间观测器(37)，瞬时/积分阈值(38)—(39)，级联预测器(40)，按检测结果切换的当前状态估计(41)|
|缓存/历史/重构|PDF6 Algorithm1保存检测起止时刻及历史输出/预测值；用已知检测迟滞界Ds/De调参考输出。PDF7 Fig1明确两处Memory|
|未知起点|检测异常起止，不预先给精确起点；但要求初始[t0,t0+μ]健康，且Ds/De已知。Assumption3约束异常最长持续与最小间隔|
|核心定理|PDF5—6 Theorem2为级联预测误差半全局一致最终有界；PDF7 Theorem3在Assumptions1—3下分别给正常分支和异常分支误差界|
|安全指标|确定性误差包络及最终有界，不是本任务有限事件总体5%误报下的危险先于确认概率|
|时延/正常性能|μ是预测器历史跨度；输出目标是当前x(t)，不是延迟d拍确认过去x(t−d)。PDF7 Remark8讨论μ、计算负担、模型不确定性的权衡；无本任务同协议η约束|

**实际差异判断：** 本文直接覆盖“历史存储＋异常检测迟滞处理＋历史信息补偿/状态重构”的研究动机；不能以这些模块本身主张原创。但它更换为非线性观测器/预测器，并不直接证明固定原共G、同标签日志重放B4与新候选的所有风险性质等价。原候选若只把这些模块接到原滤波器，仍需证明相对B4新增什么；“不是同一估计器”本身不是创新证据。

**重要适用限制：** 原文Assumption2明确限制隐蔽异常，不能作为任意Gaussian共同支撑FDI一定可检测的定理。PDF6该假设首句“cannot make…≤”与同页Remark4、PDF7 Theorem3证明采用的误差界方向存在需要澄清的语义/印式问题；已视觉核对，不静默替作者修正，不据此宣布整篇定理错误。无论该局部记号如何解释，原文保证均不能直接移植到本任务无界Gaussian、任意错误确认情形。

## 3. Bhattacharya：有限期限检测/安全动机已有，但全文证据不足

来源：[ACM原文章DOI](https://doi.org/10.1145/3777903)，TECS25(1)，Article17，17:1—17:30，2026。官方检索返回的§3.1、§4 Definition1/Claim1、§6.3—6.4及结论是**原文片段线索**；PDF/HTML直接请求403，未认定完整核读。

已可确认的线索是LTI估计/反馈、状态化CUSUM、离线最短不安全时间与在线偏置调整，以及可疑阶段改用预测数据。Claim1片段中的攻击使用当前误差/噪声与检测参数，明显比本任务预选自主ξ更强。**不得据这些片段补造核心保证的完整假设、定理页码、缓存/回退实现、未知起点保证或零误报定理。**

本轮能作的判断：不能把“在损害发生前检测、有限期限和运行开销联合考虑”当作新问题；也不能据片段断定它已实现固定d确认＋原滤波重放。完整机制覆盖和保证条件均保持未完成，不能作为第一关通过的证据。

## 4. Zhang/Yang：空间残差共享是直接近邻，核心正文缺失

来源：[出版社预览](https://www.sciencedirect.com/science/article/pii/S0016003226005107)。既有FPN10不重复登记。

预览显示其交换本地残差、用窗口χ²检查再融合，并区分攻击者有/无实时残差知识；采用异构残差映射及另行增益设计。它与原量测求和共G不是同一通信/估计协议。缓存、历史确认、重放、未知起点和时延—风险定理在完整正文未取得时一律**未核**。不能根据摘要中的mean-square bounded或本研究“空间检测”字样判断等价/原创；具体定理编号/假设不填。

本轮保留的必要约束：B1/B2空间证据基线必须合法且相同，增加交换残差属于额外通信，不能让B5免费获得、B4没有。该约束来自公平性分析，非冒称本文定理。

## 5. Gupta：严格区分2026终版与旧v3；累计未来信息已有经典来源

来源：[2026出版社预览](https://www.sciencedirect.com/science/article/pii/S0165168426000307)、[作者期刊目录](https://sites.google.com/site/arpanchattop/publications/journal-papers)、[arXiv v3版本页](https://arxiv.org/abs/2010.15785v3)。2026四作者版SRA01核心正文/补充证明未得；arXiv最新仍是2022三作者v3（SRA02），不得替代。

SRA02实际原文核对：

|项目|旧v3可核结果，不外推2026|
|---|---|
|权限/结构|PDF3—4 §II—III：远程集中KF，已知安全传感器集合；不安全创新按T线性变换加Gaussian项；§III计算已知T/起点下的恢复观测与条件分布，不是原共G自主攻击|
|核心定理|PDF6 Theorem3：Bayesian问题的后验阈值结构，阈值可依历史/时间；QUICKDET恒阈值为低复杂度受限实现，不能写成一般最优恒阈值|
|未知起点|PDF7 §V式(28)—(31)：非Bayesian起点候选的条件似然累积；PDF8式(32)—(33)为generalizedCUSUM及其引用的渐近界|
|安全/误报指标|Bayesian检测延迟与PFA，非Bayesian的FAR=1/E∞τ及WADD/CADD；不是有限13/15拍总体5%|
|缓存/重算/时延|保存似然/假设滤波信息支持检测；本次已读段未提供固定d过去状态提交、污染起点回退及原共G重放定理。不能把“恢复条件分布”改称确认协议|

对本候选的含义：等待更多数据、枚举未知起点、累计正确相关似然均已有直接先例；只有同信息、同延迟的确认/重放B4之外的新增性质才可能是本轮增量。检测延迟缩短与输出确认时延是不同指标。

## 6. 原始技术引文追溯和仍未闭合的链

这只是从已得原文参考文献追溯的**具体入口**；下列核心全文未在本分工审证，不写成已经排除重合。

- Chen参考文献[49]：Farza等，*Cascade observer design for a class of uncertain nonlinear systems with delayed outputs*，Automatica89(2018)125—134；[37] Yan/Yang，*Secure state estimation of nonlinear cyber-physical systems against DoS attacks: A multiobserver approach*，TCYB53(3)(2023)1447—1459。对应历史输出预测与安全重构。
- SRA02参考文献[40] Lai1998 *Information bounds and quick detection of parameter changes in stochastic systems*，TIT44(7)2917—2929；[41] Page1954 *Continuous inspection schemes*；[42] Lorden1971；[45] Pollak1985。对应序贯检测、未知起点、ARL与极限信息界。
- SRA02[16] Liu/Wu/Long/Wang2017 *Dynamic state recovery for cyber-physical systems under switching location attacks*，TCNS4(1)14—22。它是历史恢复/缓解方向的进一步直接入口，不因本轮不采用该攻击而忽略。

## 7. 对第一关决定的限定建议

本分工不支持A：Chen已证明历史记忆/检测迟滞/重构并非新模块；另外三篇期刊核心全文仍不足，不能称完成七篇排重。也不因全文缺失自动维持B；若主过程能证明候选在同检测信息、标签和重放规则下就是标准B4，则可依据机械等价性判C，而不必伪造这些文献的等价定理。**不能宣称所有可能的时空确认机制都被这四篇完整覆盖。**
