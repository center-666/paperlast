# 指定近邻1、6、7：原文核查与证据边界

日期：2026-11-14。第一关材料；不开发算法、不做网络仿真、不改唯一文献库。先按DOI、题名关键词检索`参考文献/文献库.json`，这三篇未检出。下列工作副本和来源记录待主过程统一登记，不建立平行文献库。

## 1. Li等，CDC2023

**题录**：Zishuo Li、Anh Tung Nguyen、André M. H. Teixeira、Yilin Mo、Karl H. Johansson，Secure State Estimation with Asynchronous Measurements against Malicious Measurement-Data and Time-Stamp Manipulation，CDC2023，7073—7080，DOI [10.1109/CDC49753.2023.10383571](https://doi.org/10.1109/CDC49753.2023.10383571)。

**实际原文**：[KTH作者官方PDF](https://people.kth.se/~kallej/papers/Secure_CDC2023_Li.pdf)，8页，SHA256 `c2b1132b1a89283a25ddccf5a34dcd6acabaf21b20938da7b2a88f314a9410e3`。实际定向读PDF2—6/印刷7074—7078及PDF8参考文献；PDF3、5渲染视觉复核。不是仅凭摘要判断。

|用户要求|原文证据与判断|
|---|---|
|问题与攻击权限|连续LTI、异步非周期采样；固定受攻传感器集合C，可改量测值/时间戳、阻断及伪造报文三元组；Definition1，PDF3/7075。是传感器集合稀疏威胁，不同于固定单链自主初值族|
|估计器/检测器|局部可观分解观测器式(12)，逐坐标median融合式(13)；Algorithm1，PDF4—5/7076—7077。不是原共G求和估计器，也不是时间奇偶＋空间报警器|
|缓存/确认/重算|PDF3/7075明确长度d缓冲、按时间戳排序、超d数据丢弃，再送估计器；其作用是处理乱序。Algorithm1不定义攻击证据确认、检查点或报警回滚；不能将普通排序缓存误写为完整候选覆盖|
|未知起点|固定C的攻击活动不要求向算法提供起点或C，Remark1/Algorithm1如此；没有未知变化点扫描、首报延迟或误报预算定理|
|核心定理/假设|Theorem2、3，PDF5—6/7077—7078：正常局部误差无偏且协方差有界，median融合的均值及协方差上界。Assumptions1—5含A特征值几何重数1、量测噪声方差上界、2p稀疏可观、非病态采样与间隔上界、缓冲后递增时间戳；另需其观测器增益条件(16)/(17)|
|安全指标|均值无穷范数及误差协方差谱半径上界；不是有限D危险概率、错误确认概率、链路定位率或Pd|
|延迟/正常性能|d是乱序处理的最大延迟参数，非最小安全确认延迟设计；理论依赖采样间隔上界。数值部分承认median保守性带来正常误差，未建立用户所需同输出延迟/正常MSE约束下的确认风险折中|
|真实差异|候选保留原共G并使用未来检测证据决定接受/拒绝和重算；Li更换为分解＋median且缓冲仅排序。两者不是同一估计器或同一安全定理；这个差异也不证明候选新颖|

**第一关用途**：普通固定缓存、时间戳排序与安全状态估计已组合出现，不能把“加d缓存”列为原创；但Li没有直接给出本候选的全部确认/回滚机制或其风险定理。不能以本文题名含“时空”就宣布完全等价。

## 2. Cheng等，TCYB2023：核心全文缺失

**题录**：Zhijian Cheng、Hongru Ren、Jiahu Qin、Renquan Lu，Security Analysis for Dynamic State Estimation of Power Systems With Measurement Delays，IEEE Transactions on Cybernetics 53(4)，2087—2096，2023（在线2021），DOI [10.1109/TCYB.2021.3108884](https://doi.org/10.1109/TCYB.2021.3108884)。

DOI请求到IEEE9541298返回202空体；网页工具访问为JavaScript机器人验证页，**没有取得核心PDF，不绕过验证**。[PubMed作者摘要](https://pubmed.ncbi.nlm.nih.gov/34543217/)及正式题录只作定位线索：延迟量测KF、FDI、两个残差和χ²是它必须比较的对象。本文**不据该摘要填定理号或参数条件**。

|要求字段|本轮能够核实的范围|
|---|---|
|问题/攻击权限、估计/检测结构|题名/作者摘要提示延迟PSSE及χ²；精确攻击知识、输入位置和KF结构须全文，未验收|
|缓存、确认、历史重算|未知；摘要没有提到不等于不存在|
|未知起点、安全指标、核心定理及假设|未知，不以“检测不受延迟影响”的摘要结论替代具体定理|
|输出延迟/正常性能与候选真实差异|未完成逐式审查；不能宣布已排除覆盖，也不能认定直接覆盖|

**第一关用途**：直接近邻缺全文的开放项；它不是允许开发算法的正面证据。

## 3. Xu等，Automatica2023：核心全文缺失

**题录**：Haoyuan Xu、Yake Yang、Jun Shang、Jun Fu、Yuzhe Li，Integrity attacks on remote estimation with spatial–temporal information sources，Automatica 155，111172，2023，DOI [10.1016/j.automatica.2023.111172](https://doi.org/10.1016/j.automatica.2023.111172)。

[出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S0005109823003333)检索可见作者摘要/节选；直接PDF请求403 HTML，不能当PDF。通讯作者[官方出版目录](https://yzli925.github.io/publications)确认题名与作者，但没有公开全文链接；未取得核心定理。

|要求字段|本轮能够核实的范围|
|---|---|
|问题与攻击权限|出版社作者摘要提示安全/可疑传感器新息、历史与跨源信息的完整性攻击；完整信息集/因果性/可改通道待全文|
|估计/检测、缓存/确认/重算、未知起点|没有原文，不能从“历史数据”推断存在防御缓冲确认或回滚|
|核心定理、假设、安全指标|出版社节选提到隐蔽约束与估计误差最优攻击；精确条件、命题号及与原自主族的关系未完成|
|输出延迟/正常性能、候选差异|未完成；它是时空互验不能自动保证安全的重要攻击侧近邻，但本轮不能证明等价/不等价|

**第一关用途**：要求对空间数据的可信性和攻击者历史权限保持谨慎；不能仅以空间一致性增加就声称已防住更强FDI。未获取全文不支持新颖性结论。

## 4. 从Li追溯的原始处理技术

Li原文PDF3/7075的缓冲段直接引[10]、[17]、[18]；PDF8/7080列出题录：

- [10] Uribe-Murcia、Shmaliy、Ahn、Zhao，Unbiased FIR Filtering for Time-Stamped Discretely Delayed and Missing Data，TAC65(5)2155—2162，2020。由[韩国大学机构题录](https://pure.korea.ac.kr/en/publications/unbiased-fir-filtering-for-time-stamped-discretely-delayed-and-mi/)核DOI `10.1109/TAC.2019.2937850`。**本轮未取得该TAC原文**，不把同作者WSEAS2019相近题目PDF替代为这篇全文。
- [17] Kaempchen与Dietmayer，Data synchronization strategies for multi-sensor fusion，ITS World Congress2003。仅追溯题录，未读其原文。
- [18] Westenberger等，State and existence estimation with out-of-sequence measurements for a collision avoidance system，IEEE Intelligent Vehicles Symposium2013，612—617。仅追溯题录，未读其原文。

因此原始技术追溯已找到明确入口，但三篇定理链没有冒称全审证。主过程若依据标准日志重放作自足等价证明，应明确那是对候选计算语义的数学审查，而不是借未得原文宣布三篇都已覆盖其全部安全定理。

## 5. 本分工结论

Li可严格支持“缓冲/异步处理/鲁棒安全估计不是新模块”，不能直接支持“候选已被Li整体等价覆盖”。Cheng和Xu的定理级比较保留缺口。候选是否停止，应结合主过程对**同信息、同标签、同检查点、同提交时刻下B4日志重放的逐路径等价证明**，不能以缺全文作为创新成立或持续B状态的借口。

工作副本、页图与文本保留在`literature_work/`临时材料内，不随研究报告公开上传第三方全文。机器证据见`literature_1_sources.json`；所有源库文件保持只读。
