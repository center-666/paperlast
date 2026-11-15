# 第一篇JFI：审稿驱动的防御定位与修订决策

2026-11-15；承接paperlast基线`a1a4bf99b2f47e403b3e6b4cb9ed33ff92e7eb4d`及此前C决定。结论：**候选I/II均C，停止本轮新防御搜索；9条已有可采用材料、15条仍需补充、5条当前无法关闭。** 核心原创性质疑尚未解决，不能宣布重投就绪。

按顺序阅读：

1. [原稿与29条意见](01_原稿与29条意见核验.md)
2. [已有理论采用清单](02_已有理论采用与待补清单.md)
3. [已有防御能力矩阵](03_原攻击下已有防御能力矩阵.md)
4. [直接近邻定理对照](04_直接近邻定理对照.md)
5. [有限时域候选](05_有限时域候选审查.md)
6. [重构补偿候选](06_攻击重构补偿候选审查.md)
7. [原50节点来源缺项](07_原实验来源缺项表.md)
8. [修订实验协议](08_修订实验协议.md)
9. [最终C决定及七个验收答案](09_研究决策_A_B_C.md)
10. [证据等级](evidence_table.md)

内部定向分工底稿：`finite_horizon_independent.md`、`reconstruction_independent.md`、`direct_literature_independent.md`。后者的下载前“缺全文”由本轮04和`fulltext_evidence.json`更新，旧过程事实保留。分工不称独立同行审证。

## 复算与保护

核心复算：在有numpy/scipy/sympy的Python环境运行`finite_horizon_checks.py`与`reconstruction_checks.py`；需仓库相邻`original_estimator_gate_20261114/five_node_aggregate_model.json`。代码独立实现定向检查，不调用旧仿真；输出区分有理证书和浮点诊断。

`audit_experiment_sources.py`、`index_original_sources.py`仅用于拥有原本机源文件的工作区，不应在缺失原档案的公开克隆上当作通用一键复现。`freeze_sources.py --check`验证冻结输入，并单独验证唯一库授权增补；不会重写基线哈希。

本机唯一库已收录用户下载七篇，6个已有ID补全文、Han新增1篇；另2条DE-FDI题录仅机构元数据/摘要，不冒充全文。库190条、135条本地PDF记录、46篇完整独立学位论文。副本重复版本不额外计数。

公开包只包含自写报告、代码和证据数据，不包含第三方PDF、全文提取/页面截图、私人邮件头、完整私有文献库。原稿、阶段稿、旧证书、第二篇及正式学位论文保持。
