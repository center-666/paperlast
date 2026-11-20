# 审稿记录：时间耦合水印、必要模态恢复及 T1—T3

2026-11-20，Asia/Shanghai。基线提交：`6a0e4985a4640181fea69511bd2d37ef81735fe8`。

用户要求记录四份新材料、上传、仔细审阅，并结合此前讨论给出建议；随后明确要求采用**审查人视角**。本目录据此区分附件原始主张、审稿判断和必要的证明核对，不把审核中的补充推导当成作者已经证明的论文贡献。

**审稿结论：指定二维区域模型下，T1 的同边缘耦合构造与统一左逆界成立；T2 的动力学一致块恢复和有限边界项可在明确假设下成立。T3 的未投影有界性与健康投影保证分别有根据，但还没有共同证明最终防御输出在受攻击时有实际优势。当前不足以认定完整算法、独立原创性或投稿就绪，建议按重大修改要求继续理论验证。**

阅读入口：

- [审稿意见与优先修改项](reviewer_report.md)：主张审定、问题分级、研究定位与下一步验收条件。
- [数学证明复核](mathematical_review.md)：T1 常数、T2 边界、T3 输出与风险量词的详细核查。
- [原始附件清单](input_manifest.json)：字节数、SHA-256 与来源 ID。
- [确定性核验代码](verify_theory.py)及[结果](checks/theory_checks.json)：分位数、左逆、块边界、区域噪声与增益。
- [交付核验](checks/delivery_checks.json)。

四份附件逐字节保留，公式转义、旧建议和引用不改：

1. [对上一轮实验的审阅及模态恢复](inputs/01_review_6a0e498_and_modal_recovery.txt)
2. [标量同边缘时间耦合](inputs/02_scalar_temporal_coupling.txt)
3. [T1：完整二维统一左逆](inputs/03_T1_two_dimensional_uniform_inverse.txt)
4. [T2 未知起点与 T3 共 G 补偿、健康预算](inputs/04_T2_unknown_onset_and_T3_compensation.txt)

本轮只做数学审查与确定性代数/数值核对，没有新增物理轨迹、性能实验或参数搜索。1,440 个水印参数网格点不能代替全域证明；全域依据在数学复核文档中单列。先前冻结代码、模型、仿真数据与原稿保持原样。

复核需要 Python 3.10+、NumPy、SciPy，沿用仓库既有环境：

```text
python -B verify_theory.py
python -B verify_delivery.py
```

本轮外部检索没有取得可核对的新正文，未声称完成查新。附件中的文献叙述按来源保留，不作为本轮已独立验证的新颖性证据。
