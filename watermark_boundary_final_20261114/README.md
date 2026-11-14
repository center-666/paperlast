# 原估计器乘性水印的最终边界筛选

日期：2026-11-14。基线提交`2d4db76`。**最终决定C：未发现足够独立的新防御机制，停止矩阵/码本搜索。** 原稿和旧产物不改。

推荐阅读顺序：

1. [最终决定与统一研究报告](最终决定_C与统一研究报告.md)：固定模型、资源、真实功效、严格首包风险反例及论文定位。
2. [数学边界](数学边界_单拍可逆水印与简单强基线.md)：G/S矩算子、增益/资源约束、时间相关与缩放边界。
3. [完整P1/P3功效对照](统一功效与P1P3比较.md)：两个模型、九项已有强基线、相同/不同攻击集合的区别。
4. [有限事件风险](有限事件风险与正交编码早期边界.md)：联合滑窗协方差、独立训练/验证、正交类严格风险>.98反例。
5. [近邻定理与投稿定位](近邻定理对照与论文定位_C.md)：Miao/Zhou/Ferrari/Gallo/Huang及JFI意见边界。
6. [证据分级表](定理与证据分级表.md)：严格证明、有限枚举、数值结果和未完成条件。

## 复算

需要Python及numpy/scipy/sympy。输入仅为同级既有`original_estimator_gate_20261114/five_node_aggregate_model.json`及`aggregate_gap_certificate.json`；不得用11月3日KF的旧F/g替换。脚本以自身目录为工作位置，默认只写本目录新结果。

```text
python watermark_moment_boundary.py
python power_comparison.py
python check_power_comparison.py
python crosscheck_moment_implementations.py
python check_probability_nonimplication.py
python certify_unitary_early_boundary.py
python event_risk_screen.py
python event_risk_p1p3.py
```

power_comparison为单次真实功效的有限平均/QMC积分，不运行网络噪声轨迹；event_risk为明确披露的联合残差噪声与水印Monte Carlo最小验证，锁定选择后独立验证。后者不能替代全族最坏风险认证；严格早期反例不依赖采样。`write_power_report.py`仅生成对应报告；`theorem_novelty_evidence.py`需本机唯一文献库，公开仓库无需第三方全文即可阅读已登记证据。

`frozen_scope.json`记录本轮执行前383项旧文件及源输入哈希。保护核验脚本需完整本机项目；精简审阅仓库不包含全部原稿与文献库。公开交付只含本轮自有报告、代码与证书；不上传`literature_work`第三方PDF提取材料、临时文件或缓存。公开文本统一为UTF-8/LF；`publish_manifest.json`分别记录本机源字节与公开字节哈希。旧输入/保护证书中的哈希仍指本机原始字节，不应与经过换行归一化的审阅副本混用。

本目录不是正式论文修订稿，不提供“新水印算法”。正确结论是：证据先于危险的缺口真实存在，但现有筛选没有给出不可被简单强基线替代的新机制。分析型论文的独立投稿价值仍需另行判断，不能将C改称“等待更多实验”。
