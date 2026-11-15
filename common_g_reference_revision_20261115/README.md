# 原共G参考基线与九条理论修订副本

基线：`72a972ccd4c3dc5fb024ef1b178769073b10c422`。2026-11-15。本轮保留两个候选及分析稿C决定，不改原主攻击，不开发防御算法。

**结论：固定共G、补充Niu聚合MMSE、理想同步、完整报文、无门控的五节点实现基础通过；原50节点复现、完整门控实验和重投稿创新性未通过。**

入口：

- [参考实现与验收](01_原共G参考实现与验收.md)
- [公式—代码—结果矩阵](02_公式代码结果对应矩阵.md)
- [历史来源差异](historical_source_audit.md)
- [九条理论修订映射](theory_patch_draft.md)及[审阅TeX](theory_review.tex)
- [29条状态更新](05_29条意见更新状态.md)
- [是否进入完整实验](06_实现基础与完整实验决策.md)
- [第二套代码核查](code_review.md)

## 最小复算

Python3.10.18，numpy2.2.5/scipy1.15.3/sympy1.14.0；安装依赖后在本目录运行：

```text
python run_checks.py
```

依次运行`test_reference.py`（227项主断言）、`code_review_checks.py`（差分残差/展开仿射式/信息损失独立路径）、`theory_patch_checks.py`（7项精确小代数）。只生成两个32拍固定种子的物理样本，不是MC或性能实验。输入模型已随目录独立复制，原始字节哈希与canonical JSON哈希区分，Git换行转换不冒充模型数据变化。

`test_results.json/npz`保存配置、容差及短轨迹；同物理样本下共G和旧自由堆叠结果只用于接口区分，不能据单样本比较MSE。评分报警不回写估计器，不能叫完整检测—隔离实验。

`audit_historical_sources.py`、`build_theory_mapping.py`、`setup_freeze.py`需本机历史路径/原稿，公开克隆无需运行。TeX已双遍draftmode验证，不生成PDF，未声称PDF视觉核验。公开包不含第三方论文全文或原稿副本；审阅TeX仅标准数学修复，不是新稿。

原稿、11月2日阶段TeX/PDF、旧研究证书、唯一文献库和第二篇均不改动；旧C决定不变。29条仍为9项材料可用、15项待补、5项原创性无法关闭。
