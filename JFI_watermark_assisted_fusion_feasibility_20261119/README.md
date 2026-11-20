# 第一篇 JFI：水印辅助污染辨识与融合的机制筛选

本轮基线 `ea62f758c8d179f3aa198c4d706f69442fed765e`。工作目录沿用 2026-11-19 的登记名；11 月 20 日续核。原共 G、原 Gaussian、BF 和十拍 A0 不变。没有新增物理轨迹、攻击、水印或融合版本。

**决定：保留问题，当前不进候选水印设计。** 相同资源的精确小例子允许信息质量与融合 MSE 出现非平凡关系。但在冻结两模型的可重跑档案中，协方差相对普通残差与均值消融没有同时通过等健康成本、前三拍 MSE 和严格报警前风险三项筛选。每模型 670/681 条旧 A0 档案只用于核查已有 V1.4 等基线，其保存结果未用于尝试新评分。

- [定向核读及待补论文](01_文献直读与排重.md)
- [条件 Gaussian 理论、资源口径与三个精确例子](02_污染后验与融合性能.md)
- [档案对照、回放消融和研究决定](03_冻结回放与研究决定.md)
- [筛选登记](registration.json)、[回放与逐路径配对结果](replay_results.json)、[精确例子](exact_gaussian_results.json)

复算需将本目录与基线的 `JFI_defensive_offline_validation/`、`JFI_secure_estimation_accuracy_safety_frontier/`、`common_g_reference_revision_20261115/` 及清单中的其他冻结目录放在同一父目录。公开仓库在基线已有这些输入。使用 Python 3.10+、NumPy、SciPy 与 SymPy。以下命令在本目录运行。

```shell
python -B check_gaussian_identities.py
python -B offline_covariance_screen.py
```

脚本只读冻结输入、在本目录更新两份结果 JSON。第二个命令在复算前后检查 34 个依赖文件的已登记哈希，复核保存的 V0/V1a、BF 和污染估计，检查旧创新裁剪与仿射回放相符。已有 670/681 条结果只读汇总；两模型各 12 条原十拍 A0 真实重新运行防御侧，未生成任何随机种子。公开附件不含有版权限制的论文 PDF；本机阅读登记仍沿用统一参考文献库。
