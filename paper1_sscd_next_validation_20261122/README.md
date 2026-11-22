# 第一篇论文：SSCD × 原50节点共 G 下一阶段严格验收

研究基线：GitHub `center-666/paperlast` 的提交 `99ee54f29d4d471d121ca1b80f6f083f46e7ff03`。本文件属于研究增强建议，没有改写冻结模型，也没有把新结果混入旧堆叠 KF。

## 1. 已有数据审计

研究对象是50节点**新共 G**、原六条登记攻击边、原
`plugin-KL` 奇偶检测器、512条健康校准路径与128条独立配对测试路径。

`screening_audit.py` 对既有报告的数值进行逐路径近似置信区间与 oracle gap 核对。`paper1_screening_summary_99ee.json` 是 GitHub 原 `common_g_small_results.json` 的明确标注的精简提取，不是重新采样的数据。任何新的攻击位置、网络拓扑、在线门控差异都需要新实验。

| 模型 | iid门控MSE | 负耦合门控MSE | 负耦合相比iid | 已知六边静态丢弃MSE | 负耦合仍高于oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不稳定 | 0.06211446 | 0.06182526 | -0.4656% | 0.06040319 | +2.354% |
| 稳定 | 0.06513043 | 0.06440376 | -1.1157% | 0.06047667 | +6.494% |

这里的改善是固定历史参数下的闭环筛查观察值，不能单凭现有数据归因于检测门控。非门控配对Monte Carlo误差尚不能分辨；虽然无门控解析风险证明了约0.2%改进，但**不能用两个不同实验的差值相减来声称实际门控贡献**。

## 2. 核心下一关：相同物理路径上的2×2门控交叉干预

设 `i`=iid 水印，`n`=负耦合水印；先在每种水印的原 plugin-KL 检测输出中取得门控序列 `g_i`、`g_n`。在相同物理轨迹与同一组已冻结攻击参数下，运行四个实际**最终共 G 输出**：

- `L_ii=Loss(delta_i,gate_i)`；
- `L_ni=Loss(delta_n,gate_i)`；
- `L_in=Loss(delta_i,gate_n)`；
- `L_nn=Loss(delta_n,gate_n)`。

这些反事实门控可使用已有 `run_common_g(...,gate_override=...)` 完成，不修改水印、估计器、原攻击或检测分数。每个世界都读取同一条物理数据；门控从未修剪的原始报文产生，与历史架构保持一致。

按两因素Shapley分解，逐条物理路径严格有

\[
\begin{aligned}
\Delta L_\mathrm{direct}&=\tfrac12[(L_{ni}-L_{ii})+(L_{nn}-L_{in})],\\
\Delta L_\mathrm{gate}&=\tfrac12[(L_{in}-L_{ii})+(L_{nn}-L_{ni})],\\
\Delta L_\mathrm{total}&=L_{nn}-L_{ii}=\Delta L_\mathrm{direct}+\Delta L_\mathrm{gate}.
\end{aligned}
\]

这使现有“改善来自更有效门控”的推论成为可检验的数学分解，而不是仅看TPR和MSE的相关变化。特别注意：这是对**实现的门控决策变量**进行干预的Shapley风险归因，既不是对所有攻击者的普遍因果定理，也不是对水印密码学安全的认证。Shapley项可能相互抵消；不能简单称“门控贡献比例”。

## 3. 评价注册与强基线

固定校准范围 `k=4..149`、每模型512条新健康路径（取488阶阈值），并给每个独立测试路径重新生成固定A0和水印流。结果应同时报告：全网与受攻击的5节点均方误差；攻击期间 `k=50..109`、停止后 `k=110..149`、整段 `k=20..149`；原插件分数的 eligible TPR 和首次报警删失时延；健康路径任意报警率；永久删去已知六条登记边的健康代价与攻击总MSE。不能在筛查中再调参数，不能把全网口径切换成仅五节点口径来制造正面结果。

第二阶段才安排未知攻击支持集 / 未知候选边实验：首先重建可审计的攻击边采样，不把六条已知候选边视作未知；保持与静态删边相同的可用通信预算。

## 4. 工程运行说明

本压缩包不包含用户私有/连接器代码和冻结配置。将 `causal_gate_ablation.py` 放入 `paper1_sscd_next_validation_20261122/`，与以下 GitHub 工程目录并列：

- `paper1_common_g50_gain_acceptance_20261121`；
- `paper1_50node_tv_gate_theory_20261121`；
- `paper1_SSCD_review_20261121`。

默认只做 dry-run，不会启动物理计算。启动新的小样本实验：

```bash
python paper1_sscd_next_validation_20261122/causal_gate_ablation.py --execute --calibration 512 --paths 128 --seed 2026112317
```

执行后生成 `crossover_results.json`，保留每模型配对样本标准误与近似95%置信区间。此实验应属于下一轮**独立筛查**，不计入既有128条样本或论文正式多路径功效结论。若再次挑选耦合参数，应另立开发集与完全冻结的评估集。

在本次会话中先完成了静态代码编译、Shapley恒等式单元测试和已有结果的独立算术审计，随后按登记命令实际运行了新的 50 节点共 G 反事实筛查。执行结果见 [crossover_results.json](crossover_results.json)，解释见 [CROSSOVER_RESULTS.md](CROSSOVER_RESULTS.md)。原下载压缩包按字节保存在 [source_package.zip](source_package.zip)，哈希和成员清单见 [package_manifest.json](package_manifest.json)。

本次执行使用 512 条健康校准路径、128 条独立评估路径、种子 `2026112317`。它仍属于小样本筛查，不计入前一轮 128 条样本，也不是最终论文功效实验。
