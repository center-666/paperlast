# 50 节点历史堆叠 KF：时变理论与长配对验证

这批计算响应“先继续理论推进”“做长时间的理论推进和实验验证”，以 `049bdc6` 找回的 50 节点原物理配置为输入。**它在后来的共 G 附件到达前完成，估计器仍是历史自由堆叠 KF，不能称为共 G 或共 G–SSCD 的验证。**

后续共 G 接口、固定增益与三项验收见 [独立新阶段](../paper1_common_g50_gain_acceptance_20261121/README.md)。两阶段文件和数据分开保留。

## 先读这些

- [理论推导](THEORY.md)：时变输入响应、多来源 Q、联合缓冲秩、非线性门控、均值/径向下界、首包不变和恢复定理。
- [完整结果与负结果](RESULTS.md)：每模型 512 条健康校准、1,024 条独立评估，误报、MSE、TPR、配对 SE、逐边延迟及强基线。
- [预先登记与后续分析范围](registration.json)、[可见对话续录](dialogue.md)。

## 关键判断

1. 修正了只看协方差就推断实际门控性能的逻辑。带同拍门控时，真实污染包含活动集合/增益差异对健康创新的作用，不能只留下水印攻击项。
2. 在原攻击、配对物理路径下，校准后的负耦合相对 iid 有小幅 MSE/TPR 改善；不门控的微小原始风险差没有被本批 MC 明确分辨，不能写成强效果。
3. 原 plugin-KL 阈值 42 有确定性启动误报：k=5 的两个八维残差样本正则化后，分数下界已经高于 42。统计从 k=20 起计算 FPR 会漏掉这次真实门控。
4. 全网 2% 健康成本**若被允许**，始终丢弃六条已知候选边是必须面对的强基线；其解析全网健康成本约 1.085%，但五个受影响节点的成本约 13.21%。严格零健康损失或未知支持下，适用性不同。

这些都是限定当前物理配置和攻击权限的结果，不证明普遍的负耦合优势、全网未知攻击位置功效或论文创新已经成立。

## 复现

在仓库根目录，以 Python 3.11+、NumPy、SciPy 运行。实际长批次环境记录于 [registration.json](registration.json)，不是找回的历史环境。

```powershell
python paper1_50node_tv_gate_theory_20261121/verify_archive.py
python paper1_50node_tv_gate_theory_20261121/theory_checks.py
```

以下命令会重新执行相应工作，按需手动运行；默认交付核验不重跑长 MC：

```powershell
# 四次原始函数体的逐路径兼容验证，保留旧源码不动。
python paper1_50node_tv_gate_theory_20261121/validate_source.py
# 重新生成确定性矩与设计屏幕。
python paper1_50node_tv_gate_theory_20261121/prepare_design.py
python paper1_50node_tv_gate_theory_20261121/observer_checks.py
# 原堆叠 KF 长验证；这不是共 G 实验。
python paper1_50node_tv_gate_theory_20261121/long_validation.py
```

`source_model.py` 只抽取哈希固定原脚本的白名单函数；不导入原顶层副作用，不调用原 main、旧 MC 包装器或任意攻击优化器。依赖同仓库的 `paper1_original50_source_20261121` 和 `paper1_SSCD_review_20261121`。没有重建拓扑或以补充 E=10 模型替代原攻击。

## 文件与数据口径

`*_paired_paths.npz` 保存每条独立路径的标量损失、事件和延迟；`*_validation.json` 保存均值、SE、配对差、时间曲线、校准最大分数和一致性检查。时间/链路不作为独立样本计算 SE。SSCD 在本物理配置下选择的负核与对应固定负核是同一结果，不能当成两组独立成功实验。

`design_results.json` 和 `moment_arrays.npz` 是无门控污染与有效支持上协方差代理的解析/数值屏幕，不是实际 plugin 分数分布。健康经验 95% 分位校准是本批既定方法，不是有限样本保证。新阶段另给精确秩规则，但不追溯替换本批阈值或结果。

`artifact_manifest.json` 列出本目录交付文件的字节数和 SHA-256；NPZ 的数学内容可复算，重跑压缩器不保证 ZIP 字节完全相同。
