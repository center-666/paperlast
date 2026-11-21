# 找回的历史 50 节点仿真源码与配置包

2026-11-21，Asia/Shanghai。用户明确要求提供、上传 50 节点源代码/工程，最好包含系统矩阵、节点量测矩阵、通信拓扑、共 G 增益、攻击生成器和检测器配置。

**已经找回三个历史原文件，SHA-256 与此前本地审计记录完全一致。** 它们实际保存在 `center-666/arcus-shixi-project`，不是当前 `paperlast` 或 `paper` 的工作树。来源固定提交为 `52e31e26302bdf24cce884c0f4f7e75b7c6aa82f`，目录为 `other/mypaper/privacyandsaety/code/`。本次只取具名三文件，没有下载/上传该仓库的其他项目或另一篇论文。

**重要边界：找回的是完整的“历史 50 节点仿真引擎源码”，不能据此认证它就是原稿最终 1000 次运行或原共 G 实现。** 它使用自由堆叠量测 KF，邻居增益块不强制相等；符合原稿共 G 接口的 K/G 数值或设计规则仍未恢复。本包不会补造这组缺失增益。

## 文件入口

- [历史主版本 complete_v2](source/simulation_revised_detection_metrics_complete_v2.py)：50 节点、150 拍，主函数受 `if __name__ == '__main__'` 保护。推荐作为历史阅读入口。
- [更早的方向修订版](source/simulation_revised_detection_metrics_direction_fixed.py)与[更早的任意攻击求解版](source/simulation_arbitrary_attack_solver.py)：均原样保留，仅作为历史版本；有未保护的顶层仿真/绘图调用，**不要 import 来读取配置**。
- [来源与原字节哈希](source_manifest.json)、[固定提交的目录元数据](remote_source_index.json)。三文件分别与历史审计的 `2f4694e6...`、`6cc7c06d...`、`1a6250a0...` SHA-256 相符。
- [可读配置快照](configuration_snapshot.json)：A、Q、50 个 H/R、50×50 邻接矩阵、坐标、邻居列表、参数及物理攻击方向。
- [原奇偶矩阵与名义协方差](parity_matrices_regenerated.npz)及[核查](parity_verification.json)：两个动力学、每个发送节点的 Z/Theta，保留旧函数的 `1e-8 I` 正则项。
- [初次自由堆叠 KF 增益](initial_stacked_gains_regenerated.npz)及[接口审计](gain_interface_audit.json)：提供第一更新时刻的实际旧增益分块与发送者次序，**不是共 G 增益**。
- [受限 AST 配置导出器](extract_config.py)：只执行白名单初始化和纯奇偶函数，不导入或运行完整仿真。
- [显式执行启动器](run_legacy.py)：默认只打印计划；必须 `--execute` 才执行原主函数。默认 MC=1，且额外任意攻击优化关闭，避免误触发原默认的长时间任务。
- [验证与打包脚本](verify_package.py)、[ZIP 构建器](build_bundle.py)、[对话及恢复过程](dialogue.md)、[状态与缺项](completeness.md)。

`historical50_source_bundle.zip` 是本次将上述材料打成的便携包，**不是声称找到了旧机器上原有的最终工程 ZIP**。根目录的 `bundle_manifest.json` 提供 ZIP 大小、哈希和成员校验，避免把新打包与历史归档混同。

## 已核实的历史配置

| 项目 | complete_v2 原代码 |
|---|---|
| 节点/状态/测量/时域 | 50 / 2 / 2 / 150 |
| 不稳定 A | `[[1.08,.05],[.01,1.05]]` |
| 稳定 A | `[[.95,.02],[.02,.97]]` |
| Q、每节点 R | `I2`、`.5 I2` |
| H 生成 | NumPy seed42，各节点对角元 `1+.5*rand()` |
| 拓扑 | 随机几何图，初始 seed42、radius .31，连通后补六条关联边 |
| 奇偶/统计窗口 | 奇偶阶数 4（5 拍量测），KL 缓冲 4 个残差 |
| 阈值 | 正则化经验 Gaussian plugin-KL 的 42，不是 39，也不是真实分布 KL |
| 编码 | `M=.5I+beta beta^T`，源/时刻字典同步 |
| 攻击支持 | 起点 50；统计/恢复图持续 60；impact 情景持续至末，共 100 拍 |
| 幅值 | 不稳定主特征向量幅值6、稳定幅值10；**没有编码能量 E=10 归一化** |
| 元组方向 | `(receiver,sender)`；第一项 `(0,1)` 是 **1→0** |
| 估计器 | 活动邻居逐包堆叠，自由块 KF；告警后剔除邻居再重算增益 |
| `use_defense` | 同时控制水印编解码与基于告警的邻居剔除，二者不是独立开关 |
| MC 默认 | 环境变量 `MC_RUNS`，默认 1000；默认值不能证明最终图确实跑过 1000 次 |

## 再生成的配置不等于原运行快照

本包用哈希固定的 `complete_v2` 初始化规则重新生成了配置，而没有执行 Monte Carlo。记录的检查环境为 Python3.11.18、NumPy2.4.6、SciPy1.17.1、NetworkX3.7。得到 seed42、50 节点、298 条无向边；所有节点 H 以及相应邻接和位置已导出。它们可作为来源绑定的新接入输入，**尚未与原稿最终图/表执行记录绑定**。

两组初次自由堆叠增益的 50 个节点都存在不相等的邻居块，最大 Frobenius 差约 0.1113。这从数值上再次确认了共 G 接口缺项；不能把它们简单平均后标成“原共同 G”。

## 使用方式

源程序所需外部包只有 NumPy、SciPy、NetworkX、Matplotlib，未发现其他本地输入模块或矩阵数据依赖。原环境版本没有完整记录，[requirements.txt](requirements.txt)只列导入依赖，不冒称原作者锁定环境。

```text
python -m pip install -r requirements.txt
python -B extract_config.py
python -B verify_package.py
python -B run_legacy.py
```

最后一条默认不运行仿真。确实需要执行历史版本时，再显式运行：

```text
python -B run_legacy.py --execute --mc-runs 1 --output-dir legacy_outputs
```

这仍会执行原主函数的全部多情景流程，并不是轻量单元测试。本次恢复请求**没有运行**该命令或原主函数。任意攻击优化还需另加 `--include-arbitrary`；本次也未执行。

若仅希望原样复跑历史入口，直接执行 `source/simulation_revised_detection_metrics_complete_v2.py` 会采用原默认设置，包括昂贵的额外攻击优化，可能耗时很长；请先读参数。

## 接入 SSCD 前必须解决的一项核心选择

现已具备真实历史 50 节点的物理网络、H、攻击与旧检测代码。但保持原稿共 G 理论时，仍需明确该网络的 K/G 设计与最终运行身份；若采用旧堆叠 KF，则必须按其时变和隔离依赖接口重新分析。不得把它默认为之前固定 F/G 的原估计器。

对原论文矩阵、阈值和图表身份的历史差异，本包 [completeness.md](completeness.md)逐项列出。恢复源码与决定最终论文采用哪个版本是两件事；当前上传不表示整网 SSCD 性能已验收。
