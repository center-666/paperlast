# D2 成果继承审计

审阅对象为 50 节点、原始量测 common-$G$ 估计器。此次交付是代码与归档的只读审计，以及[待审数学设计](D2_MODEL_DESIGN.md)。表中“验收”指**将来接入 D2 时**应完成的回归；文中单独标出已有归档的实测结果。

## 来源与迁移边界

| 冻结来源 | 版本和用途 |
|---|---|
| main | `8953e1dd9a5e2535ead2698acb6f345a85d4f532`。二维基准、四维物理模型和 D0 均取此提交。本设计分支以它为直接基线。 |
| D1 独立分支 | `7e32a41b9802438f6645d9048ee46386caa53418`。只按文件继承已审查的 U/R 模式、价值计算、已修正的回放语义和核验件。 |
| 分支关系 | 从共同祖先计数，D1 领先 main 12 个提交、落后 1 个；三点差异为 214 个文件，两个分支端点直接比较为 215 个。main 侧多出的文件是水印报文维数适配说明。[维数说明][dim] |

共同祖先为 `5919036776ace8cf5263d4d6a2078184fd58039b`。本轮直接读取 Git 对象并按 SHA-256 复核了三份 D1 清单。最终 B2 的 33 项输入、29 项输出，七权重回放的 54 项输入、24 项输出，以及清洁报文反事实的 6 项直接输入、24 项曲线输入、24 项输出均与清单相符。这些数量是清单条目数，可能引用同一资产。下表的数值结果取自相应版本的独立核验归档，本次审计没有重新运行估计器。

## 模块、公式与验收

下表的 M1–M8 对应数学设计中的编号。文件链接固定到上述 SHA，函数名称可在对应源文件查到。

| 模块 | 原始文件与入口 | D2 复用方式 | 是否修改 | 对应数学公式 | 接入验收标准 |
|---|---|---|---|---|---|
| 二维 common-$G$ 基准 | [run_common_g_50.py][m2] 的 `load_common_g / run_common_g`；[design_common_g.py][g2] 的 `aggregate_matrices / solve_mode` | 原六边、全 50 节点、原增益表；保留内置检测器的历史基准。需要对照 H1 时用显式六边掩码的 `gate_override` 另列实验身份 | 原件直接调用，新增接口仅在 D2 | M1、M3；共享 $G_i$，本地增益 $K_i$ | 两种动力学、64 个掩码及切换序列；聚合与逐邻居更新逐拍核对，最大缩放误差 $<2\times10^{-12}$。57 行是每种动力学的节点局部模式。沿用[原验收程序][test2]。 |
| 四维状态与物理路径 | [run_partial50.py][m4] 的 `design / physical / run_filter`；[physical_configuration.json][cfg] | 继承两块二维动力学、量测与噪声；原六边模式回归。U/R 扩展走下一行入口 | 原函数保持冻结 | M1、M3 | 每个节点的时域局部秩 2；原登记保留模式中实际聚合观测秩 4；逐拍状态与原四维入口一致，单列缺失块、全状态和全网。 |
| U/R 与模式增益 | [run_critical.py][crit] 的 `critical_model`；[b1_mode_models.py][gain] 的 `build_topology / make_model / design_local / build_mode_table` | 前者只作 U 回归；按 D1 构造器与配置复用 U/R、DARE 和可见块回退。R 两条目标弧分别扫描 | 择件移植 D1 模块及其输入，D2 包装索引 | M1、M3、M7 | U 邻居 16、28、35；R 邻居 1、28、35。R 无向替边后重设计 1、16、26。独立重建 42 张表、每表 128 掩码 × 50 节点；核对 4×2 的 $K,G$ 及 4×4 的 $F,\Omega$。[模式核验][ma] |
| 乘性水印与已有 SSCD | [watson_crn.py][wm] 的 `make_common_draws / beta_from_common`；[v3 结果][v3]；[物理基准][m2]的编解码 | 两个模型的报文维数同为 2。继承有符号 Watson、既有种子及离线配对方式；SSCD 留作冻结的水印来源 | 不改分布、参数与协议 | M2；$M=I_2/2+\beta\beta^\top$ | 健康编解码还原量测，$\delta_{\rm pkt}$ 只解码一次。回归各归档自己的随机流与边/发送方索引；D0 的跨水印配对和 B2 的独立水印条件分别复现。 |
| 奇偶空间与 H1 | [源奇偶构造][par]、[ablation.py][h1] 的 `residuals / scores_all`、[D0 的 h1][fast] | 复用发送方奇偶基、参考协方差、原始未门控报文和 H1；D2 只增加因果分数归一化与报警历史 | 原核心直接调用；六边适配显式传入边索引，七边使用冻结顺序 | M2、M4；$s=z^\top\Theta^\dagger z/2$ | 原始分数对齐、首个有效分数在 $k=5$、严格大于阈值、掩码逐位相同；共享发送方的两条弧仍保留各自报文。检查截断未来输入不改变既有前缀。 |
| D0 报警软融合 | [alarm_weight_replay.py][soft] 的 `run_node`；[locked_weight_check.py][fast] 的 `run_batch`；[独立归档][d0res] | 复用报警拍 $q$、无报警拍 1 的策略 A 与两个端点；保留 D0 原始六边门控语义做历史回归 | 仅 D2 决策入口接收变动权重 | M3、M4 | D0 不稳定 U、负耦合、突发：节点 26 缺失块窗口 MSE 对齐 33.83190 / 15.81386 / 15.89873（硬门控 / 报警 0.8 / 始终接纳）；独立比对同种子逐路径数组。[D0 报告][d0] |
| 静态与动态链路价值 | [link_value_design.py][value] 的 `covariance_trace / risk / transfer_gain`；[D1-A 设计及回归][design] | 静态 $L=60$、完整接纳后验初值、缺失块投影直接继承；新增“按已执行动作推进参考协方差、此刻试探一次动作”的接口 | 原件保留；D1 固定模式函数不足以完成时变计算，需 D2 组合接口 | M5；$\kappa^{(60)},\Delta J_{H,k}$ | 独立复算静态 U 的 18431.0311、12.4661 及 R 的四项；检查 D0 的 $q=0,0.8,1$ 增益；动态接口用显式有限和独立核对，重复 $q=1$ 的配对差为零。[已有核验][d1a] |
| 检测条件下的损害与反事实 | [run_crossover.py][cross] 的 `pointwise_displacement / run_fixture`；[b2_attack.py][b2] 的 `evaluate`；[b2_shadow_decomposition.py][sh] 的 `shadow` | 继承四世界对照、最终整包加权和同掩码清洁参照；将同先验单拍损害变为开发集标签 | 反事实核验继承；条件估计/UCB 新设计 | M3、M6、M8 | 仅以最终 `b2_attack_results` 为 B2 对照，两个 `pre_*_correction` 归档仅作审计。全部锁定政策点对齐最终 B2；独立双估计器复核缺失块三项恒等式。[公式修正][fix]、[独立核验][curs2] |
| 齐次收缩与受迫稳定性 | [run_critical.py][cert] 的二元七拍穷举；[四维窗口界][wth]；[gate_stability_bound.py][bound] 的 `audit_arrays`；[模式序列递推][proc] | 复用乘法顺序、选择强迫项和扰动强迫项的证明。已有单步范数程序只适用其 $c<1$ 前提；D2 另建七权重可行序列表 | 新建窗口安全集和独立核验 | M7、M8 | 每一条准许的七拍序列通过统一收缩界；验证初始历史、无报警强制接纳和下一拍仍有合法动作；验证全网其余接收节点。不稳定 U 的七次全拒应构成反例。 |
| 健康风险与消融基线 | [B0 定义][eps]、[B1 权重锁][lock]、[七权重响应][curve] | 继承整路径配对、健康 FWER、攻击/间隙/恢复分段和策略等价检查；历史曲线仅用于实验 A | 新增闭环评估和新样本登记 | M6、M8，实验 A/B/C | 给定相同初值、H1 掩码和权重表，D1 的 B/D 逐拍相同。分开记录健康风险、H1 阈值的条件 FWER 及各时间段 MSE。[B1 证据身份][b1] |
| HVL-SF 决策 | D2 新增；最接近的历史对照是[静态价值预算表][lock] | 安全集内，用开发集条件损害和历史动作形成的健康代价选报警拍权重 | 拟新实现，待审 | M6、M7 | 在线接口只接收因果量，逐拍可回放；完整闭环比较六种策略。独立终验优先检验 HVL-SF 相对仅风险策略的配对损失。 |

## 迁移时的三个接口差异

1. **检测器入口。** 二维历史 `run_common_g` 内部调用 `replay.scores`（plugin-KL）。四维 `ablation.residuals` 默认取七条边，D0 的 `h1` 输出也固定为七列。二维若采用 H1，必须使用与六边匹配的奇偶基、输出形状和单独的健康校准。二维端点回归与这个新增检测条件分别报告。[二维入口][m2]、[原 parity 上游][src]、[H1 参考][h1]
2. **其余六条边。** D0 独立复核的“始终接纳”全网参照使用 `rest_keep`，其他报警策略可使用 `rest_gate`。D2 的主要比较只改变目标边，其他六条边统一采用同一冻结 H1 二元门控。复现 D0 数字时使用它的原口径，比较 D2 全网损失时使用 D2 的配对口径。[D0 入口][fast]、[B1 锁][lock]
3. **D1 资产与最新口头审阅。** n=4 的 U/R 增益、价值和读档函数有明确源文件。后续的“有符号创新辨别阶段”在对话中报告过，冻结 D1 提交未归档该次判别器、系数或独立结果。它可以提出实验 A 的敏感性假设；主设计继续用 $s,r$，单拍改善也不能充当 HVL-SF 的闭环实测。[七权重核验][curs]

## 交给后续实现的范围

可追溯验收件分开保存：模式表对照独立 DARE 重建，七权重更新对照逐邻居实现，反事实清洁报文对照两套独立状态递推。健康校准与策略筛选还须沿用各自的样本划分记录。[模式表核验][vb1]、[七权重核验入口][qr]、[反事实核验入口][vshadow]、[B1 运行定义][cal]

| 计划文件 | 输入与输出 |
|---|---|
| `model_adapter.py`、`detector_adapter.py` | 分开暴露二维六边和四维七边模型；输出因果 H1 与冻结模式索引。历史核心按上表调用。 |
| `link_value.py`、`safe_weight_set.py` | 分别承接 D1-A 名义风险递推与七拍安全图，保存与独立重建比对的矩阵及见证序列。 |
| `conditional_risk.py`、`hvl_soft_fusion.py` | 前者从独立开发集输出带支持路径数的风险表；后者读取冻结表、价值与安全状态，返回并记录本拍动作。 |
| `run_ablation.py`、`verify_results.py` | 同路径闭环运行六策略；核验端独立展开逐邻居更新、窗口乘积及配对损失。 |

后续迁移以主分支现存模块为依赖。D1 增量仅需筛选通过的模型构造、价值计算与配套输入；历史 NPZ 和核验件可由固定 SHA 的独立检出只读访问。相对路径、冻结配置哈希、H1 构造所用源函数和模式表要一起核对。上述文件名只描述编码分工。

[m2]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_common_g50_gain_acceptance_20261121/run_common_g_50.py#L87
[g2]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_common_g50_gain_acceptance_20261121/design_common_g.py#L21
[m4]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_partial_observability_20261123/run_partial50.py#L43
[crit]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_critical_complement_20261123/run_critical.py#L40
[h1]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_history_ablation_20261123/ablation.py#L28
[fast]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_history_ablation_20261123/locked_weight_check.py#L26
[par]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_original50_source_20261121/source/simulation_revised_detection_metrics_complete_v2.py
[wm]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_sscd_commonrandom_20261122/watson_crn.py#L81
[v3]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_sscd_commonrandom_20261122/V3_RESULTS.md
[dim]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_sscd_commonrandom_20261122/WATERMARK_DIMENSION_ADAPTATION.md
[soft]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_history_ablation_20261123/alarm_weight_replay.py#L41
[d0]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_history_ablation_20261123/D0_RESULTS.md
[value]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/link_value_design.py#L228
[gain]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b1_mode_models.py#L140
[ma]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b1_mode_results/verification.json
[fix]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/B2_EXECUTION_AMENDMENT.md
[b2]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b2_attack.py#L105
[cross]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_critical_gate_crossover_20261123/run_crossover.py#L94
[sh]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b2_shadow_decomposition.py#L29
[bound]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_gate_aware_theory_20261122/gate_stability_bound.py#L30
[wth]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_partial_observability_20261123/THEORY.md
[cert]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_critical_complement_20261123/run_critical.py#L93
[proc]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b1_mode_models.py#L255
[b1]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/B1_POSTHOC_STATUS.md
[lock]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b1_health_results/policy_lock.json
[curve]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b2_weight_response_results/B2_WEIGHT_RESPONSE_REPORT.md
[qr]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/verify_b2_weight_response.py
[vshadow]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/verify_b2_shadow_decomposition.py
[vb1]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/verify_b1_mode_models.py
[d1a]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/results/verification.json
[design]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/D1A_DESIGN.md
[d0res]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_history_ablation_20261123/results/D0_locked/analysis.json
[cal]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/B1_HEALTH_RUN.json
[src]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_50node_tv_gate_theory_20261121/source_model.py#L21
[test2]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_common_g50_gain_acceptance_20261121/verify_acceptance.py
[curs]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b2_weight_response_results/independent_verification.json
[curs2]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/b2_shadow_decomposition_results/independent_verification.json
[cfg]: https://github.com/center-666/paperlast/blob/8953e1dd9a5e2535ead2698acb6f345a85d4f532/paper1_common_g50_gain_acceptance_20261121/physical_configuration.json
[eps]: https://github.com/center-666/paperlast/blob/7e32a41b9802438f6645d9048ee46386caa53418/paper1_D1_link_value_20261123/B0_REGISTRATION.json
