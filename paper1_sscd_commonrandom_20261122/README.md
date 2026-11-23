# 第一篇论文：SSCD v3（共同随机数 + 高维 Watson + 风险可接受选参）

研究日期：2026-11-22。物理基线为 `center-666/paperlast` 的冻结 common-G/物理配置；按输入 SHA-256 验证源文件未变。本包**没有**修改历史 `iid` 锁定、旧共 G、六条 A0 攻击、奇偶检测器或先前测试结果，也不属于第二篇可信事件/量化恢复论文。

## 一、审阅已有 v2 后确立的事实

已直接读取 `V2_RESULTS.md`、`locked_selection.json`、`selector_v2.py`、`run_development.py`、`run_final_evaluation.py`。v2 的12场景×256条开发路径锁定 iid，最差负耦合单场景的同时上界为 `+0.0312009007%`；在各场景等权下，旧开发数据的 **负耦合相对 iid 点估计平均约 −0.5813235%**。此外，旧固定候选在独立的12×512测试路径上仍有改进，却**不能**用于追溯修改锁定值。上述仅为旧数据的事后解释，不是 v3 新选参成果。

重要方法学问题：原 v2 在每一个场景要求“严格改善”，因此两个本来几乎没有耦合敏感性的场景可以阻止一项在多数场景有实用收益的候选。增强版考虑把“每个场景不应有超过预登记容忍度的损失”与“总体应具有足够大收益”分开，**不是**事后把原论文的 v2 iid 锁定改成负耦合。

## 二、二维 Signed Watson 精确同分布逆变换

对于 `u_t=(cos alpha_t, sin alpha_t)` 与转移核 `p(u'|u)∝exp(theta (u^T u')²)`，令 `phi=2(alpha_{t+1}-alpha_t) mod 2π`，则

\[
 f_{\theta}(\phi)=\frac{\exp[(\theta/2)\cos\phi]}{2\pi I_0(\theta/2)}.
\]

给定同一个潜在 `U~Unif(0,1)`，用 `phi=F_theta^{-1}(U)`，并用独立公平二值 `B` 在两个合法角度分支间选取 `alpha'=alpha+phi/2+πB`，则目标方向转移核与原 `replay.attack_watermarks` **同分布**（不承诺逐样本相等）。再独立抽取 `.5*chi2_2` 的半径平方，任意单拍都有 `beta_t~N(0,.5 I_2)`，`M_t=.5 I+beta_t beta_t^T`。

简化指标：

\[
 \zeta_2(\theta)=\frac{I_1(\theta/2)}{I_0(\theta/2)},\qquad
 \mathcal I_2(\theta)=\frac{\theta}{2}\zeta_2(\theta)-\log I_0(\theta/2).
\]

**注意**：这是相邻秘密水印矩阵的统计相关信息成本，并不等于窃听者观察编码载荷之后获得的秘密条件互信息，更不构成密码学不可预测性证明。

`watson_crn.py` 使用带符号 Fourier CDF 的数值逆变换，同一场景的 iid/负/正候选共享初始方向、半径、均匀分位数和方向分支，场景之间独立。原拒绝采样器保持不变；其数学目标分布与新采样器相同，不承诺逐样本相同。二维登记候选 `|theta|≤1.31524` 的CDF残差经数值检查。CRN只配对离线比较，不改变任何在线安全协议。

### 人工门控演示模型中的方差验证（不是50节点结果）

- 8组独立重采样，每组4096条物理路径；同一二维攻击、噪声和门控规则；对照仅改变候选水印之间是否共享转移驱动随机数。
- 独立转移的配对损失平均 SE：`0.00012028275`；CRN 采样平均 SE：`0.00010749312`。
- 平均 SE 减少约 `10.633%`；以 SE 比估算，相同精度需要约 `0.7987` 倍样本，即相当于独立转移设计约 `1.252` 倍有效样本数。**不是**估计 MSE 降低10%。
- 两种采样设计的试验平均风险差接近但不完全一致；单批 Monte Carlo 采样噪声可以造成差异。该演示不能预测其他系统的效率降幅；原50节点的独立审计见下节。

### 原 50 节点上的 CRN 配对标准误审计

额外在冻结50节点 common-G、原六条攻击边和 plugin-KL 门控下，进行8个独立重复、每次每模型512条路径。物理路径、每拍径向变量和初始方向保持相同；只让“负耦合与 iid 是否共享转移分位数/分支”发生变化。

| 模型 | CRN 配对MSE差 SE | 独立转移 SE | 8重复平均SE降幅 | 等效样本比近似 |
|---|---:|---:|---:|---:|
| 不稳定 | `2.746e-5` | `3.561e-5` | `22.28%` (`[14.03%,30.54%]`) | `1.73x` |
| 稳定 | `4.067e-5` | `4.547e-5` | `10.57%` (`[8.47%,12.67%]`) | `1.25x` |

这支持 CRN 可降低**该50节点实验中配对差的估计方差**。它不改变两候选各自的目标 MSE，不是水印降低物理风险的百分比。降幅来自8个重复的配对SE统计，不保证其他系统或场景也有相同效率增益。详细重复值见 `crn_variance_50node_results.json` 和 `crn_variance_50node_paths.npz`。

## 三、高维 m>=2：倾斜 Beta 表示与信息预算不对称

均匀方向下 `Z=(u_t^T u_{t+1})²~Beta(1/2,(m-1)/2)`，Watson 转移相当于该标量的指数倾斜：

\[
 f_{m,\theta}(z)=\frac{e^{\theta z} f_{m,0}(z)}{{}_1F_1(1/2;m/2;\theta)}.
\]

抽取 `Z` 的逆 CDF、独立正负号 `S`、当前方向正交补内均匀 `v` 后，`u'=S sqrt(Z) u+sqrt(1-Z) v`。若初始 `u` 为均匀方向，未来方向的单拍边缘持续均匀，配合独立 `q chi²_m` 半径给出 `beta_t~N(0,q I_m)`。

\[
 \zeta_m=\frac{m\,E[Z]-1}{m-1}, \quad
 I_m(\theta)=\theta E[Z]-\log {}_1F_1(1/2;m/2;\theta).
\]

**高维结构性发现**：二维时 Beta(1/2,1/2) 的中心对称性使 `I_2(theta)=I_2(-theta)`；m>2 时一般不对称。对相同相邻 MI 预算 `0.1 nats`，理论根为：

| m | 负 theta | 正 theta |
|---:|---:|---:|
|2|−1.315237|+1.315237|
|3|−1.728334|+1.413913|
|4|−2.192337|+1.601938|
|6|−3.168649|+2.026114|
|10|−5.182614|+2.906290|

高维正负耦合比较必须**按相同 MI 预算分别选 θ**，不能直接比较 `±θ`。`watson_nd.py` 对整数 `m>=2` 提供通用倾斜 Beta 逆变换；独立积分对照在 `m=2,3,4,6,10`、`theta=±6` 上通过。float64 的 `m=2,theta=+6` 极端上尾（z非常接近1）CDF量化误差约 `1e-8`，其余这些点典型在 `1e-12` 量级；本轮二维实际候选远离此端点。实际多维共 G/奇偶检测性能**尚未执行**。

## 四、v3 门控后最终风险选参：风险容忍与平均收益分离

对已经预先登记的 `S` 个场景，定义相对风险 `Delta_{s,theta}=(E J_{s,theta}-E J_{s,iid})/E J_{s,iid}`，其中 `J` 始终使用**最终共 G 输出的相同 MSE**，不使用 KL 替代。

v3 预备评价规则（在**新的**开发/测试集上实施，不能 retroactively 复验 v2）：

\[
 \begin{split}
 &\text{for all }s:\quad \operatorname{UCB}_{s,\theta}(\Delta)\le\varepsilon,\\
 &\operatorname{UCB}_{\theta}\bigg(\frac1{|S|}\sum_s\Delta_{s,\theta}\bigg)<-\gamma,\\
 &I_m(\theta)\le I_{\max}.
 \end{split}
\]

建议作为新设计注册起点：`epsilon=0.001` 即每场景最多 `0.1%` 相对退化；`gamma=0.001` 即跨场景至少 `0.1%` 相对改善；`Imax=0.1 nats/step`。**这些阈值在看到旧数据后提出，不能对旧 v2 终验使用，不代表已完成前瞻性预注册。** 必须说明每场景退化容忍来自工程指标和决策成本，不能根据下一次结果再调整。若无候选同时满足，退回 iid。

`selector_v3.py` 使用配对 t 分子界和 iid 分母上下界获得逐场景一侧上界；跨场景平均风险差采用配对 delta-method 影响函数（包含分子、分母协方差），再用 Satterthwaite t 近似计算上置信界。保守分配总 α=0.05：一半用于所有候选×场景非劣界，另一半用于候选间平均收益界，避免显著性选参后混用测试数据。路径损失可能偏态、时序相关；**所得不是分布无关的有限样本证书**。

该优化规则的含义是允许对少数并不敏感的场景出现工程可接受的极小退化，但确保其相对风险有显式控制，并要求跨场景能取得有意义的总收益。与 v2 的“每场景必有统计显著正改善”相比，这是一个**新的研究选择**，不是 v2 的 Bug 修复。

## 五、工程程序及已经执行的测试

| 文件 | 作用 | 此轮执行情况 |
|---|---|---|
| `watson_crn.py` | m=2 精确目标律共同分位数采样及原水印解码攻击 | CDF/KS/协方差测试已过 |
| `watson_nd.py` | m>=2 的指数倾斜 Beta 多维理论与采样 | CDF/marginal/不同维数及独立端点积分测试已过 |
| `test_nd_endpoints.py` | theta正负6、五种维度和极端分位数的独立积分核验 | 合成数值测试已过，记录float64尾端精度界 |
| `demo_variance.py` | 独立二维门控演示模型的CRN效率验证 | 8×4096条合成轨迹完成 |
| `crn_variance_50node.py` | 原50节点中 CRN 与独立转移的 paired-SE 对照 | 8重复×512路径/模型，已运行 |
| `analyze_crn_results.py` | 50节点v3终验的分层paired汇总及哈希核验 | 已运行并读回 |
| `selector_v3.py` | 风险容忍+总体收益的开发集选参规则 | 合成正例、退回 iid 测试已过 |
| `run_crn_development.py` | 读取真实冻结50节点共G源，独立开发并锁定 | 512校准×2模型；12场景×256路径；锁定边界负耦合 |
| `run_crn_final.py` | 锁定参数只读、独立健康校准与12场景测试 | 512校准/512健康测试×2模型；12场景×512路径；无重选 |
| `test_watson.py`,`test_nd_endpoints.py`,`test_adapter.py`,`test_final_adapter.py` | 数值、端点和工程接口回归 | 全部通过；adapter仍标记为mock |

`test_nd_endpoints.py` 使用独立积分验证 `m=2,3,4,6,10` 与 `theta=±6`，并发现/修复 `m=2,theta=+6` 上尾逆CDF的有限精度问题：当 z 极接近1时，float64 间距使 CDF 绝对残差约 `1e-8`；普通分位点及登记的0.1-nat候选精度显著更高。该边界误差已明确记录，不把同一个截断CDF的自洽检查称为独立精度证据。

使用方式：将本目录置于完整 `paperlast` 根目录下，并保证旁边有三个历史目录 `paper1_common_g50_gain_acceptance_20261121`, `paper1_50node_tv_gate_theory_20261121`, `paper1_sscd_dev_protocol_20261122`。需要 `numpy scipy mpmath`；使用真实原始配置时会调用已有文件SHA源保护。

```bash
# 合成、数值与mock接口测试，不运行真实50节点物理仿真
python paper1_sscd_commonrandom_20261122/test_watson.py
python paper1_sscd_commonrandom_20261122/test_nd_endpoints.py
python paper1_sscd_commonrandom_20261122/demo_variance.py
python paper1_sscd_commonrandom_20261122/selector_v3.py
python paper1_sscd_commonrandom_20261122/test_adapter.py
python paper1_sscd_commonrandom_20261122/test_final_adapter.py

# 读回已保存的真实结果，不重新生成物理路径
python paper1_sscd_commonrandom_20261122/analyze_crn_results.py
python paper1_sscd_commonrandom_20261122/verify_v3_experiment.py

# 50节点协议 dry-run，不执行物理仿真
python paper1_sscd_commonrandom_20261122/run_crn_development.py
python paper1_sscd_commonrandom_20261122/run_crn_final.py

# v3全新50节点开发数据（不会覆盖旧v2锁定/测试归档；在干净的隔离副本中执行）
python paper1_sscd_commonrandom_20261122/run_crn_development.py --execute --calibration 512 --paths 256 --seed 2026120302

# 参数锁定之后，绝对不能根据测试候选结果再改变参数；在干净的隔离副本中执行
python paper1_sscd_commonrandom_20261122/run_crn_final.py --execute --calibration 512 --paths 512 --seed 2026120705

# 真实50节点CRN方差审计；在干净的隔离副本中执行
python paper1_sscd_commonrandom_20261122/crn_variance_50node.py
```

`run_crn_development.py` 已写入 `crn_development_losses.json`、`crn_locked_selection.json` 和manifest；在本轮探索性epsilon/gamma规则下锁定 `neg_boundary`。`run_crn_final.py` 使用新随机命名空间，锁定候选不再选择，已生成独立健康校准、测试、路径NPZ及静态六边oracle结果。旧v2的iid锁定未改变。当前目录保存了真实运行输出且默认拒绝覆盖；复跑须在干净隔离副本中进行。执行时实际final runner的原始字节版保存在下载ZIP，SHA核验通过；工作区 runner 之后只增加了路径ID/覆盖审计信息。CRN方差审计使用另一组新种子，不与开发或终验共享路径。路径中的整数种子是随机数标签，不表示未来日历日期。

## 六、真实开发集与锁定后终验结果

本轮已完成新 v3 开发和独立终验；摘要与读回复核见 [V3_RESULTS.md](V3_RESULTS.md)。

| 结果 | 不稳定 | 稳定 |
|---|---:|---:|
| CRN v3 开发锁定 | θ=−1.3152369768257723 | θ=−1.3152369768257723（统一全局候选） |
| iid final network MSE post20 | 0.0615875 | 0.0646929 |
| 负边界 CRN final network MSE post20 | 0.0614204 | 0.0641072 |
| 负边界相对 iid paired 差 | −0.0001670 | −0.0005857 |
| iid eligible link-time 报警率 | 99.0300% | 65.3146% |
| 负边界 eligible 报警率 | 99.2660% | 68.8794% |
| 独立健康整路径 FWER | 3.91% | 6.64% |

v3 的开发目标容忍每场景最多0.1%相对退化并要求平均至少0.1%改善；这些目标是在审阅旧v2后提出，是新的探索性设计注册，不是旧论文的事前注册。负边界 worst scenario UCB `+0.04537%`、平均相对差 `−0.58121%` 且平均UCB `−0.53139%`，所以通过。锁定候选仍与此前固定负边界水印相同，尚未显示自动选择器胜过固定负耦合。

**强基线仍然更好**：已知六边静态删除 post20 MSE 为0.060353（不稳定）、0.060305（稳定），低于CRN v3负边界。真实网络CRN方差审计显示8×512重复中，配对SE均值下降22.28%和10.57%；跨重复相对SE降幅的未调整t区间分别约 `[14.03%,30.54%]` 和 `[8.47%,12.67%]`。这只是减少离线比较噪声，不代表风险下降同样百分比。稳定模型健康终验FWER为34/512=6.64%（精确95% CI `[4.64%,9.16%]`），超过5%，须单列作为校准稳定性问题。

上表中的 eligible link-time rate 是 eligible 链路×时刻的平均报警比例，不等于整路径检出概率。逐场景、攻击期 MSE 与 right-censored 报警时延保存在 final path archive 与 `v3_final_analysis.json`。

## 七、本轮结果与文件

真实50节点 v3 开发锁定负边界耦合；独立终验显示相对 iid 的 network post20 配对差不稳定 `−0.0001670`、稳定 `−0.0005857`，两种模型的12场景等权平均均为负。健康整路径 FWER 不稳定3.91%、稳定6.64%。已知六边静态删除 oracle 仍优于水印。新v3候选与固定负边界水印相同，尚无“自动选参优于固定负值”的证据。

本轮文件入口在 [V3_RESULTS.md](V3_RESULTS.md)。实际源 ZIP 哈希为 `45abb096a214c9bc27706e981575ab529fdd0c4c8b1b1db76702dab7dc5364dc`。`crn_variance_50node_results.json`、`crn_development_manifest.json`、`crn_locked_selection.json`、`crn_final_results.json` 保存真实执行结果；`crn_final_path_manifest.json` 重建原NPZ按行的路径ID。原压缩包中的终验runner与报告脚本SHA一致；工作树脚本增加了审计用ID字段及双输出覆盖保护。

## 八、下一步应被拒绝的论文主张

1. 不能说**已证明**新 v3 自动选择优于固定负耦合：本候选网格最终选到的就是此前固定负边界候选。
2. 不能把 CRN 的 SE 减少说成物理状态 MSE 改善，也不能将共同随机数包装成在线安全水印机制。
3. 不得把12个旋转方向/起点场景说成未知攻击**支持集**；六条攻击链仍固定。
4. 现有真实共 G 仍为 n=m=2；高维采样测试不等于高维共 G 闭环系统已验收。
5. 互信息预算不等于窃听条件下保密性或密码学不可预测性保证。
6. 已知六条边永久删除的 oracle 仍强于耦合水印，必须报告；进一步比较需设定现实且对称的信息权限。

## 九、研究决策

CRN在该二维50节点 common-G 模型中已显示配对差标准误下降；v3开发和锁定后终验也已执行。当前候选网格选中的边界负耦合与固定负边界基线相同，因此自动选参器尚未显示优于固定负参数。下一步应在新的系统几何或攻击条件中检验选参外推，同时单独建立高维部分可观测共 G 模型和未知支持集检测架构。
