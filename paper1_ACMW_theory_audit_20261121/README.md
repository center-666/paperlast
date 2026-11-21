# 第一篇论文修订：ACMW 理论审阅、反例及状态风险桥接

日期：2026-11-21。定位：对原论文的实质修改与完善，不另起第二篇。用户两份原文和两条结果来源分支快照按字节保存，未改动原稿、原分支、冻结模型或既有结果。

**审稿结论：ACMW 的单拍边缘和四阶跨时协方差推导在补足初始化、外生攻击和矩条件后成立。有限窗口里角度排斥不具有普遍检测优势；对同一攻击能量，合法不稳定旋转系统给出严格负向反例。另一组受限扩张系统中，原分支的协方差型 $D_G$ 平均改善趋势得到重跑支持；此外，同一矩公式对三个冻结补充系统给出小幅原共 G 攻击 MSE 改善。两者均不证明原 50 节点闭环优越性。**

本轮的新理论建议不是发明与原稿脱节的第二个算法，而是把有限窗口统计几何、原估计器污染代价和边缘相同约束一起放入第一篇修改：在经过限定的攻击集合内做**系统依赖的耦合选择**，允许 iid 作为无耦合对照及可行退路。

阅读顺序：

- [完整理论推导及审阅结论](theory.md)：角度 Markov 核、依赖最小化、同阶记忆、维数展开、KL 能量条件、共 G 风险 Gram、BSCW 修正。
- [理论与数值试验报告](experiments.md)：源分支复现、ACMW 多维重跑、单窗正反例、无补偿共 G 风险、BSCW 噪声和有限先验诊断。
- [本轮对话](dialogue.md)、[来源哈希与代码提交](input_manifest.json)、[预登记范围和试验口径](registration.json)。
- [解析/采样核验代码](review_checks.py)、[逐对风险恒等式核验](verify_numeric.py)、[主结果](review_results.json)和[独立风险恒等式核验](verification.json)。Python 3.11、NumPy 2.4.6、SciPy 1.17.1。
- [BSCW 两拍留出诊断代码](bscw_holdout.py)及[结果](bscw_holdout_results.json)。其代码限定在报告中，不代表十拍系统复现。
- [原 ACMW 代码原样重跑输出](reproduction/angular_coupled_watermark_results_recomputed.json)。[BSCW 原版重跑](reproduction_BSCW_original/results.json)和[噪声修正敏感性](reproduction_BSCW_corrected_noise/results.json)隔离保存，未覆盖分支快照。
- [原始 ACMW 来源快照](source_snapshots/ACMW/README.md)取自 `501d78fff4c46da23a56a6311478e5f1f7c4d14c`；[BSCW 风险前沿来源快照](source_snapshots/BSCW/README.md)取自 `c9649a62b7757def403af23f782ac2770009455c`。
- [上一轮 LC-TCW 同预算理论审阅](../research_discussions/2026-11-21_LC_TCW_review/README.md)。
- [方法和数值工具的原始参考](references.md)。本轮未声称完成论文新颖性检索。

**最重要的结果：**指定均匀方向 Markov 核的四阶时序效应全部由 degree-two 特征值 $\zeta$ 控制；协方差代理和原共 G 任务 Gram 在相同 $\zeta$ 下一致，而相同 $\zeta$ 的纯指数核最省互信息。角度排斥是否有利取决于动态与奇偶投影几何：同向扩张正例、旋转扩张负例已解析并用相同名义噪声的功效试验核对。故 $(\kappa,\rho)=(2,.7)$ 不能未经整网性能分析就称为推荐或最优参数。

## 解释证据等级

- **解析推导：**单拍边缘、四阶协方差、$D_G$ 导数、维数小耦合式、同记忆 MI 最小化命题、同一共 G 的二次污染风险均由公式给出；适用假设见 `theory.md`。
- **解析数值实例：**同向/旋转两个例子的 covariance proxy 均直接代入理论式；它们是反例/正例，不是一般定理。
- **蒙特卡罗：**标记为抽样核对。源多维随机矩阵集合与本轮实际功效均不能替代原网络闭环；置信描述以分批 SE 为主，不作跨系统外推。
- **仍开放：**原 50 节点系统的完整时序、实际 BF/阈值、检测延迟、攻击者观察过滤、端到端 MSE 和原创性排重。本轮没有对原稿进行替换性改写，也没有宣称投稿就绪。
