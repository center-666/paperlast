# 四态旋转迁移验证复现

父版本为46535058416adea9150a73ba84d81af3bfc75036，执行协议与确定性审计提交为669745e，独立确认锁提交为b8c9bd9。所有命令从本目录transfer_validation运行。

需要Python 3.11、NumPy、SciPy和原匹配投影内核使用的C编译器。冻结运行版本及平台写入各RUN.json。原物理输入通过父版本的study.py从冻结8953e1dd提交加载，Git对象库需包含该提交。

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B design_audit.py
python -B study_transfer.py dev --workers 8
python -B analyze_transfer.py dev
python -B verify_transfer.py dev
python -B study_transfer.py confirm --workers 8
python -B analyze_transfer.py confirm
python -B verify_transfer.py confirm
~~~

交付目录已含完整dev、confirm和确认锁，执行器会核验运行身份及归档哈希后回读。新checkout中独立复算时，将该checkout的dev、confirm及CONFIRM_LOCK.json移出后，按上述顺序重建并保留新运行身份。已有原始归档保留供逐数组比较。

## 数组与独立单位

RUN.json的identity依次记录arms、supports、metrics。各攻击归档中的values形状为[物理路径,9种支持,8个编码—评分组,17指标]，early、full、early_max、delay形状为[物理路径,9,8,6边]。

17列依次为受扰边前三拍检出、健康边前三拍误报、健康边任一前三拍误报、前三拍精确集合、受扰边全窗检出、健康边全窗误报、全窗精确集合、受扰期目标接收节点/原五节点/全网MSE、恢复期目标接收节点/原五节点/全网MSE、相容窗路径、大幅首报后误接纳路径、登记大损失、MSE大于1。

values中的六边全部受扰支持，健康边两个误报指标写0，语义为分母为空，报告对应列以空值或文字说明。原五节点固定为{0,2,14,15,17}。目标接收节点集合按各支持实际接收端去重。

compatible_windows形状为[路径,4编码,6边]，计数仅覆盖完整八拍受扰窗。large_late_count形状为[路径,8方法,6边]。每批转移频数transitions为[批次,3旋转规则,4前态,4后态]，states为[批次,3规则,4态]。它们统计所有50个发送端的合法密钥日程，编码按发送端同拍共享。

12场景依次为不稳定六格与稳定六格。模型内角度0°、−45°、+45°，每角度起点50、65。恢复窗各自平移到110至149、125至149。开发每场景256条独立物理路径，确认每场景1024条，确认独立物理路径共12288条，九种支持共享同一物理样本，支持条件总数110592。

## 种子与阈值

根为[20261126,61783,split,model,scene,population,batch]，开发split为21，确认为22，审计为99。组件0用于物理样本，1用于全部50发送端的旋转均匀变量，2及子组件用于负耦合随机量。population0校准、1健康留出、2攻击。每批32条，分用途探针随归档保存。

每模型分别有4096条确认校准和4096条独立健康留出路径。主阈值取健康整路径最大分的第3893顺序统计量，按严格大于判断。匹配编码族各自校准，插件健康分数共享。2.5/7.5%敏感性用相同冻结校准最大分和逐边early_max离线重新判定，MSE保留5%门控。

## 统计与独立复核

analyze_transfer.py使用20000次场景内完整物理路径bootstrap，共享九支持、全部方法及指标的路径权重。paired.csv给普通95%区间和Bonferroni8、Bonferroni3区间。八支持主定位族使用Bonferroni8列，六边全受扰的三个机制事件诊断使用Bonferroni3列。其余区间按逐项解释。

scene_events.csv给每固定场景的事件计数、精确二项区间和五节点MSE分位数。ANALYSIS.json保留模型与起点分组的绝对计数，以及零事件等权场景平均概率的95%单侧上界。health_fwer.csv单列健康整路径误报，全部六条监测边和全部决策时刻均纳入。

verify_transfer.py直接回读全部逐路径数组，重新计算端点、组均值及配对点差。它另用独立代码重建首批密钥及物理样本，对单边、同节点双边、异节点双边、全六边的混合报文重新评分和重放，验证执行器按边复用的分数。dev和confirm各独立重放2048个路径—方法—支持条件。

design_audit.py给出原iid、禁重复、禁相反的转移矩阵、条件熵、互信息、允许转移秩，以及60拍中连续8个±I的有限状态概率。THEORY.md补充熵最优性和两种等预算规则的全部固定滞后互信息相等的推导。

## D2迁移审计

四维D2固定类别权重实验在独立分支d2-fixed-class-transfer-20261126，路径paper1_D2_history_value_fusion_20261124/fixed_class_transfer。它从d6d9583继续，读取冻结D1的7e32a41。相关协议、等价性枚举、新路径结果和逐邻居核验均独立归档。
