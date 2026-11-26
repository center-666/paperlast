# 联合平台第一阶段复现

基准D2为d6d958348ec3bbc73fe42df0a9e9cf8a337a8d78，D1为7e32a41b9802438f6645d9048ee46386caa53418。任务书原件逐字节保存在TASK_ORIGINAL.txt。新程序和预执行登记提交为e951e8a，健康校准从该版本运行。

在本目录执行，环境为Python3.11.18、NumPy2.4.6、SciPy1.17.1和原C投影编译器。D1独立checkout包含冻结b1_mode_models.py、b1_health.py及模式表。

~~~powershell
$env:D2_D1_DIR = 'C:\path\to\frozen-D1\paper1_D1_link_value_20261123'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B audit_legacy.py
python -B models_quality.py
python -B physical_regression.py
python -B audit_frontend.py
python -B calibrate_frontend.py --workers 8
python -B verify_joint.py
~~~

模型与健康归档已存在时，执行器核验身份并回读。完整独立复算可在另一checkout中将新增quality_models、frontend_health输出移出，再按上面顺序重建。保留交付副本逐数组比较，运行的执行提交和墙钟信息会随重算环境变化。

## 文件映射

legacy_audit/TASK1.json包含6912格有限掩码穷举、原223735次报警回放摘要、风险训练支持与回退、每场景16条smoke及q=0/1差值。另两份B2回放JSON由原独立审计入口输出到新增目录。

quality_models/MODELS.json索引12个物理模型。每个NPZ保存所有基础模式F/K/G、目标两报警位七权重的F/K/G/Omega/C/V、初始Pplus、60拍协方差轨迹、三策略完整合法掩码与各自可达状态。每个同名子目录保留完整M7七拍乘积范数、秩、最小奇异值及合法图。每个JSON给H/V预算裕量及最短分歧见证。

FRONTEND_AUDIT.json给14配置的逐边H维数、秩、闭合、真实输入模板差异、独立最小二乘误差与因果性。frontend.py直接构造四维奇偶协方差。projection.c逐字节来自4653505的rank-aware内核，仅用于当前二维报文、最大14维有效支撑。未知幅值的恒定及突发模板采用同一个实际列空间投影。

frontend_health/RUN.json给健康预执行登记和源哈希。28个NPZ分别为14配置×校准/留出两种用途，每个2048条，字段为路径号、物理和密钥探针、三个评分族的整路径最大分。健康模型与校准配置顺序为U不稳定三质量、R不稳定三质量、U稳定三质量、R稳定三质量，再加R不稳定target1、R稳定target1。

种子根[20261126,94017,配置号,用途,批次,组件]，用途2校准、3留出，组件0初始/过程、1量测、2密钥。正式每批32条。用于输出闭合和接口smoke的用途0、1独立于这两种健康用途。两编码共用同一物理路径，单个发送端同拍跨出边共享密钥。

## 数值与修正记录

旧H1的build_parity_space对原奇偶噪声协方差对称化并加入1e−8 I。独立核验起初直接比较未正则协方差，识别出恰为1e−8的对角差，现明确重建同一冻结正则项后比较，原模型与校准数据无需改变。

物理噪声回归使用原同一随机流在倍率1逐数组验证x与y，另以分组件共享标准正态校验质量缩放。将大幅不稳定状态信号从浮点报文中相减恢复微小噪声会产生约4.46×10^−10舍入差，因此回归直接重建signal+scaled_noise并逐数组比较，同时保存相减误差。

verify_joint.py重新设计全部不同局部门控增益，独立重算60拍健康代价和κ、逐步核对协方差轨迹，对每质量配置采样2048个七拍乘积。核验程序曾因循环内重复解压同一M7数组耗时过长，中止后改为单次读取数组再逐项检查。模型、健康归档和运行身份保留。

质量倍率1的M7图逐位回归，健康代价误差按双精度输出。K、G为4×2，L水平拼接为4×4。q=0回退限U的原缺失二维块，独立核验零空间投影。各模型的Pplus为完整接纳后验DARE解，固定q的60拍轨迹与单拍降权后59拍完整接纳代价分别保存。

旧B2“all-edge”摘要只使用masks[:,:,4:]的问题保留在原始归档中。新增前端健康校准明确使用全部七条监测边和k=5至149所有决策时刻，其精确二项区间条件于实际阈值。名义目标5%与实际留出误报共同报告。

本阶段的源码Markdown检查仅覆盖控制字符、数学定界符、局部链接和文件哈希。原D2历史文档存在显示损坏的公式，接口说明按实际代码重新写出。
