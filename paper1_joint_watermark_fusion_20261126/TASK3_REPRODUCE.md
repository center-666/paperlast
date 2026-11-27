# 任务3开发筛选复现

正式开发登记为fcffbb0以后经更正并提交的版本，最终随机根为[20261126,94017,83]。质量模型源依赖为model_quality.py及其MODELS.json，校准、留出前端输入见frontend_health/RUN.json。TASK3_DEVELOPMENT_PROTOCOL.md记录最终H/V语义与D0/D1标签构造。

~~~powershell
$env:D2_D1_DIR = 'C:\path\to\frozen-D1\paper1_D1_link_value_20261123'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B task3.py
python -B analyze_task3.py
python -B audit_task3.py
~~~

正式完整运行在task3_dev，包含RUN.json和12个configXX.npz。路径缓存已存在时执行器以RUN身份和归档哈希校验，身份不符即停止。独立复算请在另一个checkout中将task3_dev移出后运行，保留交付路径用于逐数组比较。固定评估每输入256路径，连续、突发两个输入共512条攻击评价路径/配置，健康另取256条。风险拟合路径每输入256，用途4。随机路径按每批32条生成，输入目的用途、配置和组件各自是seed-tree维度。

配对统计对连续和突发场景等权，并在每格内按独立完整路径重采样。路径行对T、A、H/V、D0/D1共用。窗口为受扰[50,110)、突发活动拍按D2预登记开关、关闭拍为其补集，恢复[110,150)、健康[20,150)。所有比例、风险格阈值、回退支持均存RUN和ANALYSIS。

task3_dev/RUN.json记录每个输入文件SHA、提交、软件版本、随机用途及12个NPZ SHA。ANALYSIS.json另记录分析表SHA和bootstrap seed。VERIFY.json独立重建M7历史、风险查表和较大q并列规则，重放逐邻居递推并核对闭环损失。样本量为开发筛选，不能作为确认数据。

task3_superseded_*目录保存四轮无效/中断路径及原因。根79路径均已查看但其风险格按报警拍权重汇总而非逐路径单位，所有根79输出排除。根80在H/V权重策略语义上映射反转，也已查看并全量排除。根83为修正源代码及分析前的全新运行。TASK3_RESULTS_ZH.md披露源SHA精确保留与最初执行提交之间的EOF行尾差异。
