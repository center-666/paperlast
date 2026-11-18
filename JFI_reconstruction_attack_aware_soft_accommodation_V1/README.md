# 第一篇共 G 结构感知在线重构与软补偿 V1

> 在 GitHub 检出中，从本目录运行实验脚本时，将 `--baseline ..` 指向仓库根目录。当前研究决策见 [V1.2](../JFI_V12_budget_and_risk_aware_compensation_20261118/README.md)。

基线 [fa77d6d](https://github.com/center-666/paperlast/commit/fa77d6d5aaa59684a78a7b7b681351fb66d71a77)。V1 使用合法可信量测构造条件协方差，在线比较有限自主模板的起点假设。接收端按后验位移与风险权重补偿，发布仍由冻结 common_step 计算。接口只需逐拍的本地、健康邻居、目标原始解码值，以及同步解码矩阵。

- [第四章技术初稿](第四章_V1辨识与软补偿.md)包含联合矩、可辨识性、在线似然、权重、真实误差和有限时域保证。
- [第五章技术初稿](第五章_配对验证与适用范围.md)含 7 种估计、2 个冻结模型、4 类已存失真及健康对照。每模型 12 条旧物理路径。
- [直接近邻](直接近邻对照.md)逐式对照重构、预测替换、信息攻击与标准统计方法。
- [修改建议](研究负责人_下一轮修改建议.md)用实际权重与失配证据确定优先修改模块。
- [逐案指标](results/comparison.json)、[配对路径](results/paired_paths.npz)及[预注册配置](registration.json)提供可复算证据。

## 运行

在本目录使用 Python 3.10 和 requirements.txt 的依赖。默认从同一工作区的交付副本读取冻结目录。独立检出基线时，给验证程序传入该检出的根目录。文本文件若因换行转换而哈希不同，使用精确的冻结复制件。运行命令中的 -B 关闭字节码缓存。

~~~powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
python -B test_accommodation.py
python -B run_validation.py
~~~

验证入口支持 --baseline 指定冻结检出，--out 指定结果位置。程序重新核对七份输入的 SHA-256。合成失真测试以已存记录为输入，重新核对编解码、时序、能量及逐路径归因。P3 行是既有代理记录。当前没有针对这些防御重新优化的 P3 值。

## 调用与指标

原型从 accommodation.py 的 PreparedModel 与 OnlineSoftCompensator 接入。准备对象可供多条独立路径复用。每条路径新建接收器，按时间调用 step，依次传入本地量测、健康邻居量测、目标解码量测和当拍解码矩阵。返回后验估计、名义和混合矩、可信参考、各起点概率、失真均值与协方差、报警证据及软权重。接口无真起点、真状态或干净目标参数。全量补偿开关用于 V1a。[核心实现](accommodation.py)

每个 NPZ 键按模型、案例、方法和字段定位。模型的 states 仅供离线评价。posterior 是真实运行后的后验，alarms 为该方法所用报警序列，estimated 为解码坐标估计位移，correction 为实际扣除量。probabilities 的首列对应无事件，后续列依次对应可能起点 \(0,\ldots,31\)。已知起点和模板投影的事后诊断单独保存。JSON 的未报警时间为 null，严格危险事件使用同方法成对后验差。

本目录只新增 V1 实现、配对证据和独立章节草稿。原稿、冻结参考、旧证书和第二篇均保持。
