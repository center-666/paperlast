# V1.4 双方向在线攻击补偿

基线提交 [1feac0e](https://github.com/center-666/paperlast/commit/1feac0edd8083b3ce2ffc1d38861b9b2628e6a41)。保留节点 1 的原共 $G$、结构辨识、原始报文报警及原水印时序。

可信本地和健康邻居量测产生了 V1a 结构校正以外的动作方向。旧 363 次固定历史越界中，可信线段可解除 303 次，双方向三角形可解除 305 次。预登记的旧样本筛选通过后，两个冻结模型各完成 2,048 套独立物理和水印样本的防御回放。V1.4 在 A1 回放下相对 V1a 的报警前危险降低 11.4–17.8 个百分点，健康 MSE 样本增量小于 0.1%。同预算单可信方向的风险点值八组均更小。V1.4 保留为失配扩展，A0 继续使用 V1a。

## 阅读顺序

- [模型、联合协方差与二维账本](01_可信预测与双方向账本.md)
- [几何、闭环与强基线](02_几何与配对实验.md)
- [本机论文对照与研究取舍](03_近邻与研发决定.md)
- [第四、五章技术段落](第四第五章技术修改稿.md)
- [复核记录与文件哈希](results/delivery_checks.json)

## 可复算入口

使用 Python 3.11 或更新版本，以及冻结依赖文件。下例从本目录执行，传入含原模型与归档输入的完整仓库根目录。省略该参数时，本机默认使用既有研究镜像。复算会覆盖本目录的对应输出，可在副本运行以保存当前版本。

~~~sh
python -m pip install -r requirements.txt
python -B -X utf8 geometry_and_freeze.py --baseline ..
python -B -X utf8 test_v14.py --baseline .. -v
python -B -X utf8 run_screen.py --baseline .. --phase development
python -B -X utf8 run_screen.py --baseline .. --phase holdout
python -B -X utf8 checks/report_checks.py --baseline ..
~~~

也可直接核对 GitHub 保存的独立结果。先执行 python -B -X utf8 checks/restore_holdout.py 按 SHA-256 合并二进制分卷，再运行 checks/report_checks.py。已经存在并通过 SHA-256 校验的 NPZ 会直接保留。

本次运行 OpenBLAS 单线程。浮点复算使用代码中记录的容差。开发阶段的源代码哈希与全部门槛匹配后，独立阶段入口才抽取新样本。`dual_receiver.OnlineDualReceiver.step` 只接收实际本地、健康邻居、目标报文和同步解码矩阵，输出同拍后验、权重、可信预测、条件协方差和一致性偏差。批量复算与逐拍接口已互证。

保存的 `results/geometry.json` 逐条记录固定历史投影。开发与独立数据的 NPZ 含权重、账本、真实闭环偏移平方及危险和报警时间，离线比较使用冻结输入。要点可读 `results/paired_comparisons.json`。旧前缀数据和解码同步配置从原冻结包读取。

独立验证中的 A1 是固定档案扰动回放。旧 32 条回放保留原物理配对关系。实时因果 A1 不在本轮实证范围。
