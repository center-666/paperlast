# 复算核验

基线 `5b8364895d47c7df437d78d68cb5a979377cc503`。测试环境 Windows，Python 3.10，NumPy 2.2.5、SciPy 1.15.3；OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS 均为 1。

- 36 个已登记输入复核 SHA-256，同时与冻结提交对应 Git blob ID 一致。无原稿、原数据、冻结防御器或既有报告变更。
- 原样回放的 V1a 均值、起点后验、BF 及当拍协方差通过核对；首三拍的六组经验均方及迹与上一轮保存结果一致。计算仅复用已存的 24 条 A0 完整记录。
- 72 组条件于单条水印和时刻的已知起点闭式矩核对迹差和方向界。逐行检查三项误差及三项交叉、校准差恒等式与非活动方差分解。标量反例的两个均方用有理分数独立核对。
- 确定性积分的 \(2^{11},2^{13},2^{15}\) 三个分辨率均保存。最高两级六组均值的最大校准差变化约 \(4.89\times10^{-5}\)，单条水印条件值最大变化约 \(4.35\times10^{-4}\)。比较的是求积结果，未给严格数值误差界。

```text
unstable age=1  saved 0.027789 / 0.003891  conditional 0.028882 / 0.003924
unstable age=2  saved 0.051126 / 0.017379  conditional 0.048950 / 0.013706
unstable age=3  saved 0.046656 / 0.021069  conditional 0.054783 / 0.019022
stable   age=1  saved 0.143652 / 0.029306  conditional 0.146232 / 0.030361
stable   age=2  saved 0.069419 / 0.018914  conditional 0.067818 / 0.024147
stable   age=3  saved 0.041841 / 0.014184  conditional 0.029886 / 0.015596
```

完整计算重复运行两次，结果文件逐字节相同。`early_calibration_results.json` SHA-256：`9287d8446c637f2e1a1b3766c9616c07926b43a6cc1d6f3579fdd23d1db82ab3`。不同数值环境以文中核验容差及有效位数为准。
