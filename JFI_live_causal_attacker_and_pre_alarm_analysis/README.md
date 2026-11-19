# 严格报警前风险的定理与冻结档案复核

基线 [bebc8ae](https://github.com/center-666/paperlast/commit/bebc8ae84d511100a61a72725c8a4b56c42194d2)。本目录提供接收端理论、精确数值证书和只读档案复算。实时 A1-H/C 篡改器、攻击权限运行验收及逐防御攻击参数优化未交付，实时基础验收未打开，未新增物理轨迹或独立配对实验。

- **窗口 A** 使用 7–16 拍的既有十拍档案，另分层报告 A0 的既有起点 3、7、11。固定起点组中，30 个模型、档案、方法组合的危险样本多数严格早于原 BF。
- **窗口 B** 允许任意一拍出现满编码预算的单包扰动。对当前目标报文仿射接收器，统一零严格报警前风险的名义健康 MSE 增量至少为 **2.23414359%**（非稳定）和 **2.25729876%**（稳定）。该精确证书使用原 BF 的 5% 名义校准。它不能代替窗口 A 十拍的结论。
- **联合条件** 固定独立波形下，对给定水印才确定危险时刻的线性估计器，有限 Gaussian 平移和有限名义误报率给出非零的条件未报警概率。只要固定波形能造成正概率全窗危险，单靠同校准统计报警不能保证零严格报警前风险。

## 文件

- [联合停时与 Gaussian 重叠界](01_报警与危险的联合条件.md)
- [窗口 B 严格报警前精度下界](02_窗口B严格报警前精度证书.md)
- [窗口 A 时序、健康预算和证据判断](03_十拍档案的报警时序与健康预算.md)
- [有理数核算](results/pre_alarm_certificate.json)、[时序数据](results/archived_timing.json) 和 [交付核验](results/delivery_checks.json)

## 复算

使用 Python 3.10 或更新。从本目录执行。`--baseline ..` 指完整 GitHub 仓库副本的根，省略时使用本机登记的镜像。复算只重写本目录结果。

~~~sh
python -m pip install -r requirements.txt
python -B -X utf8 archive_timing.py
python -B -X utf8 pre_alarm_certificate.py --baseline ..
python -B -X utf8 test_pre_alarm.py -v
python -B -X utf8 review_delivery.py
~~~

本目录不含攻击生成器、攻击策略搜索器或原始扰动记录。结果来自冻结的输出统计及独立的数学复算。
