# 危险隐蔽FDI：第一轮理论筛选

日期：2026-11-14。基线`fc4f60f9904a83a67b3df0986caa64398b471585`。**本轮最终决定C：没有通过进入新水印/检测协同机制设计的门槛。**

1. [报告1：危险FDI几何](报告1_危险FDI方向几何与能量约束.md)：两模型、原五节点10条有向边；主链全角危险认证，峰值与终端分开。
2. [报告2：危险与弱检测方向](报告2_危险方向与检测弱方向的错配.md)：共同危险集真实功效、二维图、条件卡方与径向边界。
3. [报告3：统一强基线](报告3_水印与空间检测的统一强基线.md)：可信本地量测空间检测、普通匹配、损害、资源及有限首次报警。
4. [报告4：有界设计空间](报告4_有界水印的可行性与停止边界.md)：资源面、正交限制、严格首包oracle检测界及范围。
5. [报告5：审稿与决定](报告5_审稿对应与本轮决定_C.md)：三个问题的答案、阶段门控和C的准确范围。

补充完整推导：[危险集数据报告](danger_detection_共同危险集合功效与损害.md)、[资源/空间/算子边界](bounds_review.md)。图为本轮代码生成的科学静态图，不是新的网络实验。

## 复算顺序

Python依赖numpy/scipy/sympy/matplotlib。代码只写本目录，不执行旧入口。原五节点源输入在同级`original_estimator_gate_20261114`；功效脚本只读上一轮`watermark_boundary_final_20261114/power_comparison.py`的矩阵接口。请勿用旧四维堆叠KF的F/g替换。

```text
python danger_geometry.py
python danger_detection_compare.py
python danger_detection_check.py
python spatial_baseline.py
python bounds_review.py
python certify_first_packet_detection_bound.py
python first_alarm_bounds.py
python assemble_comparison.py
python plot_detection_damage.py
```

`danger_geometry.py`独立重建全部节点DARE；主链用既有严格误差包络认证有理PSD。其他链角变化是浮点诊断。`danger_detection_compare.py`使用真实非中心卡方混合，符号有限枚举、Gaussian/Haar固定Sobol，方向最小值无全局区间认证。

`first_alarm_bounds.py`不抽物理噪声：普通奇偶仅用正确边缘条件功效给相关事件界；当前空间噪声独立，先条件连乘再对完整水印路径积分。CDF/积分仍有浮点/QMC误差未认证。标准匹配只作固定起点闭式参考，不代表未知起点在线扫描已解决。

`frozen_inputs.json`记录425项已有文件哈希；原稿、阶段TeX/PDF、学位论文和唯一库保持。公开材料不包含原稿/第三方全文；精简仓库不能替代完整本机保护核查。公开文本统一UTF-8/LF，保护/输入哈希仍指本机原字节。

不增加报文、信号幅值及额外解码峰值的单位资源面会将外生M限定为正交；κ>1或额外信号能量是其他预算，不在本轮未获证明的情况下自动进入新设计。当前结果是分析与停止证据，不是新算法或可投稿认定。
