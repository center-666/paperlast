# GLRT筛选审阅入口

先读[理论与最小安全筛选报告](秘密水印条件轨迹匹配检测_理论与最小安全筛选.md)，再读[文献边界](文献定位与排重.md)。结论是保留标准GLRT为强基线，尚未证明相对普通符号评分的最坏危险风险严格改善。

`minimal_results.json`含固定攻击风险界、末拍单次功效及Gaussian求积诊断；`alarm_time_bounds.json`含首次报警CDF/逐拍质量界，不是精确联合分布。Gaussian值无严格积分误差证书。

`exact_sign_danger_cap.py/json`给冻结十进制有理模型的1/16粗风险上界，完全来自符号损害映射。它不是实际DARE解的区间证书，也不是GLRT带来的增益。

复现代码使用Python、NumPy、SciPy。无物理噪声样本；运行`python glrt_screening_20261112/minimal_screen.py`会枚举符号词并生成矩阵文件；整数证书只依赖Python标准库。固定输入位于仓库`checks/水印收束_理论与映射核验_20261105.json`。原始论文、TeX和第三方PDF不在此包中，文献报告的部分本机相对链接未随包提供。
