"""Render the numerical report from already saved data; no calculations rerun."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh
H=Path(__file__).resolve().parent
d=json.loads((H/'danger_detection_comparison.json').read_text(encoding='utf-8'))
labels={'Gaussian':'原Gaussian','normalized_Gaussian':'归一化Gaussian','whole_sign':'独立整体符号',
        'coordinate_sign':'独立逐坐标符号','Haar_SO2':'独立二维SO(2)旋转','identity':'无编码',
        'public_scalar_alternation':'公开整体交替','public_coordinate_alternation':'公开diag(1,(−1)ᵗ)',
        'public_R90_supplement':'补充公开固定R90'}
lines=['# 共同峰值危险集合内的现有水印真实检测功效', '',
'日期：2026-11-14。阶段B/C最小确定性核验；不改变原求和/common-G估计器、原自主FDI、融合、编码域注入或通信时序，不提出新水印或检测算法。', '',
'**主要结果：危险方向并未集中于狭窄子空间。** 两个补充模型的每一个W白化角方向，都能在编码域总能量10以内达到无水印峰值偏移平方.1。Au的无水印峰值在b₁₀，As在首包更新后的b₁。共同危险集合上，现有Gaussian、单位符号、SO(2)及公开交替的最低实际单次功效存在差异，但没有由此证明一个不可由简单基线替代的新设计能力。', '',
'## 1. 共同集合、统计量与径向化', '',
'攻击为aₜ=Aᵗξ，t=0,…,9。b₀=0，bₖ₊₁=Fbₖ+GMₖaₖ，停止后只按F传播。H₀=I₂，原估计器及11月14日补充五节点参数保持；该模型不是原稿50节点复现。定义共同危险集合', '',
r'\[ \mathcal A_D=\{\xi:\xi^\top W_{10}\xi\le10,\ \max_{1\le k\le15}\xi^\top S_{0,k}\xi\ge.1\}. \]', '',
'危险是无水印配对偏移的峰值，不是正常MSE或水印后实际损害。所有方案使用同一个集合；不用各自实际损害门槛重新改变攻击集合。', '',
'令u(θ)=W⁻¹ᐟ²(cosθ,sinθ)ᵀ，q(θ)=maxₖuᵀS₀,ₖu。方向可行当且仅当q(θ)≥D/E=.01。沿该方向最小危险尺度为r²=D/q(θ)，对应能量r²。条件非中心参数为r²uᵀQωu，非中心卡方右尾随r²不减，故共同危险集合的最低单次功效可放在该方向的最小危险径向边界。该结论仅降维，并不证明角度搜索的全局最优。', '',
'首5个完整受攻包t=0,…,4组成四个二维差分，普通奇偶白化df=8。阈值取单次α=1/300；它与15次监测总体误报≤5%的并集预算相容，但本报告计算的仅是t=4这一次功效：', '',
r'\[ P_D(\xi;\pi)=E_M[\operatorname{sf}_{\chi^2_8(\Lambda_M(\xi))}(\tau)]. \]', '',
'不以EΛ代替真实尾函数积分，不把首5窗功效解释为全事件发现概率，也不据此推导危险先于报警风险。As危险可能在首包已经发生；t=4功效无法倒转此前危险。', '',
'## 2. 危险方向几何', '',
'W白化角θ按π周期表示；θ不是原ξ坐标中的欧氏方向角。全部可能的二次型交点和阈值交点先解析求解，再保留上包络分支切换；本例没有有效切换点或危险角边界。另用8193角网格作第二实现核查。', '',
'|模型|固定峰值状态|全能量最小/最大潜在峰值|最小危险能量范围|危险角覆盖|',
'|---|---|---:|---:|---:|']
for name,m in d['models'].items():
    k=m['geometry']['intervals'][0]['active_bias_index'];W=np.array(m['W']);S=np.array(m['no_watermark_prefix_moment_matrices'][k-1])
    ev=eigh(S,W,eigvals_only=True)
    lines.append(f"|{'Au' if name=='unstable' else 'As'}|b{k}|{10*ev[0]:.6f} / {10*ev[-1]:.6f}|{.1/ev[-1]:.6f}—{.1/ev[0]:.6f}|全部π|" )
lines+=['', '这里的全角覆盖不等于每个小幅ξ危险；每个方向仍有由D/q(θ)给出的径向下界。As终端P3曾不可行，不能据此把本峰值危险集合判空。', '',
'## 3. 真实最低单次功效', '',
'有限符号分布分别对32个整体符号词、1024个逐坐标符号词完整平均。Gaussian/归一化Gaussian使用相同固定Sobol高斯积分节点1024/4096/8192，种子20261114、端点裁剪[10⁻¹²,1−10⁻¹²]；SO(2)角节点使用种子20261115、同层级。层级比较是积分稳定性诊断，没有积分误差区间。各角度使用129格、上包络/可行边界及局部细化。**所有随机混合方案的最小方向均未经全局区间认证；有限词平均精确不等于角度优化已认证。**', '',
'下表最高节点层级；括号为该最低功效方向使用的编码域能量。不同方案允许在共同集合中重新选择ξ；不把各行的实际损害解释为同一ξ的效果。', '',
'|已有方案|Au共同危险集合最低Pd（能量）|As共同危险集合最低Pd（能量）|Au全能量最弱Pd|As全能量最弱Pd|',
'|---|---:|---:|---:|---:|']
for scheme in labels:
    a=d['models']['unstable']['schemes'][scheme]['probes'][-1];b=d['models']['stable']['schemes'][scheme]['probes'][-1]
    x=a['danger_set_minimum'];y=b['danger_set_minimum']
    lines.append(f"|{labels[scheme]}|{100*x['Pd']:.4f}%（{x['energy']:.4f}）|{100*y['Pd']:.4f}%（{y['energy']:.4f}）|{100*a['full_energy_minimum']['Pd']:.4f}%|{100*b['full_energy_minimum']['Pd']:.4f}%|")
lines+=['', '无编码完整受攻窗对原自主族严格零残差，Pd等于1/300。公开整体交替在本共同集合最低单次功效上高于单位随机基线；这已足以阻止将一般“时间变化/秘密旋转”作为独立创新动机，但未证明其对每个ξ、每种资源或全事件风险支配。原Gaussian的正常平均编码信号能量系数ν≈.63463617，与单位编码不同；归一化Gaussian使平均信号能量系数为1，但解码峰值仍无界、编码峰值变为1/√ν。不得把原Gaussian与单位编码称为全部资源相同。', '',
'## 4. 最弱危险方向下的实际期望损害', '',
'逐时刻Sπ,ₖ=E[BₖᵀBₖ]按真实跨时刻均值及同拍矩算子推导。下表是maxₖE||bₖ||²，**不是E maxₖ||bₖ||²**。该攻击在无水印下的潜在峰值均为.1；放大比等于表中峰值除以.1。实际事件危险仍需逐实现路径，不能用此期望峰值判定风险为0。', '',
'|已有方案|Au最大期望偏移平方|Au终端期望偏移平方|As最大期望偏移平方|As终端期望偏移平方|',
'|---|---:|---:|---:|---:|']
for scheme in labels:
    a=d['models']['unstable']['schemes'][scheme]['probes'][-1]['danger_set_minimum'];b=d['models']['stable']['schemes'][scheme]['probes'][-1]['danger_set_minimum']
    lines.append(f"|{labels[scheme]}|{a['scheme_max_expected_bias_squared']:.6f}|{a['scheme_expected_terminal_bias_squared']:.6f}|{b['scheme_max_expected_bias_squared']:.6f}|{b['scheme_expected_terminal_bias_squared']:.6f}|")
lines+=['', 'As整体符号的该方向期望峰值恰为.1而终端约.00151；峰值与终端指标分离非常明显。Gaussian改善t=4功效的同时也将该方向的最大期望偏移放大到约.688；归一化仍约.437。数值显示现有检测—损害折中，尚不是新机制优越性证明。', '',
'## 5. 方向错配的准确解释', '',
'JSON同时保存三个对象：无水印全能量峰值最危险ξ、各方案全能量最难检测ξ、共同危险集合最低实际功效ξ。二维曲线在同一W白化角坐标给出这三者，以及各方向全能量/最小危险径向的实际功效和损害。不能仅由两个向量夹角认定可开发空间。', '',
'Au的全部无水印峰值由b₁₀控制，因此本共同峰值集合恰与既有P3-pot终端集合一致，最低Pd也一致。As全部峰值由b₁控制；共同危险集合最低Pd方向的θ约2.95，而全能量最弱Pd方向约1.05—1.45（不同方案），说明固定全能量方向诊断不能替代危险集合优化。这里的变化主要含危险门槛要求的径向尺度，不能称为已经发现一个可由定向矩阵解决的独立安全缺口。', '',
'## 6. 独立核查与交付', '',
'`danger_detection_check.py`以vec(B)均值/二阶矩递推另行重建全部15个前缀矩；Gaussian用三节点Hermite验证四阶矩，SO(2)用四角求积验证二阶矩。全部1024个10拍整体符号词另行逐拍递推，再对所有15个偏移状态平均，与解析矩相符；最大绝对差约3×10⁻¹⁵。W、径向能量、无水印peak=.1、8193角几何核查均通过。这是内部公式交叉核对，不是独立同行审证或物理模型区间认证。', '',
'文件：', '',
'- `danger_detection_compare.py`：计算入口，只读复用上一轮矩阵接口，写入本新目录。',
'- `danger_detection_comparison.json`：全部模型/前缀矩/危险几何/节点层级/方向/257角曲线及输入哈希。',
'- `danger_detection_check.py`、`danger_detection_checks.json`：第二套前缀矩与有限词核查。',
'- `danger_detection_compare.log`：各方案及节点层级原始输出。', '',
'本分工未重做未知起点首次报警或空间一致性检测；这些必须另行按正确联合协方差和总体误报预算判断。本报告的正面最低Pd变化不足以授权阶段D/E或宣称新颖性。']
(H/'danger_detection_共同危险集合功效与损害.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('report written')
