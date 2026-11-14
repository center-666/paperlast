from pathlib import Path
import json
H=Path(__file__).resolve().parent
d=json.loads((H/'first_alarm_bounds.json').read_text(encoding='utf-8'))
labels={'Gaussian':'原Gaussian','normalized_Gaussian':'归一化Gaussian','whole_sign':'独立整体符号',
        'coordinate_sign':'独立逐坐标符号','Haar_SO2':'独立SO(2)','identity':'无编码',
        'public_scalar_alternation':'公开整体交替','public_coordinate_alternation':'公开逐坐标交替',
        'public_R90_supplement':'公开固定R90'}
def interval(r):return f"[{r['parity_danger_before_alarm_lower']:.6f}, {r['parity_danger_before_alarm_upper']:.6f}]"
lines=['# 固定共同探针的有限首次报警界与空间检测对照', '',
'日期：2026-11-14。固定原求和/common-G估计器、原L10自主攻击、总编码域能量≤10、危险偏移平方.1。使用新目录已保存的固定探针；不优化攻击、不设计新水印或GLRT、不采样物理噪声。', '',
'## 1. 固定探针与评价时标', '',
'每模型固定三个共同探针：无水印峰值最强全能量方向、无水印峰值最弱全能量方向、前述整体符号共同危险集合最弱Pd初值乘1.000001。As另加入旧有理初值(−1.89,.42)。全部方案使用完全相同的各模型探针；不按风险再选方向。第三探针乘系数只用于使已确定的危险边界严格越过.1，所有能量仍小于10。', '',
'监测包k=0,…,14，b₀=0，本包先检验、再原融合，融合后偏移bₖ₊₁归入本包危险。攻击t=0,…,9，t≥10注入为零；停止后偏移继续按F传播至b₁₅。监测从固定候选攻击起点重置，攻击前四个健康包可用于启动奇偶窗。本报告不提供未知起点无限监测保证。', '',
'普通奇偶df8和当前空间df2分别每拍使用α=1/300；15项总体误报均≤5%。空间当前噪声独立，故健康全事件误报恰为1−(1−1/300)¹⁵≈.0488500。两类检测器作为分别运行的基线，未同时启用，未把两套检验共享同一5%预算。', '',
'|模型/固定探针|初值ξ|编码能量|无水印潜在峰值|峰值偏移状态|',
'|---|---|---:|---:|---:|']
for name,m in d['models'].items():
 for p,z in m['probes'].items():
  lines.append(f"|{'Au' if name=='unstable' else 'As'} / {p}|({z['initial_state'][0]:.9f}, {z['initial_state'][1]:.9f})|{z['encoded_energy']:.9f}|{z['no_watermark_potential_peak']:.9f}|b{z['no_watermark_peak_bias_index']}|")
lines+=['', '危险集合按无水印定义，但事件危险按每个方案的实际偏移路径定义。同一共同探针在某编码下可能根本不危险；必须与检出效果分别解释。', '',
'## 2. 不假设重叠奇偶窗独立', '',
'对整条外生水印路径ω条件化，令pₖ(ω)为正确白化、含启动/停止瞬态的真实单次非中心χ²右尾功效。普通奇偶首报警CDF满足', '',
r'\[ \max_{0\le i\le k}p_i(\omega)\le P(\tau_A\le k\mid\omega)\le\min(1,\sum_{i=0}^k p_i(\omega)). \]', '',
'先对路径取max/和，再平均；不把重叠窗口视为独立样本。JSON保存每拍CDF和由CDF差分得到的逐项PMF上下界；这些界不声称共同构成一条精确PMF。', '',
'条件于ω后，偏移首次危险拍tD(ω)确定。若无危险，事件风险为0；否则', '',
r'\[ (1-\sum_{k=0}^{t_D}p_k(\omega))_+\le P(\tau_D<\tau_A\mid\omega)\le1-\max_{k\le t_D}p_k(\omega). \]', '',
'风险积分包含本拍检验；本拍已报警不算危险先报警。普通奇偶窗口使用qₗ差分l=−3,…,14，启动q₀含δ₀，停止q₁₀=−Aδ₉，q₁₁以后攻击差分为0。第14检验攻击均值已归零，但此前偏移仍可能存在。', '',
'## 3. 当前空间检测可以条件连乘', '',
'rᵗˢₖ=ỹⱼ,ₖ−HⱼHᵢ⁻¹yᵢ,ₖ，协方差V=.5I+.5(HⱼHᵢ⁻¹)(HⱼHᵢ⁻¹)ᵀ。固定模型的本地可信量测同步可用，状态真值逐拍抵消，正常空间噪声只含独立的当前传感器噪声。对整条与物理噪声独立的外生水印条件化，当前空间检验跨拍独立。因此', '',
r'\[ P(\tau_A\le k\mid\omega)=1-\prod_{i=0}^k(1-p_i^{sp}(\omega)),\qquad P(\tau_D<\tau_A\mid\omega)=\prod_{i=0}^{t_D}(1-p_i^{sp}(\omega)) \]', '',
'无危险时风险仍为0。必须先连乘后积分；不能以各拍平均功效的乘积替代。本实现只用当前空间检验，不将五包空间能量/匹配评分的重叠序列当作独立。', '',
'## 4. 无水印峰值最强全能量共同探针', '',
'数值为最高预声明节点层级。整体符号遍历全部1024个10拍词；其他随机方案的标记见第7节。', '',
'|已有方案|Au普通奇偶风险界|Au空间当前风险|Au危险路径概率|As普通奇偶风险界|As空间当前风险|As危险路径概率|',
'|---|---:|---:|---:|---:|---:|---:|']
for scheme in labels:
 a=d['models']['unstable']['schemes'][scheme][-1]['probes']['peak_strong_fullE'];b=d['models']['stable']['schemes'][scheme][-1]['probes']['peak_strong_fullE']
 lines.append(f"|{labels[scheme]}|{interval(a)}|{a['spatial_current_danger_before_alarm']:.6f}|{a['dangerous_path_probability']:.6f}|{interval(b)}|{b['spatial_current_danger_before_alarm']:.6f}|{b['dangerous_path_probability']:.6f}|")
lines+=['', '该表只评价一个预先固定探针。普通奇偶是实际联合风险的上下界，空间当前是相应条件连乘概率；均不是整个攻击集合的最坏风险或对所有方向的排序。As多数单位编码该探针首包必危险，所以奇偶风险上下界相等，恰为首包漏警概率；后续较大功效不能倒转此前危险。', '',
'## 5. 整体符号危险弱方向的共同探针', '',
'|已有方案|Au普通奇偶风险界|Au空间当前风险|Au危险路径概率|As普通奇偶风险界|As空间当前风险|As危险路径概率|',
'|---|---:|---:|---:|---:|---:|---:|']
for scheme in labels:
 a=d['models']['unstable']['schemes'][scheme][-1]['probes']['whole_danger_weak_x1p000001'];b=d['models']['stable']['schemes'][scheme][-1]['probes']['whole_danger_weak_x1p000001']
 lines.append(f"|{labels[scheme]}|{interval(a)}|{a['spatial_current_danger_before_alarm']:.6f}|{a['dangerous_path_probability']:.6f}|{interval(b)}|{b['spatial_current_danger_before_alarm']:.6f}|{b['dangerous_path_probability']:.6f}|")
lines+=['', 'As独立SO(2)该探针仅约13%的实现达到实际危险，风险约.131主要来自危险可达性下降，不能称为报警成功率约87%。公开R90该探针风险为0也来自危险不可达，而不是100%检出。Au公开交替/部分公开编码在该探针下同样可能使危险不可达；不据此推断全攻击族安全。', '',
'## 6. 已有As有理首包探针', '',
'对固定ξ=(−1.89,.42)，整体符号普通奇偶风险约.98184、当前空间风险约.87580；SO(2)相应约.98231、.86869。这个探针在所有单拍正交解码下首包危险，故普通奇偶风险是首包漏警，不需要滑窗独立假设。数值与此前严格>.98的普通奇偶首包命题相容；本文件未把浮点CDF重新称为严格证书，也没有把该命题扩成所有检测器的不可能性定理。', '',
'## 7. 数值证据与复现范围', '',
'- 整体符号：全部1024个10拍词有限平均，普通奇偶联合界与空间当前概率在指定冻结模型上是对应真实分布的有限和；χ²尾函数仍使用浮点计算，没有区间CDF误差认证。',
'- 逐坐标符号：完整10拍需2²⁰种词，本轮使用固定20维Sobol节点1024/4096；不是全词枚举。',
'- Gaussian/归一化Gaussian：固定20维Sobol1024/4096，种子2026111432，逆正态端点裁剪[10⁻¹²,1−10⁻¹²]；二者共用相同节点。',
'- SO(2)：固定10维角Sobol1024/4096，种子2026111433；逐坐标符号种子2026111431。上述全部为确定性积分诊断，没有积分误差证书。危险首次到达是非光滑指标，节点升级稳定不能替代严格积分误差界。',
'- 确定编码：单条公开完整路径，无水印正常误差保持。全部编码每拍仍传同一二维量测，无额外发送拍数。',
'- 物理噪声采样为0。没有新增网络闭环仿真；依正确Gaussian模型解析计算条件单次功效。',
'- 脚本检查启动/停止差分、停止后差分消失、CDF界顺序、首包危险时风险=首包漏警的恒等式。攻击危险判断没有人为阈值容差，比较为保存浮点模型上的>=.1；边界探针已乘1.000001。', '',
'完整逐拍结果、峰值E max与max E、终端损害及节点层级位于`first_alarm_bounds.json`；运行入口`first_alarm_bounds.py`。本报告没有把固定探针概率界升级为全攻击族minimax风险界，没有因这些结果启动新机制设计。']
(H/'first_alarm_bounds.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('first_alarm_bounds.md written')
