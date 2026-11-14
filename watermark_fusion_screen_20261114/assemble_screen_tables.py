"""Assemble saved diagnostics without rerunning or selecting parameters."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
load=lambda n:json.loads((H/n).read_text(encoding='utf-8'))
norm=load('rho_normal_screen.json');fp=load('rho_first_packet_screen.json')
split=load('rho_first_packet_split_angle.json');clips=load('clip_baseline_results.json')
p1=load('rho_expected_damage.json');raw=load('raw_watermark_resources.json')
clipnet=load('clip_risk_net.json');cmc=load('candidate_first_mc.json')
lookup={(r['model'],r['eta'],r['scheme']):r['risk_split_angle_adaptive_radial']for r in split}
lines=['# 正常精度—安全风险—编码成本：最小筛选结果','',
 '日期：2026-11-14。以下数据均来自新聚合五节点模型，不用旧四维KF。目标链路0→1；E=10、D=.1；普通奇偶s=4，15个预定检验，每次α=.05/15。危险相对同秘密、同噪声、无报警门控的名义影子轨迹定义。', '',
 '正常预算η为目标节点稳态后验MSE trace相对原融合的容许增幅，预先冻结1%、5%、10%、20%。随机权重正常MSE由精确二阶矩递推配合浮点求积得到；裁剪采用独立训练/验证的目标节点噪声路径，实际是Monte Carlo，不将其写成解析协方差。','',
 '首包风险表是L=1预定自主攻击筛选，Gaussian列为保存方向/能量的积分诊断，含非全局搜索；普通裁剪列是预定有限攻击网格的样本最大值。它们不是全变量持续时间最坏风险。η20另有连续能量球覆盖与独立样本核对，见后文。','']
for name,mm in norm['models'].items():
 lines+=['## '+name,'',f"原融合正常trace={mm['reference_trace']:.9f}。原Gaussian首包候选风险约{fp['models'][name]['original_Gaussian']['risk']:.3f}，原符号约{fp['models'][name]['original_iid_sign']['risk']:.3f}；均不称全事件最坏值。",'',
 '|η预算|固定ρ|候选γ|Gaussian固定ρ|Gaussian感知ρ|符号匹配η/无水印固定ρ|普通裁剪无水印/同Gaussian|裁剪实际正常损失±1SE|',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
 for n in mm['rows']:
  eta=n['eta'];f=next(r for r in fp['models'][name]['rows']if r['eta']==eta)
  c=next(r for r in clips['rows']if r['model']==name and r['eta']==eta)
  lines.append(f"|{eta:.0%}|{n['fixed_rho']:.4f}|{n['high_order']['gamma']:.4f}|{lookup[name,eta,'Gaussian_fixed_rho']:.3f}|{lookup[name,eta,'Gaussian_aware']:.3f}|{f['iid_sign_aware']['risk']:.3f}|{c['risks']['none']['finite_attack_grid_empirical_max']:.3f} / {c['risks']['Gaussian']['finite_attack_grid_empirical_max']:.3f}|{c['paired_relative_MSE_increase']:.2%} ± {c['paired_relative_MSE_standard_error']:.2%}|")
 lines+=['', '固定ρ是固定线性衰减；普通裁剪是`clip_tau(G innovation)`，两者不混用。表中符号匹配η是按相同min规则独立选择γ_sign=||G||ρ_fixed的更强基线。若严格沿用Gaussian的同一个γ，本表全部γ>||G||，符号ρ=1，退化原符号，正常损失为0。','']
lines+=['## 同一Gaussian编码下的首包连续能量球核对','',
 '额外核对只固定η20和已有候选攻击，不再调候选。对普通裁剪使用128角×129半径的能量球覆盖，利用整体Gaussian噪声/奇函数裁剪的P(a)=P(−a)对称性；攻击向量覆盖半径δ=.0511594。对每网格点把危险条件放宽为`||b_g||≥sqrt(D)−||GM||δ`、未报警条件放宽为`||r_g||_{Theta^-1}≤sqrt(tau)+||Theta^-1/2 J M||δ`。由算子Lipschitz性，任意球内a的失败事件包含于某个扩张事件。','',
 '核对使用与初筛完全不同、在核对运行前冻结的新噪声/初态/水印种子。2048条独立验证路径、对16512个扩事件作Hoeffding并集；候选则在事先保存的一条攻击上另抽131072份独立M，用条件非中心卡方CDF积分物理噪声。两模型共四个单侧界各失败概率.0025，合计≤.01。以下是在理想独立抽样假设、冻结数值模型及有限预热健康初始化下的首包当拍概率界，不含正常MSE约束的严格认证，也不覆盖完整变持续攻击族。','',
 '|模型|普通裁剪：连续L1球风险上界|候选：某固定L1攻击风险下界|裁剪正常MSE损失估计|',
 '|---|---:|---:|---:|']
for c in clipnet['rows']:
 a=next(r for r in cmc['rows']if r['model']==c['model'])
 assert c['continuous_L1_risk_upper_qualified']<a['L1_worst_risk_lower_qualified']
 lines.append(f"|{c['model']}|{c['continuous_L1_risk_upper_qualified']:.3f}|{a['L1_worst_risk_lower_qualified']:.3f}|{c['normal_MSE_relative_increase_estimate']:.2%} ± {c['normal_MSE_relative_standard_error']:.2%}(SE)|")
lines+=['','裁剪正常精度只作独立验证估计；其均值约19.9%，但计入统计不确定性后不能严格认证≤20%。不能将标准误或预热后的经验MSE当严格稳态约束证明。上述比较使用完全相同Gaussian编码、载荷次数、解码数据和检测阈值，故没有把不同编码能耗混进这项停止证据。它针对L1首包子问题，不能推出整个L≤11类的全局风险排序。','',
 '## P1/P3保留，但不替代风险','',
 '|模型/η|Gaussian固定ρ P1|Gaussian感知ρ P1|感知ρ的P3-phys可行|',
 '|---|---:|---:|---|']
for name,m in p1['models'].items():
 for r in m['rows']:
  a=r['Gaussian_fixed_rho'];b=r['Gaussian_aware']
  lines.append(f"|{name}/{r['eta']:.0%}|{a['P1_expected_terminal_damage']:.6f}|{b['P1_expected_terminal_damage']:.6f}|{b['P3_phys_feasible']}|")
lines+=['','P1为L10自主初值在真实总编码能量10下最大**期望终端影子偏移平方**。随机Fρ与ρGM同拍相关，二阶递推保留该相关性。P3-phys要求期望终端损害≥.1，按每种方法重新选择方向/幅值；损害不可达时P3不可行，绝非检出率100%。首个五拍完全受攻窗的P3检出积分及初值/能量完整保存于rho_expected_damage.json；Gaussian为固定4096 Sobol求积诊断，无积分误差/全局优化证书。平均损害下降没有自动通过危险先报警目标。','',
 '## 编码资源与归一化诊断','',
 f"m=2原Gaussian的正常信号能量系数ν=E[M^-2]的标量系数约{raw['original_Gaussian_normal_packet_energy_ratio']:.9f}，独立符号/无编码为1。原Gaussian及其固定ρ、感知ρ、普通裁剪的编码成本严格相同；接收端调权不改变发送包。归一化Gaussian取Mtilde=sqrt(ν)M，使平均信号能量为1，但编码峰值增为1/sqrt(ν)≈{raw['normalized_encoder_peak_operator_norm']:.4f}，不能叫全部资源相同。归一化仅任务B诊断，任务D始终未换分布。",'',
 '|模型|原Gaussian P1|归一化Gaussian P1|符号P1|', '|---|---:|---:|---:|']
for r in raw['rows']:
 v=r['P1_expected_terminal_damage_L10'];lines.append(f"|{r['model']}|{v['Gaussian']:.6f}|{v['normalized_Gaussian']:.6f}|{v['sign']:.6f}|")
lines+=['','包数和零额外编码等待相同不等于无线总能耗相同。密钥、PRNG、时间同步、谱范数计算及有限精度开销未实测；ρ需要额外算2×2谱范数，不增加第二份秘密或报文，但没有解决原同步/失步问题。','',
 '## 本轮验收','',
 '**C：停止当前ρ候选，创新性不通过。** 低正常损失预算下没有稳定胜过同编码固定ρ；20%时对固定ρ的局部改善，被更简单的普通裁剪基线反证。即时增益封顶/能量界虽成立，却不构成已验收的新耦合能力。停止针对当前规则，不否定所有乘性水印接收端机制，也不把普通裁剪作为本论文新贡献。']
(H/'最小对照结果与停止决定.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Tables assembled from saved results; no parameter selection performed.')
