"""Evidence-linked numeric reports. No model/selection changes."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text('utf8'))
def tab(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def write(n,t):(H/n).write_text(t.strip()+'\n','utf8')
def main():
    obs=read('observer_bounds.json');res=read('validation_results.json');supp=read('strong_baseline_checks.json')
    rows=[]
    for n,z in obs['models'].items():
        for label,r in z.items():rows.append([n,label,*[f"{r[k]:.9f}" for k in ('trace_prior','trace_linear','trace_genie','trace_unseen_lower_bound')]])
    write('04_攻击者预测误差重建可行性.md',r'''# 攻击者能否重建接收端预测误差

日期2026-11-17。误差约定e⁻=接收端预测−真实状态，不与相反符号文献混用。

## 1. 需要的量不是免费可得

公开K/G、模型和初始估计0不等于知道接收端运行状态。原共G更新还含本地y₁及健康邻居y₂；目标链路q₀没有直接携带其独立量测噪声。只窃听q₀无法逐路径重建接收端估计，更无法获知真值误差。

|实际窃听权限|可重建内容|
|---|---|
|仅目标链路历史编码q₀|可按模型计算条件分布或设计线性观测器；不是精确接收端状态|
|再有当前q₀|增加当拍信息，仍未知M及本地/健康邻居噪声|
|所有接收输入均为可读原始量测、已知初值/增益/攻击输入|可重放接收端估计；真实状态仍需统计估计|
|所有输入只有编码q而无秘密M|不能直接解码；原始量测KF的Gaussian观测闭合通常失效|
|免费接收端预测误差或干净源副本|超出本轮权限|

源节点发送原始量测而非估计反馈，因此本地/健康邻居的独立测量噪声不会经目标源q重新泄露。这一点正是本原共G结构可用的独立噪声下界。

## 2. 一般不可消除的条件不确定性

记U=K₁R₁K₁ᵀ+G₁R₂G₁ᵀ。预测k的接收估计含独立噪声分量

\[
A\sum_{t=0}^{k-1}F^{k-1-t}(K_1v_{1,t}+G_1v_{2,t}).
\]

它独立于目标q全部历史以及只依赖该q的攻击策略。因此在本轮权限下，任何估计器均有

\[
\operatorname{Cov}(e^-_k\mid\mathcal I_k^a)\succeq
U_k:=A\sum_{r=0}^{k-1}F^rU(F^r)^\top A^\top .
\]

H还未读当拍q，鲜过程噪声wₖ₋₁独立于其信息，可再加Q；C读了当拍q，不能继续免费保留完整Q下界。该不可见噪声分解在攻击后仍成立，不依赖把自适应偏移误当独立变量。

**证明。** 将原共G递推展开，单独提出v₁/v₂的上述线性分量。攻击和剩余项均由物理状态、目标v₀、秘密M和目标q构成，与这些未窃听测量噪声独立。条件协方差等于其无条件协方差加另一半正定项。H下wₖ₋₁亦如此。证毕。

冻结矩阵的单步不可见协方差正定已有精确有理主子式证书。此结论没有声称条件均值不存在；它排除的是攻击者无误差地获得接收端真实预测误差。

## 3. 未知水印下不直接使用标准KF

观测q=M⁻¹(Hx+v)条件于x、未知M时为Gaussian混合。信号矩阵和噪声协方差均随机，标准固定H/R的Gaussian KF不再自动等于Bayesian MMSE。精确最优观测可形式写成E[e⁻|q历史]，但本轮未求其非线性Bayes递推，更未把真实e⁻作为攻击输入。

为量化“是否有信息”，在首次攻击前k=7作**解析最优线性**估计。令S=M⁻¹，αI=E[S]，E[SXS]=cₓX+cₜtr(X)I。对于不同t,u，

\[
\operatorname{Cov}(q_t,q_u)=\alpha^2\operatorname{Cov}(y_t,y_u);
\]

同拍使用cₓPᵧ+cₜtr(Pᵧ)I；Cov(e,q)=αCov(e,y)。LMMSE矩阵为PₑqPqq⁻¹。所有矩由原始独立Gaussian线性系数图计算，α/cₓ/cₜ仅一维确定性水印积分，无物理新样本。

下表为误差协方差迹：

'''+tab(['模型','信息','无观测误差','编码q最优线性','干净q-genie','独立噪声下界'],rows)+r'''

“干净genie”仅作下界，不给实际攻击者。此处e和干净y联合Gaussian，e=Ly+ε且ε独立y/M，因此实际Bayes编码观测的条件误差至少为干净genie误差；最优线性结果给平均Bayes误差的上界，不是Bayes精确值。

历史q对该强收缩滤波器的预测误差只提供很小线性增量；当前q更有用。不能将这个首次攻击前计算当成攻击后未知M/预算饱和条件下的精确MMSE滤波。

## 4. 与最优攻击文献的关系

用户下载的Wang、Wei全文已补齐，当前不存在这两篇的全文缺口。Wang使用相同的共G结构，其Assumption1还给攻击起点的接收端估计，递归式(44)显式含本地yᵢ及邻居原量测。Wei Proposition4/式(74)—(78)则直接处理仅可读可疑创新的MMSE观察器，并在Remark6说明其不需要该起点估计；不能把两篇权限混为一谈。两者处理的都是其信息模型下的MMSE，本文不可见噪声下界不反驳它们，也不证明条件均值无用。

本轮未知M的编码q为Gaussian混合；原始量测或创新的标准KF不能直接照搬。Wei Remark13已明确指出MMSE信息原则可推广，故“混合观测尚未求解”本身不构成新攻击原理。两篇也未提供本轮同一硬逐路径总能量和首次报警风险的完整最优性，不能反向扩大其定理量词。逐式对应及实读页见06。

源码attacker_observer_bounds.py从未把观测器输出输入主滤波器。当前H筛选只使用简单滞后q，未以“不精确KF”冒充文献最优攻击复现。
''')
    rows=[];fa=[];controls=[];p3=[]
    for n,m in res['models'].items():
        for lab,z in m['nominal'].items():fa.append([n,lab,z['df'],f"{z['threshold']:.6f}",f"{z['alarm_count']}/{z['samples']}"])
        for fam in ('A0','A1-H','A1-C'):
            for objective in ('P1_peak_mean_bias','actual_MSE','danger_joint_GLRT'):
                pid=m['selected'][fam+'/'+objective];z=m['validation_rows'][pid]['metrics']
                rows.append([n,fam,objective,pid,f"{z['peak_mean_bias']:.6f}",f"{z['mean_MSE_increment']:.6f}",
                  f"{z['mean_cross']:.6f}",f"{z['detectors']['temporal']['pre_alarm_danger_rate']:.4f}",
                  f"{z['detectors']['joint_GLRT']['pre_alarm_danger_rate']:.4f}"])
        z=supp['models'][n]
        for label,k in [('原解析A0-P1','analytic_A0_P1'),('公开首包集中对照','public_first_packet')]:
            x=z[k];r=x['validation']
            controls.append([n,label,f"{x.get('population_expected_peak',x.get('theoretical_expected_peak')):.6f}",
                f"{r['peak_mean_bias']:.6f}",f"{r['mean_MSE_increment']:.6f}"])
        for d,pid in z['population_P3_selected_using_training'].items():
            v=z['population_P3_validation'][pid];r=v['metrics'];p=v['policy']
            p3.append([n,'A0解析损害边界',d,f"{p['encoded_energy']:.6f}",f"{r['detectors'][d]['event_alarm_rate']:.4f}",
                       f"{r['detectors'][d]['pre_alarm_danger_rate']:.4f}"])
            cp=m['selected']['A1-C/P3_proxy_'+d];cz=m['validation_rows'][cp]['metrics']
            p3.append([n,'C训练可行代理',d,f"{cz['energy_mean']:.6f}",f"{cz['detectors'][d]['event_alarm_rate']:.4f}",
                       f"{cz['detectors'][d]['pre_alarm_danger_rate']:.4f}"])
    write('05_攻击损害与检测功效.md',r'''# A1：实际损害、检测和危险先报

日期2026-11-17。只做冻结五节点机理筛选；未做正式50节点实验。

## 1. 路径恒等式与总体交叉项

同噪声、同水印、同初值、全接纳固定K/G：

\[
b_k=F b_{k-1}+G_1M_ka_k,\qquad e_k^a=e_k^0+b_k ,
\quad
E\|e_k^a\|^2=E\|e_k^0\|^2+E\|b_k\|^2+2E[(e_k^0)^\top b_k].
\]

A0固定参数与物理噪声独立时最后一项期望为零，但有限样本值不必零。A1不再能删除交叉项。

**严格反例（当前负反馈首包）。** 健康e和源y联合Gaussian，令L=Cov(e,y)Pᵧ⁻¹。若首次攻击a=cα(q)q，则b=cα(q)Gy，α≥0，且

\[
2E[e^\top b]=2cE[\alpha(q)y^\top\operatorname{sym}(G^\top L)y].
\]

两冻结模型在k=7的sym(GᵀL)均由精确有理主子式认证为正定。因此对c<0、正概率实际注入的预算裁剪或可行反射，交叉项严格为负。这个总体反例满足硬预算，不只是有限样本波动；它解释了“偏移平方大”可能高估实际MSE损害。只对该首包情形给符号保证，不推广整段A1交叉项总为负。

停止后a=0，旧固定F自然衰减仍成立；正常MSE各组相同。未报警条件MSE还会因选择事件产生新相关，不能套用无条件加法；本轮未作其全局优化。

## 2. 合法检测信息与总FA

原时间奇偶仍使用受攻源5拍解码数据。联合强基线使用本地1、目标0、健康2的原始量测，消除任意窗口初态，并对**目标源五拍任意加性均值子空间**作标准GLRT，df10；没有用仅匹配A0轨迹的df2模型削弱A1检测。

目标链路已知，攻击起点不提供给检测器。扫描结束4…31共28窗，各项α=.05/28。名义数据由正确共享过程噪声协方差白化，χ²阈值使全监测FA≤5%；实际联合窗相关，未作独立窗口乘积。另一个原始噪声系数图核验相邻窗交叉协方差范数约2.81。

'''+tab(['模型','检测器','df','单窗阈值','48条健康验证事件报警'],fa)+r'''

攻击后数据依赖，不能再用固定非中心χ²算Pd。以下直接在合法实际报文计算评分和首次报警。事件报警率包含可能的预攻击误报，风险采用全监测τA；JSON另列攻击前存活条件指标。本次列出策略的验证样本无预攻击报警，不能据此假设总体不存在预误报。

## 3. 事前选择与小样本验证

每模型重放64条既有样本：16条选参数，48条验证；共128条物理输入，未新增物理随机种子。每模型378个参数配置分别针对峰值期望偏移、累计实际MSE、两个检测器的危险风险和P3代理选择。配置不等于独立算法/独立MC样本。原选择文件在验证前保存，执行数值余量修正后其哈希不变。

以下为**按训练目标选定策略**的48样本结果，均非全攻击类最优：

'''+tab(['模型','类','训练目标','策略ID','maxₖ均值偏移²','7—31均值MSE增量','均值交叉项','时间风险','联合风险'],rows)+r'''

危险率是τD<τA，τD为攻击诱导偏移²首次≥0.1，包含报警当拍的严格比较；不是真实总误差越界。首报未发生者按∞计风险；JSON的平均首报截断到32仅作删失诊断，不是检出条件延迟。

总体观察：H能改变峰值与污染时序，但部分峰值增加来自轨迹形状自由；C在稳定模型里呈现较低事件报警与较高危险先报的样本现象。C的实际MSE目标两模型都选择θ=0，H也有退化情况。没有统一支配、总体最优风险或显著性定理。

## 4. 防止弱化A0和误归因

审查补强两项对照，不重新挑选A1参数：

'''+tab(['模型','对照','解析总体期望峰值','48样本峰值','实际均值MSE增量'],controls)+r'''

公开首包对照不读q，已能产生很大峰值。因此不能把H相对自主A0的峰值提升全称为信息收益。公开首包不属于原自主A0，表中明确作为一般H集合中的确定性成员。

另用原Gaussian解析损害Sπ，将原12方向缩放至maxₖξᵀSπₖξ=.1，再仅由训练样本选择A0事件报警最小者：

'''+tab(['模型','候选','检测器','能量/平均能量','事件报警率','危险先报率'],p3)+r'''

A0行按解析总体损害可行；C行仅是训练可行P3代理。两者不构成同一总体P3的全局最优证明。A0强基线降低了原粗网格给出的“最难报警率”，说明不能只用整能量攻击替代P3对手。未因该补查变更原θ、D/E、检测阈值或验证样本。

## 5. 复核与误差说明

独立展开原共G实际更新，2976个节点拍核对攻击/健康差、实际MSE、能量与未来输入不可见性，最大误差约2.4×10⁻¹⁴；θ0严格退化A0。实际最大累计能量<10。

首次浮点求和曾出现10+1.78×10⁻¹⁵，已保留原记录并统一加入10⁻¹²预算数值余量；原参数选择哈希保持，未调物理预算。详见roundoff_guard_record.json。

有限样本比例只作机理筛选，不能把例如44/48当成92%总体风险证书。没有对A1总体P3、无限在线误报、未知目标归因或隔离闭环作认证；全部图景仍在原共G全接纳诊断范围。
''')
    print(json.dumps({'date':'2026-11-17','numeric_reports_written':2,'new_defense':False}))
if __name__=='__main__':main()
