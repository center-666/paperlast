"""Render numeric evidence reports. No change to frozen research inputs."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text('utf8'))
def write(n,s):(H/n).write_text(s.strip()+'\n','utf8')
def pct(x):return f'{100*x:.4f}%'
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
    online=read('online_bounds_results.json')
    for r in online['results'].values():
        r.pop('conditional_noise_bounds_rigorous',None)
        r['conditional_noise_bound_formula_valid']=True
        r['floating_numeric_bounds_not_interval_certified']=True
    (H/'online_bounds_results.json').write_text(json.dumps(online,ensure_ascii=False,indent=2)+'\n','utf8')
    batch=read('batch_power_results.json');prior=read('prior_baseline_results.json');checks=read('independent_checks.json')
    powerrows=[];normalrows=[];priorrows=[];p1rows=[];ncprows=[];phases=[];risks=[]
    for name,m in batch['models'].items():
        normalrows.append([name,f"{m['normal_target_MSE_average7to31']:.9f}",'全部方案相同'])
        for b,r in m['schemes'].items():
            powerrows.append([name,b,pct(r['P3']['Pd']),f"{r['P3']['energy']:.6f}",pct(r['Pd_at_P1'])])
            ncprows.append([name,b,f"{r['min_ncp_at_full_energy']:.6f}",r['df']])
        for b in ('B1','B2'):
            r=m['schemes'][b]['P1']
            p1rows.append([name,b,r['k'],f"{r['value']:.9f}",str([round(v,9) for v in r['xi']])])
        for c,z in m['auxiliary'].items():
            phases.append([name,'C'+str(int(c)+1),pct(z['min_phase_P3_power']),z['best_phase_for_aligned_batch']])
        for b,z in prior['models'][name]['schemes'].items():
            priorrows.append([name,b,pct(z['P3']['Pd']),f"{z['P3']['energy']:.6f}"])
        for b in ('B0','B1','B2','B3_coord0','B3_coord1'):
            r=online['results'][name+'/'+b]['starts']['7']['profiles']['P1']
            risks.append([name,b,f"[{r['risk_lower']:.6f}, {r['risk_upper']:.6f}]",
                f"{r['danger_reachable_fraction']:.6f}",f"[{r['truncated_delay_lower']:.3f}, {r['truncated_delay_upper']:.3f}]"])
    common='日期2026-11-16；基线28c3633，参考底座0ea9c7d。只完成G0/G1；辅助数据不进入原共G更新，不实施隔离，不称原50节点复现。\n\n'
    write('03_可信量测噪声与检测功效.md',r'''# 联合噪声、真实功效与未知起点扫描

'''+common+r'''## 1. 观测时刻与共享噪声

对窗口起点r及实际可用行hα在时刻t的观测，

\[
y_{\alpha,t}=h_\alpha A^{t-r}x_r+
\sum_{\ell=r}^{t-1}h_\alpha A^{t-1-\ell}w_\ell+v_{\alpha,t}.
\]

辅助行hα=C只在采样报文已于t+delay到达时加入。未采样行删除。堆叠Y=Oxᵣ+n+Dξ，共享过程噪声给出

\[
\Omega_{(\alpha,t),(\beta,u)}
=\sum_{\ell=r}^{\min(t,u)-1}h_\alpha A^{t-1-\ell}Q
(h_\beta A^{u-1-\ell})^\top+\mathbf1_{t=u}R_{\alpha\beta}.
\]

主计算辅助噪声独立；另对同拍本地同坐标加入ρ√(RηR₁,cc)，ρ=0.5，并以另一独立Gaussian原始量图核验。没有给各节点分别抽独立过程噪声。

当前数值Θ正定。若精确复制导致协方差奇异，必须先核噪声支撑：均值偏移落在支撑外会产生不同的仿射支撑，可确定区分；落在支撑内才可在正特征值子空间白化。直接用伪逆忽略零方差差值会漏掉保护前精确副本的信息。

取Z为O左零空间满行秩基，Θ=ZΩZᵀ，V=chol(Θ)⁻¹Z。名义白化残差v=VY服从N(0,I)，攻击均值Uξ、U=VD。原单源奇偶评分‖v‖²为χ²_d(Λ)，五拍d=8，

\[
\Lambda=\xi^\top U^\top U\xi,\qquad
P_D=\overline F_{\chi_d^2(\Lambda)}(\chi^2_{d,1-\alpha}).
\]

标准轨迹匹配GLRT用‖P_Uv‖²，df=rank U=2。闭式备择ξ无约束，评价攻击仍受E限制；不是新“能量GLRT”。Gaussian水印条件下U使用实际秘密M；无条件功效必须对尾函数积分，不能用EΛ替代Λ。

## 2. 响应和独立复核

B0无水印完整内部窗ZOⱼ=0。B1加入可信本地行后U满列秩；额外可信样本只增加标准Gaussian信息。固定rank2和单次阈值时逐方向功效不减，不代表每条实际报警轨迹都提前。

独立实现用

\[
D^\top[\Omega^{-1}-\Omega^{-1}O(O^\top\Omega^{-1}O)^{-1}O^\top\Omega^{-1}]D
\]

复算UᵀU，与奇偶白化的最大差为'''+f"{checks['errors']['GLS_info']:.3g}"+r'''。共享原始量图、正常协方差、截断模板及P3射线交叉核验合计700项通过，见[独立结果](independent_checks.json)。这不是第三方审证。

## 3. 首个完整五拍：单次FA=5%

攻击开始7，L10，首批7—11。P3要求原估计器十个后验时刻最大期望偏移平方≥0.1，编码域E≤10。各防御重新选ξ。

'''+table(['模型','方案','P3功效','P3能量','该方案P1方向功效'],powerrows)+r'''

B3_coord0/1分别为C₁/C₂，固定相位0。非水印采用解析有限驻点表征＋浮点评价；B2是128条既有秘密路径条件功效均值、181方向和局部细化，非认证无条件积分或全局概率证书。

C₁相对B1的P3增加：不稳定模型约0.336个百分点、稳定模型约0.345个百分点。这是归一化噪声下小幅单次收益，不是可靠报警或硬件实测。B2使用自己的实际期望损害矩阵，不能据表宣称统一优劣或同资源支配。

主程序还核48种Rη、相关系数和延迟组合。此首批在10采样、11结束，0/1拍延迟都能收到，因而此项相同；不能据此称延迟无关。所有敏感性点保存于batch_power_results.json，未作新调度。

## 4. 未知起点、重叠扫描与总误报

监测0—31，窗口结束4—31共28个；真实起点评价7—11五相位，检测器不知道真实起点。L10≥窗长5，所有非空截断由5个受攻后缀＋4个真前缀共9个模板覆盖；A可逆，内部偏移吸入自由ξ。另一实现核验所有实际重叠位置。

B0阈值χ²₈(1−.05/28)；B1/B2/B3各窗9个rank2标准匹配检验，单项阈值χ²₂(1−.05/(28×9))。这是同一全事件FA≤5%的保守基线，未把实际FA说成恰好5%，也未优化阈值分配。

令全期独立原始Gaussian量为ω，窗口vₕ=Lₕω，则联合协方差块为LₕLℓᵀ。相邻窗口交叉块范数约2.4495（B0）或4.6904（B1），显然不独立；相邻完整联合协方差保存NPZ，其余窗口对可由同一原始量图复算。

设pₕ为截止k各单项报警边缘概率，

\[
\max_h p_h\le P(\tau_A\le k)\le\min(1,\sum_h p_h).
\]

固定ξ/M的危险时刻τD由原估计偏移给定。严格τD<τA要求截至τD**包含当拍**未报警：

\[
1_{\tau_D<\infty}\max(0,1-\sum_{e_h\le\tau_D}p_h)
\le R\le
1_{\tau_D<\infty}(1-\max_{e_h\le\tau_D}p_h).
\]

无危险时事件为0。数学不等式成立不等于浮点端点已向外舍入认证。B2又对旧128秘密取均值，秘密分布积分误差未认证。

下表只列Ta7的各方案P1；完整五相位P1/P3、首报CDF界及风险搜索见JSON。

'''+table(['模型','方案','危险先报风险界/秘密均值诊断','危险可达比例','截断生存和界'],risks)+r'''

“截断生存和”为Σ(k=Ta…31)P(τA>k)，包含无报警删失，不是检出条件平均延迟。假警可能在攻击前；未条件于监测期此前无报警。

风险搜索为181方向×能量比例(.25,.5,1)，各方案/起点独立重选。最大下界提供合法攻击见证；最大上界只针对所搜有限集，**不是全ξ最坏风险上界**。界宽且重叠，未证明新增辅助传感降低最坏危险先报风险。

## 5. 明确保留的缺口

没有精确多窗首报概率、历史自适应起点的选择后校准、全方向最坏风险或无限在线ARL保证。28窗5%预算不能无限重复；长期FWER可另设可加总αₕ，或另定ARL目标，本轮未设计长期检测器。

无隔离动作；首报前风险仅在固定原估计轨迹评价。若未来报警改变接纳/K/G，必须重新用完整数据依赖误差模型，不能将固定F结果冒充闭环性能。KL只作分布指标，未用作不可实现在线评分。
''')
    write('04_标准联合检测强基线.md',r'''# 标准联合检测与固定采样强基线

'''+common+r'''## 1. 组间信息

|组|检测信息|新增编码/采样|主估计器|
|---|---|---|---|
|B0|受攻源0的原时间奇偶|无|共G全接纳|
|B1|本地1＋邻居0/2原始报文的标准轨迹GLRT|无|不变|
|B2|B1及合法秘密M|原Gaussian λ=q=.5|不变|
|B3|B1＋C₁或C₂|每5拍1可信标量，固定相位0|不变|
|B4|相同每10拍2次预定采样类|5/5周期、五种相位|不变|

目标链路已知为0，健康2属于合法信息。未知攻击归因、两链协调攻击未由本表认证。

## 2. 不能把窗口GLRT称为全信息最优

B1主表消除任意窗口初态干扰，是标准匹配GLRT，不是未知方向的UMP。冻结先验x₀~N(0,I)与历史可用时，还能使用更多信息。

因此补查标准**全历史GLRT**：使用0—11原始量测，Σ包含初态、共享过程及量测噪声，直接白化再投影攻击子空间，仍χ²₂。辅助可含0、5、10三次采样。它是审查强基线附录，没有改事前参数或开发新检测器；两个模型G_prior−G_B1最小特征值均约0.04。

'''+table(['模型','全历史标准基线','P3功效','P3能量'],priorrows)+r'''

全历史C₁相对对应B1仍仅增加约0.335/0.342个百分点。小幅收益不依赖故意弱化B1。全历史未知起点最优扫描/阈值分配未完成，不将已知起点单次表冒充该全局结果。

## 3. B4的有限类覆盖命题

仅对**五拍窗口内数据、未知初态任意**的匹配检验，10拍公开周期恰2次辅助采样，目标为遍历全部起点相位的最小P3边缘功效，5/5间隔周期不劣于任意非均匀两采样表。

**证明。** 两循环间距之和10；不均时最大间距≥6，存在五拍空窗，检验退化为B1，最小功效不超过B1。5/5使每个五拍窗恰有一个辅助样本。Gaussian条件化分解增加非负信息；消除初态干扰后的最小二次证据亦不减，原攻击子空间rank2保持、相同阈值下逐方向功效不降。故均匀周期最小功效≥B1；周期相移只是起点相位置换。证毕。

'''+table(['模型','候选','五相位最低P3','仅对齐首批的最好相位'],phases)+r'''

这是基础最大间距覆盖与标准信息单调性。不是新采样算法，未搜索自适应策略。该最优性**不包括**使用窗口前辅助历史的全先验检验、一般加权采样或危险先报风险最优性。不能把B4叫作一般全信息最优调度。

## 4. 正常精度及资源

辅助量测不进滤波且无门控，各组健康估计逐路径相同。目标节点7—31平均正常总状态MSE：

'''+table(['模型','正常MSE','关系'],normalrows)+r'''

原Gaussian理想同步无攻击时精确恢复量测，不改变此MSE。这里M=.5I+ββᵀ、β~N(0,.5I₂)，有E[M]=I及E[M⁻²]=2.538544683758135 I，所以正常编码信号平均能量为未编码的2.53854468倍；解码范数无有限峰值，逆编码增益≤2。旧0.634636是另一缩放参数下的数值，不能移植本表。

B3新增采样硬件/能耗另计，与B2不是同资源支配。新增信息、检测器作用、实际损害与编码成本分开报告。
''')
    write('05_最坏攻击方向与资源约束.md',r'''# 攻击响应、实际损害与最坏方向

'''+common+r'''## 1. 原更新和后验时序

\[
\hat x_{1,k}=A\hat x_{1,k-1}+K_1(y_{1,k}-H_1A\hat x_{1,k-1})
+G_1\sum_{j=0,2}(y_{j,k}-H_jA\hat x_{1,k-1}),\quad
F=(I-K_1H_1-G_1(H_0+H_2))A.
\]

采用0ea9c7d原共G参考及显式Niu聚合MMSE补充设计；不是原作者50节点配置。攻击前偏移为0，正常/受攻配对同物理噪声和水印，无门控。b=受攻估计−健康估计：

\[
b_{T_a+t}=Fb_{T_a+t-1}+G_1M_{T_a+t}H_0A^t\xi,\quad
B_t(M)=\sum_{r=0}^tF^{t-r}G_1M_{T_a+r}H_0A^r.
\]

首包b_Ta=G₁M_TaH₀ξ，不额外乘F。辅助量测只进检测，不改变该式。固定F不得用于数据依赖隔离后的实际两闭环之差。

## 2. 同一能量与实际损害

\[
\xi^\top W_{10}\xi=\sum_{r=0}^9\|H_0A^r\xi\|^2\le10.
\]

非水印Sₜ=BₜᵀBₜ；Gaussian取Cᵣ=F^(t−r)G₁、Oᵣ=H₀Aʳ，λ=q=.5时

\[
S_t^\pi=E[B_t(M)^\top B_t(M)]
=B_t(I)^\top B_t(I)+q^2\sum_{r=0}^tO_r^\top
(C_r^\top C_r+\operatorname{tr}(C_r^\top C_r)I)O_r .
\]

含解码随机放大，不能用无水印S代替。P1=E maxₜλmax(Sₜ,W)是最大**期望**时刻损害，非E[maxₜ随机损害]或逐实现最大值。

'''+table(['模型','组','攻击内后验t','最大期望偏移²','ξ'],p1rows)+r'''

检测-only的非水印组P1完全相同；没有为新传感重造同一个解。Gaussian无有限逐实现损害包络，有限秘密样本不能冒充支撑最大值。辅助传感自身不抑制污染，只提供证据；本轮没有追加警报动作。

P3=min P_D(ξ)，约束ξᵀWξ≤10且maxₜξᵀSₜξ≥.1。Gaussian使用自己的Sπ，可行集随防御变。补充相同全能量集合的最低证据：

'''+table(['模型','方案','E=10最小Λ / B2最小EΛ','单次df'],ncprows)+r'''

此表不是检出概率。固定方向u最小可行尺度r²=D/maxₜuᵀSₜu，不必用满能量。可行集是各时刻损害二次约束集的并。非水印二维问题：每个t最小化(G,Sₜ)广义Rayleigh商，约束uᵀ(Sₜ−D/E W)u≥0，最优必在广义特征驻点或能量边界射线。枚举后取最低；当前各Sₜ正定，退化模型要另作商空间分析。

采用解析有限候选＋浮点评价，另一721射线实现交叉核验；不称区间最优值证书。Gaussian真实混合功效用181方向、局部细化及128旧秘密路径均值，不称全局概率最优/认证积分。

## 3. 停止尾部

SymPy从保存K/G/H/A小数重新组成F，两模型精确认证tr(FᵀF)<1/25，故冻结十进制模型下‖F‖₂<1/5。攻击停止后任意r≥0有‖b_(stop+r)‖≤5^(−r)‖b_stop‖，最后受攻后验之后不会产生更大峰值。十拍峰值未遗漏停止后的放大。

此性质依赖当前强收缩F，不能由一般Schur稳定直接得出；非正规弱收缩需另核尾部。stop_tail_certificate是冻结输入数学证书，不是DARE/设备误差认证。

## 4. 风险口径

- 实际偏移：bₖ(M,ξ)；固定线性配对时不依赖物理噪声。
- 正常MSE：报告04协方差迹，各组相同。
- 无条件受攻MSE：确定ξ、秘密独立于物理噪声且名义误差零均值时，等于正常MSE＋ξᵀSπξ。
- 未报警条件MSE：条件化引入相关与交叉项，上述加法不成立；本轮未求。

危险τD按攻击诱导偏移平方≥.1定义，不是真实总估计误差越界。报告03提供相关事件界和有限参数重优化；不把未达危险门槛解释成100%检出。主P3均可行。

## 5. 更强知识边界

若知道当前M且允许改攻击为M⁻¹H₀Aᵗξ，解码可重获自主轨迹，编码能量矩阵亦改变；这已超出当前预定编码域自主攻击，不混入P1/P3表。可信本地/辅助仍能提供一致性证据，但不自动保证危险前报警；若这些可信源也被同步篡改，原核条件的权限前提失效。
''')
    out=dict(date='2026-11-16',decision='C',G2_started=False,reports=7,
        structural_new_capability_over_B1=False,physical_hardware_verified=False,
        worst_pre_alarm_risk_improvement_proved=False,new_physical_noise_samples=0,
        missing_fulltext=['FPN30'],finite_noise_gain='small and explained by standard Gaussian information/fixed sampling')
    (H/'research_decision.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
