"""Existing scaled orthogonal references at original Gaussian signal energy.

No new watermark distribution, direction optimization, QMC or noise simulation.
Old model/moment data are read only. Every output is next to this file.
"""
from pathlib import Path
from fractions import Fraction as Fr
from math import factorial
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.stats import norm,chi2,ncx2

H=Path(__file__).resolve().parent
P=H.parent
OLD=P/'watermark_boundary_final_20261114/power_comparison.py'
PREFIX=P/'danger_aware_watermark_screen_20261114/danger_detection_comparison.json'
PREVIOUS=P/'watermark_boundary_final_20261114/baseline_comparison.json'
MODEL=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('read_only_scaled_reference_matrices',OLD)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
prefix=json.loads(PREFIX.read_text(encoding='utf-8'))
previous=json.loads(PREVIOUS.read_text(encoding='utf-8'))
E=10.;D=.1
v,err=quad(lambda u:np.exp(-u)/(1+2*u)**2,0,np.inf,epsabs=1e-13)
nu=.5+.5*v;kappa=nu**(-.5);scale=kappa*kappa
SCHEMES=['whole_sign','coordinate_sign','Haar_SO2','public_scalar_alternation','identity']

def write_report(out):
    names={'original_Gaussian':'原Gaussian','scaled_whole_sign':'缩放整体符号',
        'scaled_coordinate_sign':'缩放逐坐标符号','scaled_Haar_SO2':'缩放独立SO(2)',
        'scaled_public_scalar_alternation':'缩放公开整体交替','scaled_identity':'常量κ*I'}
    q=out['old_As_probe_scaled_resource_diagnostic']
    lines=['# 原Gaussian平均信号能量下的简单缩放正交强基线', '',
        '日期：2026-11-14。仅用于已停止候选的资源定位。固定原邻居量测求和/common-G估计器、原编码域自主FDI、L10总能量10、D=.1，不修改融合、检测器或通信时序；不提出新水印。', '',
        '**结论：有限解码增益且匹配原Gaussian平均正常信号能量的性质，已经能由既有单位正交编码作常量缩放实现。** 这是必要的简单强基线，不是独立防御贡献。原Gaussian与该基线仍有不同的检测—损害和峰值/随机源成本，不能写成全性能支配，也不能据此否定所有κ>κ*的有界编码空间。', '',
        '## 1. 资源映射', '',
        '原Gaussian解码为M=I+ββᵀ、β∼N(0,I₂)，编码为C=M⁻¹。令r²=||β||²。M⁻²的两个特征值为1、(1+r²)⁻²，由各向同性', '',
        r'\[ E[C^\top C]=\nu I,\quad\nu=\tfrac12+\tfrac12\int_0^\infty\frac{e^{-u}}{(1+2u)^2}\,du. \]', '',
        f'数值ν≈{nu:.15f}。原Gaussian编码算子峰值为1，解码算子无有限峰值。上述数值来自确定性一维求积，求积误差估计不是严格区间证书；结构上ν∈(1/2,1)。', '',
        '对任何既有正交O，取', '',
        r'\[ \kappa_* =\nu^{-1/2},\qquad M=\kappa_*O,\quad C=\sqrt\nu\,O^\top. \]', '',
        f'κ*≈{kappa:.15f}，κ*²=1/ν≈{scale:.15f}。逐实现CᵀC=νI，故对任何正常量测，||Cy||²=ν||y||²。编码峰值√ν，解码峰值κ*，每拍仍传同一二维量测且不增加等待；理想同步精确解码下正常估计轨迹不变。', '',
        '|资源|原Gaussian|简单缩放正交|',
        '|---|---|---|',
        '|正常信号平均平方能量系数|ν|ν，且逐实现相等|',
        '|编码算子峰值|1|√ν|',
        '|解码算子峰值|无界|κ*|',
        '|随机参数|每拍二维Gaussian参数|整体/坐标符号、角参数或公开确定序列，分别计成本|',
        '|报文维数/发送拍数|保持原协议|保持原协议|', '',
        '相同平均信号能量不等于相同无线能耗、密钥成本、量化误差或全部实现资源。角参数与符号位的实现成本也不能混同。', '',
        '### 最小解码峰值的准确边界', '',
        '在要求E[CᵀC]=νI且||M||≤κ的资源类中，CᵀC≽κ⁻²I，所以κ≥ν⁻¹ᐟ²。若κ=κ*，非负矩阵CᵀC−νI的期望为0，故CᵀC=νI几乎处处，亦即上述缩放正交类。该边界依赖明确的矩阵能量约束；不是仅凭一个经验标量能量值即可断言的全类最优性。κ>κ*仍允许其他结构，本文没有将其全部排除。', '',
        '## 2. 期望检测与损害都按1/ν缩放', '',
        '对相同预定攻击ξ和相同单位正交实现O，全部解码攻击、偏移和奇偶残差都乘κ*。因此完整窗期望非中心矩阵及每个前缀损害矩阵分别为Gscaled=Gunit/ν、Sscaled,k=Sunit,k/ν。P1终端最大期望损害及supξ maxₖE||bₖ||²同样乘1/ν，广义特征方向保持。这里的max E不是E max，不提供首次危险风险。', '',
        '下表仍使用原编码域总能量≤10、危险门槛.1。EΛ不是实际检出率；本轮没有增加实际Pd角度优化。', '',
        '|模型/已有参考|最弱首5窗EΛ（能量10）|P1终端期望损害|15拍supξ max E损害|终端P3-phys可行|',
        '|---|---:|---:|---:|---|']
    for name,md in out['models'].items():
        for s,r in md['rows'].items():
            lines.append(f"|{'Au' if name=='unstable' else 'As'} / {names[s]}|{r['min_expected_first5_ncp_E10']:.6f}|{r['P1_max_expected_terminal_damage']:.6f}|{r['max_over_times_and_attack_directions_expected_damage']:.6f}|{'是' if r['P3_actual_expected_terminal_damage_feasible'] else '否'}|")
    lines+=['', 'As缩放参考的终端P3-phys仍不可行，但早期偏移可超过D；不能解释为100%检出或事件安全。常量κ*I在完整内部攻击窗仍满足ZOξ=0，因而保持全部原自主攻击的内部零残差；启动/停止瞬态另算。', '',
        '### 一项已有机制的二阶折中', '',
        '令X≽0为二维矩阵，Gaussian同拍方差算子为V(X)=X+tr(X)I。其满足V(X)≽2X、V(X)≽2diag(X)、V(X)≽tr(X)I。第一式来自tr(X)I≽X；第二式差为JXJ，其中J交换两坐标；第三式差为X。', '',
        '对iid整体符号、逐坐标符号和SO(2)，Gaussian前缀损害矩阵为4S₀,ₖ加各拍V(X)项，而相应零均值参照没有跨拍均值项。因此每个前缀都有S_Gaussian,k≽2νS_scaled,k。完整受攻窗中常量均值项被ZO=0消去，同样有G_Gaussian≽2νG_scaled。二者都更大，准确定位是检测响应与污染的折中。公开交替含确定跨拍项，不套用上述iid式。', '',
        '该Loewner关系不能排序真实检出率：两种Λ分布甚至可有相同期望10，却有不同功效。例如Λ≡10与P(Λ=0)=.99、P(Λ=1000)=.01均有均值10；后者功效≤.99/300+.01，而已有Poisson混合正项有理界证明前者严格更大。脚本保存精确分数核对。它是逻辑分布反例，不是新构造的同成本水印。', '',
        '## 3. P3强度抵消与预算边界', '',
        '对一般常量c>0，M改为cM并把ξ改为ξ/c，全部解码攻击、残差和偏移路径逐实现相同；新攻击编码能量是原来的1/c²。严格的优化恒等式是', '',
        r'\[ v(c\pi;D,E)=v(\pi;D,c^2E) \]', '',
        '其中v指P3-phys的实际损害约束及同一单次功效目标。它不是v(cπ;D,E)=v(π;D,E)的无条件定理。当前κ*>1，原可行攻击映射到ξ/κ*后仍可行；逆向映射可能超预算。脚本用既有P1方向核查15个偏移状态完全相同，同时记录逆向能量10/ν>10。若能量约束活跃，最优功效仍可能改变。共同无水印危险集合/P3-pot的门槛会随ξ缩放而变化，不能套用该P3-phys恒等式。', '',
        '## 4. 首包边界必须随资源重算', '',
        '仍取旧As有理初值ξ=(−1.89,.42)，原总编码能量9.96747434605不变。单位正交类的首包偏移平方下界7497/32000=.23428125；缩放后为', '',
        f"`(.23428125)/ν ≈ {q['scaled_first_bias_squared_lower']:.10f} > .1`。因此首包仍必危险，但检测平移也变大。", '',
        '在首包单次误报≤1/300、监测从攻击起点重置、外生O与物理噪声独立的条件下，授予真实状态、固定ξ、O及不包含源当前未篡改量测信息副本/后代的侧信息。oracle剩余独立噪声为N(0,.5I)，已知简单备择非中心参数为7.497/ν。NP界给出', '',
        f"- λoracle≈{q['oracle_known_simple_alternative_lambda']:.9f}，最大功效≈{q['oracle_first_test_power_upper']:.9f}，漏警下界≈{q['oracle_first_packet_missed_alarm_lower']:.9f}。",
        f"- 普通parity的首包Λ上界(.6·3.7485)/ν≈{q['ordinary_parity_lambda_upper']:.9f}，相应漏警下界诊断≈{q['ordinary_parity_first_miss_lower']:.9f}。",
        f"- 当前空间检测Λ上界(72/61·3.7485)/ν≈{q['current_spatial_lambda_upper']:.9f}，相应漏警下界诊断≈{q['current_spatial_first_miss_lower']:.9f}。", '',
        '**旧单位正交oracle风险>.48不能原数迁移至本同Gaussian平均能量类。** 本节概率小数均未作区间认证，不覆盖其他首包误报分配、全部κ>κ*、一般有界可逆编码或原Gaussian。它也不是新的检测器。', '',
        '## 5. 研究取舍与交付', '',
        '有限解码峰值、单拍可逆、同Gaussian平均正常信号能量的基础性质，简单缩放符号/正交参考已经实现。因此仅提出“有界归一化水印”不能回应新机制验收。另一方面，本报告没有证明缩放符号支配Gaussian的真实功效或事件风险，也没有证明全部有界分布不存在进一步折中。没有可靠的新机制及真实性能增量时，停止当前候选，保留为原Gaussian的资源与性能定位材料。', '',
        '`scaled_orthogonal_reference.py`只读调用旧矩阵接口，不运行旧main；JSON保存输入SHA256、15个前缀矩、固定既有方向诊断、P3映射误差及逻辑反例。本轮新增参数搜索、角优化、QMC节点和物理噪声样本均为0。保存矩阵及ν/概率小数仍属浮点诊断，非原始物理模型全链区间认证。']
    (H/'简单缩放正交强基线_解析核验.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def spectrum_metrics(W,Ss,G):
    terminal=eigh(Ss[9],W,eigvals_only=True)
    per=[float(E*eigh(S,W,eigvals_only=True)[-1])for S in Ss]
    return dict(min_expected_first5_ncp_E10=float(E*eigh(G,W,eigvals_only=True)[0]),
        P1_max_expected_terminal_damage=float(E*terminal[-1]),
        P3_actual_expected_terminal_damage_feasible=bool(E*terminal[-1]>=D),
        max_over_times_and_attack_directions_expected_damage=max(per),
        argmax_bias_state=int(np.argmax(per))+1,
        per_time_max_expected_damage=per,
        prefix_moment_matrices=Ss.tolist(),first5_mean_ncp_matrix=G.tolist(),
        peak_order='sup_xi max_k E||b_k||^2; not sup_xi E max_k ||b_k||^2')

def run():
    out=dict(date='2026-11-14',purpose='existing references only; original Gaussian average normal encoded signal energy',
        source_sha256={str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in [OLD,PREFIX,PREVIOUS,MODEL]},
        nu=nu,nu_quadrature_error_estimate=err,kappa_star=kappa,second_moment_scale=scale,
        nu_not_interval_certified=True,original_estimator_and_fusion_unchanged=True,
        no_new_algorithm=True,no_parameter_search=True,no_direction_optimization=True,
        physical_noise_samples=0,new_QMC_nodes=0,
        resource_original_Gaussian=dict(M='I+beta beta^T, beta~N(0,I2)',mean_encoder_gram='nu I',encoder_peak=1.,decoder_peak='unbounded'),
        resource_scaled_orthogonal=dict(M='kappa_star O',C='sqrt(nu) O^T',mean_encoder_gram='nu I',
            signal_energy_factor_pathwise=nu,encoder_peak=float(np.sqrt(nu)),decoder_peak=kappa,
            no_added_packets=True,no_added_waiting=True),
        P3_scaling_identity='v(c*pi; D,E) = v(pi; D,c^2 E); matched attack xi/c uses energy e/c^2',
        models={})
    for name,md in prefix['models'].items():
        A=np.array(md['A']);F=np.array(md['F']);Ggain=np.array(md['G']);W=np.array(md['W'])
        rows={}
        # Independent read-only matrix interface check for the saved terminal/G.
        W2,Sg,S0,Gg=base.analytic_matrices(A,F,Ggain,'Gaussian')
        assert np.max(abs(W-W2))<1e-12
        gauss_prefix=np.array(md['schemes']['Gaussian']['prefix_expected_moment_matrices'])
        assert np.max(abs(gauss_prefix[9]-Sg))<1e-12
        rows['original_Gaussian']=spectrum_metrics(W,gauss_prefix,Gg)
        diagnostics={}
        for scheme in SCHEMES:
            _,Sold,_,Gold=base.analytic_matrices(A,F,Ggain,scheme)
            Ss=np.array(md['schemes'][scheme]['prefix_expected_moment_matrices'])
            assert np.max(abs(Ss[9]-Sold))<1e-12
            scaled=scale*Ss;Q=scale*Gold
            row=spectrum_metrics(W,scaled,Q)
            row.update(reference='kappa_star times existing '+scheme,unit_reference_metrics=spectrum_metrics(W,Ss,Gold))
            rows['scaled_'+scheme]=row
            # Fixed existing P1/weak-danger directions only; no new search.
            fixed={}
            for label,xi in [('no_watermark_peak_P1',md['geometry']['peak_P1']['initial_state']),
                             ('whole_sign_danger_weak',md['schemes']['whole_sign']['probes'][-1]['danger_set_minimum']['initial_state'])]:
                x=np.array(xi);per=np.einsum('i,kij,j->k',x,scaled,x)
                fixed[label]=dict(initial_state=x.tolist(),encoded_energy=float(x@W@x),
                    max_expected_damage=float(max(per)),expected_terminal_damage=float(per[9]),
                    expected_ncp=float(x@Q@x),per_time_expected_damage=per.tolist())
            diagnostics[scheme]=fixed
        # Scaling identity checked for all15 maps, one saved public word/probe.
        xi=np.array(md['geometry']['peak_P1']['initial_state']);unit=np.zeros(2);scaled_bias=np.zeros(2)
        max_path_error=0.
        for t in range(15):
            unit=F@unit;scaled_bias=F@scaled_bias
            if t<10:
                O=(-1.)**t*np.eye(2)
                unit+=Ggain@O@np.linalg.matrix_power(A,t)@xi
                scaled_bias+=Ggain@(kappa*O)@np.linalg.matrix_power(A,t)@(xi/kappa)
            max_path_error=max(max_path_error,float(np.max(abs(unit-scaled_bias))))
        assert max_path_error<1e-12
        out['models'][name]=dict(W=W.tolist(),S0_terminal=S0.tolist(),
            rows=rows,fixed_direction_diagnostics=diagnostics,
            scaling_matched_path_error=max_path_error,
            matched_scaled_attack_energy=float((xi/kappa)@W@(xi/kappa)),
            inverse_mapping_energy_of_unit_fullE_attack=float((kappa*xi)@W@(kappa*xi)))
        loewner={}
        for scheme in ['whole_sign','coordinate_sign','Haar_SO2']:
            scaledS=np.array(rows['scaled_'+scheme]['prefix_moment_matrices'])
            scaledG=np.array(rows['scaled_'+scheme]['first5_mean_ncp_matrix'])
            ds=gauss_prefix-2*nu*scaledS;dg=Gg-2*nu*scaledG
            sm=min(float(np.linalg.eigvalsh(M)[0]) for M in ds)
            gm=float(np.linalg.eigvalsh(dg)[0])
            assert sm>=-1e-12 and gm>=-1e-12
            loewner[scheme]=dict(min_over_prefixes_eigenvalue_of_Gaussian_damage_minus_2nu_scaled=sm,
                min_eigenvalue_of_Gaussian_mean_ncp_minus_2nu_scaled=gm,
                analytic_statement='Gaussian S_k >= 2nu scaled_reference S_k; complete-window Gaussian G >=2nu scaled_reference G',
                actual_Pd_order_not_implied=True)
        out['models'][name]['moment_tradeoff_diagnostics']=loewner
    # Resource-matched interpretation of the old single-packet counterexample.
    # This is a standard known-simple-alternative NP oracle, not a new test.
    norm2=float(Fr(7497,2000));old_bias_lower=float(Fr(7497,32000))
    oracle_lambda=2*norm2*scale;z=float(norm.isf(1/300))
    parity_lambda_upper=.6*norm2*scale
    # Current spatial covariance diag(221/242,61/72); max inverse=72/61.
    spatial_lambda_upper=float(Fr(72,61))*norm2*scale
    out['old_As_probe_scaled_resource_diagnostic']=dict(initial_state=[-1.89,.42],
        original_encoded_energy=9.96747434605,encoded_energy_unchanged_when_only_decoder_scaled=True,
        squared_norm=norm2,old_unitary_first_bias_squared_lower=old_bias_lower,
        scaled_first_bias_squared_lower=old_bias_lower*scale,
        exact_definition_kappa='[1/2+1/2 integral_0^infinity exp(-u)/(1+2u)^2 du]^(-1/2)',
        oracle_known_simple_alternative_lambda=oracle_lambda,
        per_first_test_alpha=1/300,oracle_first_test_power_upper=float(norm.cdf(np.sqrt(oracle_lambda)-z)),
        oracle_first_packet_missed_alarm_lower=float(norm.cdf(z-np.sqrt(oracle_lambda))),
        ordinary_parity_lambda_upper=parity_lambda_upper,
        ordinary_parity_first_power_upper=float(ncx2.sf(chi2.isf(1/300,8),8,parity_lambda_upper)),
        ordinary_parity_first_miss_lower=float(ncx2.cdf(chi2.isf(1/300,8),8,parity_lambda_upper)),
        current_spatial_lambda_upper=spatial_lambda_upper,
        current_spatial_first_power_upper=float(ncx2.sf(chi2.isf(1/300,2),2,spatial_lambda_upper)),
        current_spatial_first_miss_lower=float(ncx2.cdf(chi2.isf(1/300,2),2,spatial_lambda_upper)),
        numerical_only_no_interval_probability_certification=True,
        assumptions=['fixed As model and old autonomous probe','first-packet test alpha<=1/300',
            'external O independent of physical noise','oracle side information excludes source-current-noise descendants/copies',
            'monitor resets at attack start; no prior attack alarm'],
        old_unit_orthogonal_risk_gt_point48_not_transferred=True,
        not_all_bounded_invertible_encoders=True,not_original_Gaussian_bound=True,
        not_all_event5percent_allocations=True)
    # Equal mean Lambda is insufficient to order actual power. This is a
    # logical distribution counterexample, not a realizable watermark proposal.
    alpha=Fr(1,300)
    assert Fr(373)*Fr(3,8)**12<alpha # central df8 tail(24)<alpha, hence tau<24.
    lo=Fr(4,11)**17*Fr(5**5,factorial(5))*sum((Fr(12**j,factorial(j)) for j in range(9)),Fr(0))
    up=Fr(99,100)*alpha+Fr(1,100)
    assert lo>up
    out['equal_mean_not_equal_power_counterexample']=dict(
        law_A='Lambda=10 surely',law_B='Lambda=0 with probability99/100,1000 with1/100',
        both_expected_Lambda=10,power_A_rigorous_lower=str(lo),power_B_rigorous_upper=str(up),
        strict_power_difference_exact_rational_check=True,not_an_allowed_watermark_or_resource_comparison=True)
    (H/'scaled_orthogonal_reference.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_report(out)
    print('nu',nu,'kappa',kappa,'scale',scale)
    for n,m in out['models'].items():
        print(n)
        for s,r in m['rows'].items():print(s,r['min_expected_first5_ncp_E10'],r['P1_max_expected_terminal_damage'],r['max_over_times_and_attack_directions_expected_damage'],r['P3_actual_expected_terminal_damage_feasible'])

if __name__=='__main__':run()
