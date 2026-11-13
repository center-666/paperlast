"""Independent reconstruction of the frozen model and an exact DARE enclosure.

Does not import or run prior research scripts. Decimal F/g are compared only
after deriving gains from the declared exact A/C/Q/R. No physical simulation.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal
import json,hashlib
import mpmath as mp
import numpy as np
import sympy as sp
from scipy.linalg import solve_discrete_are

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
ROOT=PAPER.parents[1]
SOURCE=PAPER/'checks/verify_resource_attack_20261104.py'
FROZEN=PAPER/'checks/水印收束_理论与映射核验_20261105.json'
RAW=json.loads(FROZEN.read_text(encoding='utf-8'),parse_float=Decimal)
R=sp.Rational
A=sp.diag(sp.Matrix([[1,R(1,10)],[0,1]]),sp.Matrix([[1,R(1,10)],[0,1]]))
C=sp.Matrix([[0,0,1,0],[1,0,0,0],[1,0,0,0],[0,1,0,0]])
Q=sp.eye(4)/1000
Rm=sp.eye(4)/2
J=C.T*Rm.inv()*C
assert J==sp.diag(4,2,2,0)
O=sp.Matrix.vstack(C,C*A,C*A**2,C*A**3)
assert O.rank()==4 and Q.det()>0

mp.mp.dps=110
def mm(M):return mp.matrix([[mp.mpf(str(v.p))/mp.mpf(str(v.q)) for v in row] for row in M.tolist()])
Am, Jm=mm(A),mm(J)
Pm=mp.eye(4)
for iteration in range(1,5001):
    nxt=Am*(Pm**-1+Jm)**-1*Am.T+mp.eye(4)/1000
    difference=max(abs(v) for v in nxt-Pm)
    Pm=nxt
    if difference<mp.mpf('1e-95'):break
else:raise AssertionError('high-precision Riccati iteration did not converge')

def rationalize(M,digits):
    return sp.Matrix(M.rows,M.cols,lambda i,j:R(mp.nstr(M[i,j],digits)))
P0=rationalize(Pm,65)
K0=P0*C.T*(C*P0*C.T+Rm).inv()
F0=(sp.eye(4)-K0*C)*A
g0=K0[:,1]
L0=A*(sp.eye(4)-K0*C)

# The derivative of predictive Riccati map is L dP L'. A rational weighted
# enclosure is needed: using +/-eps*I would be invalid when ||L||_2>1.
Lm=mm(L0)
operator=mp.matrix(16)
for i in range(4):
    for j in range(4):
        for a in range(4):
            for b in range(4):
                operator[4*i+j,4*a+b]=(int(i==a and j==b)-Lm[i,a]*Lm[j,b])
rhs=mp.matrix([int(i==j) for i in range(4) for j in range(4)])
zv=mp.lu_solve(operator,rhs)
Zm=mp.matrix(4,4)
for i in range(4):
    for j in range(4):Zm[i,j]=(zv[4*i+j]+zv[4*j+i])/2
Z=rationalize(Zm,35)
eps=R(1,10**30)

def pd_certificate(M):
    assert M==M.T
    minors=[sp.factor(M[:k,:k].det()) for k in range(1,M.rows+1)]
    assert all(x>0 for x in minors)
    return [str(x) for x in minors]
def riccati(X):
    # Information form is algebraically identical to the declared DARE.
    return A*(X.inv()+J).inv()*A.T+Q

Zproof=pd_certificate(Z)
lo=P0-eps*Z; hi=P0+eps*Z
pd_certificate(lo)
sub=riccati(lo)-lo
sup=hi-riccati(hi)
subproof=pd_certificate(sub)
supproof=pd_certificate(sup)
# The two exact sub/supersolutions bracket the unique positive stabilizing
# DARE solution under Q>0 and observability. Trace bounds yield operator norms.
deltaP=eps*sp.trace(Z)
traceP0=sp.trace(P0)
assert traceP0>0
assert traceP0<R(1,8) and sp.trace(Z)<57
assert deltaP<R(1,10**26)
# ||C||_2<=||C||_F=2, ||A||_2<3, ||S(P)^-1||_2<=2,
# ||K0||_2<=4 tr(P0), hence a conservative Lipschitz bound.
deltaK=(1+8*traceP0)*4*deltaP
deltaF=6*deltaK
deltag=deltaK

Fs=sp.Matrix([[R(str(v)) for v in row] for row in RAW['F']])
gs=sp.Matrix([R(str(v)) for v in RAW['g']])
Fdiffsq=sum(x*x for x in F0-Fs)
gdiffsq=sum(x*x for x in g0-gs)
# Exact rational inequality rather than an unexplained floating comparison.
assert Fdiffsq<R(1,10**26) and gdiffsq<R(1,10**26)
Ftotal=R(1,10**13)+deltaF
gtotal=R(1,10**13)+deltag
assert Ftotal<R(1,10**6) and gtotal<R(1,10**6)
assert Ftotal<R(1,10**12) and gtotal<R(1,10**12)

Af,Cf,Qf,Rf=(np.array(M,float) for M in [A,C,Q,Rm])
Pscipy=solve_discrete_are(Af.T,Cf.T,Qf,Rf)
Kscipy=np.linalg.solve(Cf@Pscipy@Cf.T+Rf,Cf@Pscipy).T
Fscipy=(np.eye(4)-Kscipy@Cf)@Af
gscipy=Kscipy[:,1]
res=Pscipy-(Af@Pscipy@Af.T-Af@Pscipy@Cf.T@np.linalg.solve(Cf@Pscipy@Cf.T+Rf,Cf@Pscipy)@Af.T+Qf)
Pstar=np.array([[float(Pm[i,j]) for j in range(4)]for i in range(4)])

# Alternative local mean-summed architecture, matching the original uniform
# scalar-neighbor convention, is compared rather than conflated with stacked KF.
# Node3 is two-dimensional in this research model, so original equal-dimension
# aggregation is not directly defined. A scalar-only diagnostic discards v_x.
Csum=sp.Matrix([[0,0,1,0],[2,0,0,0]])
Rsum=sp.diag(R(1,2),1)
Ps=solve_discrete_are(Af.T,np.array(Csum.T,float),Qf,np.array(Rsum,float))
Ks=np.linalg.solve(np.array(Csum,float)@Ps@np.array(Csum.T,float)+np.array(Rsum,float),np.array(Csum,float)@Ps).T
Fsum=(np.eye(4)-Ks@np.array(Csum,float))@Af
gsum=Ks[:,1]

def strings(M):return [[str(x) for x in row] for row in M.tolist()]
def maxabs(M):return max(abs(v) for v in M)
sources=[SOURCE,FROZEN,PAPER/'checks/verify_active_witness.py',
         PAPER/'异构模型与主动可信验证_设计及最小核验_20261103.md',
         PAPER/'有限资源自主轨迹攻击_启动结束与基准核验_20261104.md',
         ROOT/'研究规划/第一篇小论文/secure distributed safety.tex']
result=dict(date='2026-11-13',role='adversarial physical model provenance audit',
 exact_parameters=dict(A=strings(A),C=strings(C),Q=strings(Q),R=strings(Rm),H_attack=[['1','0','0','0']],
    C_row_meanings=['node2 local y-position','node1 neighbor x-position (attacked)',
                    'node3 neighbor x-position','node3 neighbor x-velocity']),
 architecture='steady-state one-hop separately stacked neighborhood Kalman filter; not original Niu summed common-G estimator',
 predictive_DARE='P=A(P^-1+C^T R^-1 C)^-1 A^T+Q',
 update='xhat_k=(I-KC)A*xhat_(k-1)+K*ybar_k',
 state_bias='postmeasurement displacement b_(t+1)=F*b_t+g*s_t*H_t*xi; time shifted notation',
 gain_design=dict(K='P C^T(CPC^T+R)^-1',F='(I-KC)A',g='K column1 zero-based, attacked row2'),
 observability_rank=int(O.rank()),Q_positive_definite=True,
 high_precision=dict(decimal_digits=110,iterations=iteration,iteration_last_step=mp.nstr(difference,12),
    P_prior=[[mp.nstr(Pm[i,j],75)for j in range(4)]for i in range(4)]),
 rational_enclosure=dict(P0=strings(P0),Z=strings(Z),epsilon=str(eps),
    form='P0-epsilon Z <= unique stabilizing P* <= P0+epsilon Z',
    P_lower_positive=True,Z_leading_minors=Zproof,
    lower_subsolution_leading_minors=subproof,upper_supersolution_leading_minors=supproof,
    deltaP_operator_upper=str(deltaP),exact_arithmetic='SymPy Rational information-form Riccati map',
    simple_exact_constants=dict(trace_P0_upper='1/8',trace_Z_upper='57',
       deltaP_operator_upper='57/10^30',deltaK_operator_upper='456/10^30',
       deltaF_operator_upper='2736/10^30',delta_g_Euclidean_upper='456/10^30',
       frozen_vs_rational_center_F_and_g_error_upper='1/10^13'),
    proof='monotonicity and unique positive stabilizing solution (Q>0, observable A/C); explicit strict sub/super solution tests'),
 reconstruction=dict(Pscipy_minus_highprecision_max=float(np.max(abs(Pscipy-Pstar))),
    scipy_DARE_residual_max=float(np.max(abs(res))),
    scipy_gain_K=Kscipy.tolist(),F=Fscipy.tolist(),g=gscipy.tolist(),
    stored_F_difference_max=float(np.max(abs(Fscipy-np.array(Fs,float)))),
    stored_g_difference_max=float(np.max(abs(gscipy-np.array(gs,float).reshape(-1)))),
    F0_minus_frozen_F_Frobenius_sq_exact=str(Fdiffsq),
    g0_minus_frozen_g_Frobenius_sq_exact=str(gdiffsq),
    exact_F_operator_total_error_upper=str(Ftotal),
    exact_g_Euclidean_total_error_upper=str(gtotal),
    conservative_simple_both_error_upper='1/1000000000000',
    certified_inside_F_g_1e_minus6_matrix_neighborhood=True),
 original_estimator_comparison=dict(original_aggregation='col(H_i,sum_j H_j), common neighbor G; original equal m',
    research_heterogeneous_m_dimensions=[1,1,2],
    original_formula_not_directly_applicable=True,
    scalar_sum_diagnostic_F_difference_norm=float(np.linalg.norm(Fsum-Fscipy,2)),
    scalar_sum_diagnostic_g_difference_norm=float(np.linalg.norm(gsum-gscipy)),
    scalar_sum_diagnostic_discards_neighbor_velocity=True),
 mathematical_parameter_model_certified=True,
 original_paper_full_model_certified=False,engineering_parameter_uncertainty_certified=False,
 time_varying_gain_not_covered=True,packet_loss_and_gate_not_covered=True,
 sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sources],
 physical_trajectories=0,noise_samples=0,new_algorithm=False)
(HERE/'physical_origin_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(iterations=iteration,DARE_residual=result['reconstruction']['scipy_DARE_residual_max'],
                      stored_F_diff=result['reconstruction']['stored_F_difference_max'],
                      stored_g_diff=result['reconstruction']['stored_g_difference_max'],
                      rigorous_simple_error_upper='1e-12',inside_1e_minus6=True,
                      original_paper_architecture_match=False),ensure_ascii=False,indent=2))
