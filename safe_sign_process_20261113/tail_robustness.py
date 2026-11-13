"""Tail certificate and a small, preset structural sensitivity set.

Only this new directory is written. Primary certificates use exact rational
interpretations of the frozen JSON decimals. Sensitivity tables are floating
diagnostics, not independently certified physical-model uncertainty bounds.
"""
from pathlib import Path
from decimal import Decimal
from fractions import Fraction as R
from itertools import product
from math import lcm
from hashlib import sha256
import json
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent/'checks/水印收束_理论与映射核验_20261105.json'
source = json.loads(INPUT.read_text(encoding='utf-8'), parse_float=Decimal)
F = [[R(x) for x in row] for row in source['F']]
g = [R(x) for x in source['g']]
W = [[R(10),R(9,2)],[R(9,2),R(57,20)]]
Fnp=np.array(F,float); gnp=np.array(g,float); Wnp=np.array(W,float)
words=list(product([-1,1],repeat=10))
word_array=np.array(words)


def determinant(a):
    return sp.Matrix(a).det(method='domain-ge')


def positive_principal(a):
    minors=[determinant([row[:n] for row in a[:n]]) for n in range(1,len(a)+1)]
    assert all(x>0 for x in minors)
    return minors


def matmul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def gram(a):
    return [[sum(a[k][i]*a[k][j] for k in range(len(a)))
             for j in range(len(a[0]))] for i in range(len(a[0]))]


def run_exact_tail():
    d=1
    for x in sum(F,[])+g:d=lcm(d,x.denominator)
    Fi=[[int(x*d) for x in row] for row in F]
    gi=[int(x*d) for x in g]
    power=[[int(i==j) for j in range(4)]for i in range(4)]
    den=1
    norm_certificates=[]
    for n in range(1,11):
        power=matmul(Fi,power);den*=d
        q=gram(power)
        # ||F|| <= 1.004, prefix powers squared <=1.03, ||F^10||<=.998.
        choices=[R(103,100)]
        if n==1:choices.append(R(251,250)**2)
        if n==10:choices.append(R(499,500)**2)
        for c in choices:
            a=[[c.numerator*den*den*int(i==j)-c.denominator*q[i][j]
                for j in range(4)]for i in range(4)]
            minors=positive_principal(a)
            norm_certificates.append(dict(n=n,squared_norm_upper=str(c),
                principal_minor_signs=[1]*4,
                integer_matrix_sha256=sha256(str(a).encode()).hexdigest(),
                leading_principal_minors=[str(x) for x in minors]))

    # Recompute every word's B13, not using the saved old B array or counts.
    numerators=[[[0,0]for _ in range(4)]for _ in words]
    dpower=1
    for k in range(13):
        for wi,w in enumerate(words):
            old=numerators[wi]
            numerators[wi]=[[sum(Fi[i][j]*old[j][ell]for j in range(4))
                +(w[k]*gi[i]*[10,k][ell]*dpower if k<10 else 0)
                for ell in range(2)]for i in range(4)]
        dpower*=d
    den=10*dpower
    WM=[[200,90],[90,57]]
    stream=sha256()
    min_q00=None;min_det=None
    for I in numerators:
        q=gram(I)
        # .0088 W - B13^T B13 > 0, i.e. energy10 gives <.088.
        a=[[11*WM[i][j]*den*den-25000*q[i][j]for j in range(2)]for i in range(2)]
        minors=positive_principal(a)
        stream.update(str(a).encode())
        min_q00=min(min_q00,minors[0])if min_q00 is not None else minors[0]
        min_det=min(min_det,minors[1])if min_det is not None else minors[1]
    assert positive_principal([[W[i][j]-R(2,3)*int(i==j)for j in range(2)]for i in range(2)])
    safe=json.loads((HERE/'safe_words.json').read_text(encoding='utf-8'))
    assert safe['safe_count']==960
    assert R(safe['common_exact_margin'])>=R(1,4000)
    assert R(103,100)*R(88,1000)==R(1133,12500)
    tail_damage=R(1133,12500) # .09064
    tail_margin=(R(1,100)-tail_damage/10)*R(2,3)
    assert tail_margin>R(1,4000)

    # Uniform perturbation domain, A/H/W fixed. The inequalities below are
    # conservative rational arithmetic, not fitted to sensitivity outcomes.
    eps=R(1,10**6)
    f=R(251,250); fp=R(201,200); gg=R(7,125); hh=R(27,20)
    assert sum(x*x for x in g)<gg*gg
    assert R(181,100)<hh*hh
    assert 10*gg*hh*f**12<R(4,5) # ||B_t||<.8, t<=13, all words
    u=R(0)
    perturb_steps=[]
    for k in range(13):
        u=fp*u+eps*(R(4,5)+hh)
        perturb_steps.append(str(u))
    assert u<R(3,100000) # 3e-5
    qerror=2*R(4,5)*R(3,100000)+R(3,100000)**2
    assert qerror<R(1,20000)
    assert R(1,4000)-qerror>R(1,5000)
    assert R(499,500)+10*fp**9*eps<R(999,1000)
    assert R(203,200)+9*fp**8*eps<R(127,125) #1.016
    # sqrt(.088)+3e-5 sqrt(15) < .297, verified with loose rational roots.
    assert R(29665,100000)**2>R(88,1000)
    assert R(3873,1000)**2>15
    assert R(29665,100000)+R(3,100000)*R(3873,1000)<R(297,1000)
    perturbed_tail=(R(127,125)*R(297,1000))**2
    assert perturbed_tail<R(1,10)
    return dict(exact_frozen_decimal_model=True,not_DARE_interval=True,
       input_sha256=sha256(INPUT.read_bytes()).hexdigest(),
       denominator=str(d),norm_certificates=norm_certificates,
       B13_all_word_checks=1024,B13_energy10_damage_upper='11/125',
       B13_comparison_integer_formula='11*[[200,90],[90,57]]*den^2 - 25000 I^T I',
       B13_matrix_stream_sha256=stream.hexdigest(),
       infinite_tail_damage_upper=str(tail_damage),infinite_tail_damage_upper_float=float(tail_damage),
       infinite_tail_margin_lower=str(tail_margin),all_time_safe_word_margin='1/4000',
       common_margin_from_fresh_safe_words=True,
       perturbation=dict(F_operator_norm_error=str(eps),g_Euclidean_error=str(eps),
         A_H_W_fixed=True,prefix_B_norm_upper='4/5',prefix_B_error_upper='3/100000',
         prefix_quadratic_margin_loss_upper=str(qerror),
         remaining_prefix_margin_lower='1/5000',perturbed_F10_norm_upper='999/1000',
         perturbed_F_power_remainders_upper='127/125',
         perturbed_B13_state_norm_upper='297/1000',
         infinite_tail_damage_upper=str(perturbed_tail),
         infinite_tail_damage_upper_float=float(perturbed_tail),
         recursive_error_bounds=perturb_steps))


def floating_case(name,L=10,ratio=.01,Fm=None,gm=None,scaleA=1.,dt=.1,horizon=None):
    Fm=Fnp if Fm is None else Fm
    gm=gnp if gm is None else gm
    H=np.array([[scaleA**t,scaleA**t*dt*t]for t in range(L)])
    Wm=H.T@H
    ev,V=np.linalg.eigh(Wm)
    Winv=(V/np.sqrt(ev))@V.T
    ws=np.array(list(product([-1,1],repeat=L)))
    B=np.zeros((len(ws),4,2))
    N=L+21 if horizon is None else horizon
    maxima=np.zeros(len(ws));early=np.zeros(len(ws));margin=np.full(len(ws),np.inf)
    all_cap=[]
    for k in range(N):
        B=np.einsum('ij,wja->wia',Fm,B)
        if k<L:B+=gm[None,:,None]*ws[:,k,None,None]*H[k,None,:]
        Q=np.einsum('wki,wkj->wij',B,B)
        raw=np.linalg.eigvalsh(ratio*Wm[None,:,:]-Q)[:,0]
        margin=np.minimum(margin,raw)
        U=B@Winv
        cap=np.linalg.eigvalsh(np.einsum('wki,wkj->wij',U,U))[:,-1]
        maxima=np.maximum(maxima,cap)
        if k<L+3:early=np.maximum(early,cap)
        all_cap.append(float(10*cap.max()))
    safe=maxima<ratio
    return dict(name=name,L=L,ratio=ratio,meaning='fixed-map deterministic structural diagnostic; not a refitted physical KF',
                checked_k=list(range(N)),safe_count_in_checked_horizon=int(safe.sum()),
                early_safe_count=int((early<ratio).sum()),
                early_safe_but_later_danger_count=int(((early<ratio)&(~safe)).sum()),
                minimum_safe_margin_float=float(margin[safe].min())if safe.any()else None,
                safe_margin_quantiles_float=np.quantile(margin[safe],[0,.5,1]).tolist()if safe.any()else [],
                spectral_radius_float=float(max(abs(np.linalg.eigvals(Fm)))),
                F_norm_float=float(np.linalg.norm(Fm,2)),
                maximum_damage_by_k=all_cap,
                infinite_tail_certified=False,physical_trajectories=0)


def exact_nonnormal_counterexample():
    Fbad=[[R(49,50)*int(i==j)+int(i==0 and j==1)for j in range(4)]for i in range(4)]
    word=words[216]
    xi=[R(103,1000),R(-2013,1000)]
    energy=sum(xi[i]*W[i][j]*xi[j]for i in range(2)for j in range(2))
    assert energy<10
    B=[[R(0),R(0)]for _ in range(4)]
    early=[];terminal=None
    for k in range(59):
        B=matmul(Fbad,B)
        if k<10:
            B=[[B[i][j]+g[i]*word[k]*([R(1),R(k,10)][j])for j in range(2)]for i in range(4)]
        b=[sum(row[j]*xi[j]for j in range(2))for row in B]
        harm=sum(x*x for x in b)
        if k<13:early.append(harm);assert harm<R(1,10)
        if k==58:terminal=harm
    assert terminal>R(9,10)
    return dict(F='(49/50)I4 + e1 e2^T',spectral_radius='49/50',Schur=True,
                word_index=216,word=list(word),xi=[str(x)for x in xi],energy=str(energy),
                energy_float=float(energy),maximum_damage_k0_to12_float=float(max(early)),
                first_13_fixed_attack_safe_exact=True,
                damage_k58_exact=str(terminal),damage_k58_float=float(terminal),
                late_danger_exact=True,
                lesson='Schur stability alone does not make a finite k0..12 safety check a full-tail guarantee')


def main():
    exact=run_exact_tail()
    cases=[]
    for ratio in [.009,.01,.011]:cases.append(floating_case('ratio_'+str(ratio),ratio=ratio))
    for L in [9,11]:cases.append(floating_case('duration_'+str(L),L=L))
    for a in [.98,1.02]:cases.append(floating_case('A_scale_'+str(a),scaleA=a))
    for s in [.99,1.01]:cases.append(floating_case('g_scale_'+str(s),gm=s*gnp))
    cases.append(floating_case('F_plus_0.0001I',Fm=Fnp+.0001*np.eye(4)))
    cases.append(floating_case('stable_scalar_F',Fm=.95*np.eye(4)))
    cases.append(floating_case('unit_circle_F',Fm=np.eye(4)))
    nonnormal=.98*np.eye(4);nonnormal[0,1]=1
    cases.append(floating_case('Schur_nonnormal_F',Fm=nonnormal,horizon=130))
    counter=exact_nonnormal_counterexample()
    result=dict(date='2026-11-13',primary=exact,sensitivity_cases=cases,
                Schur_nonnormal_finite_horizon_failure=counter,
                model_perturbation_formula={
                  'B_error':'u_{t+1}<=(||F||+eps_F)u_t+eps_F||B_t||+eps_g||h_t||+(||g||+eps_g)eps_h_t',
                  'W_error':'eps_W<=sum_t(2||h_t|| eps_h_t+eps_h_t^2)',
                  'Q_error':'||(r Wprime-Bprime^T Bprime)-(r W-B^T B)||<=r eps_W+2||B||u+u^2',
                  'A_error':'eps_h_t<=||H|| t max(||A||,||Aprime||)^(t-1) eps_A, for fixed H'},
                physical_trajectories=0,random_search=False,
                large_parameter_search=False,
                preset_structural_sensitivity=True,
                sensitivity_preset_cases=len(cases))
    (HERE/'tail_robustness_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tail_upper=exact['infinite_tail_damage_upper'],
         robust_F_g_radius=exact['perturbation']['F_operator_norm_error'],
         counts=[(c['name'],c['safe_count_in_checked_horizon'],c['early_safe_but_later_danger_count'])for c in cases],
         nonnormal_counterexample=counter['damage_k58_float']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
