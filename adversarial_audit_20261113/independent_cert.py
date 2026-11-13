"""Independent reconstruction: exact closed-form convolution, not old recurrence.

Inputs are only original frozen F/g/W and H_t=[1,t/10]. No safe-word or old
certificate JSON is read until optional final comparison. SymPy Rational is
used to form F powers and every individual convolution coefficient. At each
time the resulting finite rational sum is independently integer-scaled to
certify principal minors. No original script is imported or run.
"""
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
from itertools import product
from functools import reduce
from math import gcd, log2
import hashlib
import json
import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

HERE=Path(__file__).resolve().parent
P=HERE.parent
INPUT=P/'checks/水印收束_理论与映射核验_20261105.json'
OUTPUT=HERE/'independent_cert.json'

def lcm(a,b): return a//gcd(a,b)*b

def bareiss_det(matrix):
    """Fraction-free exact determinant, including permutations if needed."""
    a=[list(map(int,row)) for row in matrix]
    n=len(a)
    if n==0:return 1
    sign=1; previous=1
    for k in range(n-1):
        if a[k][k]==0:
            swap=next((i for i in range(k+1,n) if a[i][k]),None)
            if swap is None:return 0
            a[k],a[swap]=a[swap],a[k];sign=-sign
        pivot=a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                numerator=a[i][j]*pivot-a[i][k]*a[k][j]
                assert numerator%previous==0
                a[i][j]=numerator//previous
            a[i][k]=0
        previous=pivot
    return sign*a[-1][-1]

def integer_matrix(exact):
    denominator=reduce(lcm,(int(v.q) for v in exact),1)
    num=[[int(exact[i,j]*denominator) for j in range(exact.cols)] for i in range(exact.rows)]
    return num,denominator

def gram4(num):
    return [[sum(row[i]*row[j] for row in num) for j in range(2)] for i in range(2)]

def q_integer(Bnum,Bden, c, W):
    # Exact Q=cW-B'B, independently integer-scaled.
    base=sp.Matrix([[c*W[i,j] for j in range(2)] for i in range(2)])
    A,aden=integer_matrix(base)
    G=gram4(Bnum)
    return [[A[i][j]*Bden**2-aden*G[i][j] for j in range(2)] for i in range(2)],aden*Bden**2

def signs_of_Q(Q):
    a,b,c=Q[0][0],Q[0][1],Q[1][1]
    det=a*c-b*b
    if a>0 and det>0:return 'strict_safe',a,det
    if a>=0 and c>=0 and det>=0:return 'boundary',a,det
    return 'danger_reachable',a,det

def norm_certificate(M,c):
    num,den=integer_matrix(M)
    n=len(num)
    G=[[sum(num[k][i]*num[k][j] for k in range(n)) for j in range(n)]for i in range(n)]
    Q=[[int(c.p)*den**2*int(i==j)-int(c.q)*G[i][j]for j in range(n)]for i in range(n)]
    minors=[bareiss_det([r[:k]for r in Q[:k]])for k in range(1,n+1)]
    assert all(x>0 for x in minors)
    return dict(squared_norm_upper=str(c),leading_minors=[str(x) for x in minors],
                integer_matrix_sha256=hashlib.sha256(json.dumps(Q).encode()).hexdigest())

def main():
    raw=INPUT.read_bytes()
    d=json.loads(raw.decode('utf-8-sig'),parse_float=Decimal)
    R=lambda x:sp.Rational(str(x))
    F=sp.Matrix([[R(x)for x in row]for row in d['F']])
    g=sp.Matrix([R(x)for x in d['g']])
    W=sp.Matrix([[R(x)for x in row]for row in d['W10']])
    H=[sp.Matrix([[1,sp.Rational(t,10)]])for t in range(10)]
    # Matrix powers/convolutions are formed from their closed formulas.
    # None of the old scaled integer B_{k+1} recurrences is used.
    powers=[F**i for i in range(14)]
    conv=[]
    for t in range(1,14):
        coefficients=[powers[t-1-j]*g*H[j]for j in range(min(t,10))]
        den=reduce(lcm,(int(x.q) for C in coefficients for x in C),1)
        columns=[[[int(C[i,a]*den)for a in range(2)]for i in range(4)]for C in coefficients]
        conv.append((columns,den))
    words=list(product([-1,1],repeat=10))
    safe=[];danger=[];boundary=[];records=[]
    counts=[[0,0,0]for _ in range(13)]
    B13cert=[];stream=hashlib.sha256(); margin=sp.Rational(1,4000)
    min_display=float('inf')
    for wi,word in enumerate(words):
        status=[];margin_ok=[];first_failure=None
        local_min=float('inf')
        for t,(columns,den) in enumerate(conv,1):
            Bnum=[[sum(word[j]*C[i][a]for j,C in enumerate(columns))for a in range(2)]for i in range(4)]
            Q,qden=q_integer(Bnum,den,sp.Rational(1,100),W)
            typ,a,det=signs_of_Q(Q);status.append(typ)
            col={'strict_safe':0,'boundary':1,'danger_reachable':2}[typ]
            counts[t-1][col]+=1
            stream.update(f'{wi}:{t}:{Q}\n'.encode())
            # Scale Q-margin I by 4000 to test the strict common margin.
            Qm=[[4000*Q[i][j]-qden*int(i==j)for j in range(2)]for i in range(2)]
            margin_ok.append(signs_of_Q(Qm)[0]=='strict_safe')
            Bfloat=np.array(Bnum,float)/float(den) if len(str(den))<300 else np.array([[float(sp.Rational(v,den))for v in row]for row in Bnum])
            local_min=min(local_min,float(np.linalg.eigvalsh(np.array(W,float)/100-Bfloat.T@Bfloat)[0]))
            if typ!='strict_safe' and first_failure is None:first_failure=t
            if t==13:
                Qt,_=q_integer(Bnum,den,sp.Rational(11,1250),W)
                assert signs_of_Q(Qt)[0]=='strict_safe'
                B13cert.append(hashlib.sha256(str(Qt).encode()).hexdigest())
        if 'danger_reachable'in status:
            danger.append(wi);overall='danger_reachable'
        elif 'boundary'in status:
            boundary.append(wi);overall='boundary'
        else:
            safe.append(wi);overall='strict_safe';assert all(margin_ok)
            min_display=min(min_display,local_min)
        records.append(dict(index=wi,word=list(word),classification=overall,
                            first_non_strict_state_index=first_failure,
                            status_by_state_index_1_to13=status,
                            Q_minus_I_over4000_strict_by_state=margin_ok,
                            minimum_margin_float=local_min))
    assert len(safe)==960 and len(danger)==64 and not boundary
    assert all(records[i]['classification']==records[1023-i]['classification']for i in range(1024))
    norms={}
    for r in range(10):norms[str(r)]=norm_certificate(powers[r],sp.Rational(103,100))
    norms['10']=norm_certificate(powers[10],sp.Rational(499,500)**2)
    # W >= 2/3 I independently, by exact 2x2 minors.
    Wlower=W-sp.Rational(2,3)*sp.eye(2)
    assert Wlower[0,0]>0 and Wlower.det()>0
    tail=sp.Rational(103,100)*sp.Rational(11,125)
    tailmargin=(sp.Rational(1,100)-tail/10)*sp.Rational(2,3)
    assert tail==sp.Rational(1133,12500) and tailmargin>margin

    # Independent graph rebuild. State words and edges come only from fresh safe.
    edges=[(w>>1,w&511)for w in safe]
    A=csr_matrix((np.ones(len(edges)),([e[0]for e in edges],[e[1]for e in edges])),shape=(512,512))
    ncomp,labels=connected_components(A,directed=True,connection='strong')
    comps=[np.flatnonzero(labels==i).tolist()for i in range(ncomp)]
    cyc=[c for c in comps if len(c)>1 or A[c[0],c[0]]]
    C=max(cyc,key=len);local=A[C,:][:,C]; rowcounts=np.diff(local.indptr)
    rho=float(np.real(eigs(local,k=1,which='LM',v0=np.ones(len(C)),tol=1e-13)[0][0]))
    assert ncomp==43 and len(cyc)==1 and len(C)==470 and local.nnz==912
    assert sum(rowcounts==1)==28
    # Cycle period from depth differences, independent of prior graph code.
    depth={C[0]:0};todo=[C[0]]; period=0
    adj={v:[w for a,w in edges if a==v and w in C]for v in C}
    while todo:
        v=todo.pop(0)
        for w in adj[v]:
            if w not in depth:depth[w]=depth[v]+1;todo.append(w)
    for v in C:
        for w in adj[v]:period=gcd(period,abs(depth[v]+1-depth[w]))
    assert period==1
    # Reconstruct the Parry/PF chain and compare entropy/predictability notions.
    er,vr=eigs(local,k=1,which='LM',v0=np.ones(len(C)),tol=1e-13)
    el,vl=eigs(local.T,k=1,which='LM',v0=np.ones(len(C)),tol=1e-13)
    right=np.real(vr[:,0]);left=np.real(vl[:,0])
    if right.sum()<0:right=-right
    if left.sum()<0:left=-left
    assert right.min()>0 and left.min()>0
    right/=right.max();left/=left@right
    stationary=left*right
    dense=local.toarray()
    transition=dense*right[None,:]/(rho*right[:,None])
    assert np.max(abs(transition.sum(axis=1)-1))<1e-12
    assert np.max(abs(stationary@transition-stationary))<1e-12
    entropy=-sum(stationary[i]*sum(p*log2(p)for p in transition[i]if p>0)for i in range(len(C)))
    rational_right=[sp.Rational(str(float(v)))for v in right]
    ratios=[]
    for i in range(len(C)):
        ratios.append(sum(rational_right[j]for j in np.flatnonzero(dense[i]))/rational_right[i])
    predictions=[]
    for known in [0,1,5,9]:
        groups={}
        mask=(1<<known)-1
        for i,state in enumerate(C):
            suffix=state&mask
            mass=stationary[i]
            plus=sum(transition[i,j]for j,w in enumerate(C)if w&1)
            a,b=groups.get(suffix,(0.,0.))
            groups[suffix]=(a+mass,b+mass*plus)
        avg=sum(max(plus,mass-plus)for mass,plus in groups.values())
        worst=max(max(plus,mass-plus)/mass for mass,plus in groups.values())
        hc=0.
        for mass,plus in groups.values():
            p=plus/mass
            hc+=mass*sum(-x*log2(x)for x in [p,1-p]if x>0)
        predictions.append(dict(known_past_bits=known,average_optimal_guess=avg,
                                 worst_history_guess=worst,mean_conditional_entropy_bits=hc))
    ancestors=set(C)
    while True:
        nxt=ancestors|{a for a,b in edges if b in ancestors}
        if nxt==ancestors:break
        ancestors=nxt
    Wupper=13*sp.eye(2)-W
    assert Wupper[0,0]>0 and Wupper.det()>0
    global_damage_upper=sp.Rational(1,10)-10*margin/13
    Ffloat=np.array(F,float)
    commutator=F.T*F-F*F.T
    assert commutator[0,0]!=0
    out=dict(date='2026-11-13',role='independent adversarial certificate reconstruction',
             input_file=str(INPUT.relative_to(P)),input_sha256=hashlib.sha256(raw).hexdigest(),
             implementation='SymPy Rational matrix powers plus direct closed-form convolution sums; independent per-time scaling; no old code imported',
             classification_input_old_results=False,safe_count=len(safe),boundary_count=len(boundary),danger_count=len(danger),
             safe_indices=safe,danger_indices=danger,boundary_indices=boundary,
             all_sign_reversal_pairs_passed=True,word_time_checks=1024*13,
             status_counts_by_state_1_to13=counts,common_margin_strict='1/4000',
             minimum_safe_margin_display_float=min_display,
             matrix_stream_sha256=stream.hexdigest(),records=records,
             norm_certificates=norms,Schur_stability_certified_by_F10_norm=True,
             F_spectral_radius_float=float(max(abs(np.linalg.eigvals(Ffloat)))),
             F_non_normal_commutator_norm_float=float(np.linalg.norm(Ffloat.T@Ffloat-Ffloat@Ffloat.T,2)),
             F_non_normal_exact_entry_00_nonzero=True,
             B13_all_word_damage_strict_upper='11/125',B13_matrix_hashes=B13cert,
             tail_state_index='b_(Ta+13+n), n>=0; b13 is after all 10 injections and 3 homogeneous updates',
             tail_strict_upper=str(tail),tail_uniform_matrix_margin=str(tailmargin),
             all_time_uniform_margin_at_least='1/4000',
             uniform_all_time_damage_strict_upper_using_Wmax13=str(global_damage_upper),
             graph=dict(states=512,edges=len(edges),scc_count=ncomp,cyclic_scc_count=len(cyc),
                        cyclic_states=C,cyclic_state_count=len(C),cyclic_internal_edges=local.nnz,
                        forced_states=[C[i]for i,x in enumerate(rowcounts)if x==1],
                        forced_count=int(sum(rowcounts==1)),period=period,
                        rho_float=rho,topological_entropy_bits_float=log2(rho),
                        collatz_rational_lower=str(min(ratios)),collatz_rational_upper=str(max(ratios)),
                        markov_entropy_rate_bits_float=entropy,
                        stationary_next_plus_probability_float=sum(stationary[i]*sum(transition[i,j]for j,w in enumerate(C)if w&1)for i in range(len(C))),
                        forced_state_stationary_mass_float=float(sum(stationary[i]for i,x in enumerate(rowcounts)if x==1)),
                        history_predictions=predictions,
                        infinitely_extendible_state_count=len(ancestors),
                        no_uniform_positive_one_step_entropy=True,
                        finite_seed_information_theoretic_entropy_rate_zero=True),
             physical_trajectories=0,old_scripts_run=0,proof_scope='frozen exact decimal rational model, not physical DARE interval certificate')
    # Only now compare against existing published classifications.
    prior=json.loads((P/'safe_sign_process_20261113/safe_words.json').read_text(encoding='utf-8'))
    out['final_comparison_to_prior']=dict(safe_match=safe==prior['safe_indices'],danger_match=danger==prior['danger_indices'])
    assert out['final_comparison_to_prior']['safe_match']and out['final_comparison_to_prior']['danger_match']
    OUTPUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items()if k not in ['records','safe_indices','danger_indices','norm_certificates','B13_matrix_hashes','graph']},ensure_ascii=False,indent=2))
    print(json.dumps(out['graph'],indent=2))

if __name__=='__main__':main()
