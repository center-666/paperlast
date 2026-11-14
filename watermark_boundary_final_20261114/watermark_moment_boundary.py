"""Independent expectation/geometry reconstruction for the frozen model.

No prior verification script is imported or run. Detector/energy matrices are
rebuilt with SymPy rationals from A,Q,R. DARE is separately recomputed only for
the expected-damage diagnostic. No physical/noise simulation or new design.
"""
from pathlib import Path
from itertools import product
import hashlib,json
import numpy as np
import sympy as sp
from scipy.linalg import solve_discrete_are,block_diag,eigh
from scipy.integrate import quad
H=Path(__file__).resolve().parent;R=sp.Rational
scope=json.loads((H/'frozen_scope.json').read_text(encoding='utf-8'))
SOURCE=H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
saved=json.loads(SOURCE.read_text(encoding='utf-8'))
I=sp.eye(2);L=10;w=5;E=10
As={'unstable':sp.Matrix([[R(108,100),R(5,100)],[R(1,100),R(105,100)]]),
    'stable':sp.Matrix([[R(80,100),R(5,100)],[R(1,100),R(90,100)]])}
J90=sp.Matrix([[0,-1],[1,0]])
nu_int,nu_err=quad(lambda x:np.exp(-x)/(1+2*x)**2,0,np.inf,epsabs=1e-12,epsrel=1e-12)
nu=.5+.5*nu_int
def arr(M):return np.array(M,float)
def strings(M):return [[str(x)for x in row]for row in M.tolist()]
def twirl(X,kind):
    if kind=='whole_sign':return X
    if kind=='coordinate_sign':return np.diag(np.diag(X))
    if kind=='Haar_SO2':return np.trace(X)*np.eye(2)/2
    if kind=='Gaussian':return 5*X+np.trace(X)*np.eye(2)
    if kind=='normalized_Gaussian':return nu*(5*X+np.trace(X)*np.eye(2))
    raise KeyError(kind)
def exact_twirl(X,kind):
    if kind=='whole_sign':return X
    if kind=='coordinate_sign':return sp.diag(X[0,0],X[1,1])
    if kind=='Haar_SO2':return sp.trace(X)*I/2
    if kind=='Gaussian':return 5*X+sp.trace(X)*I
    raise KeyError(kind)
def publics(kind,t):
    if kind=='identity':return I
    if kind=='public_scalar_alternation':return (-1)**t*I
    if kind=='public_coordinate_alternation':return sp.diag(1,(-1)**t)
    if kind=='public_R90':return J90
    raise KeyError(kind)
def gen2_values(G,W):
    z=sp.symbols('z');pol=sp.Poly((G-z*W).det(),z)
    aa,bb,cc=pol.all_coeffs();disc=sp.factor(bb**2-4*aa*cc)
    r1=(-bb-sp.sqrt(disc))/(2*aa);r2=(-bb+sp.sqrt(disc))/(2*aa)
    lo,hi=sorted([r1,r2],key=lambda r:float(r))
    return dict(characteristic_coefficients=[str(x)for x in [aa,bb,cc]],
        lower_exact=str(lo),upper_exact=str(hi),
        lower_float=float(lo),upper_float=float(hi),energy10_weakest_expected_ncp=E*float(lo))
def rank_histogram(Aint,whole=False):
    # Common denominator 100^4 for each exact differencing coefficient.
    powers=[np.linalg.matrix_power(np.array(Aint,dtype=object),j)for j in range(5)]
    hist={0:0,1:0,2:0};byrank={0:[],1:[],2:[]}
    words=product([-1,1],repeat=5 if whole else 10)
    for word in words:
        mats=[np.eye(2,dtype=object)*word[t]if whole else np.diag(word[2*t:2*t+2]).astype(object)for t in range(5)]
        D=np.vstack([(mats[t]@powers[t]-powers[1]@mats[t-1]@powers[t-1])*100**(4-t)for t in range(1,5)])
        q=D.T@D;a,b,c=int(q[0,0]),int(q[0,1]),int(q[1,1])
        rk=0 if a==0 and c==0 else (1 if a*c-b*b==0 else 2)
        hist[rk]+=1;byrank[rk].append(list(word))
    return dict(counts=hist,total=sum(hist.values()),rank0_words=byrank[0],
       rank1_words=byrank[1],method='exact integers in common denominator 100^4; Gram determinant',
       rank2_word_list_omitted=True)
out=dict(date='2026-11-14',model_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    scope='fixed common-G aggregate estimator; same L10 autonomous attack and actual encoded energy',
    construction='Z adjacent state-space differencing, complete left nullspace; Theta from primitive noise',
    Gaussian_m=2,Gaussian_lambda=1,Gaussian_q=1,normalization_energy_coefficient=nu,
    normalization_scalar_integral_error_estimate=nu_err,
    exact_detector_and_W=True,DARE_diagnostic_only=True,physical_noise_samples=0,models={})
for name,A in As.items():
    O=sp.Matrix.vstack(*[A**t for t in range(5)]);W=sum(((A**t).T*A**t for t in range(L)),start=sp.zeros(2))
    Z=sp.zeros(8,10);Theta=sp.zeros(8)
    for j in range(4):
        Z[2*j:2*j+2,2*j:2*j+2]=-A;Z[2*j:2*j+2,2*j+2:2*j+4]=I
        Theta[2*j:2*j+2,2*j:2*j+2]=I+I/2+A*A.T/2
        if j<3:
            Theta[2*j:2*j+2,2*j+2:2*j+4]=-A.T/2
            Theta[2*j+2:2*j+4,2*j:2*j+2]=-A/2
    assert Z*O==sp.zeros(8,2) and Z.rank()==8 and W.det()>0
    P=Z.T*Theta.inv()*Z
    matrices={}
    # Raw Gaussian cross-mean sum cancels in the complete attacked window.
    for kind in ['whole_sign','coordinate_sign','Haar_SO2','Gaussian']:
        G=sum(((A**t).T*(exact_twirl(P[2*t:2*t+2,2*t:2*t+2],kind)
            -(4*P[2*t:2*t+2,2*t:2*t+2]if kind=='Gaussian'else sp.zeros(2)))*(A**t)
            for t in range(5)),start=sp.zeros(2))
        matrices[kind]=G
    for kind in ['identity','public_scalar_alternation','public_coordinate_alternation','public_R90']:
        DD=Z*sp.diag(*[publics(kind,t)for t in range(5)])*O
        matrices[kind]=DD.T*Theta.inv()*DD
    exp={kind:dict(G=strings(G),generalized_eigenvalues=gen2_values(G,W))for kind,G in matrices.items()}
    # Do not present the integrated normalization coefficient as exact rational.
    gn=nu*arr(matrices['Gaussian']);exn=eigh(gn,arr(W),eigvals_only=True)
    exp['normalized_Gaussian']=dict(G=gn.tolist(),weakest_expected_ncp_diagnostic=float(E*exn[0]))
    # Independent aggregate DARE, not importing saved gains as algorithm input.
    C=np.array([[1.1,0],[0,1.2],[2.2,0],[0,2.4]])
    V=block_diag(.5*np.eye(2),np.eye(2));Af=arr(A)
    Pred=solve_discrete_are(Af.T,C.T,np.eye(2),V)
    gain=np.linalg.solve(C@Pred@C.T+V,C@Pred).T
    F=(np.eye(2)-gain@C)@Af;Gg=gain[:,2:]
    target=saved['models'][name]['nodes'][1]
    assert np.max(abs(F-np.array(target['posterior_F'])))<1e-12
    assert np.max(abs(Gg-np.array(target['G'])))<1e-12
    Bs=[np.linalg.matrix_power(F,L-1-t)@Gg for t in range(L)]
    Oall=[np.linalg.matrix_power(Af,t)for t in range(L)]
    Cdamage=np.hstack(Bs);Y=np.vstack(Oall);Qd=Cdamage.T@Cdamage
    sm={}
    for kind in ['whole_sign','coordinate_sign','Haar_SO2','Gaussian','normalized_Gaussian']:
        mu={'whole_sign':0,'coordinate_sign':0,'Haar_SO2':0,'Gaussian':2,'normalized_Gaussian':2*np.sqrt(nu)}[kind]
        S=np.zeros((2,2))
        for t in range(L):
            S+=Oall[t].T@twirl(Bs[t].T@Bs[t],kind)@Oall[t]
            for u in range(L):
                if u!=t:S+=mu**2*Oall[t].T@Bs[t].T@Bs[u]@Oall[u]
        ev=eigh(S,arr(W),eigvals_only=True)
        sm[kind]=dict(S=S.tolist(),P1_expected_terminal_damage=float(E*ev[-1]),
           P3_phys_feasible_at_D_point1=bool(E*ev[-1]>=.1))
    for kind in ['identity','public_scalar_alternation','public_coordinate_alternation','public_R90']:
        D=sum((Bs[t]@arr(publics(kind,t))@Oall[t]for t in range(L)),start=np.zeros((2,2)))
        S=D.T@D;ev=eigh(S,arr(W),eigvals_only=True)
        sm[kind]=dict(S=S.tolist(),P1_expected_terminal_damage=float(E*ev[-1]),
           P3_phys_feasible_at_D_point1=bool(E*ev[-1]>=.1))
    comm=J90*A-A*J90
    assert comm.det()!=0
    # All unit orthogonal laws share the first-packet pathwise envelope.
    Q1=Gg.T@Gg;instant={k:float(E*np.linalg.eigvalsh(twirl(Q1,k))[-1])for k in ['whole_sign','coordinate_sign','Haar_SO2']}
    first_autonomous={k:float(E*eigh(twirl(Q1,k),arr(W),eigvals_only=True)[-1])for k in ['whole_sign','coordinate_sign','Haar_SO2']}
    haar_pathwise=float(E*np.linalg.norm(Gg,2)**2/np.linalg.eigvalsh(arr(W))[0])
    Aint=[[int(x*100)for x in row]for row in A.tolist()]
    out['models'][name]=dict(A=strings(A),W10=strings(W),Theta=strings(Theta),Z=strings(Z),
       expected_detector=exp,expected_terminal_damage=sm,
       R90_commutator=strings(comm),R90_commutator_determinant=str(comm.det()),
       single_free_packet_energy10_max_expected_damage=instant,
       single_free_packet_unit_orthogonal_pathwise_envelope=float(E*np.linalg.norm(Gg,2)**2),
       original_L10_autonomous_first_packet_max_expected_damage=first_autonomous,
       original_L10_first_packet_Haar_pathwise_envelope=haar_pathwise,
       first5_whole_sign_exact_rank_counts=rank_histogram(Aint,True),
       first5_coordinate_sign_exact_rank_counts=rank_histogram(Aint,False))
(H/'watermark_moment_boundary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for n,m in out['models'].items():
    print(n,{k:v['generalized_eigenvalues']['energy10_weakest_expected_ncp']for k,v in m['expected_detector'].items()if'generalized_eigenvalues'in v})
    print('rank counts',m['first5_whole_sign_exact_rank_counts']['counts'],m['first5_coordinate_sign_exact_rank_counts']['counts'])
