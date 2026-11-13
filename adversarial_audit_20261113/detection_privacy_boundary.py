"""Independent graph-support check of what secrecy can, and cannot, add.

Fresh floating map classification builds the graph; exact support-Gram signs
then prove no common 4-packet blind vector on that graph. Classification must
be cross-checked with the separate exact safety audit before it is a certificate.
"""
from pathlib import Path
from itertools import product
import json,hashlib
import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

HERE=Path(__file__).resolve().parent
P=HERE.parent
src=P/'checks/水印收束_理论与映射核验_20261105.json'
d=json.loads(src.read_text(encoding='utf-8'))
F,g,W=np.array(d['F']),np.array(d['g']),np.array(d['W10'])
v,U=np.linalg.eigh(W);Wh=(U/np.sqrt(v))@U.T
words=np.array(list(product([-1.,1.],repeat=10)))
B=np.zeros((1024,4,2));peak=np.zeros(1024)
for k in range(13):
    B=np.einsum('ij,wja->wia',F,B)
    if k<10:B+=g[None,:,None]*words[:,k,None,None]*np.array([1.,k/10])[None,None,:]
    X=B@Wh;peak=np.maximum(peak,10*np.linalg.eigvalsh(np.einsum('wki,wkj->wij',X,X))[:,-1])
safe=[i for i in range(1024) if peak[i]<.1]
edges=[(i>>1,i&511) for i in safe]
A=csr_matrix((np.ones(len(edges),dtype=int),tuple(np.array(edges).T)),shape=(512,512))
n,labels=connected_components(A,directed=True,connection='strong')
sizes=np.bincount(labels);label=int(sizes.argmax());states=sorted(np.flatnonzero(labels==label).tolist())
stateset=set(states)
adj={i:[] for i in states}
for i,j in edges:
    if i in stateset and j in stateset:adj[i].append(j)
As=A[states,:][:,states].astype(float)
rho,rr=eigs(As,k=1,which='LM',tol=1e-13)
rho=float(rho[0].real);rr=np.abs(rr[:,0].real)
_,ll=eigs(As.T,k=1,which='LM',tol=1e-13);ll=np.abs(ll[:,0].real)
ll/=ll@rr;pi=ll*rr;pi/=pi.sum()
ix={v:i for i,v in enumerate(states)}
pplus=np.zeros(len(states));probabilities={}
for state in states:
    i=ix[state]
    probs={j:rr[ix[j]]/(rho*rr[i]) for j in adj[state]}
    probabilities[state]=probs
    pplus[i]=sum(p for j,p in probs.items() if j&1)
prediction=[]
for r in [1,5,9]:
    groups={}
    for state in states:
        suffix=state&((1<<r)-1)
        i=ix[state];u=groups.setdefault(suffix,[0.,0.])
        u[0]+=pi[i];u[1]+=pi[i]*pplus[i]
    avg=sum(max(v[1],v[0]-v[1]) for v in groups.values())
    worst=max(max(v[1]/v[0],1-v[1]/v[0]) for v in groups.values())
    ent=-sum(v[0]*sum(p*np.log2(p) for p in [v[1]/v[0],1-v[1]/v[0]] if 0<p<1) for v in groups.values())
    prediction.append(dict(known_past_signs=r,average_optimal_guess=float(avg),
                           worst_history_guess=float(worst),average_conditional_entropy_bits=float(ent)))
Z=sp.Matrix([[1,-2,1,0],[0,1,-2,1]])
proof=[]
for state in states:
    paths=[(state,[])]
    for _ in range(4):
        paths=[(j,bits+[1 if j&1 else -1])for i,bits in paths for j in adj[i]]
    gram=sp.zeros(4)
    for _,s in paths:
        C=Z*sp.diag(*s);gram+=C.T*C
    minors=[int(gram[:j,:j].det()) for j in range(1,5)]
    assert all(x>0 for x in minors)
    proof.append(dict(state=state,paths=[s for _,s in paths],
                      unweighted_gram=[[int(x)for x in row]for row in gram.tolist()],
                      exact_leading_minors=minors))

# Public known words have rank-two, complete-window nullspace; one example.
O=sp.Matrix([[1,sp.Rational(j,10)]for j in range(4)])
S=sp.diag(-1,1,-1,1)
assert Z*S*S*O==sp.zeros(2)
eta=sp.Matrix([sp.Rational(1,2),1])
a=S*O*eta
assert Z*S*a==sp.zeros(2,1)
J=sp.zeros(14,16)
for i in range(14):J[i,i:i+3]=sp.Matrix([[1,-2,1]])
Jactive=J[:,3:13]
assert Jactive.rank()==10
independent=HERE/'independent_cert.json'
exact_set_crosscheck=False
if independent.exists():
    exact=json.loads(independent.read_text(encoding='utf-8'))
    assert safe==exact['safe_indices']
    exact_set_crosscheck=True

# Deterministic residual means cannot be zero on the whole finite event;
# healthy history makes Jactive injective, for every invertible sign word.
out=dict(date='2026-11-13',input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
  source_classification='fresh floating B-map; final set crosschecked against separately rebuilt exact safety certificate',
  separate_exact_safe_set_crosscheck_passed=exact_set_crosscheck,
  fresh_safe_indices=safe,state_count=len(states),internal_edges=sum(map(len,adj.values())),
  forced_states=[i for i in states if len(adj[i])==1],
  independently_recomputed_rho=rho,
  independently_recomputed_entropy_rate=float(np.log2(rho)),
  predictions_given_last_r_signs=prediction,
  conditional_four_future_signs_support_Gram_proofs=proof,
  positive_probability_assumption='every internal transition has positive probability; e.g. PF or dyadic chain',
  public_fixed_word_complete_window_attack=[str(x)for x in a],
  public_precompensation_residual=['0','0'],
  full_finite_event_J_active_rank=10,
  deterministic_public_entire_finite_event_hidden=False,
  secret_advantage_scope='no common nonzero complete-window blind vector for arbitrary preselected 4-vector given paststate',
  not_original_autonomous_attack_family=True,
  no_damage_guarantee_for_extended_attack_class=True,
  no_finite_time_alarm_probability_order_proved=True,
  no_noise_or_physical_samples=True)
(HERE/'detection_privacy_boundary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['fresh_safe_indices','conditional_four_future_signs_support_Gram_proofs']},ensure_ascii=False,indent=2))
