"""Adversarial finite searches; explicit witnesses are separately rationalized.

Uses raw frozen F/g and independent closed convolution maps. It does not call
the previous scripts or read their word classifications to choose test words.
"""
from pathlib import Path
from itertools import product
from fractions import Fraction as Q
import json
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE=Path(__file__).resolve().parent; P=HERE.parent
inp=json.loads((P/'checks/水印收束_理论与映射核验_20261105.json').read_text(encoding='utf-8'))
F=np.array(inp['F']);g=np.array(inp['g']);W=np.array([[10,4.5],[4.5,2.85]])
powF=[np.linalg.matrix_power(F,k) for k in range(45)]
H=lambda L:np.array([[1,t/10] for t in range(L)])
words=np.array(list(product([-1,1],repeat=10)))
def mapword(s,L=10,N=14):
    return np.array([sum((np.outer(powF[k-1-j]@g,H(L)[j])*s[j]
                        for j in range(min(k,L))),np.zeros((4,2)))for k in range(1,N+1)])
maps=np.array([mapword(s) for s in words])
caps=np.array([[10*eigh(b.T@b,W,eigvals_only=True)[-1] for b in row[:13]]for row in maps])
safe=(caps<.1).all(axis=1); assert safe.sum()==960
# Legal graph from freshly computed floating classification for exploration.
adj={i:[] for i in range(512)}
for w in np.flatnonzero(safe):adj[w>>1].append((w&511,w&1))
rows=[w>>1 for w in np.flatnonzero(safe)]; cols=[w&511 for w in np.flatnonzero(safe)]
ncomp,labels=connected_components(csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(512,512)),directed=True,connection='strong')
largest=max(range(ncomp),key=lambda i:sum(labels==i));states=set(np.flatnonzero(labels==largest))
supported=[w for w in np.flatnonzero(safe) if w>>1 in states and w&511 in states]
def legal(seq):
    return all(safe[sum(int(x>0)<<(9-i)for i,x in enumerate(seq[t:t+10]))]
               for t in range(len(seq)-9))
residual=[]
for wi in supported:
    for k,b in enumerate(maps[wi,:13],1):
        ev,V=eigh(b.T@b,W);xi=np.sqrt(10)*V[:,-1];forced=b@xi
        v=powF[k].T@forced;v/=np.linalg.norm(v)
        fv=powF[k]@v
        a=fv@fv;bb=2*fv@forced;cc=forced@forced-.1
        r=(-bb+np.sqrt(bb*bb-4*a*cc))/(2*a)
        residual.append((r,wi,k,xi,v,forced))
r,wi,k,xi,v,b=min(residual,key=lambda x:x[0])
bstart=(r*1.001)*v
res=dict(word=int(wi),signs=words[wi].tolist(),state_index=k,
         xi=xi.tolist(),energy=float(xi@W@xi),b_start=bstart.tolist(),
         b_start_norm=float(np.linalg.norm(bstart)),damage=float(np.linalg.norm(powF[k]@bstart+b)**2))
# Two attacks on deterministic legal sequences. Search a few fixed periodic
# candidate histories (no random sampling); each attack keeps energy<=10.
two=[]
for period in range(2,11):
    for bits in product([-1,1],repeat=period):
        if not legal((list(bits)*((40//period)+2))[:40]):continue
        seq=(list(bits)*((45//period)+2))[:45]
        # Nonoverlapping second attack, gap0,1,3,10 (budget10 per event).
        for Ta in [10,11,13,20]:
            for end in range(Ta+1,min(Ta+14,40)):
                B1=mapword(seq[:10],N=end)[-1]
                B2=mapword(seq[Ta:Ta+10],N=end-Ta)[-1]
                # Same initial trajectory in the two events is the simplest
                # physically meaningful witness; optimize this 2D restriction.
                BB=B1+B2;ev,V=eigh(BB.T@BB,W);x=np.sqrt(9.99)*V[:,-1]
                dmg=float(np.linalg.norm(BB@x)**2)
                if dmg>.1:
                    two.append((dmg,period,bits,Ta,end,x,B1@x,B2@x));break
            if two:break
        if two:break
    if two:break
tw=None
if two:
    dmg,period,bits,Ta,end,x,b1,b2=two[0]
    tw=dict(period=list(bits),second_start=Ta,first_start=0,evaluation_state=end,
            xi_first=x.tolist(),xi_second=x.tolist(),energy_each=float(x@W@x),
            total_energy=float(2*x@W@x),damage=dmg,
            isolated_first_damage=float(b1@b1),isolated_second_damage=float(b2@b2),
            scope='10 energy per event; not total10 across both events')
# Duration9/11: exhaust all legal graph-supported paths of that length.
duration=[]
for L in [9,11]:
    WL=H(L).T@H(L);best=None
    paths=[(v,[1 if v&(1<<(8-j)) else -1 for j in range(9)]) for v in sorted(states)]
    for _ in range(L-9):
        paths=[(n,seq+[1 if bit else -1])for v,seq in paths for n,bit in adj[v] if n in states]
    for _,bits in paths:
        Bs=mapword(bits,L=L,N=L+20)
        for t,B in enumerate(Bs,1):
            ev,V=eigh(B.T@B,WL);x=np.sqrt(9.999)*V[:,-1];dmg=float(np.linalg.norm(B@x)**2)
            if best is None or dmg>best[0]:best=(dmg,bits,t,x)
    duration.append(dict(L=L,maximum_found_damage=best[0],word=list(best[1]),state=best[2],
                         xi=best[3].tolist(),energy=float(best[3]@WL@best[3]),
                         words_checked=len(paths),all_supported_paths_of_length=True,
                         finite_horizon_only=True))
# Small nonautonomous perturbation, total coded energy unchanged. Rotate the
# original trajectory on the energy sphere towards a damaging tangent vector.
pert=[]
for wi in supported:
    for kk in range(6,14):
        B=maps[wi,kk-1]; _,V=eigh(B.T@B,W);xx=np.sqrt(9.999)*V[:,-1]
        a0=H(10)@xx; norm=np.linalg.norm(a0);u=a0/norm
        C=np.column_stack([words[wi,j]*(powF[kk-1-j]@g) if j<kk else np.zeros(4)for j in range(10)])
        v=C.T@C@u;v-=u*(u@v)
        if np.linalg.norm(v)<1e-12:continue
        v/=np.linalg.norm(v)
        harm=lambda th:float(np.linalg.norm(C@(norm*(np.cos(th)*u+np.sin(th)*v)))**2)-.1
        if harm(.3)>0:
            th=brentq(harm,0,.3)*1.001
            aa=norm*(np.cos(th)*u+np.sin(th)*v)
            pert.append((np.linalg.norm(aa-a0),wi,kk,xx,aa,np.sin(th)*norm*v,th))
pe=None
if pert:
    distance,wi,kk,xx,aa,ee,th=min(pert,key=lambda r:r[0])
    pe=dict(word=int(wi),signs=words[wi].tolist(),state=kk,
            autonomous_xi=(np.cos(th)*xx).tolist(),additive_perturbation=ee.tolist(),
            coded_sequence=aa.tolist(),total_energy=float(aa@aa),
            perturbation_L2=float(np.linalg.norm(ee)),perturbation_max=float(np.max(abs(ee))),
            difference_from_original_autonomous_L2=float(distance),
            damage=float(np.linalg.norm(np.column_stack([words[wi,j]*(powF[kk-1-j]@g) if j<kk else np.zeros(4)for j in range(10)])@aa)**2),
            xi_selected_before_secret=True,word_event_has_positive_probability=True)
# One event with a single virtual-state reset after five slots; total energy10.
Tpiece=np.zeros((10,4));Tpiece[:5,:2]=H(5);Tpiece[5:,2:]=H(5)
Wp=Tpiece.T@Tpiece;piece=None
for wi in supported:
    for kk in range(6,14):
        C=np.column_stack([words[wi,j]*(powF[kk-1-j]@g)if j<kk else np.zeros(4)for j in range(10)])
        B=C@Tpiece;ev,V=eigh(B.T@B,Wp);x=np.sqrt(9.999)*V[:,-1]
        dmg=float(np.linalg.norm(B@x)**2)
        if piece is None or dmg>piece[0]:piece=(dmg,wi,kk,x)
pdmg,pwi,pkk,px=piece
piecewise=dict(signs=words[pwi].tolist(),word=int(pwi),state=pkk,
               split_after=5,xi_first=px[:2].tolist(),xi_second=px[2:].tolist(),
               total_energy=float(px@Wp@px),damage=pdmg)
# Arbitrary preselected sequence: no access to fresh signs; matched to a
# PUBLIC alternating word, energy10, one rank-one terminal convolution.
s=np.array([-1,1]*5); C=np.column_stack([s[j]*(powF[9-j]@g)for j in range(10)])
ev,V=np.linalg.eigh(C.T@C);a=np.sqrt(9.99)*V[:,-1]
arbitrary=dict(public_signs=s.tolist(),coded_sequence=a.tolist(),energy=float(a@a),
               terminal_damage=float(np.linalg.norm(C@a)**2),
               autonomous_fit_residual=float(np.linalg.norm(a-H(10)@np.linalg.lstsq(H(10),a,rcond=None)[0])))
out=dict(date='2026-11-13',residual_witness=res,two_attack_witness=tw,
         duration_witnesses=duration,small_additive_perturbation=pe,public_arbitrary_sequence=arbitrary,
         piecewise_autonomous=piecewise,
         raw_input_closed_convolution=True,physical_trajectories=0,
         finite_deterministic_candidate_search=True,random_search=False)
(HERE/'counterexample_search.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
