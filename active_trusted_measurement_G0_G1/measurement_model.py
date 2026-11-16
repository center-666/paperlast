"""Joint raw-measurement Gaussian model. Auxiliary rows are detection-only."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
import numpy as np
from scipy.linalg import null_space,cholesky,solve_triangular
from scipy.stats import chi2,ncx2
CFG=json.loads((H/'registration.json').read_text('utf8'))
_CACHE={}
def schedule(cycle_phase=0,delay=0,variance=.5,coordinate=0,rho=0.):
    return dict(phase=cycle_phase,delay=delay,variance=variance,coordinate=coordinate,rho=rho)
def rows_for_window(end,kind,aux=None):
    start=end-4;ids=(0,) if kind=='B0' else (1,0,2);rows=[(t,j,c) for t in range(start,end+1) for j in ids for c in (0,1)]
    if aux is not None:
        for t in range(start,end+1):
            if t%5==aux['phase'] and t+aux['delay']<=end:rows.append((t,3,aux['coordinate']))
    return rows
def raw_matrices(m,rows,aux=None):
    key=(m.name,tuple(rows),None if aux is None else tuple(sorted(aux.items())))
    if key in _CACHE:return _CACHE[key]
    start=min(t for t,j,c in rows);end=max(t for t,j,c in rows);n=len(m.A)
    def hrow(j,c):return m.H[j][c] if j<3 else np.eye(2)[c]
    O=np.stack([hrow(j,c)@np.linalg.matrix_power(m.A,t-start) for t,j,c in rows])
    N=len(rows);cov=np.zeros((N,N))
    for a,(t,j,c) in enumerate(rows):
        for b,(u,l,d) in enumerate(rows):
            value=0.
            for r in range(start,min(t,u)):
                v=hrow(j,c)@np.linalg.matrix_power(m.A,t-1-r);w=hrow(l,d)@np.linalg.matrix_power(m.A,u-1-r)
                value+=v@m.Q@w.T
            if t==u:
                if j==l and j<3:value+=m.R[j][c,d]
                elif j==l==3:value+=aux['variance']
                elif {j,l}=={1,3} and c==d:
                    value+=aux['rho']*np.sqrt(aux['variance']*m.R[1][c,c])
            cov[a,b]=value
    Z=null_space(O.T,rcond=1e-12).T;theta=Z@cov@Z.T
    L=cholesky(theta,lower=True);V=solve_triangular(L,Z,lower=True)
    assert np.max(np.abs(V@cov@V.T-np.eye(len(V))))<1e-9
    out=dict(rows=rows,O=O,cov=cov,Z=Z,theta=theta,white=V);_CACHE[key]=out;return out
def attack_map(m,rows,Ta,M=None):
    D=np.zeros((len(rows),2))
    for i,(t,j,c) in enumerate(rows):
        if j==0 and Ta<=t<Ta+10:
            mat=m.H[0]@np.linalg.matrix_power(m.A,t-Ta)
            if M is not None:mat=M[t]@mat
            D[i]=mat[c]
    return D
def template_maps(m,rows,M=None):
    start=min(t for t,j,c in rows);result=[]
    # Full autonomous pattern plus four later starts and four early endings.
    for first,last in [(a,4) for a in range(5)]+[(0,b) for b in range(4)]:
        D=np.zeros((len(rows),2))
        for i,(t,j,c) in enumerate(rows):
            r=t-start
            if j==0 and first<=r<=last:
                mat=m.H[0]@np.linalg.matrix_power(m.A,r-first)
                if M is not None:mat=M[t]@mat
                D[i]=mat[c]
        result.append(D)
    return result
def info_for_attack(m,end,kind,aux,Ta,M=None):
    raw=raw_matrices(m,rows_for_window(end,kind,aux),aux);U=raw['white']@attack_map(m,raw['rows'],Ta,M)
    if kind=='B0':return [(U.T@U,len(raw['white']))],raw
    out=[]
    for D in template_maps(m,raw['rows'],M):
        C=raw['white']@D;Z,sv,_=np.linalg.svd(C,full_matrices=False);rank=int((sv>1e-10).sum())
        if rank:
            V=Z[:,:rank].T@U;out.append((V.T@V,rank))
    return out,raw
def one_batch_info(m,Ta,kind,aux=None,M=None):
    raw=raw_matrices(m,rows_for_window(Ta+4,kind,aux),aux);U=raw['white']@attack_map(m,raw['rows'],Ta,M)
    return U.T@U,int(len(raw['white']) if kind=='B0' else np.linalg.matrix_rank(U,tol=1e-10)),raw
def physical_damage_matrices(m,Gaussian=False):
    F,G=m.F(1),m.G[1];out=[];q=.5 if Gaussian else 0.
    O=[m.H[0]@np.linalg.matrix_power(m.A,t) for t in range(10)]
    for k in range(10):
        C=[np.linalg.matrix_power(F,k-t)@G for t in range(k+1)];B=sum(C[t]@O[t] for t in range(k+1));S=B.T@B
        for t,c in enumerate(C):S+=q*q*O[t].T@(c.T@c+np.trace(c.T@c)*np.eye(2))@O[t]
        out.append(S)
    return np.stack(out),sum(t.T@t for t in O)
def normal_covariances(m):
    i=1;J=m.J(i);F=m.F(i);K,G=m.K[i],m.G[i]
    V=K@m.R[i]@K.T+G@(m.R[0]+m.R[2])@G.T;P=J@np.eye(2)@J.T+V;out=[P]
    for k in range(1,32):P=F@P@F.T+J@m.Q@J.T+V;out.append(P)
    return np.stack(out)
