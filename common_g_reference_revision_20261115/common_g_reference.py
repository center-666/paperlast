"""Original measurement-sum/common-G reference. No gating or gain redesign.

Index k means the measurement time: prior[k]=A@posterior[k-1].
The initial saved estimate is posterior[-1]. Every node receives raw measurements.
Niu aggregate MMSE is an explicit supplementary gain design, not a 50-node claim.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.linalg import block_diag, null_space, solve_discrete_are
from scipy.stats import chi2

@dataclass
class Model:
    name:str
    A:np.ndarray
    Q:np.ndarray
    H:list
    R:list
    neighbors:list
    K:list
    G:list
    source:dict
    def validate(self):
        n=self.A.shape[0];m=self.H[0].shape[0];N=len(self.H)
        assert self.A.shape==(n,n) and self.Q.shape==(n,n)
        assert len(self.R)==len(self.K)==len(self.G)==len(self.neighbors)==N
        for i in range(N):
            if self.H[i].shape!=(m,n):raise ValueError('Common-G raw measurement sum requires a common measurement dimension')
            assert self.R[i].shape==(m,m) and self.K[i].shape==self.G[i].shape==(n,m)
            assert i not in self.neighbors[i] and len(set(self.neighbors[i]))==len(self.neighbors[i])
            if any(not isinstance(j,int) or not 0<=j<N for j in self.neighbors[i]):raise ValueError('invalid neighbor identity')
        return self
    def J(self,i):
        return np.eye(self.A.shape[0])-self.K[i]@self.H[i]-self.G[i]@sum((self.H[j] for j in self.neighbors[i]),np.zeros_like(self.H[i]))
    def F(self,i):return self.J(i)@self.A

def load_model(name,path=None):
    path=Path(path) if path else Path(__file__).parent/'inputs/five_node_aggregate_model.json'
    a=json.loads(path.read_text('utf8'));b=a['models'][name]
    return Model(name,np.array(b['A']),np.array(a['Q']),[np.array(x) for x in a['H']],
        [np.array(x) for x in a['R']],[list(v) for v in a['neighbors']],
        [np.array(x['K']) for x in b['nodes']],[np.array(x['G']) for x in b['nodes']],b).validate()

def aggregate_design(model,i):
    """Niu9--13 supplemental aggregate MMSE design; does not replace saved gains."""
    C=np.vstack([model.H[i],sum((model.H[j] for j in model.neighbors[i]),np.zeros_like(model.H[i]))])
    V=block_diag(model.R[i],sum((model.R[j] for j in model.neighbors[i]),np.zeros_like(model.R[i])))
    P=solve_discrete_are(model.A.T,C.T,model.Q,V)
    L=np.linalg.solve(C@P@C.T+V,C@P).T
    Pc=(np.eye(model.A.shape[0])-L@C)@P
    return C,V,P,L,(Pc+Pc.T)/2

def common_step(previous,local_y,neighbor_y,A,Hi,H_neighbors,K,G):
    """Literal original formula, with all innovations using the SAME prior."""
    if len(neighbor_y)!=len(H_neighbors):raise ValueError('neighbor measurement/matrix lists must have equal lengths')
    prior=A@previous
    local=local_y-Hi@prior
    total=np.zeros_like(local)
    for y,Hj in zip(neighbor_y,H_neighbors):total+=y-Hj@prior
    return prior,prior+K@local+G@total

def common_block_step(previous,local_y,neighbor_y,A,Hi,H_neighbors,K,G):
    """Equivalent two-block compressed observation, no freely optimized G_j."""
    if len(neighbor_y)!=len(H_neighbors):raise ValueError('neighbor measurement/matrix lists must have equal lengths')
    prior=A@previous
    m=Hi.shape[0]
    total=sum(neighbor_y,np.zeros(m));Hsum=sum(H_neighbors,np.zeros_like(Hi))
    Y=np.r_[local_y,total];C=np.vstack([Hi,Hsum]);L=np.hstack([K,G])
    return prior+L@(Y-C@prior)

def finite_autonomous_attack(model,sender,Ta,L,xi,T):
    """Original attack only; support is [Ta,Ta+L), energy in coding domain."""
    if not (0<=Ta<T and 1<=L<=T-Ta):raise ValueError('attack support outside declared horizon')
    a=np.zeros((T,model.H[sender].shape[0]))
    for t in range(L):a[Ta+t]=model.H[sender]@np.linalg.matrix_power(model.A,t)@xi
    return a,float(np.einsum('tm,tm->',a,a))

def draw_physical_path(model,T,seed):
    """One seeded Gaussian realization, shared by ALL nodes; not Monte Carlo."""
    rng=np.random.default_rng(seed);N=len(model.H);n=len(model.A);m=model.H[0].shape[0]
    x=np.empty((T,n));x[0]=rng.standard_normal(n)
    w=rng.standard_normal((T-1,n))@np.linalg.cholesky(model.Q).T
    for k in range(T-1):x[k+1]=model.A@x[k]+w[k]
    v=np.empty((T,N,m))
    for j in range(N):v[:,j]=rng.standard_normal((T,m))@np.linalg.cholesky(model.R[j]).T
    y=np.stack([x@Hj.T for Hj in model.H],axis=1)+v
    return {'x':x,'w':w,'v':v,'y':y}

def original_watermarks(model,T,seed,lam=.5,q=.5):
    if lam<=0 or q<0:raise ValueError('lambda>0 and q>=0 required')
    rng=np.random.default_rng(seed);N=len(model.H);m=model.H[0].shape[0]
    beta=rng.normal(size=(T,N,m))*np.sqrt(q)
    return lam*np.eye(m)[None,None,:,:]+np.einsum('tnm,tnp->tnmp',beta,beta),beta

def transmit_decode(model,y,M,attack=None,sender=0,receiver=1):
    """q=M^-1 y; an optional addition to one encoded edge; decode M(q+a).

    M[k,j] is shared by source j and its receivers at transmitted timestamp k.
    Local measurements are trusted and never replaced by a decoded neighbor.
    """
    T,N,m=y.shape
    if M.shape!=(T,N,m,m):raise ValueError('watermark shape/time mismatch')
    if receiver not in range(N) or sender not in model.neighbors[receiver]:raise ValueError('not a declared edge')
    encoded=np.linalg.solve(M,y[...,None])[...,0]
    received=np.zeros((T,N,N,m))
    for i in range(N):
        for j in model.neighbors[i]:
            payload=encoded[:,j].copy()
            if attack is not None and (i,j)==(receiver,sender):payload+=attack
            received[:,i,j]=np.einsum('tij,tj->ti',M[:,j],payload)
    return encoded,received

def run_common(model,y,received,initial=None):
    T,N,_=y.shape;n=len(model.A)
    previous=np.zeros((N,n)) if initial is None else np.array(initial,copy=True)
    prior=np.empty((T,N,n));post=np.empty_like(prior)
    for k in range(T):
        for i in range(N):
            js=model.neighbors[i]
            prior[k,i],post[k,i]=common_step(previous[i],y[k,i],[received[k,i,j] for j in js],
                model.A,model.H[i],[model.H[j] for j in js],model.K[i],model.G[i])
        previous=post[k].copy()
    return {'prior':prior,'posterior':post}

def bias_recursion(model,i,decoded_effect):
    """b+_k=J b-_k+G eta_k, b-_k=A b+_(k-1). No gates or isolation."""
    T=len(decoded_effect);n=len(model.A);prev=np.zeros(n)
    prior=np.empty((T,n));post=np.empty_like(prior)
    for k in range(T):
        prior[k]=model.A@prev
        post[k]=model.J(i)@prior[k]+model.G[i]@decoded_effect[k]
        prev=post[k].copy()
    return prior,post

def legacy_stacked_step(model,i,previous,P_previous,local_y,neighbor_y):
    """Reimplement complete_v2 lines580--617 without importing old module.

    Its neighbor+self ordering, pinv and 1e-9 covariance jitter are retained.
    This is the comparison baseline, NEVER the original/common-G estimator.
    """
    ids=model.neighbors[i]+[i]
    C=np.vstack([model.H[j] for j in ids]);V=block_diag(*(model.R[j] for j in ids))
    Y=np.concatenate(list(neighbor_y)+[local_y]);prior=model.A@previous
    Pp=model.A@P_previous@model.A.T+model.Q
    L=Pp@C.T@np.linalg.pinv(C@Pp@C.T+V)
    post=prior+L@(Y-C@prior)
    Pc=(np.eye(len(model.A))-L@C)@Pp;Pc=(Pc+Pc.T)/2+1e-9*np.eye(len(model.A))
    return prior,post,Pc,L

def run_legacy_stacked(model,y,received):
    T,N,_=y.shape;n=len(model.A);prev=np.zeros((N,n));P=[np.eye(n) for _ in range(N)]
    prior=np.empty((T,N,n));post=np.empty_like(prior)
    for k in range(T):
        for i in range(N):
            prior[k,i],post[k,i],P[i],_=legacy_stacked_step(model,i,prev[i],P[i],y[k,i],
                [received[k,i,j] for j in model.neighbors[i]])
        prev=post[k].copy()
    return {'prior':prior,'posterior':post}

def parity_model(A,H,Q,R,s):
    n=len(A);m=H.shape[0]
    O=np.vstack([H@np.linalg.matrix_power(A,t) for t in range(s+1)])
    if np.linalg.matrix_rank(O)!=n:raise ValueError('full-column observability required by this reference')
    Z=null_space(O.T,rcond=1e-13).T
    if Z.shape[0]==0:raise ValueError('empty parity space')
    W=np.zeros(((s+1)*m,s*n))
    for a in range(s+1):
        for b in range(a):W[a*m:(a+1)*m,b*n:(b+1)*n]=H@np.linalg.matrix_power(A,a-1-b)
    V=W@np.kron(np.eye(s),Q)@W.T+np.kron(np.eye(s+1),R)
    theta=Z@V@Z.T
    np.linalg.cholesky(theta)
    return O,Z,W,theta

def parity_series(stream,Z,s):
    return np.stack([Z@stream[k-s:k+1].reshape(-1) for k in range(s,len(stream))])

def global_parity_map(A,H,Q,R,T,s,Z):
    """Independent primitive-noise expansion, including actual overlaps.

    Primitive order x0, w0..w(T-2), v0..v(T-1). x0 covariance I.
    This joint covariance can be singular; no inverse/independence is assumed.
    """
    n=len(A);m=len(R);dim=T*n+T*m
    state=np.zeros((n,dim));state[:,:n]=np.eye(n)
    ys=[]
    for k in range(T):
        ymap=H@state;ymap[:,T*n+k*m:T*n+(k+1)*m]+=np.eye(m);ys.append(ymap)
        if k<T-1:
            state=A@state;state[:,n+k*n:n+(k+1)*n]+=np.eye(n)
    maps=np.vstack([Z@np.vstack(ys[k-s:k+1]) for k in range(s,T)])
    V=block_diag(np.eye(n),np.kron(np.eye(T-1),Q),np.kron(np.eye(T),R))
    return maps,maps@V@maps.T

def nominal_scores(residuals,theta,event_alpha=.05):
    if not 0<event_alpha<1:raise ValueError('event_alpha must be in (0,1)')
    if residuals.ndim!=2 or len(residuals)==0:raise ValueError('a nonempty, fixed finite monitoring family is required')
    if theta.shape!=(residuals.shape[1],residuals.shape[1]):raise ValueError('residual/covariance dimension mismatch')
    whitened=np.linalg.solve(np.linalg.cholesky(theta),residuals.T).T
    scores=np.einsum('td,td->t',whitened,whitened)
    count=len(scores);threshold=float(chi2.isf(event_alpha/count,theta.shape[0]))
    return scores,threshold,scores>threshold
