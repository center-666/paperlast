"""Reconstruct the frozen T innovation histories for the registered numerical fixtures."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task3
import frontend
import models_quality as quality

INDEX = 2

def replay_T(archive,prefix,fixture,model):
    tag="no_repeat_"
    y=archive["y"];x=archive["x"];delta=archive[tag+"delta"]
    keys=archive[tag+"keys"][:,:,6];gates=archive[tag+"gates"].astype(bool)
    actions=archive[tag+"S_fixed_table_T_actions"];n=len(x)
    states=np.zeros((n,150,4));innov=np.zeros((n,150,2));raw=(y.swapaxes(-1,-2)@model.adjacency.T).swapaxes(-1,-2)
    sender=int(model.links[6,1]);lookup={tuple(map(int,e)):i for i,e in enumerate(model.links[:6])}
    for t in range(1,150):
        prior=states[:,t-1]@model.A.T
        packet=y[:,t,sender]+delta[:,t,6]
        innov[:,t]=packet-prior@model.H[sender].T
        received=raw[:,t,26].copy()
        for (receiver,src),edge in lookup.items():
            if receiver==26:received-=gates[:,t,edge,None]*y[:,t,src]
        alarm=gates[:,t,6];q=np.where(alarm,quality.frozen.Q[actions[:,t]],1.)
        packet26=received+delta[:,t,6]+(q-1.)[:,None]*(y[:,t,sender]+delta[:,t,6])
        mode=alarm.astype(int);idx=actions[:,t]
        f=model.target["F"][mode,idx];k=model.target["K"][mode,idx];g=model.target["G"][mode,idx]
        states[:,t]=(np.einsum("bij,bj->bi",f,states[:,t-1])
                     +np.einsum("bij,bj->bi",k,y[:,t,26])
                     +np.einsum("bij,bj->bi",g,packet26))
    mse=((states-x)**2)[:,:,2:].sum(axis=2)
    if fixture=="health":np.testing.assert_allclose(mse[:,20:].mean(1),archive[tag+"S_fixed_table_T_losses"][:,5],rtol=0,atol=2e-6)
    else:np.testing.assert_allclose(mse[:,50:110].mean(1),archive[tag+"S_fixed_table_T_losses"][:,0],rtol=0,atol=2e-6)
    return innov,keys,gates[:,:,6],delta[:,:,6]


def ridge(X,Y,alpha=1.):
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1.
    Z=(X-mu)/sd;X1=np.column_stack((Z,np.ones(len(Z))))
    reg=np.eye(X1.shape[1])*alpha;reg[-1,-1]=0.
    coef=np.linalg.solve(X1.T@X1+reg,X1.T@Y)
    return mu,sd,coef


def apply(f,X):
    mu,sd,b=f
    return np.column_stack(((X-mu)/sd,np.ones(len(X))))@b


def target_replay(archive,fixture,vector,model,ops,threshold):
    y=archive["y"];x=archive["x"];keys=archive["no_repeat_keys"]
    active=np.ones(60) if fixture=="continuous" else (np.arange(60)%10<6).astype(float)
    delta=np.zeros((len(x),150,7,2))
    delta[:,50:110,6]=np.einsum("btij,tj->bti",keys[:,50:110,6],active[:,None]*vector)
    packets=(y[:,:,model.links[:,1]]+delta)
    score=frontend.scores(packets,keys,ops)
    gates=score>threshold
    n=len(x);state=np.zeros((n,150,4));residual=np.zeros((n,150,2))
    actions=np.full((n,150),6,dtype=np.int8)
    history=np.full(n,7**6-1,dtype=np.int64)
    graph=task3.configure(INDEX)[1]
    raw=(y.swapaxes(-1,-2)@model.adjacency.T).swapaxes(-1,-2)
    lookup={tuple(map(int,e)):i for i,e in enumerate(model.links[:6])}
    sender=int(model.links[6,1]);neighbors=model.neighbors[26]
    for tick in range(1,150):
        prior_state=state[:,tick-1]@model.A.T
        residual[:,tick]=y[:,tick,sender]+delta[:,tick,6]-prior_state@model.H[sender].T
        alarm=gates[:,tick,6]
        legal=graph[history]
        requested=np.where(alarm,4,6)
        feasible=legal & (np.arange(7)[None,:]>=requested[:,None])
        action=feasible.argmax(axis=1)
        actions[:,tick]=action
        history=(history*7+action)%7**6
        received=raw[:,tick,26].copy()
        for src in neighbors:
            edge=lookup.get((26,int(src)))
            if edge is not None:
                received-=gates[:,tick,edge,None]*y[:,tick,int(src)]
        q=np.where(alarm,quality.Q[action],1.)
        packet=(received+delta[:,tick,6]
                +(q-1.)[:,None]*(y[:,tick,sender]+delta[:,tick,6]))
        mode=alarm.astype(int)
        f=model.target["F"][mode,action];k=model.target["K"][mode,action];g=model.target["G"][mode,action]
        state[:,tick]=(np.einsum("bij,bj->bi",f,state[:,tick-1])
                       +np.einsum("bij,bj->bi",k,y[:,tick,26])
                       +np.einsum("bij,bj->bi",g,packet))
    return {"residual":residual,"delta":delta[:,:,6],"keys":keys[:,:,6],
            "gates":gates[:,:,6],"actions":actions,"state":state,"x":x,
            "alarm_score":score[:,:,6]}
