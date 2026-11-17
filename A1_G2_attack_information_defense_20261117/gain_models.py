"""Original common-G architecture; all gain schedules are deterministic and separate."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import block_diag,solve_discrete_are
from scipy.optimize import brentq
H=Path(__file__).resolve().parent;P=H.parent;T=32
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
def propagation(m,Ks,Gs):
    Hc=np.vstack([m.H[1],m.H[0]+m.H[2]])
    R=block_diag(m.R[1],m.R[0]+m.R[2])
    prevP=None;prevC=None;Ps=[];Cs=[];Fs=[];Js=[];ey=[];iv=[]
    for k,(K,G) in enumerate(zip(Ks,Gs)):
        L=np.hstack([K,G]);J=np.eye(2)-L@Hc;F=J@m.A
        V=np.eye(2) if k==0 else m.A@prevP@m.A.T+m.Q
        D=-np.eye(2) if k==0 else m.A@prevC@m.A.T-m.Q
        pc=J@V@J.T+L@R@L.T;ce=J@D
        Ps.append(pc);Cs.append(ce);Fs.append(F);Js.append(J);ey.append(ce@m.H[0].T+G@m.R[0])
        iv.append(np.hstack([-J@V@m.H[1].T+K@m.R[1],-J@V@(m.H[0]+m.H[2]).T+G@(m.R[0]+m.R[2])]))
        prevP=(pc+pc.T)/2;prevC=ce
    return dict(P=np.array(Ps),C=np.array(Cs),F=np.array(Fs),J=np.array(Js),
      target_cov=np.array(ey),innovation_orthogonality=np.array(iv),
      healthy_cost=float(np.trace(np.array(Ps),axis1=1,axis2=2).mean()),
      healthy_7_31=float(np.trace(np.array(Ps)[7:],axis1=1,axis2=2).mean()))
def constrained(m,kind):
    Hc=np.vstack([m.H[1],m.H[0]+m.H[2]]);R=block_diag(m.R[1],m.R[0]+m.R[2])
    K=[];G=[];prevP=prevC=None;constraint=[]
    for k in range(T):
        V=np.eye(2) if k==0 else m.A@prevP@m.A.T+m.Q
        D=-np.eye(2) if k==0 else m.A@prevC@m.A.T-m.Q
        S=Hc@V@Hc.T+R;Si=np.linalg.inv(S);L0=V@Hc.T@Si
        L=L0
        if kind=='raw_orthogonal':
            B=-Hc@D@m.H[0].T+np.vstack([np.zeros((2,2)),m.R[0]])
            E=-D@m.H[0].T-L0@B
            L=L0+E@np.linalg.solve(B.T@Si@B,B.T@Si)
            constraint.append(float(np.linalg.norm(D@m.H[0].T+L@B)))
        J=np.eye(2)-L@Hc;prevP=J@V@J.T+L@R@L.T;prevP=(prevP+prevP.T)/2
        prevC=J@D;K.append(L[:,:2]);G.append(L[:,2:])
    return np.array(K),np.array(G),max(constraint,default=0.)
def inflated(m,eta):
    Hc=np.vstack([m.H[1],m.H[0]+m.H[2]])
    R=block_diag(m.R[1],m.R[0]+m.R[2]+eta*np.eye(2))
    Pp=solve_discrete_are(m.A.T,Hc.T,m.Q,R);L=np.linalg.solve(Hc@Pp@Hc.T+R,Hc@Pp).T
    return np.repeat(L[:,:2][None],T,axis=0),np.repeat(L[:,2:][None],T,axis=0)
def match_inflation(m,target):
    def fun(x):
        K,G=inflated(m,x);return propagation(m,K,G)['healthy_cost']-target
    lo=0.;hi=1.
    while fun(hi)<0:
        hi*=2
        if hi>1e8:raise ValueError('healthy-cost target not crossed on regularization path')
    root=brentq(fun,lo,hi,xtol=1e-12)
    return root,*inflated(m,root)
def main():
    allmodels={}
    for name in ('unstable','stable'):
        m=ref.load_model(name);baseK=np.repeat(m.K[1][None],T,axis=0);baseG=np.repeat(m.G[1][None],T,axis=0)
        base=propagation(m,baseK,baseG)['healthy_cost']
        tvK,tvG,_=constrained(m,'normal_MMSE');ck,cg,res=constrained(m,'raw_orthogonal')
        cost=propagation(m,ck,cg)['healthy_cost'];eta,rk,rg=match_inflation(m,cost)
        capeta,capk,capg=match_inflation(m,base*1.20*(1-1e-10))
        def shrinkcost(alpha):return propagation(m,baseK,baseG*alpha)['healthy_cost']-cost
        assert shrinkcost(0)>0 and shrinkcost(1)<0
        alpha=brentq(shrinkcost,0,1)
        specs=[('Niu',baseK,baseG,{}),('TV_MMSE',tvK,tvG,{}),
          ('raw_orthogonal',ck,cg,{'constraint_residual':res}),
          ('robust_matched',rk,rg,{'eta':eta}),('robust_cap20',capk,capg,{'eta':capeta}),
          ('shrink_matched',baseK,baseG*alpha,{'alpha':alpha})]
        result={}
        for label,K,G,meta in specs:
            st=propagation(m,K,G)
            Fnorm=max(np.linalg.norm(F,2) for F in st['F'])
            result[label]=dict(K=K.tolist(),G=G.tolist(),**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in st.items()},
              healthy_relative=st['healthy_cost']/base-1,max_F_operator_norm=float(Fnorm),
              max_target_correlation=float(np.linalg.norm(st['target_cov'],axis=(1,2)).max()),**meta)
            assert Fnorm<1
            if label!='raw_orthogonal':assert st['healthy_cost']<=base*1.20000001
        allmodels[name]=result
        print(json.dumps({name:{k:{f:r[f] for f in ('healthy_relative','max_F_operator_norm','max_target_correlation')} for k,r in result.items()}}),flush=True)
    (H/'gain_designs.json').write_text(json.dumps(dict(date='2026-11-17',models=allmodels,
      healthy_cap=.20,selection_used_attacked_paths=False,source_model_unchanged=True),ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
