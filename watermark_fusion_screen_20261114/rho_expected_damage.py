"""P1 expected terminal damage with random transition/gain correlation retained.

Only deterministic finite moment quadrature; not a dangerous-before-alarm
probability. Original autonomous sequence and actual encoded W_L retained.
"""
import json
import numpy as np
from scipy.special import roots_laguerre
from scipy.linalg import eigh
from scipy.optimize import brentq,minimize_scalar
from scipy.stats import qmc,norm,ncx2
from rho_numeric import H,data,model_params,normGM,TAU,E,D

def gaussian_moments(p,kind,param=1.,nrad=96,nang=96):
    G=p['G'];F0=p['F0'];X=p['X']
    angles=np.pi*(np.arange(nang)+.5)/nang;g0=np.linalg.norm(G,2)
    nodes,weights=roots_laguerre(nrad)
    # For rho-aware, smooth integration split at rho=1 threshold.
    Ms=[];Ws=[];Rs=[]
    for t in angles:
        v=np.array([np.cos(t),np.sin(t)])
        if kind=='aware' and param>g0:
            hi=max(1.,param/g0)
            while normGM(G,hi,t)<param:hi*=2
            cut=brentq(lambda u:normGM(G,u,t)-param,0,hi)
            # finite [0,cut] part with Legendre and tail shifted Laguerre.
            x,w=np.polynomial.legendre.leggauss(nrad)
            us=np.r_[cut*(x+1)/2,cut+nodes]
            ww=np.r_[w*cut/2*np.exp(-cut*(x+1)/2),weights*np.exp(-cut)]/nang
        else:us=nodes;ww=weights/nang
        M=np.eye(2)[None,:,:]+2*us[:,None,None]*np.outer(v,v)[None,:,:]
        rho=np.minimum(1.,param/normGM(G,us,t)) if kind=='aware' else np.full(len(us),param)
        Ms.extend(M);Ws.extend(ww);Rs.extend(rho)
    M=np.array(Ms);w=np.array(Ws);r=np.array(Rs)
    Wm=r[:,None,None]*np.einsum('ij,njk->nik',G,M)
    z=Wm.reshape(-1,4)
    return dict(m1=float(w@r),m2=float(w@(r*r)),d=w@z,rd=(w*r)@z,dd=np.einsum('n,ni,nj->ij',w,z,z))

def deterministic_moments(p,rho,water='none'):
    z=(rho*p['G']).reshape(4)
    if water=='none':return dict(m1=rho,m2=rho*rho,d=z,rd=rho*z,dd=np.outer(z,z))
    if water=='sign':return dict(m1=rho,m2=rho*rho,d=np.zeros(4),rd=np.zeros(4),dd=np.outer(z,z))
    raise ValueError(water)

def damage(p,m,L=10):
    A=p['A'];K0=np.kron(p['F0'],np.eye(2));Kx=np.kron(p['X'],np.eye(2))
    mu=np.zeros(4);Z=np.zeros((4,4));T=np.vstack([np.linalg.matrix_power(A,j)for j in range(L)])
    W=T.T@T
    for j in range(L):
        AA=np.kron(np.eye(2),np.linalg.matrix_power(A,j).T)
        ed=AA@m['d'];erd=AA@m['rd'];edd=AA@m['dd']@AA.T
        cross=np.outer(K0@mu,ed)-np.outer(Kx@mu,erd)
        Z=K0@Z@K0.T-m['m1']*(Kx@Z@K0.T+K0@Z@Kx.T)+m['m2']*Kx@Z@Kx.T+cross+cross.T+edd
        mu=(K0-m['m1']*Kx)@mu+ed
    S=np.array([[sum(Z[2*r+i,2*r+j]for r in range(2))for j in range(2)]for i in range(2)])
    assert np.linalg.eigvalsh(S).min()>-1e-12
    eig,U=eigh(S,W);xi=np.sqrt(E)*U[:,-1]
    return dict(S=S.tolist(),W=W.tolist(),P1_expected_terminal_damage=float(E*eig[-1]),
                P1_initial_state=xi.tolist(),P3_phys_threshold=D,P3_phys_feasible=bool(E*eig[-1]>=D),
                L=L,objective='expected terminal shadow displacement, no alarm gating')

def p3_pd(p,row,water,nq=4096):
    S=np.array(row['S']);W=np.array(row['W']);A=p['A']
    if not row['P3_phys_feasible']:return dict(feasible=False,Pd=None,not_detection_100_percent=True)
    # first five attacked samples ordinary s=4, finite-event budget per test.
    Z=np.zeros((8,10));Theta=np.zeros((8,8));R=.5*np.eye(2)
    for j in range(4):
        Z[2*j:2*j+2,2*j:2*j+2]=-A;Z[2*j:2*j+2,2*j+2:2*j+4]=np.eye(2)
        Theta[2*j:2*j+2,2*j:2*j+2]=np.eye(2)+R+A@R@A.T
        if j<3:
            Theta[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T
            Theta[2*j+2:2*j+4,2*j:2*j+2]=-A@R
    Thinv=np.linalg.inv(Theta);Os=np.stack([np.linalg.matrix_power(A,j)for j in range(5)])
    if water=='Gaussian':
        uniforms=qmc.Sobol(10,scramble=False).random_base2(int(np.log2(nq)))
        # Endpoints deterministically clipped; no random seed or error certificate.
        beta=norm.ppf(np.clip(uniforms,2**-24,1-2**-24)).reshape(-1,5,2)
        MM=np.eye(2)[None,None,:,:]+beta[:,:,:,None]*beta[:,:,None,:]
    elif water=='sign':
        import itertools
        signs=np.array(list(itertools.product([-1.,1.],repeat=5)))
        MM=signs[:,:,None,None]*np.eye(2)[None,None,:,:]
    elif water=='none':MM=np.eye(2)[None,None,:,:]*np.ones((1,5,1,1))
    else:raise ValueError(water)
    mapped=np.einsum('bsij,sjk->bsik',MM,Os).reshape(len(MM),10,2)
    J=np.einsum('ri,bik->brk',Z,mapped)
    Qs=np.einsum('bri,rs,bsj->bij',J,Thinv,J)
    def objective(theta,ret=False):
        u=np.array([np.cos(theta),np.sin(theta)])
        ss=float(u@S@u)
        if ss<=0:return (np.inf,None) if ret else np.inf
        rr=D/ss
        if rr*float(u@W@u)>E*(1+1e-10):return (np.inf,None)if ret else np.inf
        lam=rr*np.einsum('i,bij,j->b',u,Qs,u)
        pd=float(np.mean(ncx2.sf(TAU,8,np.maximum(lam,0))))
        return (pd,np.sqrt(rr)*u)if ret else pd
    ts=np.linspace(0,np.pi,513);vs=np.array([objective(t)for t in ts]);i=int(np.argmin(vs))
    lo,hi=ts[max(0,i-1)],ts[min(512,i+1)]
    opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-11})
    val,xi=objective(float(opt.x),True)
    return dict(feasible=True,Pd=val,initial_state=xi.tolist(),energy=float(xi@W@xi),
        expected_terminal_damage=float(xi@S@xi),global_certified=False,
        integration=('finite32sign average'if water=='sign'else 'fixed unscrambled Sobol Gaussian integral diagnostic, clipped endpoints'if water=='Gaussian'else'no watermark deterministic'),
        nodes=len(MM),first_five_attacked_samples_only=True)

def main():
    normal=json.loads((H/'rho_normal_screen.json').read_text())
    out=dict(date='2026-11-14',kind='P1/P3-phys expected terminal response, floating moment quadrature',
        L=10,E=E,D=D,full_event_worst_danger_risk=False,models={})
    for name in normal['models']:
        p=model_params(name);rows=[]
        base={}
        for label,mm,w in [('original_Gaussian',gaussian_moments(p,'fixed',1),'Gaussian'),
            ('original_sign',deterministic_moments(p,1,'sign'),'sign')]:
            dd=damage(p,mm);dd['P3_phys_first_window']=p3_pd(p,dd,w);base[label]=dd
        for rr in normal['models'][name]['rows']:
            r=rr['fixed_rho'];g=rr['high_order']['gamma'];row={'eta':rr['eta']}
            for label,mm,w in [('no_watermark_fixed_rho',deterministic_moments(p,r),'none'),
                ('Gaussian_fixed_rho',gaussian_moments(p,'fixed',r),'Gaussian'),
                ('Gaussian_aware',gaussian_moments(p,'aware',g),'Gaussian'),
                ('sign_aware',deterministic_moments(p,r,'sign'),'sign')]:
                dd=damage(p,mm);dd['P3_phys_first_window']=p3_pd(p,dd,w)
                row[label]=dd
            rows.append(row)
        out['models'][name]=dict(baselines=base,rows=rows)
        print(name,[(r['eta'],r['Gaussian_fixed_rho']['P1_expected_terminal_damage'],r['Gaussian_aware']['P1_expected_terminal_damage'])for r in rows],flush=True)
        (H/'rho_expected_damage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (H/'rho_expected_damage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
