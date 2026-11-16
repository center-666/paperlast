"""Finite, registered comparison of standard Gaussian and consistency gates.

Defense functions receive only actual local/decoded measurements, public models,
and declared prior target knowledge. Truth and clean paths enter evaluation only.
"""
from pathlib import Path
import sys,json,hashlib,itertools
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent;OLD=P/'attack_gain_switch_audit_20261115'
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
import numpy as np
from scipy.stats import chi2,norm
from scipy.linalg import eigh
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
CFG=json.loads((H/'registration.json').read_text('utf8'))
PRE=json.loads((H/'analytic_preflight.json').read_text('utf8'))
assert PRE['registration_sha256']==sha(H/'registration.json')
SCHEMES=('B0','B1','B2','B4','B5','B4H')
def summary(x):
    x=np.asarray(x,float);n=len(x)
    return dict(mean=float(x.mean()),sample_SE=float(x.std(ddof=1)/np.sqrt(n)) if n>1 else None,n=n)
def prop(x):
    x=np.asarray(x,bool);n=len(x);k=int(x.sum());p=k/n;z=1.959963984540054
    a=(p+z*z/(2*n))/(1+z*z/n);b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(count=k,n=n,estimate=p,Wilson95=[float(a-b),float(a+b)])
def gaussian_info(m):
    inv=[np.linalg.inv(h) for h in m.H];S=[v@r@v.T for v,r in zip(inv,m.R)]
    V=np.block([[S[0]+S[1],S[1]],[S[1],S[2]+S[1]]]);VV=np.linalg.inv(V)
    out=dict(inv=inv,S=S,V=V,Vi=VV)
    out['C']=[S[1]@np.linalg.inv(S[j]+S[1]) for j in (2,0)]
    out['Gamma']=[S[j]+S[1]-S[1]@np.linalg.inv(S[o]+S[1])@S[1] for j,o in ((0,2),(2,0))]
    out['pairV']=[S[0]+S[1],S[2]+S[1],S[0]+S[2]]
    return out
def scores(m,y):
    info=gaussian_info(m)
    z=[y[:,:,j]@info['inv'][j].T for j in (0,1,2)]
    e=[z[0]-z[1],z[2]-z[1]];r=np.concatenate(e,axis=2)
    pair=np.stack([np.einsum('sti,ij,stj->st',v,np.linalg.inv(c),v) for v,c in zip((e[0],e[1],z[0]-z[2]),info['pairV'])],axis=2)
    u=[e[j]-e[1-j]@info['C'][j].T for j in range(2)]
    focused=np.stack([np.einsum('sti,ij,stj->st',v,np.linalg.inv(c),v) for v,c in zip(u,info['Gamma'])],axis=2)
    omnibus=np.einsum('sti,ij,stj->st',r,info['Vi'],r)
    return dict(pair=pair,focused=focused,omnibus=omnibus,r=r)
def decision(sc,scheme,known):
    pair,focused=sc['pair'],sc['focused'];N,T,_=pair.shape
    flags=np.zeros((N,T,2),bool);monitor=np.zeros(T,bool);monitor[7:]=True
    target=None if known is None else (0 if known==0 else 1)
    if scheme in ('B0','B1','B2'):
        threshold=float(chi2.isf(.05/(25*(2 if target is None else 1)),2))
        flags=pair[:,:,:2]>threshold
        if target is not None:flags[:,:,1-target]=False
        healthy=pair[:,:,:2]<=chi2.isf(.05,2)
    elif scheme in ('B4','B4H'):
        threshold=float(chi2.isf(.05/(25*(2 if target is None else 1)),2))
        if target is None:
            which=focused.argmax(axis=2);anyflag=focused.max(axis=2)>threshold
            for j in range(2):flags[:,:,j]=anyflag&(which==j)
        else:flags[:,:,target]=focused[:,:,target]>threshold
        healthy=focused<=chi2.isf(.05,2)
    else:
        # Standard maximum pairwise-consistent subset containing trusted local.
        # [local], [local,0], [local,2], [local,0,2]. Known case keeps healthy edge.
        threshold=float(chi2.isf(.05/(25*3),2));good=pair<=threshold
        eligible=np.ones((N,T,4),bool)
        eligible[:,:,1]=good[:,:,0];eligible[:,:,2]=good[:,:,1];eligible[:,:,3]=good.all(axis=2)
        if target is not None:
            healthy_bit=2 if target==0 else 1
            for mask in range(4):
                if not mask&healthy_bit:eligible[:,:,mask]=False
            # A known-healthy pair can itself disagree due to noise: retain it;
            # accept the target only if ALL three pair checks pass.
            eligible[:,:,healthy_bit]=True
        utility=np.broadcast_to(np.array([0.,10.,10.,20.]),(N,T,4)).copy()
        utility[:,:,1]-=pair[:,:,0]/(1+pair[:,:,0])*1e-2
        utility[:,:,2]-=pair[:,:,1]/(1+pair[:,:,1])*1e-2
        utility[~eligible]=-np.inf
        chosen=utility.argmax(axis=2)
        for j in range(2):flags[:,:,j]=(chosen&(1<<j))==0
        healthy=~flags
    flags[:,~monitor]=False
    if scheme=='B4H':
        active=np.zeros((N,2),bool);passes=np.zeros((N,2),int);phi=np.ones((N,T,2),np.int8)
        for k in range(T):
            passes=np.where(healthy[:,k],passes+1,0)
            active|=flags[:,k];passes=np.where(flags[:,k],0,passes)
            active&=passes<5;phi[:,k]=~active
    else:
        remaining=np.zeros((N,2),int);phi=np.ones((N,T,2),np.int8)
        for k in range(T):
            remaining=np.where(flags[:,k],5,remaining);phi[:,k]=remaining==0
            remaining=np.maximum(remaining-1,0)
    return flags,phi,threshold
def run(m,modelrow,scheme,y,phi):
    N,T,_,n=y.shape;prev=np.zeros((N,n));post=np.empty((N,T,n));prior=np.empty_like(post)
    ph=np.ones_like(phi) if scheme=='B0' else phi
    mask=ph[:,:,0]+2*ph[:,:,1]
    rows=modelrow['fixed_modes' if scheme in ('B0','B1') else 'redesigned_modes']
    Ks=np.array([r['K'] for r in rows]);Gs=np.array([r['G'] for r in rows]);Js=np.array([r['J'] for r in rows]);Fs=np.array([r['F'] for r in rows])
    for k in range(T):
        prior[:,k]=prev@m.A.T
        local=y[:,k,1]-prior[:,k]@m.H[1].T
        neigh=sum(ph[:,k,h,None]*(y[:,k,j]-prior[:,k]@m.H[j].T) for h,j in enumerate((0,2)))
        post[:,k]=prior[:,k]+np.einsum('sij,sj->si',Ks[mask[:,k]],local)+np.einsum('sij,sj->si',Gs[mask[:,k]],neigh)
        prev=post[:,k].copy()
    return dict(post=post,prior=prior,phi=ph,mask=mask,K=Ks[mask],G=Gs[mask],J=Js[mask],F=Fs[mask])
def closed(m,row,scheme,y,known):
    sc=scores(m,y);a,phi,th=decision(sc,scheme,known)
    return run(m,row,scheme,y,phi),a,th
def first(a):
    active=a[:,7:];return np.where(active.any(axis=1),7+active.argmax(axis=1),np.inf)
def metrics(run,clean,b0clean,x,alarms,target):
    delta=run['post']-clean['post'];fixed=run['post']-b0clean['post'];d2=(delta**2).sum(axis=2)
    loss=((run['post']-x)**2).sum(axis=2);normal=((clean['post']-x)**2).sum(axis=2)
    tauA=first(alarms.any(axis=2));tauD=first(d2>=.1);tf=first((fixed**2).sum(axis=2)>=.1)
    tauC=first(alarms[:,:,target]);clock=np.arange(32)[None,:]
    anypost=((d2>=.1)&(clock>tauA[:,None])).any(axis=1)
    read=np.zeros_like(d2,bool);read[:,1:]=(run['phi'][:,1:,target]==1)&(run['phi'][:,:-1,target]==0)
    read&=clock>tauA[:,None];read[:,:7]=False
    near=read.copy();near[:,1:]|=read[:,:-1];near[:,2:]|=read[:,:-2]
    return dict(risk=tauD<tauA,risk_fixed=tf<tauA,risk_before_correct_alarm=tauD<tauC,
      postalarm_danger=anypost,reentry_danger=((d2>=.1)&read).any(axis=1),reentry3_danger=((d2>=.1)&near).any(axis=1),
      reentry_event=read.any(axis=1),first_packet_danger=d2[:,7]>=.1,
      mse=loss[:,7:].mean(axis=1),cumulative_bias=d2[:,7:].sum(axis=1),peak_bias=d2[:,7:].max(axis=1),
      Pd=alarms[:,7:17].any(axis=(1,2)),correct_Pd=alarms[:,7:17,target].any(axis=1),
      wrong_first=((tauA<tauC)&np.isfinite(tauA)),FA=alarms[:,7:].any(axis=(1,2)),tauA=tauA,
      stop_bias=d2[:,17],late_bias=d2[:,21],terminal_bias=d2[:,-1],rejected_fraction=1-run['phi'][:,7:].mean(axis=(1,2)),
      cross=2*np.einsum('sti,sti->st',clean['post']-x,delta)[:,7:].mean(axis=1),
      mse_identity_error=float(np.max(np.abs(loss-normal-d2-2*np.einsum('sti,sti->st',clean['post']-x,delta)))))
BOOL=('risk','risk_fixed','risk_before_correct_alarm','postalarm_danger','reentry_danger','reentry3_danger','reentry_event','first_packet_danger','Pd','correct_Pd','wrong_first')
def report(m):return {k:prop(v) if k in BOOL else summary(v) for k,v in m.items() if k not in ('tauA','FA','mse_identity_error')}
def sample_inputs(m,name):
    bank=np.load(OLD/'screen_arrays.npz');data={k:bank[name+'__physical__'+k] for k in ('x','y','w','v')}
    seeds=json.loads((OLD/'sampling_seed_map.json').read_text('utf8'))[name]['watermark']
    M=np.stack([ref.original_watermarks(m,32,s,.5,.5)[0] for s in seeds])
    assert np.array_equal(M[:,:,0],bank[name+'__M_target'])
    return data,M
def attack_grid(W):
    vals=[]
    for f in (.25,.5,1.):
        for t in range(24):
            a=2*np.pi*t/24;u=np.array([np.cos(a),np.sin(a)])
            vals.append(u*np.sqrt(10*f/(u@W@u))*(1-1e-12))
    return np.array(vals)
def signals(m,j,xis):
    O=np.stack([m.H[j]@np.linalg.matrix_power(m.A,t) for t in range(10)])
    out=np.zeros((len(xis),32,2));out[:,7:17]=np.einsum('tij,sj->sti',O,xis)
    return out
def prefix_glrt(m,y,M,j,scope,local_only=False):
    """Ordinary rank-two GLRT; onset7 aligned benchmark, not a new detector."""
    info=gaussian_info(m);sc=scores(m,y);r=sc['r'];N=len(y)
    flags=np.zeros((N,32),bool);scs=np.zeros((N,32,2))
    candidates=[j] if scope=='known' else [0,2]
    for h in candidates:
        idx=0 if h==0 else 1;V=info['pairV'][idx] if local_only else info['V'];Vi=np.linalg.inv(V)
        I=np.zeros((N,2,2));b=np.zeros((N,2))
        for k in range(7,17):
            L=np.einsum('ij,sjk,kl->sil',info['inv'][h],M[:,k,h],m.H[h]@np.linalg.matrix_power(m.A,k-7))
            if not local_only:
                Z=np.zeros((N,4,2));Z[:,2*idx:2*idx+2]=L;L=Z
            rr=r[:,k,2*idx:2*idx+2] if local_only else r[:,k]
            I+=np.einsum('sji,jk,skl->sil',L,Vi,L);b+=np.einsum('sji,jk,sk->si',L,Vi,rr)
            scs[:,k,idx]=np.einsum('si,sij,sj->s',b,np.linalg.inv(I),b)
    threshold=chi2.isf(.05/(10*len(candidates)),2)
    flags[:,7:17]=scs[:,7:17].max(axis=2)>threshold
    return flags,scs
def identities(m,run,clean,data,eta,j):
    N,T,_=run['post'].shape;err=run['post']-data['x'];delta=run['post']-clean['post'];full=bad=truth=0.
    for k in range(T):
        ph=run['phi'][:,k];K=run['K'][:,k];G=run['G'][:,k];J=run['J'][:,k];F=run['F'][:,k]
        e=(-np.einsum('sij,sj->si',J,data['x'][:,0])) if k==0 else np.einsum('sij,sj->si',F,err[:,k-1])-np.einsum('sij,sj->si',J,data['w'][:,k-1])
        e+=np.einsum('sij,sj->si',K,data['v'][:,k,1])
        noise=sum(ph[:,h,None]*(data['v'][:,k,hj]+(eta[:,k] if hj==j else 0)) for h,hj in enumerate((0,2)))
        e+=np.einsum('sij,sj->si',G,noise);truth=max(truth,float(np.max(np.abs(e-err[:,k]))))
        d=np.zeros((N,2)) if k==0 else np.einsum('sij,sj->si',F,delta[:,k-1])
        d+=np.einsum('sij,sj->si',G,ph[:,0 if j==0 else 1,None]*eta[:,k])
        wrong=d.copy();p0=clean['prior'][:,k]
        d+=np.einsum('sij,sj->si',K-clean['K'][:,k],data['y'][:,k,1]-p0@m.H[1].T)
        for h,hj in enumerate((0,2)):
            C=G*ph[:,h,None,None]-clean['G'][:,k]*clean['phi'][:,k,h,None,None]
            d+=np.einsum('sij,sj->si',C,data['y'][:,k,hj]-p0@m.H[hj].T)
        full=max(full,float(np.max(np.abs(d-delta[:,k]))));bad=max(bad,float(np.max(np.abs(wrong-delta[:,k]))))
    return dict(true_error=truth,paired_full=full,paired_without_mode_cross=bad)
def evaluate(name,scope,j,enc,xis,indices,schemes=SCHEMES,save=False):
    m=ref.load_model(name);row=PRE['models'][name];data,M=sample_inputs(m,name)
    data={k:v[indices] for k,v in data.items()};M=M[indices]
    if enc=='identity':M=np.broadcast_to(np.eye(2),M.shape).copy()
    K,N=len(xis),len(indices);a=signals(m,j,xis)
    eta=np.einsum('ntij,ktj->knti',M[:,:,j],a)
    y=np.tile(data['y'],(K,1,1,1));y[:,:,j]+=eta.reshape(K*N,32,2)
    x=np.tile(data['x'],(K,1,1));known=j if scope=='known' else None
    sc=scores(m,y);nc=scores(m,data['y']);clean={};normal={}
    ones=np.ones((N,32,2),np.int8);b0=run(m,row,'B0',data['y'],ones)
    metrics_all={};stored={};checks={}
    for b in schemes:
        ca,cp,th=decision(nc,b,known);clean[b]=run(m,row,b,data['y'],cp)
        aa,ph,th=decision(sc,b,known);actual=run(m,row,b,y,ph)
        ck={key:np.tile(v,(K,)+(1,)*(v.ndim-1)) for key,v in clean[b].items()}
        bk={key:np.tile(v,(K,)+(1,)*(v.ndim-1)) for key,v in b0.items()}
        mm=metrics(actual,ck,bk,x,aa,0 if j==0 else 1)
        metrics_all[b]={key:val.reshape(K,N) for key,val in mm.items() if key!='mse_identity_error'}
        normal[b]=dict(target_MSE=summary(((clean[b]['post']-data['x'])**2).sum(axis=2)[:,7:].mean(axis=1)),
          event_FA=prop(ca[:,7:].any(axis=(1,2))),reject_fraction=float(1-clean[b]['phi'][:,7:].mean()),threshold=th)
        if save:
            stored[b+'__post']=actual['post'];stored[b+'__phi']=actual['phi'];stored[b+'__alarms']=aa
            for key,val in metrics_all[b].items():stored[b+'__'+key]=val
            dd={key:np.tile(val,(K,)+(1,)*(val.ndim-1)) for key,val in data.items()}
            checks[b]=identities(m,actual,ck,dd,eta.reshape(K*N,32,2),j)
            checks[b]['mse_identity_error']=mm['mse_identity_error']
        del actual
    if save:
        for b in ('B1','B2'):
            if b in schemes:assert np.array_equal(stored['B1__alarms'],stored[b+'__alarms'])
        for b in schemes:
            for key in ('true_error','paired_full','mse_identity_error'):assert checks[b][key]<1e-9,(b,key,checks[b][key])
    return metrics_all,normal,stored,checks
def main():
    assert sha(OLD/'screen_arrays.npz')==CFG['sample_bank_sha256']
    output={};arrays={};checkrows={};searchlog={};selection={}
    for name in CFG['models']:
      for scope in ('known','unknown'):
       for j in (0,2):
        for enc in CFG['encodings']:
            case=f'{name}/{scope}/j{j}/{enc}';row=PRE['models'][name]['attacks'][str(j)][enc];W=np.array(row['W']);grid=attack_grid(W)
            tr,_,_,_=evaluate(name,scope,j,enc,grid,CFG['search_indices'])
            selections={};logs={}
            for b in SCHEMES:
                selections[b]={};logs[b]={}
                for objective,key in (('pre','risk'),('reentry','reentry_danger')):
                    values=tr[b][key].mean(axis=1);ind=int(np.argmax(values))
                    selections[b][objective]=dict(index=ind,xi=grid[ind].tolist(),train_count=int(tr[b][key][ind].sum()),train_n=64)
                    logs[b][objective]=[int(v) for v in tr[b][key].sum(axis=1)]
            searchlog[case]=dict(xi_grid=grid.tolist(),coding_energy=[float(x@W@x) for x in grid],counts=logs)
            selection[case]=selections
            # Save selection BEFORE accessing the evaluation split for this case.
            dump(H/'attack_selection_before_evaluation.json',selection)
            unique=[row['ordinary']['xi'],row['P1']['xi']];labels=['ordinary','P1']
            for b in SCHEMES:
                for ob in ('pre','reentry'):unique.append(selections[b][ob]['xi']);labels.append(b+'_'+ob+'_selected')
            mm,nor,st,checks=evaluate(name,scope,j,enc,np.array(unique),CFG['evaluation_indices'],save=True)
            reports={label:{b:report({k:v[h] for k,v in r.items()}) for b,r in mm.items()} for h,label in enumerate(labels)}
            own={b:{ob:reports[b+'_'+ob+'_selected'][b] for ob in ('pre','reentry')} for b in SCHEMES}
            output[case]=dict(normal=nor,profiles=reports,own_selected_attacks=own,
                paired_P1_B4_minus_B2={k:summary(mm['B4'][k][1].astype(float)-mm['B2'][k][1].astype(float)) for k in ('risk','reentry_danger','mse')})
            checkrows[case]=checks
            tag=case.replace('/','__')
            # Keep only P1 full trajectory and all selected paired outcome arrays.
            N=64
            for key,v in st.items():
                if key.endswith(('__post','__phi','__alarms')):arrays[tag+'__'+key]=v[N:2*N]
                else:arrays[tag+'__'+key]=v
            print(json.dumps(dict(completed=case,P1={b:reports['P1'][b]['risk']['count'] for b in SCHEMES})),flush=True)
    dump(H/'finite_attack_grid.json',searchlog)
    dump(H/'screen_results.json',dict(registration_sha256=sha(H/'registration.json'),cases=output,identity_checks=checkrows,
      newly_generated_physical_realizations=0,old_bank_realizations=256,evaluation_per_case=64,not_global_worst_case=True))
    np.savez_compressed(H/'screen_arrays.npz',**arrays)
if __name__=='__main__':main()
