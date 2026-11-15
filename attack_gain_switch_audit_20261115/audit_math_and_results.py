"""Independent exact mode certificates and saved-data checks. No new samples."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent
import numpy as np
import sympy as sp
from scipy.stats import chi2
from run_screen import closed_loop
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
def sm(x):return sp.Matrix([[sp.Rational(str(v)) for v in row] for row in x])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def main():
    cfg=json.loads((H/'registration.json').read_text('utf8'));pre=json.loads((H/'analytic_preflight.json').read_text('utf8'))
    res=json.loads((H/'screen_results.json').read_text('utf8'));data=np.load(H/'screen_arrays.npz')
    model=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
    allcert={};afferr=0.;trutherr=0.;bayeserr=0.;closederr=0.;gate_equal=True;primary_bound=True;extra={}
    for name,z in pre['models'].items():
        A=sm(model['models'][name]['A']);HH=[sm(h) for h in model['H']];AR=np.array(A,float);Hn=[np.array(h,float) for h in HH]
        runtime_model=ref.load_model(name)
        modes=z['mode_gains'];cert=[]
        for b in ('B0','B1','B2','B3'):
            for mode in modes[b]:
                K,G=sm(mode['K']),sm(mode['G']);phi=mode['effective_phi'];J=sp.eye(2)-K*HH[1]-G*(HH[2]+phi*HH[0]);F=J*A
                V=sp.eye(2)/4-F.T*F;assert V[0,0]>0 and V.det()>0
                cert.append({'baseline':b,'phi':mode['phi'],'F_exact':str(F),'quarter_I_minus_FtF_exact_minors':[str(V[0,0]),str(V.det())],'positive':True})
        # Rejection-mode first-stage information fusion, a standard Bayesian identity.
        Pp=np.array(z['rejection_design']['prior_reject']);Ri=np.array(model['R'][1]);Rh=np.array(model['R'][2])
        Pc=np.linalg.inv(np.linalg.inv(Pp)+Hn[1].T@np.linalg.solve(Ri,Hn[1])+Hn[2].T@np.linalg.solve(Rh,Hn[2]))
        Kbayes=Pc@Hn[1].T@np.linalg.inv(Ri);Gbayes=Pc@Hn[2].T@np.linalg.inv(Rh)
        bayeserr=max(bayeserr,float(np.max(np.abs(Kbayes-np.array(modes['B2'][0]['K'])))),float(np.max(np.abs(Gbayes-np.array(modes['B2'][0]['G'])))))
        X=data[name+'__physical__x'];W=data[name+'__physical__w'];Y=data[name+'__physical__y'];Vn=data[name+'__physical__v']
        trutherr=max(trutherr,float(np.max(np.abs(X[:,1:]-X[:,:-1]@AR.T-W))))
        for j in range(5):trutherr=max(trutherr,float(np.max(np.abs(Y[:,:,j]-X@Hn[j].T-Vn[:,:,j]))))
        D=Hn[0]@np.linalg.inv(Hn[1]);Vs=np.array(model['R'][0])+D@np.array(model['R'][1])@D.T
        clean_r=Y[:,:,0]-Y[:,:,1]@D.T
        clean_score=np.einsum('sti,ij,stj->st',clean_r,np.linalg.inv(Vs),clean_r)
        fa_clean=(clean_score[:,7:]>chi2.isf(.05/25,2)).any(axis=1)
        ex={}
        for enc in cfg['encodings']:
            MM=np.broadcast_to(np.eye(2),(128,32,2,2)) if enc=='identity' else data[name+'__M_target']
            for profile in cfg['attack_profiles']:
                stem=name+'__'+enc+'__'+profile;attack=data[stem+'__attack'];eta=np.einsum('stij,tj->sti',MM,attack)
                yja=Y[:,:,0]+eta
                score=data[stem+'__spatial_score'];a=score>chi2.isf(.05/25,2);a[:,:7]=False
                # Different gate implementation: scan last five alarms, not a countdown.
                phi=np.ones((128,32),np.int8)
                for k in range(32):phi[:,k]=~a[:,max(0,k-4):k+1].any(axis=1)
                detail={}
                for b in ('B0','B1','B2','B3'):
                    saved=data[stem+'__'+b+'__post'];masks=np.ones_like(phi) if b=='B0' else phi
                    assert np.array_equal(masks,data[stem+'__'+b+'__phi'])
                    own,own_detector,own_gate=closed_loop(runtime_model,modes,b,Y[:,:,1],yja,Y[:,:,2],cfg)
                    closederr=max(closederr,float(np.max(np.abs(own['post']-saved))))
                    assert np.array_equal(own_gate,phi) and np.array_equal(own['phi'],masks)
                    Kg=np.array([v['K'] for v in modes[b]]);Gg=np.array([v['G'] for v in modes[b]])
                    prev=np.zeros((128,2));calc=np.zeros_like(saved)
                    for k in range(32):
                        ix=masks[:,k];K=Kg[ix];G=Gg[ix]
                        J=np.eye(2)[None]-K@Hn[1]-G@(Hn[2][None]+ix[:,None,None]*Hn[0])
                        F=J@AR
                        calc[:,k]=np.einsum('sij,sj->si',F,prev)+np.einsum('sij,sj->si',K,Y[:,k,1])+np.einsum('sij,sj->si',G,Y[:,k,2]+ix[:,None]*yja[:,k])
                        prev=calc[:,k].copy()
                    afferr=max(afferr,float(np.max(np.abs(calc-saved))))
                    clean=data[name+'__normal__'+b+'__post'];d2=np.sum((saved-clean)**2,axis=2)
                    re=(masks[:,1:]==1)&(masks[:,:-1]==0)
                    active_re=np.zeros_like(masks,dtype=bool);active_re[:,1:]=re;active_re[:,:7]=False;active_re[:,17:]=False
                    detail[b]={'readmit_during_attack_runs':int(active_re.any(axis=1).sum()),
                        'dangerous_readmit_during_attack_runs':int((active_re&(d2>=.1)).any(axis=1).sum()),
                        'mean_dangerous_posterior_updates_in_event':float((d2[:,7:]>=.1).sum(axis=1).mean()),
                        'any_danger_in_event_count':int((d2[:,7:]>=.1).any(axis=1).sum()),
                        'max_squared_bias_after_stop_mean':float(d2[:,17:].max(axis=1).mean())}
                # Primary risk differences can occur ONLY on a clean-shadow false-alarm path.
                r1=data[stem+'__B1__risk'];r2=data[stem+'__B2__risk']
                primary_bound &= bool(np.all(~(r1!=r2)|fa_clean))
                ex[enc+'/'+profile]=detail
        allcert[name]=cert;extra[name]=ex
    assert afferr<1e-10 and trutherr<1e-11 and bayeserr<1e-12 and closederr<1e-12 and primary_bound
    exactFA=1-(1-.05/25)**25
    out={'date':'2026-11-15','mode_certificates':allcert,'common_quadratic_P':'I2','common_r':.5,
      'independent_affine_recomputation_error':afferr,'physical_state_measurement_identity_error':trutherr,
      'B2_rejection_information_form_identity_error':bayeserr,'primary_risk_difference_implies_clean_FA':primary_bound,
      'all_four_independent_closed_loop_reexecution_error':closederr,
      'raw_spatial_nominal_finite_event_FA_exact_under_iid_noise':exactFA,
      'new_physical_paths':0,'new_parameters_or_seed_changes':False,
      'extra_diagnostics':'readmission and continuing pollution already requested in registration; derived from saved arrays, no added runs',
      'readmission_diagnostics':extra,'input_hashes':{n:sha(H/n) for n in ('registration.json','analytic_preflight.json','screen_results.json','screen_arrays.npz')}}
    dump(H/'independent_math_audit.json',out)
    print(json.dumps({'affine_error':afferr,'state_error':trutherr,'information_gain_error':bayeserr,'mode_certificates':sum(map(len,allcert.values())),
      'exact_space_FA':exactFA,'primary_difference_only_clean_FA':primary_bound,
      'readmit_danger_counts':{n:{c:{b:v['dangerous_readmit_during_attack_runs'] for b,v in row.items()} for c,row in z.items()} for n,z in extra.items()}},ensure_ascii=False))
if __name__=='__main__':main()
