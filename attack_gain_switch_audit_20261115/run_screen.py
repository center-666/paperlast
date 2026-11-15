"""Frozen small screen of standard B0--B3 baselines, not a new algorithm.

No defense reads truth, attack labels/parameters, or clean target replicas.
Clean shadows and truth below are OFFLINE evaluation and identity checks only.
"""
from pathlib import Path
import sys,json,hashlib,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
import numpy as np
from scipy.stats import chi2

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def proportion(x):
    x=np.asarray(x,dtype=bool);n=len(x);k=int(x.sum());p=k/n;z=1.959963984540054
    c=(p+z*z/(2*n))/(1+z*z/n);r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'count':k,'n':n,'estimate':p,'Wilson95':[float(c-r),float(c+r)]}
def avg(x):
    x=np.asarray(x,float)
    return {'mean':float(x.mean()),'sample_SE':float(x.std(ddof=1)/np.sqrt(len(x))),'n':len(x)}
def gate_from_alarms(alarms,hold):
    N,T=alarms.shape;remaining=np.zeros(N,dtype=int);phi=np.ones_like(alarms,dtype=np.int8)
    for k in range(T):
        remaining=np.where(alarms[:,k],hold,remaining);phi[:,k]=(remaining==0)
        remaining=np.maximum(remaining-1,0)
    return phi
def scores_and_alarms(m,yi,yj,start,end,alpha,s):
    """Only available local and decoded target data; estimator-independent."""
    N,T,_=yi.shape;times=np.arange(start,end+1)
    D=m.H[0]@np.linalg.inv(m.H[1]);V=m.R[0]+D@m.R[1]@D.T
    spatial=yj-yi@D.T
    space_score=np.einsum('sti,ij,stj->st',spatial,np.linalg.inv(V),spatial)
    _,Z,_,theta=ref.parity_model(m.A,m.H[0],m.Q,m.R[0],s)
    rr=np.stack([ref.parity_series(v,Z,s) for v in yj])
    temp_score=np.zeros((N,T));temp_score[:,s:]=np.einsum('sti,ij,stj->st',rr,np.linalg.inv(theta),rr)
    th_s=float(chi2.isf(alpha/len(times),2));th_t=float(chi2.isf(alpha/len(times),len(theta)))
    a_s=np.zeros((N,T),bool);a_t=np.zeros_like(a_s)
    a_s[:,times]=space_score[:,times]>th_s;a_t[:,times]=temp_score[:,times]>th_t
    return {'spatial_score':space_score,'temporal_score':temp_score,'spatial_alarm':a_s,'temporal_alarm':a_t,
            'spatial_threshold':th_s,'temporal_threshold':th_t,'Z':Z,'theta':theta,'spatial_covariance':V}
def run_target(m,mode_rows,scheme,yi,yj,yh,phi):
    """Defense update: ONLY data, model, declared gains and observed gate."""
    N,T,n=yi.shape;prior=np.zeros((N,T,n));post=np.zeros_like(prior);prev=np.zeros((N,n))
    effective=np.ones_like(phi) if scheme=='B0' else phi
    modes=mode_rows[scheme]
    Ks=np.array([z['K'] for z in modes]);Gs=np.array([z['G'] for z in modes]);Js=np.array([z['J'] for z in modes]);Fs=np.array([z['F'] for z in modes])
    for k in range(T):
        mask=effective[:,k].astype(int);K=Ks[mask];G=Gs[mask]
        prior[:,k]=prev@m.A.T;local=yi[:,k]-prior[:,k]@m.H[1].T
        other=yh[:,k]-prior[:,k]@m.H[2].T
        if scheme=='B3':
            predicted=prior[:,k]@m.H[0].T
            replaced=mask[:,None]*yj[:,k]+(1-mask[:,None])*predicted
            target=replaced-predicted
        else:target=mask[:,None]*(yj[:,k]-prior[:,k]@m.H[0].T)
        post[:,k]=prior[:,k]+np.einsum('sij,sj->si',K,local)+np.einsum('sij,sj->si',G,other+target)
        prev=post[:,k].copy()
    return {'prior':prior,'post':post,'phi':effective,'K':Ks[effective],'G':Gs[effective],'J':Js[effective],'F':Fs[effective]}
def closed_loop(m,modes,scheme,yi,yj,yh,config):
    d=scores_and_alarms(m,yi,yj,config['monitor_start'],config['monitor_end_inclusive'],config['event_FA_budget_per_detector'],config['parity_order'])
    phi=gate_from_alarms(d['spatial_alarm'],config['rejection_hold_updates_including_alarm'])
    return run_target(m,modes,scheme,yi,yj,yh,phi),d,phi
def true_error_identity(m,run,data,eta):
    post=run['post'];N,T,n=post.shape;e=post-data['x'];error=0.
    for k in range(T):
        J,K,G,F=run['J'][:,k],run['K'][:,k],run['G'][:,k],run['F'][:,k];ph=run['phi'][:,k]
        if k==0:pred=-np.einsum('sij,sj->si',J,data['x'][:,0])
        else:pred=np.einsum('sij,sj->si',F,e[:,k-1])-np.einsum('sij,sj->si',J,data['w'][:,k-1])
        pred+=np.einsum('sij,sj->si',K,data['v'][:,k,1])+np.einsum('sij,sj->si',G,data['v'][:,k,2]+ph[:,None]*(data['v'][:,k,0]+eta[:,k]))
        error=max(error,float(np.max(np.abs(pred-e[:,k]))))
    return error
def paired_identity(m,actual,clean,yi,yj0,yh,eta):
    delta=actual['post']-clean['post'];err=wrong=0.;N,T,n=delta.shape
    for k in range(T):
        prev=np.zeros((N,n)) if k==0 else delta[:,k-1]
        Ka,Ga,Fa=actual['K'][:,k],actual['G'][:,k],actual['F'][:,k]
        K0,G0=clean['K'][:,k],clean['G'][:,k];pa=actual['phi'][:,k];p0=clean['phi'][:,k]
        pi0=clean['prior'][:,k];nui=yi[:,k]-pi0@m.H[1].T
        nuj=yj0[:,k]-pi0@m.H[0].T;nuh=yh[:,k]-pi0@m.H[2].T
        direct=np.einsum('sij,sj->si',Fa,prev)+np.einsum('sij,sj->si',Ga,pa[:,None]*eta[:,k])
        cross=np.einsum('sij,sj->si',Ka-K0,nui)+np.einsum('sij,sj->si',Ga-G0,nuh)
        cross+=np.einsum('sij,sj->si',Ga*pa[:,None,None]-G0*p0[:,None,None],nuj)
        err=max(err,float(np.max(np.abs(delta[:,k]-direct-cross))));wrong=max(wrong,float(np.max(np.abs(delta[:,k]-direct))))
    return err,wrong
def fixed_original_nodes(m,y):
    """Raw sharing: four unaffected nodes need no attacked-state feedback."""
    N,T,_,n=y.shape;post=np.zeros((N,T,5,n));prev=np.zeros((N,5,n))
    for k in range(T):
        for i in range(5):
            prior=prev[:,i]@m.A.T;nu=y[:,k,i]-prior@m.H[i].T
            neigh=sum((y[:,k,j]-prior@m.H[j].T for j in m.neighbors[i]),np.zeros_like(nu))
            post[:,k,i]=prior+nu@m.K[i].T+neigh@m.G[i].T
        prev=post[:,k].copy()
    return post
def first_true(a,start,end):
    times=np.arange(start,end+1);active=a[:,times];return np.where(active.any(axis=1),times[active.argmax(axis=1)],np.inf)
def eval_attack(m,run,clean,b0_clean,data,eta,det,config,other_loss):
    ta,L=config['attack_start'],config['attack_length'];tb=ta+L;T=config['T'];times=np.arange(ta,T)
    delta=run['post']-clean['post'];fixed=run['post']-b0_clean['post']
    d2=np.sum(delta**2,axis=2);f2=np.sum(fixed**2,axis=2)
    tauA=first_true(det['spatial_alarm'],ta,T-1);tauD=first_true(d2>=config['danger_threshold_posterior'],ta,T-1)
    tauF=first_true(f2>=config['danger_threshold_posterior'],ta,T-1)
    true_a=run['post']-data['x'];true_0=clean['post']-data['x']
    mse_a=np.sum(true_a**2,axis=2);mse_0=np.sum(true_0**2,axis=2)
    cross=2*np.einsum('sti,sti->st',true_0,delta)
    risk=proportion(tauD<tauA);riskF=proportion(tauF<tauA)
    exact_mse=float(np.max(np.abs(mse_a-mse_0-d2-cross)))
    readmit=(run['phi'][:,1:]==1)&(run['phi'][:,:-1]==0)
    re_values=[]
    for s,k in zip(*np.nonzero(readmit)):
        k+=1
        if k>=ta:re_values.extend(mse_a[s,k:min(T,k+3)].tolist())
    result={'physical_target_MSE':avg(mse_a[:,times].mean(axis=1)),
      'physical_network_MSE':avg(((mse_a+other_loss)/5)[:,times].mean(axis=1)),
      'cumulative_policy_shadow_bias_squared':avg(d2[:,times].sum(axis=1)),
      'cumulative_fixed_nominal_difference_squared':avg(f2[:,times].sum(axis=1)),
      'max_policy_shadow_bias_squared':avg(d2[:,times].max(axis=1)),
      'first_packet_danger_policy_shadow':proportion(d2[:,ta]>=config['danger_threshold_posterior']),
      'risk_policy_shadow_danger_before_alarm':risk,'risk_fixed_nominal_danger_before_alarm':riskF,
      'alarm_event':proportion(np.isfinite(tauA)),
      'Pd_spatial_during_attack':proportion(det['spatial_alarm'][:,ta:tb].any(axis=1)),
      'Pd_temporal_during_attack':proportion(det['temporal_alarm'][:,ta:tb].any(axis=1)),
      'Pd_temporal_whole_monitored_event':proportion(det['temporal_alarm'][:,ta:T].any(axis=1)),
      'spatial_delay_among_alarm_events_only':avg((tauA-ta)[np.isfinite(tauA)]) if np.isfinite(tauA).sum()>1 else None,
      'unreported_infinite_delays_count':int((~np.isfinite(tauA)).sum()),
      'reject_fraction_monitored':float(1-run['phi'][:,times].mean()),
      'stop_residual_policy_shadow':{str(k-tb):avg(d2[:,k]) for k in (tb-1,tb,tb+1,tb+4,T-1)},
      'readmission_event_count':int(readmit.sum()),'readmission_next3_physical_loss_pooled_mean':float(np.mean(re_values)) if re_values else None,
      'MSE_cross_term_identity_error':exact_mse,'MSE_increment_mean':float((mse_a-mse_0)[:,times].mean()),
      'policy_shadow_bias_mean':float(d2[:,times].mean()),'normal_error_cross_term_mean':float(cross[:,times].mean()),
      'not_a_global_closed_loop_P3_solution':True}
    return result,{'mse':mse_a[:,times].mean(axis=1),'cumulative_bias':d2[:,times].sum(axis=1),'risk':(tauD<tauA),'risk_fixed':(tauF<tauA),
      'delta':delta,'d2':d2,'loss':mse_a,'tauA':tauA,'tauD':tauD,'cross':cross}

def main():
    config=json.loads((H/'registration.json').read_text('utf8'));pre=json.loads((H/'analytic_preflight.json').read_text('utf8'))
    cert=json.loads((H/'attack_feasibility_certificate.json').read_text('utf8'))
    assert cert['analytic_preflight_sha256']==sha(H/'analytic_preflight.json')
    registered_hash=sha(H/'registration.json');N,T=config['samples_per_model'],config['T'];start=config['monitor_start'];end=T-1
    assert pre['registration_sha256']==registered_hash,'registered protocol changed after preflight'
    assert (T,config['attack_start'],config['attack_length'],start,end,config['rejection_hold_updates_including_alarm'],config['parity_order'])==(32,7,10,7,31,5,4)
    assert config['energy_budget_coding_domain']==10 and config['danger_threshold_posterior']==.1 and config['event_FA_budget_per_detector']==.05
    seeds={name:{'physical':[config['physical_seed_base']+100000*i+r for r in range(N)],
      'watermark':[config['watermark_seed_base']+100000*i+r for r in range(N)]} for i,name in enumerate(config['models'])}
    seedpath=H/'sampling_seed_map.json'
    if seedpath.exists():assert json.loads(seedpath.read_text('utf8'))==seeds
    else:dump(seedpath,seeds)
    output={};stored={};proof={'true_error_identity_max':0.,'paired_full_identity_max':0.,'paired_without_cross_terms_error_max':0.,
      'B3_vs_B1_identity_max':0.,'B2_shared_path_vs_independent_closed_max':0.,'all_accept_reference_identity_max':0.,
      'same_alarm_paths_across_schemes':True,'after_stop_mode_synchronization':True,'after_sync_tail_identity_max':0.,'MSE_cross_identity_max':0.}
    for name in config['models']:
        m=ref.load_model(name);modes=pre['models'][name]['mode_gains']
        paths=[ref.draw_physical_path(m,T,s) for s in seeds[name]['physical']]
        data={k:np.stack([z[k] for z in paths]) for k in paths[0]};y=data['y'];yi,yj0,yh=y[:,:,1],y[:,:,0],y[:,:,2]
        Ms=[]
        for s in seeds[name]['watermark']:Ms.append(ref.original_watermarks(m,T,s,config['lambda'],config['q'])[0][:,0])
        MG=np.stack(Ms);MI=np.broadcast_to(np.eye(2),(N,T,2,2)).copy()
        full=fixed_original_nodes(m,y);other_loss=(np.sum((full-data['x'][:,:,None,:])**2,axis=3).sum(axis=2)-np.sum((full[:,:,1]-data['x'])**2,axis=2))
        normaldet=scores_and_alarms(m,yi,yj0,start,end,.05,4)
        healthy_gate=gate_from_alarms(normaldet['spatial_alarm'],5)
        normal={};cleans={}
        for b in ('B0','B1','B2','B3'):
            clean=run_target(m,modes,b,yi,yj0,yh,healthy_gate);cleans[b]=clean
            loss=np.sum((clean['post']-data['x'])**2,axis=2)
            normal[b]={'target_MSE':avg(loss[:,start:].mean(axis=1)),
              'network_MSE':avg(((loss+other_loss)/5)[:,start:].mean(axis=1)),
              'reject_fraction_monitored':float(1-clean['phi'][:,start:].mean()),
              'spatial_event_FA':proportion(normaldet['spatial_alarm'][:,start:].any(axis=1)),
              'temporal_event_FA':proportion(normaldet['temporal_alarm'][:,start:].any(axis=1))}
            stored[name+'__normal__'+b+'__post']=clean['post']
        for r in range(2):
            U=np.broadcast_to(np.eye(2),(T,5,2,2)).copy();_,rec=ref.transmit_decode(m,y[r],U)
            expected=ref.run_common(m,y[r],rec)['posterior'][:,1]
            proof['all_accept_reference_identity_max']=max(proof['all_accept_reference_identity_max'],float(np.max(np.abs(expected-cleans['B0']['post'][r]))))
        proof['B3_vs_B1_identity_max']=max(proof['B3_vs_B1_identity_max'],float(np.max(np.abs(cleans['B3']['post']-cleans['B1']['post']))))
        rows={}
        for enc,M in [('identity',MI),('original_Gaussian',MG)]:
            encenergy=np.sum(np.linalg.solve(M,yj0[...,None])[...,0]**2,axis=2).mean()
            rawenergy=np.sum(yj0**2,axis=2).mean()
            for profile in ('ordinary','P1'):
                ap=pre['models'][name]['attacks'][enc][profile];xi=np.array(ap['xi'])
                attack,energy=ref.finite_autonomous_attack(m,0,7,10,xi,T);assert energy<=10+1e-12
                eta=np.einsum('stij,tj->sti',M,attack)
                # Validate the actual frozen sender/inject/decode order on two paths.
                for r in range(2):
                    MM=np.broadcast_to(np.eye(2),(T,5,2,2)).copy();MM[:,0]=M[r]
                    _,rec=ref.transmit_decode(m,y[r],MM,attack,0,1)
                    assert np.max(np.abs(rec[:,1,0]-(yj0[r]+eta[r])))<1e-11
                yja=yj0+eta;det=scores_and_alarms(m,yi,yja,start,end,.05,4);gate=gate_from_alarms(det['spatial_alarm'],5)
                scenario={};details={};actuals={}
                for b in ('B0','B1','B2','B3'):
                    run=run_target(m,modes,b,yi,yja,yh,gate);actuals[b]=run
                    if b=='B2':
                        own,dd,gg=closed_loop(m,modes,b,yi,yja,yh,config)
                        proof['same_alarm_paths_across_schemes'] &= bool(np.array_equal(gg,gate))
                        proof['B2_shared_path_vs_independent_closed_max']=max(proof['B2_shared_path_vs_independent_closed_max'],float(np.max(np.abs(own['post']-run['post']))))
                    proof['true_error_identity_max']=max(proof['true_error_identity_max'],true_error_identity(m,run,data,eta))
                    pe,pw=paired_identity(m,run,cleans[b],yi,yj0,yh,eta)
                    proof['paired_full_identity_max']=max(proof['paired_full_identity_max'],pe);proof['paired_without_cross_terms_error_max']=max(proof['paired_without_cross_terms_error_max'],pw)
                    metric,detail=eval_attack(m,run,cleans[b],cleans['B0'],data,eta,det,config,other_loss);scenario[b]=metric;details[b]=detail
                    proof['MSE_cross_identity_max']=max(proof['MSE_cross_identity_max'],metric['MSE_cross_term_identity_error'])
                    sync=7+10+5-1
                    proof['after_stop_mode_synchronization'] &= bool(np.array_equal(run['phi'][:,sync:],cleans[b]['phi'][:,sync:]))
                    for k in range(sync,T):
                        d=detail['delta'];prev=d[:,k-1]
                        tail=np.einsum('sij,sj->si',run['F'][:,k],prev)
                        proof['after_sync_tail_identity_max']=max(proof['after_sync_tail_identity_max'],float(np.max(np.abs(d[:,k]-tail))))
                    base=name+'__'+enc+'__'+profile+'__'+b
                    stored[base+'__post']=run['post'];stored[base+'__prior']=run['prior'];stored[base+'__phi']=run['phi']
                    for k in ('mse','cumulative_bias','risk','risk_fixed','tauA','tauD'):stored[base+'__'+k]=detail[k]
                proof['B3_vs_B1_identity_max']=max(proof['B3_vs_B1_identity_max'],float(np.max(np.abs(actuals['B3']['post']-actuals['B1']['post']))))
                diffs={}
                for key in ('mse','cumulative_bias','risk','risk_fixed'):
                    d=details['B2'][key].astype(float)-details['B1'][key].astype(float);diffs[key]=avg(d)
                # Against all-accept clean reference every algorithm is identical before the SAME first spatial alarm.
                for b in ('B1','B2','B3'):
                    assert np.array_equal(details['B0']['risk_fixed'],details[b]['risk_fixed'])
                tag=enc+'/'+profile
                rows[tag]={'attack':ap,'sample_coding_energy':energy,'normal_encoded_to_raw_signal_energy_ratio':float(encenergy/rawenergy),
                  'baselines':scenario,'B2_minus_B1_paired':diffs,
                  'same_secondary_pre_alarm_risk_all_baselines':True,
                  'policy_shadow_risk_difference_count_B2_vs_B1':int(np.sum(details['B2']['risk']!=details['B1']['risk'])),
                  'healthy_shadow_FA_count_upper_for_difference':int(normaldet['spatial_alarm'][:,start:].any(axis=1).sum()),
                  'spatial_threshold':det['spatial_threshold'],'temporal_threshold':det['temporal_threshold']}
                stored[name+'__'+enc+'__'+profile+'__spatial_score']=det['spatial_score'];stored[name+'__'+enc+'__'+profile+'__temporal_score']=det['temporal_score']
                stored[name+'__'+enc+'__'+profile+'__attack']=attack
        output[name]={'normal':normal,'cases':rows}
        for k,v in data.items():stored[name+'__physical__'+k]=v
        stored[name+'__M_target']=MG
    for k in ('true_error_identity_max','paired_full_identity_max','B3_vs_B1_identity_max','B2_shared_path_vs_independent_closed_max','all_accept_reference_identity_max','after_sync_tail_identity_max','MSE_cross_identity_max'):assert proof[k]<1e-10,(k,proof[k])
    assert proof['same_alarm_paths_across_schemes'] and proof['after_stop_mode_synchronization']
    assert proof['paired_without_cross_terms_error_max']>1e-4
    assert sha(H/'registration.json')==registered_hash
    result={'date':'2026-11-15','baseline':config['baseline'],'registration_sha256':registered_hash,'analytic_preflight_sha256':sha(H/'analytic_preflight.json'),
      'models':output,'identity_checks':proof,'unique_physical_paths':256,'samples_per_model':128,'T':32,
      'small_screen_not_original50_or_formal_performance_validation':True,'no_parameter_retuning':True,
      'defense_truth_or_attack_label_inputs':False,'closed_loop_P3_optimized':False}
    dump(H/'screen_results.json',result);np.savez_compressed(H/'screen_arrays.npz',**stored)
    print(json.dumps({'identities':proof,'normal':{n:{b:round(v['target_MSE']['mean'],6) for b,v in z['normal'].items()} for n,z in output.items()},
      'cases':{n:{k:{'risk_B1':v['baselines']['B1']['risk_policy_shadow_danger_before_alarm']['estimate'],'risk_B2':v['baselines']['B2']['risk_policy_shadow_danger_before_alarm']['estimate'],
      'risk_fixed':v['baselines']['B1']['risk_fixed_nominal_danger_before_alarm']['estimate'],'B2_B1_MSE_diff':v['B2_minus_B1_paired']['mse']['mean']} for k,v in z['cases'].items()} for n,z in output.items()}},ensure_ascii=False))
if __name__=='__main__':main()
