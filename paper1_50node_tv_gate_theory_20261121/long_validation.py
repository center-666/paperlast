"""Independent paired 50-node validation: fixed schedule and endogenous gates.

Uses all 50 physical sensor streams. The exact no-estimate-sharing structure
allows recomputing only the five affected receivers under a changed gate.
No original source file or old artifact is overwritten.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import time
import numpy as np
from scipy.stats import beta as beta_distribution
from source_model import load_model,covariance_schedule,physical_paths
from replay import base_filter,scores,affected_filter,attack_watermarks,paired_forcing

HERE=Path(__file__).resolve().parent


def cp_interval(success,n,alpha=.05):
    return [float(beta_distribution.ppf(alpha/2,success,n-success+1)) if success else 0.,
            float(beta_distribution.ppf(1-alpha/2,success+1,n-success)) if success<n else 1.]


def calibrate(model,mi):
    maxima=[];maxima20=[];rates=[];startup=[]
    for b in range(8):
        physical=physical_paths(model,64,202612010000+1000*mi+b)
        sc=scores(model,physical,np.zeros((64,150,6,2)))
        maxima.extend(np.max(sc[:,4:],axis=(1,2)).tolist())
        maxima20.extend(np.max(sc[:,20:],axis=(1,2)).tolist())
        rates.extend(np.mean(sc[:,20:]>42,axis=(1,2)).tolist())
        startup.extend(np.min(sc[:,5],axis=1).tolist())
    cutoff=float(np.quantile(maxima,.95,method='higher'))
    return dict(paths=512,decision_times=[4,149],calibrated_pathmax_threshold=cutoff,
                legacy_threshold=42.,legacy_full_fwer=float(np.mean(np.array(maxima)>42.)),
                legacy_post20_fwer=float(np.mean(np.array(maxima20)>42.)),
                legacy_post20_pointwise_fpr=float(np.mean(rates)),
                calibrated_training_fwer=float(np.mean(np.array(maxima)>cutoff)),
                minimum_observed_k5_score=float(min(startup)),path_maxima=maxima,
                post20_maxima=maxima20)


def base_affected(model,base,schedule,sc):
    B=len(base);ids=model.receiver_ids
    shape=(B,150,len(ids),2,2)
    return SimpleNamespace(post=base[:,:,ids],P=np.broadcast_to(schedule.P[:,ids][None],shape),
                           star=np.broadcast_to(schedule.star[:,ids][None],shape),
                           F=np.broadcast_to(schedule.F[:,ids][None],shape),
                           gates=np.zeros_like(sc,dtype=bool),scores=sc)


def scalar_metrics(model,physical,base,world,healthy=None,pair=None):
    ids=model.receiver_ids
    baseerr=np.sum((base-physical.x[:,:,None])**2,axis=-1)
    afferr=np.sum((world.post-physical.x[:,:,None])**2,axis=-1)
    others=np.sum(baseerr,axis=2)-np.sum(baseerr[:,:,ids],axis=2)
    net=(others+np.sum(afferr,axis=2))/50
    affected=afferr.mean(axis=2)
    fields=dict(network_mse_all=net.mean(axis=1),network_mse_post20=net[:,20:].mean(axis=1),
          affected_mse_post20=affected[:,20:].mean(axis=1),
          attack_mse_network=net[:,50:110].mean(axis=1),
          attack_mse_affected=affected[:,50:110].mean(axis=1),
          after_support_mse_network=net[:,110:].mean(axis=1),
          after_buffer_recovery_mse_network=net[:,117:].mean(axis=1),
          source_mean_path_rmse_post20=np.sqrt(net[:,20:]).mean(axis=1),
          first_attacked_tick_mse_network=net[:,50])
    curves=dict(network_mse=net.sum(axis=0),affected_mse=affected.sum(axis=0),
                source_mean_path_rmse=np.sqrt(net).sum(axis=0))
    if healthy is not None:
        herror=healthy.post-physical.x[:,:,None]
        aerror=world.post-physical.x[:,:,None]
        b=world.post-healthy.post
        rhs=np.sum(b*b+2*herror*b,axis=(2,3))/50
        actual=(np.sum(aerror*aerror,axis=(2,3))-np.sum(herror*herror,axis=(2,3)))/50
        identity=float(np.max(abs(rhs-actual)))
        fields['paired_mse_increment_network_post20']=actual[:,20:].mean(axis=1)
        fields['displacement_mse_network_post20']=np.sum(b*b,axis=(2,3))[:,20:].mean(axis=1)/50
        fields['state_cross_term_network_post20']=(2*np.sum(herror*b,axis=(2,3)))[:,20:].mean(axis=1)/50
        curves['paired_displacement_mse']=np.sum(b*b,axis=(0,2,3))/50
        curves['paired_state_cross_term']=2*np.sum(herror*b,axis=(0,2,3))/50
        if pair is not None:
            curves['gain_mismatch_forcing_mse']=np.sum(pair.mismatch**2,axis=(0,2,3))/50
            curves['admitted_attack_forcing_mse']=np.sum(pair.forcing**2,axis=(0,2,3))/50
        return fields,curves,identity
    return fields,curves,0.


def detection_metrics(sc,threshold):
    alarms=sc>threshold
    anywindow=alarms[:,50:110];fullwindow=alarms[:,57:110]
    anyhit=np.any(anywindow,axis=1);fullhit=np.any(fullwindow,axis=1)
    delay_any=np.where(anyhit,np.argmax(anywindow,axis=1),150)
    delay_full=np.where(fullhit,np.argmax(fullwindow,axis=1)+7,150)
    return dict(tpr_full_window=fullwindow.mean(axis=(1,2)),first_alarm_delay=delay_any,
                first_full_window_alarm_delay=delay_full,
                alarm_any_full=np.any(alarms[:,4:],axis=(1,2)),
                alarm_any_post20=np.any(alarms[:,20:],axis=(1,2)),
                pointwise_alarm_rate_post20=alarms[:,20:].mean(axis=(1,2)),
                startup_all_links_alarm=np.all(alarms[:,5],axis=1))


def empirical_summary(arr):
    a=np.asarray(arr,float)
    if a.ndim!=1:return None
    return dict(mean=float(a.mean()),path_se=float(a.std(ddof=1)/np.sqrt(len(a))),paths=len(a))


def run_model(label,mi,design_entry):
    model=load_model(label);schedule=covariance_schedule(model)
    cal=calibrate(model,mi);thresholds={'legacy42':42.,'calibrated':cal['calibrated_pathmax_threshold']}
    print(label,'calibrated threshold',thresholds['calibrated'],flush=True)
    arrays={};curve_sums={};max_identity=0.;max_bias_identity=0.;max_riccati_violation=0.;max_coalescence_mismatch=0
    first_packet_max=0.;endpoints=[];diagnostics=[]
    def save(key,fields,curves):
        for field,value in fields.items():arrays.setdefault(key+'__'+field,[]).append(np.asarray(value))
        for field,value in curves.items():
            k=key+'__'+field
            curve_sums[k]=curve_sums.get(k,np.zeros_like(value))+value
    policies={'iid':0.,'negative':-1.315236976827179,'positive':1.315236976827179}
    selector_theta=design_entry['design']['conservative_proxy_policy']['theta']
    selector_alias=min(policies,key=lambda k:abs(policies[k]-selector_theta))
    assert abs(policies[selector_alias]-selector_theta)<1e-10
    for batch in range(16):
        started=time.perf_counter();B=64
        physical=physical_paths(model,B,202612020000+1000*mi+batch)
        base=base_filter(model,physical,schedule)
        zero=np.zeros((B,150,6,2));health_sc=scores(model,physical,zero)
        health={'wm_no_gate':base_affected(model,base,schedule,health_sc)}
        for mode,c in thresholds.items():health[mode]=affected_filter(model,physical,zero,health_sc,c,True)
        static_mask=np.ones_like(health_sc,dtype=bool);static_mask[:,0]=False
        health['static_known_candidates']=affected_filter(model,physical,zero,health_sc,42.,True,gate_override=static_mask)
        for mode,w in health.items():
            fields,curves,_=scalar_metrics(model,physical,base,w)
            if mode in thresholds:
                events=detection_metrics(health_sc,thresholds[mode])
                fields.update({k:v for k,v in events.items() if k in {'alarm_any_full','alarm_any_post20','pointwise_alarm_rate_post20','startup_all_links_alarm'}})
            save('healthy_'+mode,fields,curves)
        first_reference={};delta_first=None
        for pi,(policy,theta) in enumerate(policies.items()):
            delta,beta=attack_watermarks(model,B,theta,202612030000+1000*mi+batch,
                    202612040000+1000*mi+batch,202612050000+10000*mi+100*pi+batch)
            if delta_first is None:delta_first=delta[:,50].copy()
            first_packet_max=max(first_packet_max,float(np.max(abs(delta[:,50]-delta_first))))
            sc=scores(model,physical,delta)
            for mode in ('wm_no_gate','legacy42','calibrated'):
                gate=mode!='wm_no_gate';c=thresholds.get(mode,42.)
                w=affected_filter(model,physical,delta,sc,c,gate)
                pair=paired_forcing(model,physical,health[mode],w,delta)
                max_bias_identity=max(max_bias_identity,float(np.max(abs(pair.residual))))
                fields,curves,err=scalar_metrics(model,physical,base,w,health[mode],pair)
                max_identity=max(max_identity,err)
                fields.update(detection_metrics(sc,c))
                save(mode+'_'+policy,fields,curves)
                if mode not in first_reference:first_reference[mode]=w.post[:,50].copy()
                first_packet_max=max(first_packet_max,float(np.max(abs(w.post[:,50]-first_reference[mode]))))
                if gate:
                    mismatch=np.any(w.gates[:,117:]!=health[mode].gates[:,117:])
                    max_coalescence_mismatch=max(max_coalescence_mismatch,int(mismatch))
                    D=w.P-health[mode].P
                    # Spectral norm of symmetric 2x2 covariance difference.
                    tr=.5*(D[...,0,0]+D[...,1,1]);rad=np.sqrt((.5*(D[...,0,0]-D[...,1,1]))**2+D[...,0,1]**2)
                    norm=np.maximum(abs(tr-rad),abs(tr+rad))
                    disagreement=np.zeros((B,150,5),bool)
                    for e,(receiver,sender) in enumerate(model.links):
                        disagreement[:,:,model.receiver_index[int(receiver)]]|=w.gates[:,:,e]!=health[mode].gates[:,:,e]
                    v=norm[:,1:]-model.rbar**2*norm[:,:-1]-.5*disagreement[:,1:]
                    max_riccati_violation=max(max_riccati_violation,float(v.max()))
                    diagnostics.append(dict(batch=batch,mode=mode,policy=policy,
                        mean_gate_disagreement=float(disagreement[:,50:117].mean()),
                        max_cov_difference_after117=float(norm[:,117:].max()),
                        displacement_rms_116=float(np.sqrt(np.mean(pair.b[:,116]**2))),
                        displacement_rms_149=float(np.sqrt(np.mean(pair.b[:,149]**2)))))
        print(label,'batch',batch+1,'/16','seconds',round(time.perf_counter()-started,2),flush=True)
    arrays={k:np.concatenate(v,axis=0) for k,v in arrays.items()}
    np.savez_compressed(HERE/f'{label}_paired_paths.npz',**arrays)
    summaries={k:empirical_summary(v) for k,v in arrays.items() if np.asarray(v).ndim==1}
    for mode in thresholds:
        k='healthy_'+mode+'__alarm_any_full';success=int(arrays[k].sum())
        summaries[k]['binomial_95_interval']=cp_interval(success,len(arrays[k]))
    comparisons={}
    for mode in ('wm_no_gate','legacy42','calibrated'):
        for policy in ('negative','positive'):
            for metric in ('network_mse_post20','affected_mse_post20','attack_mse_network','after_support_mse_network','tpr_full_window'):
                d=arrays[f'{mode}_{policy}__{metric}']-arrays[f'{mode}_iid__{metric}']
                comparisons[f'{mode}_{policy}_minus_iid__{metric}']=empirical_summary(d)
    assert max_coalescence_mismatch==0
    assert first_packet_max<1e-9 and max_riccati_violation<1e-9
    assert max_bias_identity<1e-5 and max_identity<1e-6
    delay_summaries={}
    for mode in ('wm_no_gate','legacy42','calibrated'):
        for policy in policies:
            for name in ('first_alarm_delay','first_full_window_alarm_delay'):
                a=arrays[f'{mode}_{policy}__{name}'];detected=a<150
                delay_summaries[f'{mode}_{policy}__{name}']=dict(
                    per_link_detected_fraction=detected.mean(axis=0).tolist(),
                    per_link_conditional_mean=[float(a[:,e][detected[:,e]].mean()) if np.any(detected[:,e]) else None for e in range(6)])
    out=dict(model=label,estimator='recovered 50-node stacked KF, not paper common-G',
             paths=1024,calibration=cal,policy_thetas=policies,SSCD_selector_alias=selector_alias,
             summaries=summaries,paired_comparisons=comparisons,
             delay_summaries=delay_summaries,
             curves={k:(v/1024).tolist() for k,v in curve_sums.items()},
             identity_checks=dict(first_attacked_tick_max_difference_across_copulas=first_packet_max,
                 maximum_paired_bias_recursion_error=max_bias_identity,maximum_MSE_decomposition_error=max_identity,
                 maximum_Riccati_path_bound_violation=max_riccati_violation,post117_gate_disagreement=max_coalescence_mismatch),
             gate_path_diagnostics=diagnostics)
    (HERE/f'{label}_validation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(label,'validation complete',flush=True)


if __name__=='__main__':
    design=json.loads((HERE/'design_results.json').read_text())
    for i,row in enumerate(design['models']):run_model(row['model'],i,row)
