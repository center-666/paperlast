"""Minimal executable identity suite: two 32-step physical paths, zero MC.

Run python test_reference.py; all declared tests fail loudly on any mismatch.
Tests validate the original formula, not innovation or old 50-node results.
"""
from pathlib import Path
import argparse,json,hashlib,sys,importlib.metadata
import numpy as np
from scipy.linalg import block_diag
from common_g_reference import (load_model,aggregate_design,common_step,common_block_step,
 finite_autonomous_attack,draw_physical_path,original_watermarks,transmit_decode,
 run_common,bias_recursion,run_legacy_stacked,legacy_stacked_step,
 parity_model,parity_series,global_parity_map,nominal_scores)
H=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=H/'test_results.json');args=parser.parse_args()
    config={'T':32,'Ta':7,'L':10,'s':4,'sender':0,'receiver':1,'xi':[.25,-.15],
        'physical_seeds':{'unstable':1729,'stable':1730},'watermark_seeds':{'unstable':2718,'stable':2719},
        'lambda':.5,'q':.5,'total_energy_bound':10.,'danger_threshold_reference_only':.1,'event_FA_budget':.05,
        'monitoring_scope':'one declared target link, all end-times s..T-1; diagnostics only, NO gating'}
    T,Ta,L,s=32,7,10,4;sender,receiver=0,1;tests=[];cases={};arrays={}
    def check(name,value,tolerance=None):
        ok=bool(value) if tolerance is None else bool(np.isfinite(value) and value<=tolerance)
        tests.append({'test':name,'value':float(value) if tolerance is not None else bool(value),'tolerance':tolerance,'passed':ok})
        assert ok,(name,value,tolerance)
    for name in ('unstable','stable'):
        m=load_model(name);data=draw_physical_path(m,T,config['physical_seeds'][name]);y=data['y'];n=len(m.A)
        gainerr=dareerr=0.
        for i in range(5):
            C,V,P,B,Pc=aggregate_design(m,i)
            gainerr=max(gainerr,float(np.max(np.abs(B-np.hstack([m.K[i],m.G[i]])))))
            dareerr=max(dareerr,float(np.max(np.abs(P-(m.A@Pc@m.A.T+m.Q)))))
            J=m.J(i);noise=m.K[i]@m.R[i]@m.K[i].T+m.G[i]@sum((m.R[j] for j in m.neighbors[i]),np.zeros_like(m.R[i]))@m.G[i].T
            check(name+f'.node_{i}_Joseph_covariance_from_common_G',np.max(np.abs(Pc-(J@P@J.T+noise))),1e-11)
            check(name+f'.node_{i}_stationary_posterior_covariance',np.max(np.abs(Pc-(m.F(i)@Pc@m.F(i).T+J@m.Q@J.T+noise))),1e-11)
        check(name+'.supplementary_Niu_saved_gain_agreement',gainerr,1e-11)
        check(name+'.supplementary_DARE_residual',dareerr,1e-11)
        # Independent one-packet calculation, including a purely local update.
        prev=np.array([.3,-.2]);i=receiver;js=m.neighbors[i];neigh=[y[0,j] for j in js]
        prior,out=common_step(prev,y[0,i],neigh,m.A,m.H[i],[m.H[j] for j in js],m.K[i],m.G[i])
        block=common_block_step(prev,y[0,i],neigh,m.A,m.H[i],[m.H[j] for j in js],m.K[i],m.G[i])
        tied=np.hstack([m.K[i]]+[m.G[i] for _ in js]);Craw=np.vstack([m.H[i]]+[m.H[j] for j in js]);Yraw=np.concatenate([y[0,i]]+neigh)
        check(name+'.common_sum_equals_two_block',np.max(np.abs(out-block)),1e-13)
        check(name+'.common_sum_equals_tied_raw_blocks',np.max(np.abs(out-(prior+tied@(Yraw-Craw@prior)))),1e-13)
        _,local=common_step(prev,y[0,i],[],m.A,m.H[i],[],m.K[i],m.G[i])
        check(name+'.single_node_local_innovation',np.max(np.abs(local-(prior+m.K[i]@(y[0,i]-m.H[i]@prior)))),1e-13)
        # Equal-sum neighbor measurements: common-G invariant, free stack not.
        perturb=np.array([.4,-.3]);shift=[neigh[0]+perturb,neigh[1]-perturb]
        shifted=common_block_step(prev,y[0,i],shift,m.A,m.H[i],[m.H[j] for j in js],m.K[i],m.G[i])
        check(name+'.opposite_neighbor_changes_cancel_for_common_G',np.max(np.abs(shifted-out)),1e-13)
        _,old0,_,oldgain=legacy_stacked_step(m,i,prev,np.eye(n),y[0,i],neigh)
        _,old1,_,_=legacy_stacked_step(m,i,prev,np.eye(n),y[0,i],shift)
        oldexpected=(oldgain[:,:2]-oldgain[:,2:4])@perturb
        check(name+'.free_stack_equal_sum_changes_output',np.linalg.norm(old1-old0)>1e-4)
        check(name+'.free_stack_difference_explained_by_unequal_blocks',np.max(np.abs(old1-old0-oldexpected)),1e-13)
        # Match the SAME prior covariance, removing initialization as an excuse.
        Cagg,Vagg,Pmatched,Lagg,Pcmatched=aggregate_design(m,i)
        _,matched_old,_,matched_gain=legacy_stacked_step(m,i,prev,Pcmatched,y[0,i],neigh)
        _,matched_shift,_,_=legacy_stacked_step(m,i,prev,Pcmatched,y[0,i],shift)
        check(name+'.matched_prior_covariance_in_legacy',np.max(np.abs(m.A@Pcmatched@m.A.T+m.Q-Pmatched)),1e-11)
        check(name+'.matched_prior_common_from_Niu_gain',np.max(np.abs(out-(prior+Lagg@(np.r_[y[0,i],sum(neigh,np.zeros(2))]-Cagg@prior)))),1e-12)
        check(name+'.matched_prior_free_stack_still_not_common_G',np.linalg.norm(matched_old-out)>1e-5)
        check(name+'.matched_prior_unequal_neighbor_gain_blocks',np.linalg.norm(matched_gain[:,:2]-matched_gain[:,2:4])>1e-3)
        check(name+'.matched_prior_equal_sum_witness',np.max(np.abs(matched_shift-matched_old-(matched_gain[:,:2]-matched_gain[:,2:4])@perturb)),1e-12)
        attack,energy=finite_autonomous_attack(m,sender,Ta,L,np.array(config['xi']),T)
        check(name+'.original_attack_actual_energy_within_E10',energy<=10)
        check(name+'.finite_support_before_and_after',not np.any(attack[:Ta]) and not np.any(attack[Ta+L:]))
        U=np.broadcast_to(np.eye(2),(T,5,2,2)).copy()
        _,recv0=transmit_decode(m,y,U);clean=run_common(m,y,recv0)
        legacy=run_legacy_stacked(m,y,recv0)
        legacydifference=float(np.max(np.abs(clean['posterior']-legacy['posterior'])))
        check(name+'.same_physical_sample_common_vs_legacy_different',legacydifference>1e-4)
        gauss,beta=original_watermarks(m,T,config['watermark_seeds'][name],lam=.5,q=.5)
        case={}
        for scheme,M in [('identity',U),('Gaussian',gauss)]:
            encoded,healthy=transmit_decode(m,y,M)
            _,attacked=transmit_decode(m,y,M,attack,sender,receiver)
            nominal=run_common(m,y,healthy);compromised=run_common(m,y,attacked)
            eta=np.einsum('tij,tj->ti',M[:,sender],attack)
            theoretical_prior,theoretical_post=bias_recursion(m,receiver,eta)
            actualpost=compromised['posterior'][:,receiver]-nominal['posterior'][:,receiver]
            actualprior=compromised['prior'][:,receiver]-nominal['prior'][:,receiver]
            prefix=name+'.'+scheme
            check(prefix+'.ideal_encode_decode',np.max(np.abs(np.einsum('tnij,tnj->tni',M,encoded)-y)),1e-11)
            check(prefix+'.all_nodes_nominal_unchanged',np.max(np.abs(nominal['posterior']-clean['posterior'])),1e-10)
            check(prefix+'.coded_injection_decodes_to_Ma',np.max(np.abs(attacked[:,receiver,sender]-healthy[:,receiver,sender]-eta)),1e-11)
            check(prefix+'.paired_posterior_bias_recursion',np.max(np.abs(actualpost-theoretical_post)),1e-10)
            check(prefix+'.paired_prediction_bias_recursion',np.max(np.abs(actualprior-theoretical_prior)),1e-10)
            check(prefix+'.prediction_equals_A_previous_posterior',np.max(np.abs(actualprior[1:]-actualpost[:-1]@m.A.T)),1e-10)
            check(prefix+'.prediction_initial_offset_zero',np.max(np.abs(actualprior[0])),1e-13)
            other=[j for j in range(5) if j!=receiver]
            check(prefix+'.no_multi_hop_estimate_pollution',np.max(np.abs((compromised['posterior']-nominal['posterior'])[:,other])),1e-12)
            stop=Ta+L-1
            tail=max(float(np.max(np.abs(actualpost[k]-np.linalg.matrix_power(m.F(receiver),k-stop)@actualpost[stop]))) for k in range(stop+1,T))
            check(prefix+'.stopped_attack_natural_F_tail',tail,1e-10)
            O,Z,W,theta=parity_model(m.A,m.H[sender],m.Q,m.R[sender],s)
            r0=parity_series(healthy[:,receiver,sender],Z,s)
            r1=parity_series(attacked[:,receiver,sender],Z,s)
            deta=parity_series(eta,Z,s)
            check(prefix+'.actual_residual_shift_equals_parity_of_Ma',np.max(np.abs(r1-r0-deta)),1e-10)
            check(prefix+'.ZO_zero',np.max(np.abs(Z@O)),1e-13)
            scores,threshold,alarms=nominal_scores(r1,theta,.05)
            rows=[]
            for q,k in enumerate(range(s,T)):
                offsets=list(range(k-s,k+1));count=sum(Ta<=t<Ta+L for t in offsets)
                kind='clean' if count==0 else ('internal' if count==s+1 else ('onset' if k<Ta+s else 'stopping'))
                # Independent masked support map T_window, not assuming a full O block.
                Tw=np.vstack([m.H[sender]@np.linalg.matrix_power(m.A,t-Ta) if Ta<=t<Ta+L else np.zeros_like(m.H[sender]) for t in offsets])
                blockM=block_diag(*(M[t,sender] for t in offsets))
                mapshift=Z@blockM@Tw@np.array(config['xi'])
                check(prefix+f'.window_{k}_masked_support_identity',np.max(np.abs(deta[q]-mapshift)),1e-12)
                rows.append({'end':k,'kind':kind,'residual_bias_norm':float(np.linalg.norm(deta[q]))})
            if scheme=='identity':
                internal=[v['residual_bias_norm'] for v in rows if v['kind']=='internal']
                boundaries=[v['residual_bias_norm'] for v in rows if v['kind'] in ('onset','stopping')]
                check(prefix+'.complete_internal_windows_stealthy',max(internal),1e-12)
                check(prefix+'.onset_stopping_windows_nonzero_for_this_witness',min(boundaries)>1e-4)
            else:
                check(prefix+'.selected_secret_realization_breaks_internal_alignment',max(v['residual_bias_norm'] for v in rows if v['kind']=='internal')>1e-4)
            case[scheme]={'max_posterior_bias_norm_squared':float(np.max(np.sum(actualpost**2,axis=1))),
                'tail_absolute_identity_error':tail,'windows':rows,'diagnostic_alarms':int(alarms.sum()),
                'single_window_dimension':len(theta),'finite_monitoring_test_count':T-s,
                'Bonferroni_threshold':threshold,'not_an_empirical_false_alarm_or_power_estimate':True}
            for k,v in [('posterior_clean',nominal['posterior']),('posterior_attacked',compromised['posterior']),('prior_clean',nominal['prior']),('prior_attacked',compromised['prior']),('bias_theory',theoretical_post),('residual_clean',r0),('residual_attacked',r1),('decoded_effect',eta),('M',M)]:arrays[name+'__'+scheme+'__'+k]=v
        # Independent exact-coefficient covariance reconstruction, no sampling fit.
        O,Z,W,theta=parity_model(m.A,m.H[sender],m.Q,m.R[sender],s)
        maps,joint=global_parity_map(m.A,m.H[sender],m.Q,m.R[sender],T,s,Z)
        d=len(theta);cov_err=max(np.max(np.abs(joint[q*d:(q+1)*d,q*d:(q+1)*d]-theta)) for q in range(T-s))
        check(name+'.nominal_Theta_matches_global_noise_expansion',cov_err,1e-11)
        primitives=np.r_[data['x'][0],data['w'].reshape(-1),data['v'][:,sender].reshape(-1)]
        rdirect=parity_series(y[:,sender],Z,s)
        check(name+'.global_primitive_map_matches_actual_nominal_residual',np.max(np.abs(maps@primitives-rdirect.reshape(-1))),1e-10)
        off=float(np.linalg.norm(joint[:d,d:2*d]))
        check(name+'.overlapping_residuals_not_independent',off>1e-4)
        check(name+'.joint_covariance_symmetric',np.max(np.abs(joint-joint.T)),1e-12)
        cases[name]={'actual_attack_energy':energy,'max_saved_gain_difference':gainerr,'max_DARE_residual':dareerr,
            'max_common_vs_legacy_same_path_difference':legacydifference,
            'equal_sum_stack_difference_norm':float(np.linalg.norm(old1-old0)),
            'matched_prior_free_stack_vs_common_difference':float(np.linalg.norm(matched_old-out)),
            'nominal_theta_global_difference':float(cov_err),'adjacent_window_cross_cov_norm':off,
            'joint_covariance_rank':int(np.linalg.matrix_rank(joint,tol=1e-10)),
            'joint_covariance_dimension':len(joint),'scenarios':case}
        for k,v in data.items():arrays[name+'__physical__'+k]=v
        arrays[name+'__attack']=attack;arrays[name+'__legacy_stacked_posterior']=legacy['posterior']
    # Explicit dimension error: never silently substitute a heterogeneous stack.
    model=load_model('stable');model.H[2]=model.H[2][:1]
    failed=False
    try:model.validate()
    except ValueError:failed=True
    check('incompatible_neighbor_dimension_rejected',failed)
    model=load_model('stable')
    failed=False
    try:common_step(np.zeros(2),np.zeros(2),[np.zeros(2)],model.A,model.H[1],[],model.K[1],model.G[1])
    except ValueError:failed=True
    check('unequal_neighbor_lists_rejected_not_silently_zipped',failed)
    for label,res,alpha in [('empty_monitoring',np.zeros((0,2)),.05),('zero_alpha',np.zeros((1,2)),0.),('unit_alpha',np.zeros((1,2)),1.)]:
        failed=False
        try:nominal_scores(res,np.eye(2),alpha)
        except ValueError:failed=True
        check(label+'_rejected',failed)
    modelpath=H/'inputs/five_node_aggregate_model.json';raw=modelpath.read_bytes();modeldata=json.loads(raw)
    result={'date':'2026-11-15','baseline':'72a972ccd4c3dc5fb024ef1b178769073b10c422','config':config,
        'model_raw_sha256':hashlib.sha256(raw).hexdigest(),
        'model_canonical_json_sha256':hashlib.sha256(json.dumps(modeldata,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest(),
        'tests':tests,'passed':sum(t['passed'] for t in tests),'failed':sum(not t['passed'] for t in tests),
        'cases':cases,'python':sys.version.split()[0],'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','scipy')},
        'physical_realizations':2,'timesteps_per_realization':T,'Monte_Carlo':False,'old_simulation_modules_imported':False,
        'scope':'common-G fixed gains; ideal synchronized encoding; complete packets; no data-dependent gating',
        'not_original_50node_reproduction':True,'C_novelty_decisions_retained':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    print(json.dumps({'passed':result['passed'],'failed':result['failed'],'physical_paths':2,'MC':False,
      'models':{k:{x:v[x] for x in ('actual_attack_energy','max_common_vs_legacy_same_path_difference','nominal_theta_global_difference','adjacent_window_cross_cov_norm')} for k,v in cases.items()}},ensure_ascii=False))
if __name__=='__main__':main()
